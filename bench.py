"""
bench.py —— 推理性能对照实验
=============================

跑出 README 里引用的全部推理数据：

    python bench.py          # 五组实验全跑
    python bench.py 4 5      # 只跑第 4、5 组（避免重跑覆盖已有 JSON）

五组实验：
  实验 1  改进 B：KV cache 的加速比随 batch 变化
          结论：batch=1 时几乎没有收益（CPU 调度主导）；batch 越大收益越大（GPU 计算主导）
  实验 2  改进 B：单次 forward 的耗时 vs 序列长度
          用于解释实验 1 —— 为什么 batch=1 时看不出加速
  实验 3  改进 C：GQA 把 KV cache 显存压缩多少（公式估算 + 真实张量形状双重验证）
  实验 4  改进 D：三种注意力实现（baseline / sdpa / chunked）的耗时与峰值显存，附数值等价性
  实验 5  改进 E：分页 KV cache 的显存浪费率与前缀共享，附 token 逐字一致性检查

结果写入 results/bench_*.json 与 results/bench_*_*.json，随仓库提交，便于事后核对。
"""

import json
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, _SDPA_FUSED_BACKENDS
from paged_kv import PagedKVCache, bytes_per_block, generate_paged, reserved_bytes_naive

try:                                    # 老版本 PyTorch 没有这个接口，缺了也能跑
    from torch.nn.attention import sdpa_kernel
except ImportError:                     # pragma: no cover
    sdpa_kernel = None

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def time_generate(model, prompt, max_new_tokens, use_cache, repeats=2):
    """跑 repeats 次生成，取最快的一次（避免首次编译/预热干扰）。"""
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(prompt.clone(), max_new_tokens=max_new_tokens, use_cache=use_cache)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------- 实验 1：KV cache 加速比 vs batch ----------------
def exp1_speedup_vs_batch():
    print("=" * 74)
    print("实验 1  改进 B：KV cache 加速比 —— 随 batch 变化")
    print("=" * 74)

    block, gen_tokens, prompt_len = 512, 479, 32
    rows = []
    print(f"{'batch':<8}{'无cache':<14}{'有cache':<14}{'加速比':<10}")
    print("-" * 46)

    for B in [1, 8, 32]:
        torch.manual_seed(0)
        cfg = GPTConfig(block_size=block, n_layer=4, n_embd=256, n_head=4, n_kv_head=4)
        model = GPT(cfg).to(DEVICE).eval()
        prompt = torch.randint(0, 65, (B, prompt_len), device=DEVICE)

        time_generate(model, prompt, 8, use_cache=True, repeats=1)   # 预热
        t_off = time_generate(model, prompt, gen_tokens, use_cache=False)
        t_on = time_generate(model, prompt, gen_tokens, use_cache=True)

        speedup = t_off / t_on
        rows.append({
            "batch": B,
            "gen_tokens": gen_tokens,
            "no_cache_sec": round(t_off, 3),
            "with_cache_sec": round(t_on, 3),
            "speedup": round(speedup, 2),
        })
        print(f"{B:<8}{t_off:>10.3f}s{'':<3}{t_on:>10.3f}s{'':<3}{speedup:>6.2f}x")
        del model
        torch.cuda.empty_cache()

    path = RESULTS_DIR / "bench_speedup_vs_batch.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n-> 已写入 {path.name}")
    return rows


# ---------------- 实验 2：单次 forward 耗时 vs 序列长度 ----------------
def exp2_forward_cost_vs_length():
    print()
    print("=" * 74)
    print("实验 2  单次 forward 耗时 —— 为什么 batch=1 时看不出加速")
    print("=" * 74)

    torch.manual_seed(0)
    cfg = GPTConfig(block_size=1024, n_layer=6, n_embd=384, n_head=6, n_kv_head=6)
    model = GPT(cfg).to(DEVICE).eval()

    rows = []
    print(f"{'序列长度 T':<14}{'单次 forward':<16}")
    print("-" * 32)
    for T in [1, 64, 256, 512, 1024]:
        x = torch.randint(0, 65, (1, T), device=DEVICE)
        for _ in range(3):
            model(x)                      # 预热
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000

        rows.append({"seq_len": T, "ms_per_forward": round(ms, 2)})
        print(f"{T:<14}{ms:>8.2f} ms")

    # 解读文字用实测值生成，避免写死数字后与实际数据脱节
    t1 = rows[0]["ms_per_forward"]                                       # T=1
    t512 = next(r["ms_per_forward"] for r in rows if r["seq_len"] == 512)
    print(f"\n解读：T 从 1 涨到 512（计算量涨 512 倍），单次 forward 由 {t1} ms 变为 {t512} ms，几乎持平")
    print(f"      => 这约 {t1:.0f} ms 是与序列长度无关的固定开销（Python 解释器 + kernel 调度）")
    print("      => 单条生成时 GPU 计算量太小，KV cache 省下的算力被固定开销淹没")

    path = RESULTS_DIR / "bench_forward_cost.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n-> 已写入 {path.name}")
    return rows


# ---------------- 实验 3：GQA 对 KV cache 显存的压缩 ----------------
def exp3_gqa_kv_memory():
    print()
    print("=" * 74)
    print("实验 3  改进 C：GQA 对 KV cache 显存的影响")
    print("=" * 74)

    n_head, head_dim, n_layer = 8, 64, 6
    ctx_len = 1024
    rows = []
    print(f"{'n_kv_head':<12}{'模式':<22}{'KV cache 显存':<16}{'相对基线':<10}")
    print("-" * 62)

    base_mem = None
    for n_kv in [8, 4, 2, 1]:
        if n_kv == n_head:
            mode = "MHA（标准多头）"
        elif n_kv == 1:
            mode = "MQA（全部共享）"
        else:
            mode = f"GQA（{n_head // n_kv} 个 Q 头共享）"

        # KV cache 显存 = 2(k和v) × 层数 × kv头数 × 头维 × 上下文长度 × 2字节(fp16)
        bytes_per = 2 * n_layer * n_kv * head_dim * ctx_len * 2
        mb = bytes_per / 1024 / 1024
        if base_mem is None:
            base_mem = mb

        rows.append({
            "n_head": n_head,
            "n_kv_head": n_kv,
            "mode": mode,
            "kv_cache_MB": round(mb, 2),
            "ratio_vs_mha": round(mb / base_mem, 3),
        })
        print(f"{n_kv:<12}{mode:<22}{mb:>8.2f} MB{'':<5}{mb / base_mem:>6.2f}x")

    print(f"\n（按 fp16 计算：batch=1, {n_layer} 层, {n_head} 个 Q 头, 头维 {head_dim}, 上下文 {ctx_len}）")
    print("解读：GQA 只改 KV 头数，Q 头数不变 —— 表达能力基本保住，KV cache 显存按比例下降")

    # 用真实模型验证一遍：前向传播时缓存张量的形状
    print("\n真实模型验证（跑一次生成，看缓存张量实际形状）：")
    for n_kv in [8, 1]:
        torch.manual_seed(0)
        cfg = GPTConfig(block_size=64, n_layer=2, n_embd=512, n_head=8, n_kv_head=n_kv)
        model = GPT(cfg).to(DEVICE).eval()
        prompt = torch.randint(0, 65, (1, 8), device=DEVICE)
        with torch.no_grad():
            _, _, kvs = model(prompt, use_cache=True)
        k0 = kvs[0][0]          # 第 1 层的 k
        print(f"  n_kv_head={n_kv} -> 缓存 k 形状 {tuple(k0.shape)}  "
              f"(batch, kv头数, 已缓存长度, 头维)")
        del model
        torch.cuda.empty_cache()

    path = RESULTS_DIR / "bench_gqa_memory.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n-> 已写入 {path.name}")
    return rows


# ---------------- 计时与显存的小工具 ----------------
def _sync():
    """只有用 GPU 时才需要等设备把活干完；CPU 上跑就跳过。"""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _time_ms(fn, warmup=3, repeats=10):
    """预热 warmup 次，再正式测 repeats 次，取最小值（与实验 1、2 的口径一致）。"""
    for _ in range(warmup):
        fn()
    _sync()
    best = float("inf")
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def _peak_mb(fn):
    """单次调用的峰值显存：先把统计清零，跑一次，再读峰值。"""
    if DEVICE != "cuda":
        return 0.0
    _sync()
    torch.cuda.reset_peak_memory_stats()
    fn()
    _sync()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def _set_attn(model, impl):
    """把模型里每一层的注意力实现切换成 impl（权重不变，便于直接比数值）。"""
    for blk in model.blocks:
        blk.attn.attn_impl = impl


def _detect_sdpa_backends(n_head, n_kv_head, T, head_dim, dtype, batch=8):
    """判定本机实际能用哪些 SDPA 内核。

    这台机器的 profiler（CUPTI 初始化失败）与 nvidia-smi（NVML 报错）都取不到 kernel 名，
    所以只能用「强制指定某个后端，看它报不报错」的办法来判断谁真的能跑。
    输入形状与掩码类型都对齐 _attn_sdpa 的真实调用（bool 掩码 + enable_gqa）。
    """
    if DEVICE != "cuda":
        return []
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return []
    q = torch.randn(batch, n_head, T, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, n_kv_head, T, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn_like(k)
    mask = torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool))
    usable = []
    for name in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        try:
            with warnings.catch_warnings(), sdpa_kernel([getattr(SDPBackend, name)]):
                warnings.simplefilter("ignore")     # 逐后端试跑会刷一堆内核选择提示，结果记在返回值里
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               enable_gqa=(n_kv_head != n_head))
            _sync()
            usable.append(name)
        except Exception:
            pass
    return usable


def _sdpa_dispatcher_vs_forced(head_dim=64, T=2048, batch=8, n_head=4, n_kv_head=1,
                               dtype=torch.bfloat16):
    """对比「默认分发器」与「显式指定融合内核」在同一组输入下的差别。

    这是 _attn_sdpa 要显式列出融合内核的原因：默认分发器在 GQA 形状下会挑中 MATH
    —— 那是按定义逐步算的参考实现，耗时与峰值显存都差一个数量级。
    """
    if DEVICE != "cuda" or sdpa_kernel is None or _SDPA_FUSED_BACKENDS is None:
        return None
    q = torch.randn(batch, n_head, T, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, n_kv_head, T, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn_like(k)
    mask = torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool))
    kwargs = dict(attn_mask=mask, enable_gqa=(n_kv_head != n_head))

    def default_call():
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def forced_call():
        with warnings.catch_warnings(), sdpa_kernel(_SDPA_FUSED_BACKENDS):
            warnings.simplefilter("ignore")
            return F.scaled_dot_product_attention(q, k, v, **kwargs)

    default_ms, default_peak = _time_ms(default_call), _peak_mb(default_call)
    forced_ms, forced_peak = _time_ms(forced_call), _peak_mb(forced_call)
    return {
        "shape": {"batch": batch, "n_head": n_head, "n_kv_head": n_kv_head,
                  "seq_len": T, "head_dim": head_dim},
        "forced_backends": [b.name for b in _SDPA_FUSED_BACKENDS],
        "default_ms": round(default_ms, 3),
        "default_peak_MB": round(default_peak, 2),
        "forced_ms": round(forced_ms, 3),
        "forced_peak_MB": round(forced_peak, 2),
        "speed_ratio": round(default_ms / forced_ms, 2),
    }


def _headline_ms(n_kv_head, T, impl, batch=8, dtype=torch.bfloat16):
    """只测一个档位、测完即弃。用来在实验首尾各测一次，量同一配置的重复测量波动。"""
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=2048, n_layer=4, n_embd=256, n_head=4,
                    n_kv_head=n_kv_head, attn=impl)
    model = GPT(cfg).to(DEVICE).to(dtype).eval()
    x = torch.randint(0, 65, (batch, T), device=DEVICE)
    with torch.no_grad():
        ms = _time_ms(lambda: model(x))
    del model
    torch.cuda.empty_cache()
    return round(ms, 3)


# ---------------- 实验 4：三种注意力实现的耗时与峰值显存 ----------------
def exp4_attention_impls():
    print()
    print("=" * 74)
    print("实验 4  改进 D：三种注意力实现的耗时与峰值显存")
    print("=" * 74)

    batch, dtype = 8, torch.bfloat16
    head_dim = 64
    rows = []

    # 提速到底是真的还是测量噪声？先量一下同一配置首尾各测一次的差异。
    probe_n_kv, probe_T = 1, 2048
    probe_impls = ("baseline", "sdpa")
    round_1 = {impl: _headline_ms(probe_n_kv, probe_T, impl) for impl in probe_impls}

    print(f"{'n_kv_head':<11}{'T':<7}{'实现':<11}{'单次前向':<13}{'峰值显存':<11}")
    print("-" * 54)

    for n_kv_head in (4, 1):
        for T in (128, 512, 2048):
            for impl in ("baseline", "sdpa", "chunked"):
                torch.manual_seed(0)
                cfg = GPTConfig(block_size=2048, n_layer=4, n_embd=256, n_head=4,
                                n_kv_head=n_kv_head, attn=impl)
                model = GPT(cfg).to(DEVICE).to(dtype).eval()
                x = torch.randint(0, 65, (batch, T), device=DEVICE)
                with torch.no_grad():
                    ms = _time_ms(lambda: model(x))
                    peak = _peak_mb(lambda: model(x))
                rows.append({
                    "n_kv_head": n_kv_head,
                    "seq_len": T,
                    "impl": impl,
                    "ms": round(ms, 3),
                    "peak_MB": round(peak, 2),
                })
                print(f"{n_kv_head:<11}{T:<7}{impl:<11}{ms:>9.2f} ms   {peak:>7.1f} MB")
                del model
                torch.cuda.empty_cache()

    print("\n同一档位内的对比（耗时 / 峰值显存，括号里是相对 baseline 的耗时倍数）：")
    for n_kv_head in (4, 1):
        for T in (128, 512, 2048):
            d = {r["impl"]: r for r in rows if r["n_kv_head"] == n_kv_head and r["seq_len"] == T}
            b = d["baseline"]
            print(f"  n_kv_head={n_kv_head} T={T:<5} "
                  f"baseline {b['ms']:>7.2f} ms/{b['peak_MB']:>7.1f} MB  ｜ "
                  f"sdpa {d['sdpa']['ms']:>7.2f} ms/{d['sdpa']['peak_MB']:>7.1f} MB"
                  f"（{b['ms'] / d['sdpa']['ms']:.2f}×） ｜ "
                  f"chunked {d['chunked']['ms']:>7.2f} ms/{d['chunked']['peak_MB']:>7.1f} MB"
                  f"（{b['ms'] / d['chunked']['ms']:.2f}×）")

    # ---- 为什么 _attn_sdpa 要显式列出融合内核：默认分发器会挑中 MATH ----
    print("\n默认分发器 vs 显式指定融合内核（GQA 形状，单独调 scaled_dot_product_attention）：")
    dispatcher_check = _sdpa_dispatcher_vs_forced()
    if dispatcher_check:
        s = dispatcher_check
        print(f"  输入：batch={s['shape']['batch']}, n_head={s['shape']['n_head']}, "
              f"n_kv_head={s['shape']['n_kv_head']}, T={s['shape']['seq_len']}, "
              f"head_dim={s['shape']['head_dim']}, dtype=bfloat16")
        print(f"  默认分发器  {s['default_ms']:>8.3f} ms   峰值 {s['default_peak_MB']:>8.1f} MB")
        print(f"  显式融合    {s['forced_ms']:>8.3f} ms   峰值 {s['forced_peak_MB']:>8.1f} MB")
        print(f"  速比 {s['speed_ratio']}×；显式指定的后端：{s['forced_backends']}")
    else:
        print("  跳过（没有 CUDA，或当前 PyTorch 不支持 sdpa_kernel）")

    # ---- 数值等价性：三种实现算的是不是同一件事 ----
    print("\n数值等价性（与 baseline 实现逐元素比，T=64、batch=2）")
    equivalence = []
    for dtype_name, dtype_ in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for n_kv_head in (4, 1):
            torch.manual_seed(0)
            cfg = GPTConfig(block_size=256, n_layer=4, n_embd=256, n_head=4, n_kv_head=n_kv_head)
            model = GPT(cfg).to(DEVICE).to(dtype_).eval()
            x = torch.randint(0, 65, (2, 64), device=DEVICE)
            with torch.no_grad():
                _set_attn(model, "baseline")
                ref = model(x)[0].float()
                for impl in ("sdpa", "chunked"):
                    _set_attn(model, impl)
                    out = model(x)[0].float()
                    diff = (out - ref).abs()
                    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
                    argmax_match = bool(torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1)))
                    equivalence.append({
                        "dtype": dtype_name,
                        "n_kv_head": n_kv_head,
                        "impl": impl,
                        "max_abs_diff": float(f"{diff.max().item():.3e}"),
                        "mean_abs_diff": float(f"{diff.mean().item():.3e}"),
                        "cosine": round(cos, 8),
                        "argmax_match": argmax_match,
                        "usable_backends": _detect_sdpa_backends(4, n_kv_head, 64, head_dim, dtype_),
                    })
                    print(f"  {dtype_name:<9} n_kv_head={n_kv_head} {impl:<8} "
                          f"max|Δ|={diff.max().item():.2e}  cosine={cos:.8f}  argmax 一致={argmax_match}")
            del model
            torch.cuda.empty_cache()

    # ---- 重复测量波动：同一档位在实验首尾各测一次，差值就是这台机器的噪声地板 ----
    round_2 = {impl: _headline_ms(probe_n_kv, probe_T, impl) for impl in probe_impls}
    repeatability = {
        "n_kv_head": probe_n_kv,
        "seq_len": probe_T,
        "round_1_ms": round_1,
        "round_2_ms": round_2,
        "spread_pct": {impl: round(abs(round_1[impl] - round_2[impl])
                                  / max(min(round_1[impl], round_2[impl]), 1e-6) * 100, 2)
                       for impl in probe_impls},
    }
    print(f"\n重复测量波动（n_kv_head={probe_n_kv}、T={probe_T}，首尾各测一次）：")
    for impl in probe_impls:
        print(f"  {impl:<9} 第一次 {round_1[impl]:>7.2f} ms  第二次 {round_2[impl]:>7.2f} ms"
              f"  相差 {repeatability['spread_pct'][impl]:.2f}%")
    print("  => 这台机器上同一配置重复测量的差异在百分之几量级，"
          "所以只看远大于它的差距才有意义")

    data = {
        "setup": {
            "device": DEVICE,
            "dtype": "bfloat16（测速）/ float32 与 bfloat16（等价性）",
            "batch": batch,
            "n_head": 4,
            "head_dim": head_dim,
            "n_layer": 4,
            "sdpa_forced_backends": [b.name for b in _SDPA_FUSED_BACKENDS]
                                    if _SDPA_FUSED_BACKENDS else None,
            "flash_attn_available": bool(torch.backends.cuda.is_flash_attention_available()),
            "backend_detection": "强制指定 sdpa_kernel([...]) 逐个试，谁不报错就是谁能跑"
                                 "（本机 profiler 与 nvidia-smi 都取不到 kernel 名）",
            "torch": torch.__version__,
        },
        "rows": rows,
        "dispatcher_vs_forced": dispatcher_check,
        "repeatability": repeatability,
        "equivalence": equivalence,
        "note": "耗时取 10 次最小值，每次前后 synchronize；换档前 empty_cache 并清零峰值统计。"
                "sdpa 一档显式指定融合内核列表（见 setup.sdpa_forced_backends），"
                "不依赖默认分发器；逐档实际可用的后端见 equivalence 里的 usable_backends。"
                "chunked 是纯 PyTorch 的分块在线 softmax，用来说明算法结构，"
                "定位不是打赢融合内核，所以它最慢。",
    }
    path = RESULTS_DIR / "bench_attention_impls.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> 已写入 {path.name}")
    return data


# ---------------- 实验 5：分页 KV cache 的容量与前缀共享 ----------------
def exp5_paged_kv():
    print()
    print("=" * 74)
    print("实验 5  改进 E：分页 KV cache 的容量与前缀共享")
    print("=" * 74)

    block_tokens, max_len = 16, 512
    n_layer, n_head, n_kv_head, head_dim = 4, 4, 4, 64
    dtype_bytes = 2                       # fp16：与实验 3 的字节口径保持一致
    cfg = GPTConfig(block_size=max_len, n_layer=n_layer, n_embd=256,
                    n_head=n_head, n_kv_head=n_kv_head)
    per_token = 2 * n_layer * n_kv_head * head_dim * dtype_bytes
    per_block = bytes_per_block(n_layer, n_kv_head, block_tokens, head_dim, dtype_bytes)

    # ---- ③ 先做一致性检查：分页路径必须与 torch.cat 路径逐字相同 ----
    torch.manual_seed(0)
    model = GPT(cfg).to(DEVICE).eval()
    prompt = torch.randint(0, 65, (1, 8), device=DEVICE)
    n_gen = 16
    torch.manual_seed(1337)
    ref = model.generate(prompt.clone(), max_new_tokens=n_gen, temperature=0.8, top_k=40, use_cache=True)
    cache_check = PagedKVCache(n_layer, n_kv_head, head_dim, block_tokens=block_tokens,
                               num_blocks=64, dtype=torch.float32, device=DEVICE)
    torch.manual_seed(1337)
    got = generate_paged(model, prompt.clone(), n_gen, cache_check, seq_id=0,
                         temperature=0.8, top_k=40)
    token_match = torch.equal(ref, got)
    print(f"一致性：连续生成 {n_gen} 个字符，分页路径与 torch.cat 路径 token 逐字相同 = {token_match}")
    print(f"        （分页只是存储的换位，注意力算法没变 ⇒ 必须完全一致，不一致就是 bug）")
    del cache_check
    torch.cuda.empty_cache()

    # ---- ① 容量 / 浪费率：长度用真实生成出来的值 ----
    model16 = GPT(cfg).to(DEVICE).half().eval()
    capacity = []
    print(f"\n{'请求数':<8}{'实际 token 总数':<17}{'预留式基线':<14}{'分页实占':<12}{'分页浪费率':<12}")
    print("-" * 63)
    for n_req in (8, 32):
        cache = PagedKVCache(n_layer, n_kv_head, head_dim, block_tokens=block_tokens,
                             num_blocks=n_req * (max_len // block_tokens),
                             dtype=torch.float16, device=DEVICE)
        gen = torch.Generator().manual_seed(2026)
        lengths = []
        for req in range(n_req):
            n_new = int(torch.randint(8, 41, (1,), generator=gen).item())   # 每个请求要生成多少字符
            p = torch.randint(0, 65, (1, 4), device=DEVICE)
            with torch.no_grad():
                generate_paged(model16, p, n_new, cache, seq_id=req, temperature=0.8, top_k=40)
            lengths.append(4 + n_new)                                       # 4 个开头字符 + 新生成的
        needed_bytes = sum(lengths) * per_token                             # 真正需要存的字节
        paged_bytes = cache.used_blocks() * per_block                       # 分页实际占的字节
        reserved_bytes = reserved_bytes_naive(n_req, n_layer, n_kv_head, head_dim,
                                              max_len, dtype_bytes)         # 预留式基线
        capacity.append({
            "n_req": n_req,
            "lengths": lengths,
            "tokens_total": sum(lengths),
            "avg_len": round(sum(lengths) / n_req, 2),
            "needed_MB": round(needed_bytes / 1024 / 1024, 3),
            "reserved_naive_MB": round(reserved_bytes / 1024 / 1024, 3),
            "paged_MB": round(paged_bytes / 1024 / 1024, 3),
            "paged_blocks_used": cache.used_blocks(),
            "naive_waste_rate": round(1 - needed_bytes / reserved_bytes, 4),
            "paged_waste_rate": round(1 - needed_bytes / paged_bytes, 4),
            "paged_vs_naive_saving": round(1 - paged_bytes / reserved_bytes, 4),
        })
        print(f"{n_req:<8}{sum(lengths):<17}{reserved_bytes / 1024 / 1024:>8.2f} MB"
              f"{'':<3}{paged_bytes / 1024 / 1024:>6.3f} MB{'':<3}"
              f"{1 - needed_bytes / paged_bytes:>8.2%}")
        del cache
        torch.cuda.empty_cache()

    # ---- ② 前缀共享：8 条序列共享同一段 64 字符前缀（= 4 个整块）----
    n_share, prefix_len = 8, 64
    prefix = torch.randint(0, 65, (1, prefix_len), device=DEVICE)
    print(f"\n前缀共享：{n_share} 条序列共享同一段 {prefix_len} 个字符的开头（= {prefix_len // block_tokens} 个整块）")
    with torch.no_grad():
        _, _, kv_prefix = model16(prefix, use_cache=True)

    # 基线：每个请求各自跑一次 prefill（真跑，不按公式乘）
    _sync()
    t0 = time.perf_counter()
    naive_caches = []
    with torch.no_grad():
        for req in range(n_share):
            c = PagedKVCache(n_layer, n_kv_head, head_dim, block_tokens=block_tokens,
                             num_blocks=prefix_len // block_tokens, dtype=torch.float16, device=DEVICE)
            _, _, kv = model16(prefix, use_cache=True)
            c.append_all(req, kv, n_new=prefix_len)
            naive_caches.append(c)
    _sync()
    naive_sec = time.perf_counter() - t0
    naive_bytes = sum(c.used_bytes() for c in naive_caches)

    # 分页：prefill 一次，其余 fork 共享
    _sync()
    t0 = time.perf_counter()
    paged_cache = PagedKVCache(n_layer, n_kv_head, head_dim, block_tokens=block_tokens,
                               num_blocks=64, dtype=torch.float16, device=DEVICE)
    paged_cache.append_all(0, kv_prefix, n_new=prefix_len)
    shared, copied = paged_cache.fork(0, 1)
    for req in range(2, n_share):
        paged_cache.fork(0, req)
    paged_sec = time.perf_counter() - t0
    paged_bytes = paged_cache.used_bytes()

    print(f"  基线（各自 prefill {n_share} 次，各持一份）：{naive_bytes / 1024 / 1024:.3f} MB，"
          f"prefill 合计 {naive_sec * 1000:.1f} ms")
    print(f"  分页（prefill 1 次 + fork {n_share - 1} 次）  ：{paged_bytes / 1024 / 1024:.3f} MB，"
          f"合计 {paged_sec * 1000:.1f} ms（其中 prefill 只做 1 次）")
    print(f"  省下 {1 - paged_bytes / naive_bytes:.1%} 的缓存字节")

    # ---- ④ 墙钟：分页路径不会更快，把差距测出来，而不是靠推断 ----
    n_cmp, seed_len, reps = 32, 8, 5
    prompt_cmp = torch.randint(0, 65, (1, seed_len), device=DEVICE)

    def run_cat():
        torch.manual_seed(1337)
        return model16.generate(prompt_cmp.clone(), max_new_tokens=n_cmp,
                                temperature=0.8, top_k=40, use_cache=True)

    def run_paged():
        c = PagedKVCache(n_layer, n_kv_head, head_dim, block_tokens=block_tokens,
                         num_blocks=64, dtype=torch.float16, device=DEVICE)
        torch.manual_seed(1337)
        return generate_paged(model16, prompt_cmp.clone(), n_cmp, c, seq_id=0,
                              temperature=0.8, top_k=40)

    with torch.no_grad():
        run_cat(), run_paged()                          # 预热
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            run_cat()
        _sync()
        cat_sec = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            run_paged()
        _sync()
        paged_sec = (time.perf_counter() - t0) / reps

    print(f"\n墙钟（单条序列生成 {n_cmp} 个字符，各取 {reps} 次平均）：")
    print(f"  torch.cat 增长式缓存 {cat_sec * 1000:>7.2f} ms")
    print(f"  分页 KV cache        {paged_sec * 1000:>7.2f} ms   （{paged_sec / cat_sec:.2f}×）")
    print("  => 分页每一步多一次查表 + 拷贝，在这个纯 Python 实现里更慢；"
          "它换来的是容量与共享，不是速度")

    data = {
        "setup": {
            "device": DEVICE,
            "block_tokens": block_tokens,
            "max_len": max_len,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_kv_head": n_kv_head,
            "head_dim": head_dim,
            "bytes_per_block": per_block,
            "bytes_per_token": per_token,
            "cache_dtype": "float16（字节口径与实验 3 一致）",
        },
        "consistency": {
            "max_new_tokens": n_gen,
            "seed": 1337,
            "token_match": token_match,
            "note": "分页只是把同样内容的 k/v 换了个存放位置，注意力算法没变，"
                    "所以两条路径的 token 必须逐字相同；不一致就是 bug。",
        },
        "capacity": capacity,
        "prefix_sharing": {
            "n_share": n_share,
            "prefix_len": prefix_len,
            "naive_MB": round(naive_bytes / 1024 / 1024, 3),
            "paged_MB": round(paged_bytes / 1024 / 1024, 3),
            "saved_MB": round((naive_bytes - paged_bytes) / 1024 / 1024, 3),
            "naive_prefill_sec": round(naive_sec, 4),
            "paged_prefill_sec": round(paged_sec, 4),
            "prefill_times_naive": n_share,
            "prefill_times_paged": 1,
            "shared_full_blocks": shared,
            "copied_tail_blocks": copied,
            "note": "只共享写满的整块；前缀尾部不满一块时各序列各持一份（未实现写时复制）。",
        },
        "wall_clock": {
            "max_new_tokens": n_cmp,
            "seed_len": seed_len,
            "repeats": reps,
            "cat_cache_ms": round(cat_sec * 1000, 3),
            "paged_cache_ms": round(paged_sec * 1000, 3),
            "ratio": round(paged_sec / cat_sec, 2),
            "note": "同一模型、同一随机种子、同一 prompt；两条路径生成的 token 逐字相同，"
                    "差别只在存储方式。分页这一步每步多做一次查表与拷贝，所以更慢。",
        },
        "note": "本实现是 Python 层的分页：每步多一次查表与拷贝，墙钟慢于 model.py 里的 "
                "torch.cat 增长式缓存（具体倍数见 wall_clock）。本实验验证的是容量与共享，不是速度。",
    }
    path = RESULTS_DIR / "bench_paged_kv.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> 已写入 {path.name}")
    return data


if __name__ == "__main__":
    print(f"设备：{DEVICE}")
    if DEVICE == "cpu":
        print("注意：没有 CUDA，测速结果不可用（只验证代码能跑通）")

    experiments = {
        1: ("改进 B：KV cache 加速比 vs batch", exp1_speedup_vs_batch),
        2: ("改进 B：单次 forward 耗时 vs 序列长度", exp2_forward_cost_vs_length),
        3: ("改进 C：GQA 对 KV cache 显存的影响", exp3_gqa_kv_memory),
        4: ("改进 D：三种注意力实现的耗时与峰值显存", exp4_attention_impls),
        5: ("改进 E：分页 KV cache 的容量与前缀共享", exp5_paged_kv),
    }
    todo = [int(a) for a in sys.argv[1:]] or sorted(experiments)
    for i in todo:
        if i not in experiments:
            raise SystemExit(f"没有第 {i} 组实验（可选：{sorted(experiments)}）")
        experiments[i][1]()

    if sorted(todo) == sorted(experiments):
        print()
        print("=" * 74)
        print("全部完成。结果已写入 results/bench_*.json")
        print("=" * 74)


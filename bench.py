"""
bench.py —— 推理性能对照实验
=============================

一条命令跑出 README 里引用的全部推理数据：

    python bench.py

三组实验：
  实验 1  改进 B：KV cache 的加速比随 batch 变化
          结论：batch=1 时几乎没有收益（CPU 调度主导）；batch 越大收益越大（GPU 计算主导）
  实验 2  改进 B：单次 forward 的耗时 vs 序列长度
          用于解释实验 1 —— 为什么 batch=1 时看不出加速
  实验 3  改进 C：GQA 把 KV cache 显存压缩多少（公式估算 + 真实张量形状双重验证）

结果写入 results/bench_*.json，随仓库提交，便于事后核对。
"""

import json
import sys
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig

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


if __name__ == "__main__":
    print(f"设备：{DEVICE}")
    if DEVICE == "cpu":
        print("注意：没有 CUDA，测速结果不可用（只验证代码能跑通）")
    exp1_speedup_vs_batch()
    exp2_forward_cost_vs_length()
    exp3_gqa_kv_memory()
    print()
    print("=" * 74)
    print("全部完成。结果已写入 results/bench_*.json")
    print("=" * 74)

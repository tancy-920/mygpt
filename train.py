# -*- coding: utf-8 -*-
"""
train.py —— 在 tinyshakespeare 上训练字符级 GPT
================================================

用法（在本目录下执行）：
    python train.py --run_name base                         # 基线：LayerNorm + 标准多头注意力
    python train.py --run_name rmsnorm --norm rmsnorm       # 改进 A：换 RMSNorm
    python train.py --run_name gqa --n_kv_head 1            # 改进 C：KV 头 4 → 1（GQA/MQA）

产出：
    out/<run_name>.pt                 模型权重 + 词表（体积较大，不进 git）
    results/<run_name>_metrics.json   本次实验的全部指标（参数量/耗时/显存/loss 曲线，进 git）

为什么把改进都做成命令行开关？
    对照实验只允许变更一个变量。开关化之后，基线与改进两组除该开关外完全一致，
    参数量、显存、loss 的差异才能干净地归因到这一项改动上，而不是靠「感觉更快」。
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig

BASE = Path(__file__).resolve().parent
TEXT_PATH = BASE / "data" / "tinyshakespeare" / "input.txt"
OUT_DIR = BASE / "out"            # 权重：每份约 13MB，用 .gitignore 挡掉
RESULTS_DIR = BASE / "results"    # 指标：几 KB，随仓库一起提交，供 README 引用


# ---------------- 数据 ----------------
def load_data():
    """整本书 → 字符表 → 整数张量 → 9:1 切分训练集/验证集。"""
    text = TEXT_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], chars


def get_batch(data, block_size, batch_size, device):
    """随机抓一批 (x, y)；y 是 x 向右错开一位 —— 每个位置都要预测它的下一个字符。"""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=50):
    """在给定数据集上取 eval_iters 个批次，返回平均 loss。"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss, _ = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def get_lr(step, max_steps, learning_rate, warmup):
    """学习率调度：先线性升温（warmup），再余弦式降温。"""
    if step < warmup:
        return learning_rate * (step + 1) / warmup
    if step > max_steps:
        return learning_rate * 0.1
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * ratio)))


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, default="base")
    ap.add_argument("--norm", type=str, default="layernorm", choices=["layernorm", "rmsnorm"])
    ap.add_argument("--n_kv_head", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = "cuda" in device

    train_data, val_data, chars = load_data()
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=len(chars),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        norm=args.norm,
        bias=True,                           # 固定 True：对照实验只改 norm 一个变量，结论才干净
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95))

    print("=" * 68)
    print(f"实验名: {args.run_name}")
    print(f"设备: {device}    归一化: {args.norm}    KV 头数: {args.n_kv_head}/{args.n_head}")
    print(f"参数量: {model.num_params():,}")
    print(f"训练步数: {args.max_steps}    批大小: {args.batch_size}    上下文: {args.block_size}")
    print("=" * 68)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    t0 = time.time()
    model.train()

    for step in range(args.max_steps + 1):
        lr = get_lr(step, args.max_steps, args.learning_rate, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % args.eval_interval == 0:
            train_loss = estimate_loss(model, train_data, args.block_size, args.batch_size, device)
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, device)
            history.append({"step": step, "train": round(train_loss, 4), "val": round(val_loss, 4)})
            print(f"step {step:5d}   train loss {train_loss:.4f}   val loss {val_loss:.4f}   lr {lr:.2e}")

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
        else:
            _, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    train_time = time.time() - t0

    # ---------------- 汇总指标 ----------------
    tokens_per_step = args.batch_size * args.block_size
    metrics = {
        "run_name": args.run_name,
        "norm": args.norm,
        "n_kv_head": args.n_kv_head,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "num_params": model.num_params(),
        "train_time_sec": round(train_time, 2),
        "ms_per_step": round(train_time / (args.max_steps + 1) * 1000, 2),
        "tokens_per_sec": round(tokens_per_step * (args.max_steps + 1) / train_time, 1),
        "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
        "final_train_loss": history[-1]["train"],
        "final_val_loss": history[-1]["val"],
        "history": history,
    }

    print("-" * 68)
    print(f"训练耗时 {metrics['train_time_sec']}s   {metrics['ms_per_step']} ms/step   "
          f"{metrics['tokens_per_sec']:,} tokens/s   峰值显存 {metrics['peak_mem_MB']} MB")
    print(f"最终 train {metrics['final_train_loss']}   最终 val {metrics['final_val_loss']}")
    print("-" * 68)

    # ---------------- 保存 ----------------
    OUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "chars": chars,
        },
        OUT_DIR / f"{args.run_name}.pt",
    )
    (RESULTS_DIR / f"{args.run_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存: out/{args.run_name}.pt   results/{args.run_name}_metrics.json")

    # ---------------- 采样一段文本作为参考 ----------------
    itos = {i: ch for i, ch in enumerate(chars)}
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
    print("\n--- 采样 200 个字符 ---")
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()

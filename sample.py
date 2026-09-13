# -*- coding: utf-8 -*-
"""
sample.py —— 加载训练好的权重，生成一段文本
============================================

用法（在本目录下执行）：
    python sample.py --run_name base                        # 用真实前缀续写
    python sample.py --run_name rmsnorm --gen_tokens 300 --temperature 0.7

参数说明：
    --gen_tokens   生成多少个字符
    --temperature  采样温度：越低越保守（趋近贪心），越高越随机
    --top_k        只从概率最高的 k 个字符里采样；设 0 表示不限制
    --from_newline 不带任何前缀，从换行符开始生成（默认用一个真实剧本片段开头）

注：推理性能（KV cache 加速比、GQA 显存）的对照实验统一在 bench.py 中完成，
    本文件只负责生成文本样本。
"""
import argparse
from pathlib import Path

import torch

from model import GPT

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "out"

PROMPT = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\n"


def load_model(run_name, device):
    """从 out/<run_name>.pt 加载模型权重与词表。"""
    ckpt = torch.load(OUT_DIR / f"{run_name}.pt", map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["chars"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, default="base")
    ap.add_argument("--gen_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--from_newline", action="store_true", help="从换行符开始生成，不带前缀")
    ap.add_argument("--use_cache", action="store_true", help="生成时启用 KV cache（结果文本相同）")
    ap.add_argument("--seed", type=int, default=1337, help="固定随机种子，保证同一命令可复现")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, chars = load_model(args.run_name, device)
    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in enumerate(chars)}

    if args.from_newline:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        prefix = ""
    else:
        ids = [stoi[ch] for ch in PROMPT if ch in stoi]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        prefix = "".join(chars[i] for i in ids)

    print("=" * 70)
    print(f"权重: out/{args.run_name}.pt    参数量 {model.num_params():,}    设备 {device}")
    print("=" * 70)

    n = min(args.gen_tokens, model.config.block_size - idx.size(1))
    out = model.generate(
        idx,
        max_new_tokens=n,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        use_cache=args.use_cache,
    )
    text = "".join(itos[i] for i in out[0].tolist())

    print(f"\n--- temperature={args.temperature}  top_k={args.top_k} "
          f"use_cache={args.use_cache} ---")
    print(text if args.from_newline else text[len(prefix):])


if __name__ == "__main__":
    main()

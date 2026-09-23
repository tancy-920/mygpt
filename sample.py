# -*- coding: utf-8 -*-
"""
sample.py —— 加载训练好的权重，生成一段文本
============================================

用法（在本目录下执行）：
    python sample.py --prompt "Shakespeare"                 # 自己给一个开头，让模型往后续写
    python sample.py --run_name rmsnorm --gen_tokens 300 --temperature 0.7
    python sample.py --from_newline                         # 不带开头，直接从头生成

参数说明：
    --prompt       开头文字。模型会把它当作已有的上文，往后接着写
    --gen_tokens   想生成多少个字符（受上下文长度上限约束，见下方注 2）
    --temperature  采样温度：越低越保守（趋近贪心），越高越随机
    --top_k        只从概率最高的 k 个字符里采样；设 0 表示不限制
    --from_newline 不带任何前缀，从换行符开始生成（默认用一个真实剧本片段开头）

注 1：本模型是字符级的，词表里只有 65 个字符（英文字母大小写、数字、常用标点、空格、换行）。
      开头里若出现词表之外的字符（如中文），会被忽略，并在运行时提示。
注 2：上下文上限为 block_size（默认 128）。开头 + 生成部分合计不超过这个数，
      所以开头越长，能续写的长度越短。

注 3：推理性能（KV cache 加速比、GQA 显存）的对照实验统一在 bench.py 中完成，
      本文件只负责生成文本样本。

注 4：--temperature 设 0 表示贪心 —— 每一步都取分数最高的那个字符，结果完全确定。
      这个值不能取负数。
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
    ap.add_argument("--prompt", type=str, default=None,
                    help="开头文字；不给则用内置的剧本片段")
    ap.add_argument("--gen_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="采样温度：越低越保守；设 0 表示贪心（每步取分数最高的字符）")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--from_newline", action="store_true", help="从换行符开始生成，不带前缀")
    ap.add_argument("--use_cache", action="store_true", help="生成时启用 KV cache（结果文本相同）")
    ap.add_argument("--seed", type=int, default=1337, help="固定随机种子，保证同一命令可复现")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.temperature < 0:
        raise SystemExit("--temperature 不能为负数：设 0 表示贪心，或者给一个正数。")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, chars = load_model(args.run_name, device)
    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in enumerate(chars)}

    if args.from_newline:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        prefix = ""
    else:
        text_in = args.prompt if args.prompt is not None else PROMPT
        dropped = sorted({ch for ch in text_in if ch not in stoi})
        if dropped:
            print(f"提示：以下字符不在词表内，已忽略 —— {''.join(dropped)!r}")
        ids = [stoi[ch] for ch in text_in if ch in stoi]
        if not ids:
            raise SystemExit("开头里没有任何词表内的字符，无法生成。请改用英文字母、数字或标点。")
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        prefix = "".join(chars[i] for i in ids)

    # 开头比上下文上限还长时，只保留最后 block_size 个字符（与 generate 内部的处理一致）
    idx = idx[:, -model.config.block_size:]
    prefix = prefix[-idx.size(1):]

    print("=" * 70)
    print(f"权重: out/{args.run_name}.pt    参数量 {model.num_params():,}    设备 {device}")
    print(f"开头: {prefix!r}")
    print("=" * 70)

    n = max(0, min(args.gen_tokens, model.config.block_size - idx.size(1)))
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
    print(text)


if __name__ == "__main__":
    main()

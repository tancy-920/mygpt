# -*- coding: utf-8 -*-
"""
model.py —— 约 3.2M 参数的字符级 GPT，含三项可开关的推理优化
==============================================================

三项改进均由配置项或调用参数控制，以便逐项做「改前 / 改后」对照实验：

    GPTConfig.norm       "layernorm" | "rmsnorm"   改进 A：RMSNorm 替换 LayerNorm
    generate(use_cache=) True | False              改进 B：KV cache 增量解码
    GPTConfig.n_kv_head  < n_head 即启用           改进 C：GQA（多个 Q 头共享一组 KV 头）

文件结构（自上而下 6 块）：
    GPTConfig            超参数
    LayerNorm / RMSNorm  两种归一化，改进 A 在此二选一
    CausalSelfAttention  因果自注意力，改进 B / C 都落在这个类里
    MLP                  前馈网络
    Block                一层 = 注意力 + MLP，各带残差与归一化
    GPT                  整体：嵌入 → N 层 Block → 输出层

命令行入口：
    train.py   训练 + 记录指标
    sample.py  加载权重生成文本
    bench.py   推理性能对照（KV cache 加速比 / GQA 显存）
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# 配置
# ============================================================
@dataclass
class GPTConfig:
    """模型超参数集中在此处。"""

    block_size: int = 128      # 上下文长度：一次最多看 128 个字符
    vocab_size: int = 65       # 字符种类数（tinyshakespeare 为 65 种）
    n_layer: int = 4           # 堆叠多少个 Block
    n_head: int = 4            # 注意力划分为几个 Query 头
    n_kv_head: int = 4         # 改进 C：KV 头数；等于 n_head 即标准多头注意力，小于 n_head 即 GQA
    n_embd: int = 256          # 每个字符用多少维向量表示
    dropout: float = 0.0       # 小数据集设 0（更快）
    bias: bool = True          # 线性层是否带 bias
    norm: str = "layernorm"    # 改进 A 的开关："layernorm" 或 "rmsnorm"


# ============================================================
# 归一化
# ============================================================
class LayerNorm(nn.Module):
    """逐位置归一化：减均值 → 除标准差 → 缩放 + 平移。

    「减均值」与「平移」这最后一步是 RMSNorm 省掉的，两者对照看即知差异：
        mean = x.mean(-1, keepdim=True)            # RMSNorm 不算均值
        var  = x.var(-1, keepdim=True)
        x    = (x - mean) / sqrt(var)
        x    = x * weight + bias                   # RMSNorm 没有 bias
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))                  # 可学习缩放
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # 可学习平移

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class RMSNorm(nn.Module):
    """RMSNorm（Zhang & Sennrich, 2019；Llama 系列采用的归一化方式）。

    相对 LayerNorm 只少了两件事：
        1. 不减均值（省掉一次 mean 计算）
        2. 没有 bias（少一组参数）
    代价是表达力略降，实践中几乎无损，且参数量与显存占用都更小。
    """

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)   # 均方根
        return x / rms * self.weight


def make_norm(config, ndim):
    """按配置构造归一化层；改进 A 的开关在此处落地。"""
    if config.norm == "rmsnorm":
        return RMSNorm(ndim)
    return LayerNorm(ndim, bias=config.bias)


# ============================================================
# 因果自注意力
# ============================================================
class CausalSelfAttention(nn.Module):
    """因果自注意力：每个位置回顾它之前的所有位置，做一次加权平均。

    q / k / v 三个投影的分工：
        q（query）当前位置在找什么信息
        k（key）  各位置有哪些特征可供匹配
        v（value）各位置能提供的具体内容
    q·k 越大代表越相关，经 softmax 归一化成权重后，用权重对 v 加权求和。

    「因果」体现在掩码上：位置 i 只能看到 j <= i，训练时不允许提前看到答案。
    改进 B（KV cache）与改进 C（GQA）都在这一个类里。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 要能被 n_head 整除"
        assert config.n_head % config.n_kv_head == 0, "n_head 要能被 n_kv_head 整除（GQA 要求）"

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head       # 改进 C：KV 头数
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # 三个投影。nanoGPT 上游把 qkv 合并为单个 c_attn，这里拆开是为了支持 GQA（k/v 的输出维度可以更小）
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 下三角掩码：第 i 行第 j 列 = 1 表示「位置 i 可以看位置 j」，仅对角线及左侧为 1
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        """
        x       : (B, T, C)
        past_kv : 改进 B 用 —— 历史缓存 (past_k, past_v)，形状 (B, n_kv_head, 已缓存长度, head_dim)
        use_cache: 是否把本次算出的 k/v 也一并返回
        返回     : (输出 y (B,T,C), 新的 kv 缓存 或 None)
        """
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)      # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)   # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # ---- 改进 B：KV cache ----
        # 生成时每步只喂 1 个新字符，但注意力需要「前面所有字符」的 k/v。
        # 无缓存：每次重算前面全部；有缓存：直接接上历史，只算新字符。
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_kv = (k, v) if use_cache else None

        # ---- 改进 C：GQA ----
        # k/v 头数少于 q 头数时，把每个 k/v 头复制若干份，各自服务一组 q 头。
        # 效果：KV cache 的显存占用按 n_kv_head / n_head 缩小。
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        Tq, Tk = q.size(2), k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))   # (B, nh, Tq, Tk)
        # 取掩码的对应切片：增量解码时 Tq=1，该行掩码天然全为 1，无需屏蔽任何位置
        att = att.masked_fill(self.mask[:, :, Tk - Tq : Tk, :Tk] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                                          # (B, nh, Tq, hd)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.resid_dropout(self.c_proj(y)), new_kv


# ============================================================
# 前馈网络
# ============================================================
class MLP(nn.Module):
    """前馈网络：每个位置独立过一遍，位置之间不交换信息。维度 n_embd → 4·n_embd → n_embd。"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ============================================================
# 一层 Block
# ============================================================
class Block(nn.Module):
    """一层 = 先自注意力（每个位置汇总前文）+ 再 MLP（各自加工）。

    `x = x + ...` 是残差连接：把输入原样加回输出，使梯度沿这条路径直接回传，
    深层网络才训得动。
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = make_norm(config, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config, config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


# ============================================================
# 完整模型
# ============================================================
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # 字符编号 → 向量
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # 位置编号 → 向量
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # 参考 nanoGPT：对参与残差相加的投影层，按层数缩小初始化标准差，训练更稳定
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 权重共享（weight tying）：输入嵌入与输出投影共用同一张表，减少参数量
        self.lm_head.weight = self.wte.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        """参数量（权重共享只算一次）。"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False, pos_offset=0):
        """
        idx       : (B, T) 字符编号
        targets   : (B, T) 正确答案；训练时提供，生成时不提供
        past_kvs  : 改进 B —— 每一层各自的 KV 缓存组成的列表
        pos_offset: 改进 B —— 增量解码时，新字符的位置编号从该偏移量往后数
        返回       : (logits (B,T,vocab), loss 或 None, 新的 past_kvs 或 None)
        """
        B, T = idx.size()
        pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))            # 字符向量 + 位置向量

        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_kvs

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """所有参数交给 AdamW；仅对二维及以上的张量做 weight_decay（沿用常见做法）。"""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    # --------------------------------------------------------
    # 生成
    # --------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
        """自回归生成：每一步追加一个字符。

        use_cache=False —— 基线写法：每追加一个字符，都要把整个前缀重新计算一遍
        use_cache=True  —— 改进 B：只在第一步计算整个前缀，之后每步只算新字符，历史 k/v 从缓存读取
        """
        self.eval()
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _, _ = self(idx_cond)
                idx = self._sample_and_append(idx, logits[:, -1, :], temperature, top_k)
            return idx

        past_kvs = None
        pos_offset = 0
        idx_cond = idx[:, -self.config.block_size :]     # 第一步：把已有前缀整个喂进去
        for _ in range(max_new_tokens):
            logits, _, past_kvs = self(
                idx_cond, past_kvs=past_kvs, use_cache=True, pos_offset=pos_offset
            )
            pos_offset += idx_cond.size(1)
            idx = self._sample_and_append(idx, logits[:, -1, :], temperature, top_k)
            idx_cond = idx[:, -1:]                        # 之后每步只喂刚生成的那个字符
        return idx

    @staticmethod
    def _sample_and_append(idx, logits, temperature, top_k):
        """从最后一组分数中采样一个字符，并追加到序列末尾。

        temperature 约定为正数。设为 0 时按贪心处理：直接取分数最大的那个字符。
        贪心单独走一条分支，是为了避开 softmax 与 multinomial —— 除以 0 会得到 ±inf，
        过完 softmax 是一整片 NaN，而 NaN 送进采样内核会让 CUDA 内核报错退出。
        另外 argmax 的结果不受正数缩放影响，所以这一支也无需先做除法。
        """
        if temperature <= 0:
            idx_next = logits.argmax(dim=-1, keepdim=True)
            return torch.cat((idx, idx_next), dim=1)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return torch.cat((idx, idx_next), dim=1)

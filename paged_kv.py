# -*- coding: utf-8 -*-
"""
paged_kv.py —— 改进 E：分页 KV cache
=====================================

把每个序列的 k/v 按固定长度切成「块」，存进一块预先分配好的显存里；序列自己只保留一张
「第 n 块 → 池内物理块号」的对照表。与 model.py 里那套逐字符拼接的缓存相比，三种做法是：

    增长式（model.py 的默认路径）  每步重新分配一整段连续显存，并把全部历史拷进去
    预留式（对照基线）             每个请求一开始就按 max_len 预留整段连续 buffer，用不完也占着
    分页（本文件）                 按需分配 block_tokens 大小的块，只有最后一块可能有内部碎片

**分页不会让生成变快**：Python 层每步多一次查表与拷贝，实测墙钟是增长式的 1.10 倍（见
bench.py 实验 5 的 wall_clock）。它换来的是**容量**——显存浪费率大幅下降；而且多条共享同一段
前缀的序列可以共享整块物理显存。

本文件与 model.py 解耦：只用 GPT 的公开接口（past_kvs / use_cache / pos_offset），
所以 model.py 不需要为改进 E 做任何改动，默认路径也不受影响。

用法：

    import torch
    from model import GPT, GPTConfig
    from paged_kv import PagedKVCache, generate_paged

    model = GPT(GPTConfig()).eval()
    cache = PagedKVCache(n_layer=4, n_kv_head=4, head_dim=64, dtype=torch.float32)
    idx = torch.tensor([[1, 2, 3, 4]])
    out = generate_paged(model, idx, max_new_tokens=50, cache=cache, seq_id=0)
"""
import math

import torch

from model import GPT


def bytes_per_block(n_layer, n_kv_head, block_tokens, head_dim, dtype_bytes):
    """一个物理块的字节数：k 和 v 各占一份。可按下面的例子自行核对。

    例：2 × 4 层 × 4 个 KV 头 × 16 个 token × 64 维 × 2 字节(fp16) = 65,536 字节 = 64 KiB
    """
    return 2 * n_layer * n_kv_head * block_tokens * head_dim * dtype_bytes


def reserved_bytes_naive(n_req, n_layer, n_kv_head, head_dim, max_len, dtype_bytes):
    """预留式基线的字节数：每个请求按 max_len 预留一整段连续 buffer，与实际用掉多少无关。

    例：8 个请求 × 2 × 4 层 × 4 个 KV 头 × 512 长 × 64 维 × 2 字节(fp16) = 16,777,216 字节 = 16 MiB
    """
    return n_req * 2 * n_layer * n_kv_head * max_len * head_dim * dtype_bytes


class PagedKVCache:
    """分页 KV cache：一块显存池 + 每序列一张块表。

    参数：
        n_layer / n_kv_head / head_dim 要与模型一致
        block_tokens  每块存多少个 token（16 表示 16 个）
        num_blocks    池子里一共多少块（预先分配，用完即报错，不再向系统要）
        dtype         必须与模型参数的数据类型一致，否则拼接时会因类型不符报错
    """

    def __init__(self, n_layer, n_kv_head, head_dim, block_tokens=16, num_blocks=256,
                 dtype=torch.float32, device=None):
        self.n_layer = n_layer
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.block_tokens = block_tokens
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        shape = (num_blocks, n_layer, n_kv_head, block_tokens, head_dim)
        self.k_pool = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=self.device)

        self.ref = [0] * num_blocks            # 每块的引用计数（前缀共享时 > 1）
        self.block_len = [0] * num_blocks      # 每块实际写了多少个 token
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.tables = {}                       # seq_id -> [物理块号, ...]，按顺序排列

    # ---------------- 池子的分配与回收 ----------------
    def _alloc(self):
        if not self.free_blocks:
            raise RuntimeError(
                f"分页缓存池已用尽（{len(self.ref)} 块）。调大 num_blocks 或减小 block_tokens 后重试。"
            )
        bid = self.free_blocks.pop()
        self.ref[bid] = 1
        self.block_len[bid] = 0
        return bid

    def _retain(self, bid):
        self.ref[bid] += 1

    def _release(self, bid):
        self.ref[bid] -= 1
        if self.ref[bid] == 0:
            self.block_len[bid] = 0
            self.free_blocks.append(bid)

    # ---------------- 与 GPT 对接的两个接口 ----------------
    def gather_all(self, seq_id):
        """把该序列的 k/v 按顺序拼回连续张量，交给 GPT 当 past_kvs 用。

        返回 [(k, v), ...]，每层一个 (1, n_kv_head, 已用长度, head_dim)；
        序列还没有任何内容时返回 None（与「没有缓存」是同一种语义）。
        """
        table = self.tables.get(seq_id)
        if not table:
            return None
        out = []
        for layer in range(self.n_layer):
            ks, vs = [], []
            for bid in table:
                n = self.block_len[bid]
                ks.append(self.k_pool[bid, layer, :, :n, :])
                vs.append(self.v_pool[bid, layer, :, :n, :])
            k = torch.cat(ks, dim=1).unsqueeze(0)          # (1, n_kv_head, 已用长度, head_dim)
            v = torch.cat(vs, dim=1).unsqueeze(0)
            out.append((k, v))
        return out

    def append_all(self, seq_id, new_kvs, n_new):
        """把上一步新算出的 k/v 写进池子。

        new_kvs 是 GPT 返回的逐层缓存，其中 k/v 是整个序列（历史 + 本次），
        所以只取最后 n_new 个位置写进块里，避免把历史重复存一遍。

        注意：一个物理块存的是**所有层**在同一段 token 上的 k/v（池子的形状里 n_layer 是
        块内的一维），所以分配块的动作必须放在最外层、按 token 数走，不能按层各分配一次。
        """
        table = self.tables.setdefault(seq_id, [])
        new_k = [k[:, :, -n_new:, :] for k, _ in new_kvs]      # 逐层取出「本次新增」的部分
        new_v = [v[:, :, -n_new:, :] for _, v in new_kvs]

        done = 0
        while done < n_new:
            if not table or self.block_len[table[-1]] == self.block_tokens:
                table.append(self._alloc())
            bid = table[-1]
            # 只有「最后一块且没写满」才会续写；能续写的块必然不被共享
            assert self.ref[bid] == 1, "末尾块被共享时不能续写，说明前缀共享的边界处理有误"
            start = self.block_len[bid]
            take = min(self.block_tokens - start, n_new - done)
            for layer in range(self.n_layer):
                self.k_pool[bid, layer, :, start:start + take, :] = new_k[layer][0, :, done:done + take, :]
                self.v_pool[bid, layer, :, start:start + take, :] = new_v[layer][0, :, done:done + take, :]
            self.block_len[bid] = start + take
            done += take

    # ---------------- 前缀共享 ----------------
    def fork(self, src_id, dst_id):
        """把 src 已经写满的整块共享给 dst（引用计数 +1），尾部不满一块的部分复制一份。

        生产实现会用写时复制（copy-on-write）处理尾部；这里用「整块共享 + 尾部各持一份」的
        等价简化：每个序列最多多占一块的显存，换来实现简单、语义好核对。
        """
        src_table = self.tables[src_id]
        n_full = sum(1 for bid in src_table if self.block_len[bid] == self.block_tokens)
        full, tail = src_table[:n_full], src_table[n_full:]

        new_table = list(full)
        for bid in full:
            self._retain(bid)
        for bid in tail:                      # 尾部：复制到一块新的私有块
            nb = self._alloc()
            self.k_pool[nb] = self.k_pool[bid]
            self.v_pool[nb] = self.v_pool[bid]
            self.block_len[nb] = self.block_len[bid]
            new_table.append(nb)
        self.tables[dst_id] = new_table
        return len(full), len(tail)

    def free(self, seq_id):
        for bid in self.tables.pop(seq_id, []):
            self._release(bid)

    # ---------------- 容量统计 ----------------
    def used_blocks(self):
        return sum(1 for r in self.ref if r > 0)

    def used_bytes(self):
        return self.used_blocks() * bytes_per_block(
            self.n_layer, self.n_kv_head, self.block_tokens, self.head_dim, self.dtype.itemsize
        )

    def used_tokens(self):
        return sum(self.block_len[bid] for bid in self.ref_blocks())

    def ref_blocks(self):
        return [bid for bid in range(len(self.ref)) if self.ref[bid] > 0]

    def stats(self):
        return {
            "block_tokens": self.block_tokens,
            "blocks_total": len(self.ref),
            "blocks_used": self.used_blocks(),
            "used_bytes": self.used_bytes(),
            "used_mb": round(self.used_bytes() / 1024 / 1024, 3),
        }


@torch.no_grad()
def generate_paged(model, idx, max_new_tokens, cache, seq_id=0, temperature=1.0, top_k=None):
    """用分页 KV cache 生成，逐步逻辑与 GPT.generate(use_cache=True) 完全一致。

    只调用 GPT 的公开接口，因此模型文件不需要任何改动。
    """
    model.eval()
    idx_cond = idx[:, -model.config.block_size :]
    pos = 0
    for _ in range(max_new_tokens):
        past = cache.gather_all(seq_id)                                  # 历史从池子里取
        logits, _, new_kvs = model(idx_cond, past_kvs=past, use_cache=True, pos_offset=pos)
        cache.append_all(seq_id, new_kvs, n_new=idx_cond.size(1))        # 新的写进池子
        pos += idx_cond.size(1)
        idx = GPT._sample_and_append(idx, logits[:, -1, :], temperature, top_k)
        idx_cond = idx[:, -1:]
    return idx


def blocks_needed(n_tokens, block_tokens):
    """n_tokens 个 token 需要多少块（向上取整）。"""
    return math.ceil(n_tokens / block_tokens)

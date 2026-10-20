# mygpt

A character-level GPT (3.2M parameters) used as a controlled testbed for measuring the effect of five common training/inference optimizations. Each optimization is implemented as an independent switch: the baseline and the modified run differ by exactly one flag, so any difference between them can be attributed to that single change. Every number in this document is backed by raw measurements committed under `results/*.json`.

## Results at a glance

| # | Optimization | Measured effect |
|---|---|---|
| A | LayerNorm → RMSNorm | Parameters −2,304; peak training memory 365.7 → 329.6 MB (−10%); step time unchanged |
| B | KV cache | Generation 30.5× faster at batch 32; 0.98× (no gain) at batch 1 |
| C | GQA / MQA | Per-sequence KV cache 12.0 → 6.0 → 3.0 → 1.5 MB (−87.5% for MQA) |
| D | Fused attention (SDPA) | 3.68× faster at T=2048; peak memory 607.6 → 143.6 MB |
| E | Paged KV cache | Reserved-memory waste 93.8% → 16.45%; shared prefix 2.0 → 0.25 MB |

## Purpose

The primary goal is to quantify, through controlled before/after experiments, how much each of five widely used optimizations actually improves a Transformer model, and at what cost. The emphasis is on the measurements and their interpretation rather than on building a language model or a training stack.

A secondary goal is understanding the Transformer architecture itself: causal self-attention, multi-head attention, the MLP, residual connections, normalization, and positional embeddings.

## Model and setup

```
input "F" "i" "r" "s" "t"  →  token ids (B, T)
  │
  ├─ wte  token embedding     (65 × 256)
  └─ wpe  position embedding  (128 × 256)   →  summed to (B, T, 256)
  │
  ▼   repeated 4× (each Block)
  ├─ normalization              ← A switch: LayerNorm / RMSNorm
  ├─ causal self-attention      ← B (KV cache) / C (GQA) / D (attention path)
  ├─ residual add
  ├─ normalization
  ├─ MLP  256 → 1024 → 256
  └─ residual add
  │
  ▼
  final normalization → Linear(256 → 65)  →  scores over 65 characters
                                             (weights shared with wte)
```

Default configuration: 4 layers / 4 attention heads / 256 embedding dims / context 128 / vocab 65 (character-level). The model is trained on TinyShakespeare (≈1.1 MB).

### Parameter breakdown

| Module | Computation | Parameters |
|---|---|---|
| `wte` token embedding (shared with output) | 65 × 256 | 16,640 |
| `wpe` position embedding | 128 × 256 | 32,768 |
| Per layer · attention (q/k/v/c_proj) | 4 × (256×256 + 256) | 263,168 |
| Per layer · MLP | (256×1024 + 1024) + (1024×256 + 256) | 525,568 |
| Per layer · normalization ×2 | 2 × (256 + 256) | 1,024 |
| **4 blocks total** | | **3,159,040** |
| Final normalization `ln_f` | 256 + 256 | 512 |
| **Total** | | **3,208,960** |

With RMSNorm, each of the 9 normalization layers drops one bias vector → −9 × 256 = −2,304, giving 3,206,656, matching the measurement.

## Training comparison

All runs share the same configuration — 3,000 steps, batch 32, context 128, 4 layers / 4 heads / 256 dims — on a single RTX 5060 Laptop 8 GB, PyTorch 2.11.0+cu128, bf16 automatic mixed precision. Only the stated switch differs.

| run | norm | n_kv_head | params | train loss | val loss | train time | ms/step | peak mem |
|---|---|---|---|---|---|---|---|---|
| `base` | LayerNorm | 4 | 3,208,960 | 1.1612 | 1.5134 | 74.2 s | 24.73 | 365.7 MB |
| `rmsnorm` | **RMSNorm** | 4 | **3,206,656** | 1.1628 | 1.5104 | 74.0 s | 24.66 | **329.6 MB** |
| `gqa2` | LayerNorm | **2** | 2,945,792 | 1.1512 | 1.4994 | 83.6 s | 27.84 | 361.2 MB |
| `gqa1` | LayerNorm | **1** | 2,814,208 | 1.1614 | 1.4970 | 82.4 s | 27.44 | 359.3 MB |

A uniform random guess over 65 characters would give ln(65) ≈ 4.174; all four runs are far below this. Train and validation loss decrease together with no divergence, so 3,000 steps is well short of overfitting.

## The five optimizations

### A — RMSNorm (replacing LayerNorm)

LayerNorm subtracts the mean, divides by the standard deviation, and applies a learned scale and shift. RMSNorm ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467), used by the Llama family) drops the mean subtraction and the bias, keeping only the root-mean-square scaling:

```python
rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
return x / rms * weight
```

**Measured**: parameters −2,304; peak training memory 365.7 → 329.6 MB (−10%); step time essentially unchanged (24.73 → 24.66 ms); val loss 1.5134 → 1.5104 (a difference of 0.003, within noise, and therefore not claimed as a quality improvement).

Step time does not change because, at this scale, each step is dominated by Python and kernel-launch overhead rather than tensor arithmetic. The measurable gain is memory.

### B — KV cache (incremental decoding)

Without a cache, every generation step recomputes the entire prefix, so the cost grows with the square of the generated length. Caching each layer's k/v reduces each step to a single new position:

```
no cache: prefix → recompute all T positions → sample 1 token     cost ∝ T²
cache:    prefix → cache k/v, then 1 position per step → sample   cost ∝ T
```

Correctness is verified before speed: with a fixed random seed, cached and uncached generation produce byte-identical text. A cache that changes the output is a bug, not an optimization; this check is part of the repository (`sample.py --use_cache`).

| batch | no cache | KV cache | speedup |
|---|---|---|---|
| 1 | 2.055 s | 2.103 s | 0.98× |
| 8 | 3.566 s | 2.008 s | 1.78× |
| **32** | 88.93 s | **2.912 s** | **30.5×** |

(479 generated tokens, context 512, 4 layers / 256 dims.)

This is the plainest implementation (a `torch.cat` per step, no pre-allocated buffer, no paging); production implementations are faster still.

### C — GQA / MQA (grouped-query attention)

Standard multi-head attention (MHA) gives each query head its own key/value head, and KV cache size scales with the number of KV heads. GQA/MQA let several query heads share one key/value head.

| mode | Q heads | KV heads | sharing |
|---|---|---|---|
| MHA | 8 | 8 | one K/V per Q head |
| **GQA** | 8 | 4 / 2 | 2 or 4 Q heads share one K/V |
| MQA | 8 | 1 | all Q heads share one K/V |

KV cache size scales with the KV-head count:

| n_kv_head | mode | KV cache | vs MHA |
|---|---|---|---|
| 8 | MHA | 12.0 MB | 1.0× |
| 4 | GQA | 6.0 MB | 0.5× |
| 2 | GQA | 3.0 MB | 0.25× |
| 1 | MQA | 1.5 MB | 0.125× |

The figures are also verified against the actual cached tensor shapes — the KV-head dimension is reduced structurally, so the memory reduction is not merely an estimate. On this small model GQA also happens to lower validation loss (1.4970 vs 1.5134); this is a small-model artifact and must not be read as "GQA is generally better" — at real scale, GQA trades a small, controlled quality loss for memory and bandwidth. Training speed is essentially unchanged, consistent with the original paper: GQA's benefit is on the inference side, not the training side.

### D — Fused attention (SDPA)

The baseline materializes the full Tq × Tk score matrix before softmax; at T=2048, batch 8, 4 heads this matrix alone is `8 × 4 × 2048 × 2048`. `F.scaled_dot_product_attention` (SDPA) fuses the computation so the full matrix is never written out.

Measured on a full forward pass (batch 8, bf16, 4 layers / 4 heads / 256 dims):

| n_kv_head | impl | T=128 | T=512 | T=2048 | peak mem (T=2048) |
|---|---|---|---|---|---|
| 4 | baseline (full matrix) | 3.73 ms | 4.63 ms | 53.91 ms | 604.3 MB |
| 4 | **fused (SDPA)** | 3.40 ms | 3.19 ms | 19.28 ms | 144.3 MB |
| 4 | chunked (online softmax) | 6.26 ms | 14.31 ms | 131.66 ms | 185.3 MB |
| 1 | baseline (full matrix) | 3.33 ms | 4.45 ms | 51.09 ms | 607.6 MB |
| 1 | **fused (SDPA)** | 3.09 ms | 3.03 ms | **13.88 ms** | **143.6 MB** |
| 1 | chunked (online softmax) | 6.12 ms | 13.65 ms | 129.37 ms | 188.6 MB |

At T=2048 the fused path is 3.68× faster (n_kv_head=1) with peak memory 607.6 → 143.6 MB. The gap grows with sequence length; at T=128 both paths are in the fixed-overhead regime and show no difference.

Three points worth noting:

- This is PyTorch's fused entry point, not the FlashAttention kernel itself. On this machine the backend that actually runs is cuDNN (`FLASH_ATTENTION` is not built, `is_flash_attention_available()` returns `False`), and the speedup holds only in bf16 — in fp32 the fused backends are mostly unavailable for grouped-query shapes.
- The default dispatcher silently selects `MATH` for grouped-query shapes (21.73 ms / 1257.3 MB); pinning the fused backend explicitly gives 1.292 ms / 41.25 MB (16.8×). The code therefore pins the backend list rather than relying on the dispatcher.
- A third path, `chunked` (tiling + online softmax in plain PyTorch), cuts memory (604 → 185 MB) but is slower than the baseline (53.91 → 131.66 ms). This demonstrates that the gain comes from fusing the whole computation into one kernel, not from tiling alone.

Numerical equivalence (T=64, batch 2, vs baseline): max |Δ| ≈ 6e-07 in fp32 and ≈ 1.2e-02 in bf16, argmax identical in all cases. Back-to-back runs of the same configuration vary by 1.6%–7.7%, so only gaps well beyond that are treated as results.

### E — Paged KV cache

The KV cache from improvement B stores each sequence as one contiguous tensor grown with `torch.cat`. This wastes memory in two ways: reserved allocation (a request that generates fewer tokens than its reserved maximum still holds the unused space) and duplicated prefixes (requests sharing a prompt or document each store their own copy).

Paged caching splits the cache into fixed-size blocks (16 tokens each) held in a shared pool; each sequence keeps only a block table pointing to its blocks. This enables on-demand allocation and prefix sharing via reference counting.

Capacity (4 layers / 4 heads / head dim 64 / fp16, block 16 tokens, max 512):

| requests | tokens total | reserved baseline | paged | reserved waste | paged waste |
|---|---|---|---|---|---|
| 8 | 254 | 16.00 MB | 1.188 MB | 93.8% | 16.45% |
| 32 | 899 | 64.00 MB | 4.375 MB | 94.5% | 19.73% |

Prefix sharing (8 sequences sharing a 64-token prefix):

| | cache | prefill passes |
|---|---|---|
| one copy each | 2.000 MB | 8 |
| **paged + fork** | **0.250 MB (−87.5%)** | **1** |

Correctness: paging only changes where the same k/v is stored, so the two paths must emit byte-identical tokens (verified, `token_match = true`).

Wall-clock: this pure-Python implementation is 1.10× slower (128.68 → 141.61 ms) because each step adds a table lookup and a copy. The experiment validates capacity and sharing, not speed; real systems such as vLLM move these operations into GPU kernels and combine them with continuous batching.

## The batch-size result

The first KV-cache benchmark returned 0.98× — the cache was slower. Rather than retuning until the number looked favorable, the time was measured where it actually goes:

| seq len | 1 | 64 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| ms per forward | 10.24 | 11.50 | 11.85 | 11.59 | 13.77 |

A 1024× increase in sequence length costs only ~3.5 ms, so roughly 10 ms of every forward pass is fixed overhead (Python + kernel launches), not arithmetic. At batch 1 there is almost no GPU work to remove, so the cache cannot help. As the batch grows, the real computation appears and the cache pays off (30.5× at batch 32). The conclusion: KV cache removes arithmetic, not overhead.

## Why speculative decoding was not used

Speculative decoding accelerates a large target model by having a cheaper draft model propose several tokens and then verifying them in a single target-model pass. Its premise is that the target model's per-token cost is dominated by expensive arithmetic that the draft model can avoid.

That premise does not hold here. mygpt's per-token cost is dominated by the ~10 ms fixed overhead measured above, not by arithmetic (a 1024× increase in sequence length changes the forward time by only ~3.5 ms). A smaller draft model cannot remove this fixed overhead, and the verification step adds one more target-model forward pass on top. The theoretical speedup ceiling is about 1.4× under the most optimistic assumptions, and under sampling (the model's default mode) the result is actually slower. Speculative decoding was therefore evaluated and not implemented: at this model scale there is no arithmetic to amortize.

This is consistent with the other findings here — KV cache gives 0.98× at batch 1, SDPA is swamped at short sequences, and GQA shows no quality cost at this size — techniques whose benefit requires a much larger model or a much larger batch.

## Reproducibility

```bash
git clone https://github.com/tancy-920/mygpt.git
cd mygpt
pip install -r requirements.txt    # RTX 50 series: use the cu128 index, see the file header

python train.py --run_name base      # ~75 s on a laptop GPU
python sample.py --prompt "Shakespeare"
python bench.py                      # reproduces every inference number below
```

Reproducing the comparison experiments:

```bash
# A — RMSNorm
python train.py --run_name base    --norm layernorm
python train.py --run_name rmsnorm --norm rmsnorm

# B / C / D / E — inference side
python bench.py                     # all five benchmark groups
python bench.py 4 5                 # run only groups 4 and 5

# C — training-side quality comparison
python train.py --run_name gqa2 --n_kv_head 2
python train.py --run_name gqa1 --n_kv_head 1
```

`train.py` writes `results/<run_name>_metrics.json`; `bench.py` writes `results/bench_*.json`. The switch for improvement D (`GPTConfig.attn`) is used only on the inference-side forward comparison and is not wired into `train.py`, so training always uses the baseline path.

### Repository structure

```
mygpt/
├── model.py      GPTConfig / LayerNorm / RMSNorm / CausalSelfAttention (three attention paths) / MLP / Block / GPT
├── paged_kv.py   Paged KV cache: block pool + block table + reference counting + prefix sharing
├── train.py      Training loop + metric logging
├── sample.py     Load weights and generate (also the KV-cache equivalence check)
├── bench.py      Five inference benchmarks producing the inference data in this README
├── data/tinyshakespeare/input.txt
├── results/      Raw measurements (committed, each number here can be re-derived)
└── requirements.txt
```

`out/` (model weights, ~13 MB each) is excluded by `.gitignore`; `results/` (JSON, a few KB) is committed.

## Limitations

- This is not a usable language model: 3.2M parameters, character-level, 1.1 MB of text, 3,000 steps. It is a measurement vehicle.
- Single run, single seed, single laptop GPU. Validation-loss differences around 0.005 are noise.
- The 30.5× is not a universal constant; it is the intersection of O(T²) vs O(T) generation on this hardware at batch 32.
- The architecture is GPT-2 style (no RoPE, no SwiGLU, no gradient checkpointing). The five optimizations are introduced in isolation.
- Attention runs through PyTorch's fused entry point, not the FlashAttention kernel; the speedup is bf16-only.
- The paged KV cache is a Python-level proof of concept (no copy-on-write, no scheduler, attention still gathers blocks back into a contiguous tensor).
- The default KV cache remains the plainest `torch.cat` version; the paged path coexists in `paged_kv.py`.
- Repeated measurements vary by 1.6%–7.7%, and timing is taken on a thermally-throttling laptop.

## References

**Code and data**

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — architecture, weight initialization, and residual scaling follow its approach. This repository differs in five ways: attention split into three projections (required for GQA), switchable normalization, KV-cache generation (absent in upstream nanoGPT), three switchable attention paths, and an additional paged KV-cache implementation.
- [karpathy/char-rnn](https://github.com/karpathy/char-rnn) — the TinyShakespeare dataset (≈1.1 MB).

**Papers**

| Topic | Reference |
|---|---|
| Transformer | Vaswani et al., *Attention Is All You Need*, 2017 · [1706.03762](https://arxiv.org/abs/1706.03762) |
| GPT-2 | Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019 |
| RMSNorm | Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019 · [1910.07467](https://arxiv.org/abs/1910.07467) |
| MQA | Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019 · [1911.02150](https://arxiv.org/abs/1911.02150) |
| GQA | Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 · [2305.13245](https://arxiv.org/abs/2305.13245) |
| Llama 2 (RMSNorm + GQA in production) | Touvron et al., 2023 · [2307.09288](https://arxiv.org/abs/2307.09288) |
| FlashAttention | Dao et al., 2022 · [2205.14135](https://arxiv.org/abs/2205.14135) |
| PagedAttention / vLLM | Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 · [2309.06180](https://arxiv.org/abs/2309.06180) |

## License

[MIT](LICENSE)

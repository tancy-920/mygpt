# mygpt

**一个从零手写的字符级 GPT（3.2M 参数），以及三项可一键开关的推理优化 —— 每一项都配「改前 / 改后」的实测数据。**

不依赖 `transformers` / `torchtext`，整个模型不到 350 行 Python。

> **三条结论（完整数据见第四节）**
>
> - `LayerNorm → RMSNorm`：参数 **−2,304**，训练峰值显存 **−36 MB（−10%）**，速度持平
> - `无缓存 → KV cache`：batch=1 时 **0.98×（没有加速）**，batch=32 时 **30.5×**
> - `KV 头 4 → 1`（GQA/MQA）：单条序列的 KV cache 从 12.0 MB 降到 **1.5 MB**（−87.5%）

---

## 一、项目目的

我想真的弄明白现代大语言模型内部发生了什么，而不是停在「听说注意力很强」这一层。

触发点很具体：我读到过太多「KV cache 能大幅加速推理」「GQA 省显存」的说法，但很难找到一个**代码量足够小、能一次只改一个开关、并且把两组数据都摆在明面上**的仓库 —— 现成实现要么藏在庞大的推理框架里，要么只贴一个漂亮的最终数字。

所以我自己写了一遍，并把每个结论都变成可复现的对照实验。

| 想弄明白的事 | 具体做到什么程度 |
|---|---|
| **Transformer 架构本身** | 不借助任何封装库，从零实现因果自注意力、多头、MLP、残差、归一化、位置编码、权重共享 |
| **大语言模型的底层逻辑** | 把 Llama 系模型标配的三项技术（RMSNorm / KV cache / GQA）亲手实现，弄清"改了什么、换来什么、代价是什么" |
| **结论是怎么来的** | 所有改动做成开关；基线组与改动组除该开关外完全一致；参数、显存、loss、墙钟时间全部落成 JSON 入库，随时可重新跑出来 |
| **留下的东西能被人复现** | 别人 clone 下来，一条命令训练，一条命令复现全部推理数据 |

## 二、项目亮点

1. **三项改进全部开关化** —— 同一个数据管道、同一套训练循环，`--norm` / `use_cache` / `--n_kv_head` 一次只动一个变量，所以差异可以干净地归因到这一项改动上。
2. **不回避反常结果** —— 第一次测 KV cache 得到 **0.98×**（缓存反而更慢）。我没有换一组好看的配置交差，而是先查清时间花在哪，再找到真正能体现收益的场景（第五节）。
3. **正确性先于性能** —— KV cache 是「结果等价」的优化：固定随机种子后，开 / 关缓存生成的文本 `diff` **逐字相同**。会改变输出的缓存是 bug，不是优化。
4. **所有原始数据入库** —— `results/*.json` 随仓库提交，README 里的每个数字都能自己核对，没有一个是凭记忆写的。
5. **明确标注局限** —— 见第八节。小模型上 GQA 的表现**不能**推广到真实大模型。

## 三、模型架构

```
输入字符 "F" "i" "r" "s" "t"   →   字符编号 (B, T)
  │
  ├─ wte 字符嵌入  (65 × 256)
  └─ wpe 位置嵌入 (128 × 256)      →  两者相加，得到 (B, T, 256)
  │
  ▼  以下结构重复 4 层（每个 Block）
  ├─ 归一化                     ← 改进 A 开关：LayerNorm / RMSNorm
  ├─ 因果自注意力                ← 改进 B（KV cache）/ 改进 C（GQA）
  ├─ 残差相加
  ├─ 归一化
  ├─ MLP  256 → 1024 → 256
  └─ 残差相加
  │
  ▼
  最终归一化 → Linear(256 → 65)   →   下一个字符的 65 个分数
                                      （与 wte 共享权重）
```

默认配置：4 层 / 4 个注意力头 / 256 维 / 上下文 128 / 词表 65（字符级）。

### 参数量构成（与实测值逐项对得上）

| 模块 | 计算 | 参数量 |
|---|---|---|
| `wte` 字符嵌入（与输出层共享） | 65 × 256 | 16,640 |
| `wpe` 位置嵌入 | 128 × 256 | 32,768 |
| 单层 · 注意力（q/k/v/c_proj） | 4 × (256×256 + 256) | 263,168 |
| 单层 · MLP | (256×1024 + 1024) + (1024×256 + 256) | 525,568 |
| 单层 · 归一化 × 2 | 2 × (256 + 256) | 1,024 |
| **4 层 Block 合计** | | **3,159,040** |
| 最终归一化 `ln_f` | 256 + 256 | 512 |
| **总计** | | **3,208,960** |

换成 RMSNorm 后，9 个归一化层各少一组 bias → **−9 × 256 = −2,304**，得 3,206,656，与实测完全一致。

## 四、三项改进与实测结果

### 训练对照

配置完全相同：**3000 步、batch 32、上下文 128、4 层 / 4 头 / 256 维**，单张 **RTX 5060 Laptop 8 GB**，`torch 2.11.0+cu128`，bf16 自动混合精度。除标注的开关外，其他一个字没改。

| 实验 | 归一化 | KV 头数 | 参数量 | train loss | val loss | 训练耗时 | ms/步 | 吞吐 | 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|
| `base` | LayerNorm | 4 | 3,208,960 | 1.1612 | 1.5134 | 74.2 s | 24.73 | 165,600 tok/s | 365.7 MB |
| `rmsnorm` | **RMSNorm** | 4 | **3,206,656** | 1.1628 | 1.5104 | **74.0 s** | **24.66** | 166,096 tok/s | **329.6 MB** |
| `gqa2` | LayerNorm | **2** | 2,945,792 | 1.1512 | 1.4994 | 83.6 s | 27.84 | 147,112 tok/s | 361.2 MB |
| `gqa1` | LayerNorm | **1** | **2,814,208** | 1.1614 | **1.4970** | 82.4 s | 27.44 | 149,272 tok/s | 359.3 MB |

> 参考基准：65 类字符的完全随机猜测是 `ln(65) = 4.174`，四组都远低于随机水平。训练 loss 与验证 loss 同步下降、验证 loss 未出现反弹，说明 3000 步远未到过拟合点。

### 推理对照

| 改动 | 场景 | 改前 | 改后 |
|---|---|---|---|
| **KV cache** | 生成 479 个字符，batch=1 | 2.055 s | 2.103 s（**0.98×，无收益**） |
| | batch=8 | 3.566 s | 2.008 s（1.78×） |
| | **batch=32** | 88.930 s | **2.912 s（30.5×）** |
| **GQA / MQA** | 单条序列的 KV cache 显存 | MHA 12.00 MB | GQA 6.00 / 3.00 MB，MQA **1.50 MB** |

### 改进 A —— RMSNorm 替代 LayerNorm

LayerNorm 会减均值、除标准差，再做一次可学习的缩放**和**平移。RMSNorm（[Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)，Llama 系列采用）去掉了减均值和 bias，只保留均方根缩放：

```python
rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
return x / rms * weight
```

**实测**：参数 −2,304，训练峰值显存 365.7 → **329.6 MB（−10%）**，val loss 1.5134 → 1.5104（相差 0.003，落在噪声范围内，**不作为质量提升来宣称**）。

**诚实说明**：单步耗时几乎没变（24.73 → 24.66 ms）。在这个规模下，每步的时间主要被 Python 开销和 kernel 启动吃掉，少几次张量运算不会体现在墙钟时间上。**真正拿到的是显存。**

### 改进 B —— KV cache 增量解码

基线生成时，每一步都要把整个前缀重新过一遍网络，计算量正比于已生成长度的**平方**；缓存每一层的 k/v 之后，每步只计算新字符：

```
无缓存：前缀 → 重算全部 T 个位置 → 采样 1 个字符      总代价 ∝ T²
有缓存：前缀 → 缓存 k/v，之后每步只算 1 个位置 → 采样   总代价 ∝ T
```

实现在 `model.py` 的 `past_kv` / `use_cache`，经由 `CausalSelfAttention` → `Block` → `GPT` 逐层串联，并用 `pos_offset` 保证增量解码时位置编码不会错位。

**先验正确性，再谈速度。** 固定随机种子后，`use_cache=False` 与 `use_cache=True` 生成的文本 **`diff` 逐字相同**：

```
What is the devil, where you come?

COMINIUS:
I will despend
```

会改变输出的缓存不是优化，是 bug —— 所以这项检查也写进了仓库（`sample.py` 开关 `--use_cache` 对比）。

> 这是最朴素的一种实现（每步 `torch.cat` 拼接，没有预分配 buffer、没有分页）。生产级实现会比这里更快。

### 改进 C —— GQA / MQA（分组查询注意力）

标准多头注意力（MHA）给每个 Query 头配一套独立的 Key/Value 头；而 KV cache 的大小正比于 **KV 头数**，所以推理显存基本都花在这里。

| 模式 | Q 头 | KV 头 | 共享方式 |
|---|---|---|---|
| MHA | 8 | 8 | 每个 Q 头独占一套 K/V |
| **GQA** | 8 | 4 / 2 | 每 2 或 4 个 Q 头共享一套 K/V |
| MQA | 8 | 1 | 所有 Q 头共用一套 K/V |

实现方式：用更少的输出维度投影 K 和 V，再在注意力矩阵乘之前用 `repeat_interleave` 展开。只改 `n_kv_head`，Query 头数不动。

**实测**（fp16、batch=1、6 层、8 个 Q 头、头维 64、上下文 1024）：

| n_kv_head | 模式 | KV cache 显存 | 相对 MHA |
|---|---|---|---|
| 8 | MHA | 12.00 MB | 1.000× |
| 4 | GQA | 6.00 MB | 0.500× |
| 2 | GQA | 3.00 MB | 0.250× |
| 1 | MQA | 1.50 MB | **0.125×** |

估算值同时用真实的缓存张量形状验证过：`n_kv_head=8` 时缓存的 k 形状是 `(1, 8, L, 64)`，`n_kv_head=1` 时是 `(1, 1, L, 64)` —— 第 2 维就是 KV 头数，**显存下降是结构性的，不是估算出来的**。

**诚实说明**：上表里 GQA 的 val loss 反而更低（1.4970 vs 1.5134），这**不能**解读为「GQA 更好」—— 4 层 / 256 维 / 1.1 MB 文本的规模太小，压不出 GQA 应付出的质量代价。在真实大模型上，GQA 是用少量、可控的质量损失换取显存和带宽。另外 GQA 让**训练**慢了约 12%（82–84 s vs 74 s），因为展开 KV 头本身有开销 —— **它的收益在推理侧，不在训练侧。**

## 五、一个我没料到的结果：KV cache 的收益取决于 batch

第一次基准测试，加速比是 **0.98×** —— 缓存**更慢**。与其悄悄换一组更好看的配置，不如先查清时间花在哪。

诊断：单次 forward 耗时 vs 序列长度（batch=1，6 层，384 维）：

| 序列长度 T | 1 | 64 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| 单次 forward | 10.24 ms | 11.50 ms | 11.85 ms | 11.59 ms | 13.77 ms |

序列长度涨 **512 倍**，耗时几乎不变。这约 10 ms 是与长度无关的**固定开销**（Python 解释器 + kernel 调度），不是算术。**batch=1 时根本没有多少 GPU 计算可以被省掉**，缓存自然无从发挥。

把 batch 提上去，真实的 GPU 计算量才出现（上下文 512，4 层，256 维，生成 479 个字符）：

| batch | 无缓存 | 有缓存 | 加速比 |
|---|---|---|---|
| 1 | 2.055 s | 2.103 s | 0.98× |
| 8 | 3.566 s | 2.008 s | 1.78× |
| **32** | **88.930 s** | **2.912 s** | **30.5×** |

**结论：KV cache 省的是算力，不是开销。** 只有当算力超过调度开销时，它才体现为墙钟加速 —— 在小模型上，单条请求几乎没有收益，而批量推理服务可以有几十倍。这也解释了为什么真实的推理框架都在做 continuous batching。

## 六、快速开始

```bash
git clone https://github.com/tancy-920/mygpt.git
cd mygpt
pip install -r requirements.txt          # RTX 50 系请用 cu128 索引，见文件头注释

python train.py --run_name base          # 笔记本 GPU 上约 75 秒
python sample.py --run_name base         # 用真实剧本片段续写
```

## 七、复现全部对照实验

```bash
# 改进 A —— RMSNorm
python train.py --run_name base      --norm layernorm
python train.py --run_name rmsnorm   --norm rmsnorm

# 改进 B / C —— 推理侧（KV cache 加速比曲线 + GQA 显存）
python bench.py

# 改进 C —— 训练侧质量对比
python train.py --run_name gqa2 --n_kv_head 2
python train.py --run_name gqa1 --n_kv_head 1
```

每次运行都会写出 `results/<run_name>_metrics.json`（参数量、耗时、峰值显存、完整 loss 曲线）。表格里的数字都可以重新跑出来。

### 仓库结构

```
mygpt/
├── model.py          模型：GPTConfig / LayerNorm / RMSNorm / CausalSelfAttention / MLP / Block / GPT
├── train.py          训练循环 + 指标记录
├── sample.py         加载权重生成文本（同时是 KV cache 的等价性检查）
├── bench.py          三组推理基准，产出 README 中全部推理数据
├── data/tinyshakespeare/input.txt
├── results/          原始实测 JSON（随仓库提交，可逐项核对）
└── requirements.txt
```

`out/`（模型权重，每个约 13 MB）被 `.gitignore` 排除；`results/`（几 KB 的 JSON）**有意提交**。

## 八、局限性

- **这不是一个能用的语言模型。** 3.2M 参数、字符级、1.1 MB 文本、3000 步 —— 它是用来做测量的载体。生成的文本只是「长得像英文」的乱码（能正确输出说话人名字和换行结构）。
- **单次运行、单一随机种子、单台笔记本 GPU。** val loss 上 0.005 量级的差异是**噪声**，我照实记录，但不据此下结论。
- **30.5× 不是一个通用常数。** 它来自「无缓存一侧是 O(T²)、有缓存一侧是 O(T)」在本机 batch=32 上的交点；换模型规模或换硬件，这个数字会变。
- **架构是 GPT-2 风格，不是 Llama 风格。** 没有 RoPE、没有 SwiGLU、没有 FlashAttention、没有梯度检查点。三项改进是「点状」引入的，不代表完整复现了现代 LLM 的全部工程实践。
- **KV cache 实现是最朴素的版本**（逐步 `torch.cat`、无预分配、无 paged attention），生产实现会更快、显存更可控。
- **计时取自会降频的笔记本。** 每次测量都用 `torch.cuda.synchronize()` 包围并对重复运行取最小值，但它仍然是墙钟数据，不宜当作精确基准。

## 九、致谢与参考资料

**代码与数据来源**

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) —— 架构、权重初始化方案、残差缩放均沿用其思路。本仓库与它的差别有三处：注意力拆成三个投影（GQA 必需）、归一化层可切换、生成时带 KV cache（上游 nanoGPT 未实现）。
- [karpathy/char-rnn](https://github.com/karpathy/char-rnn) —— TinyShakespeare 数据集，约 1.1 MB。

**论文与文献**

| 主题 | 文献 |
|---|---|
| Transformer | Vaswani et al., *Attention Is All You Need*, 2017 · [1706.03762](https://arxiv.org/abs/1706.03762) |
| GPT-2 | Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019 |
| RMSNorm | Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019 · [1910.07467](https://arxiv.org/abs/1910.07467) |
| MQA | Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019 · [1911.02150](https://arxiv.org/abs/1911.02150) |
| GQA | Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 · [2305.13245](https://arxiv.org/abs/2305.13245) |
| Llama 2（RMSNorm + GQA 的工程化落地） | Touvron et al., 2023 · [2307.09288](https://arxiv.org/abs/2307.09288) |

## License

[MIT](LICENSE)

---
---

# mygpt (English)

**A character-level GPT (~3.2M params) written from scratch — no `transformers`, no `torchtext` — with three switchable inference optimisations, each backed by an actual before/after measurement.**

I built this to understand the *internals* of modern LLM inference (KV cache, GQA, RMSNorm) rather than take the claims on faith, and I could not find a small enough repo where you can flip one switch at a time and see both sides of the numbers.

```bash
pip install -r requirements.txt
python train.py --run_name base      # ~75 s on a laptop GPU
python bench.py                      # reproduces every number below
```

**Training** — 3000 steps, batch 32, context 128, 4 layers / 4 heads / 256 dims, RTX 5060 Laptop 8 GB, `torch 2.11.0+cu128`. Only the stated switch differs between rows.

| run | norm | n_kv_head | params | val loss | train time | peak mem |
|---|---|---|---|---|---|---|
| `base` | LayerNorm | 4 | 3,208,960 | 1.5134 | 74.2 s | 365.7 MB |
| `rmsnorm` | RMSNorm | 4 | 3,206,656 | 1.5104 | 74.0 s | **329.6 MB** |
| `gqa2` | LayerNorm | 2 | 2,945,792 | 1.4994 | 83.6 s | 361.2 MB |
| `gqa1` | LayerNorm | 1 | 2,814,208 | 1.4970 | 82.4 s | 359.3 MB |

- **A · RMSNorm** — −2,304 params, −10% peak training memory, step time unchanged. The val-loss difference (0.003) is within noise; I do not claim a quality win.
- **B · KV cache** — generation costs O(T²) without a cache and O(T) with one. Correctness first: with a fixed seed, cached and uncached generation produce **byte-identical text**. Speed depends entirely on batch size:

  | batch | no cache | KV cache | speedup |
  |---|---|---|---|
  | 1 | 2.055 s | 2.103 s | 0.98× |
  | 8 | 3.566 s | 2.008 s | 1.78× |
  | 32 | 88.930 s | 2.912 s | **30.5×** |

- **C · GQA / MQA** — KV cache size scales with the KV-head count: 12.0 → 6.0 → 3.0 → **1.5 MB**, verified against the real cached tensor shapes.

**The result I did not expect.** My first benchmark gave **0.98×** — the cache was *slower*. Instead of retuning until the number looked good, I measured where the time actually goes: one forward pass takes 10.24 ms at T=1 and 11.59 ms at T=512. A 512× increase in sequence length barely moves the time, so the cost is **fixed overhead** (Python + kernel launches) — and at batch 1 there is almost no GPU work to remove. Raise the batch and the cache starts paying off: 30.5× at batch 32. *KV cache removes arithmetic, not overhead.*

**Caveats.** 3.2M params, 1.1 MB of text — a measurement vehicle, not a usable language model. Single seed, single run, one thermally-throttling laptop; val-loss deltas of ~0.005 are noise. The 30.5× is not a universal constant. The architecture is GPT-2 style (no RoPE / SwiGLU / FlashAttention) and the KV cache is the naive `torch.cat` version. Note also that in this small setting GQA *lowers* val loss too — that is a small-model artefact and must not be read as "GQA is generally better".

**References.** Architecture follows [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT); TinyShakespeare comes from [char-rnn](https://github.com/karpathy/char-rnn). RMSNorm — [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467); MQA — [Shazeer, 2019](https://arxiv.org/abs/1911.02150); GQA — [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245).

**License:** [MIT](LICENSE)

# mygpt

**一个从零搭建的字符级 GPT（3.2M 参数），以及五项可一键开关的推理优化 —— 每一项都配「改前 / 改后」的实测数据。**

不依赖 `transformers` / `torchtext`，模型本体 473 行 Python，五个 `.py` 合计 1,716 行。

> **五条结论（完整数据见第四节）**
>
> - `LayerNorm → RMSNorm`：参数 **−2,304**，训练峰值显存 **−36 MB（−10%）**，速度持平
> - `无缓存 → KV cache`：batch=1 时 **0.98×（没有加速）**，batch=32 时 **30.5×**
> - `KV 头 8 → 1`（GQA/MQA，估算）：单条序列的 KV cache 从 12.0 MB 降到 **1.5 MB**（−87.5%）
> - `逐位置完整矩阵 → 融合注意力`（SDPA，T=2048）：**3.7×** 提速，峰值显存 **608 MB → 144 MB**
> - `连续缓存 → 分页 KV cache`：8 条序列共享一段前缀，缓存字节 **−87.5%**；但本实现墙钟 **1.10×（更慢）**—— 它换的是容量，不是速度


---

## 一、项目目的

我想真的弄明白现代大语言模型内部发生了什么，而不是停在「听说注意力很强」这一层。

触发点很具体：我读到过太多「KV cache 能大幅加速推理」「GQA 省显存」的说法，但很难找到一个**代码量足够小、能一次只改一个开关、并且把两组数据都摆在明面上**的仓库 —— 现成实现要么藏在庞大的推理框架里，要么只贴一个漂亮的最终数字。

所以我把每个结论都变成可复现的对照实验。

| 想弄明白的事 | 具体做到什么程度 |
|---|---|
| **Transformer 架构本身** | 不借助任何封装库，从零实现因果自注意力、多头、MLP、残差、归一化、位置编码、权重共享 |
| **大语言模型的底层逻辑** | 把 Llama 系模型标配的三项技术（RMSNorm / KV cache / GQA）自己实现，弄清"改了什么、换来什么、代价是什么" |
| **推理引擎里的常见优化** | 再把另两项做出来：融合注意力（SDPA，也会用到 GQA 的 `enable_gqa`）与分页 KV cache —— 并明确说清哪一项在本机真的拿到收益、哪一项只是概念验证 |
| **结论是怎么来的** | 所有改动做成开关；基线组与改动组除该开关外完全一致；参数、显存、loss、墙钟时间全部落成 JSON 入库，随时可重新跑出来 |
| **留下的东西能被人复现** | 别人 clone 下来，一条命令训练，一条命令复现全部推理数据 |

## 二、项目亮点

1. **五项改进全部开关化** —— 同一个数据管道、同一套训练循环，`--norm` / `use_cache` / `--n_kv_head` / `attn` 一次只动一个变量，所以差异可以干净地归因到这一项改动上。
2. **不回避反常结果** —— 第一次测 KV cache 得到 **0.98×**（缓存反而更慢）。我没有换一组好看的配置交差，而是先查清时间花在哪，再找到真正能体现收益的场景（第五节）。
3. **正确性先于性能** —— 优化分两类，检查方式也不同。KV cache 与分页缓存是「结果等价」的：固定随机种子后，两条路径生成的文本必须 `diff` **逐字相同**。三种注意力实现算的也是同一件事：逐元素差值必须落在数值精度之内、`argmax` 完全一致。会改变输出的「优化」是 bug，不是优化。
4. **没拿到收益的也照实写** —— 分页 KV cache 在本实现里墙钟 **1.10×（更慢）**，这一点直接写进结论，而不是含糊成「效果有待观察」。README 里没有只挑好看数字的地方。
5. **所有原始数据入库** —— `results/*.json` 随仓库提交，README 里的每个数字都能自己核对。
6. **明确标注局限** —— 见第八节。小模型上 GQA 的表现**不能**推广到真实大模型；本机也没有可用的 FlashAttention 内核（原因见「改进 D」一节）。

## 三、模型架构

```
输入字符 "F" "i" "r" "s" "t"   →   字符编号 (B, T)
  │
  ├─ wte 字符嵌入  (65 × 256)
  └─ wpe 位置嵌入 (128 × 256)      →  两者相加，得到 (B, T, 256)
  │
  ▼  以下结构重复 4 层（每个 Block）
  ├─ 归一化                     ← 改进 A 开关：LayerNorm / RMSNorm
  ├─ 因果自注意力                ← 改进 B（KV cache）/ C（GQA）/ D（注意力实现三选一）
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

### 参数量构成

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

## 四、五项改进与实测结果

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
| **融合注意力** | 单次前向，T=2048、batch=8、bf16 | 完整矩阵 51.09 ms / 607.6 MB | 融合内核 **13.88 ms / 143.6 MB（3.68×）** |
| **分页 KV cache** | 8 条序列共享 64 字符前缀 | 各持一份 2.000 MB | 共享同一批物理块 **0.250 MB（−87.5%）** |
| | 单条序列生成 32 个字符（墙钟） | 连续缓存 128.68 ms | 分页 **141.61 ms（1.10×，更慢）** |

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

**估算**（按下方公式计算，另用真实缓存张量形状验证；fp16、batch=1、6 层、8 个 Q 头、头维 64、上下文 1024）：

| n_kv_head | 模式 | KV cache 显存 | 相对 MHA |
|---|---|---|---|
| 8 | MHA | 12.00 MB | 1.000× |
| 4 | GQA | 6.00 MB | 0.500× |
| 2 | GQA | 3.00 MB | 0.250× |
| 1 | MQA | 1.50 MB | **0.125×** |

估算值同时用真实的缓存张量形状验证过：`n_kv_head=8` 时缓存的 k 形状是 `(1, 8, L, 64)`，`n_kv_head=1` 时是 `(1, 1, L, 64)` —— 第 2 维就是 KV 头数，**显存下降是结构性的，不是估算出来的**。

**诚实说明**：上表里 GQA 的 val loss 反而更低（1.4970 vs 1.5134），这**不能**解读为「GQA 更好」—— 4 层 / 256 维 / 1.1 MB 文本的规模太小，压不出 GQA 应付出的质量代价。在真实大模型上，GQA 是用少量、可控的质量损失换取显存和带宽。另外更正一条：早期版本写过「GQA 让训练慢了约 12%，因为展开 KV 头本身有开销」。事后核对发现，那两组 GQA 与 `base` 并不是同一时段训练的（同一台笔记本上 GPU 长时间运行会降频），这个归因不成立；把三种配置放在同一时段交替重测后，差异缩小到与重复测量自身的波动同阶（该重测未随仓库提交）。**结论：训练速度上 GQA 与 MHA 基本持平** —— 这与原论文一致：**GQA 的收益在推理侧，不在训练侧。**

### 改进 D —— 融合注意力（SDPA）

基线实现会把完整的 `Tq × Tk` 分数矩阵算出来、存下来，再 softmax、加权求和。T=2048、batch=8、4 头时这个矩阵是 `8 × 4 × 2048 × 2048` —— 光它就要几百 MB，而这些中间值**除了一次 softmax 之外没有别的用处**。融合内核的思路是根本不让它落地：把计算切块，边算边累加。

PyTorch 把这类实现收进一个统一入口 `F.scaled_dot_product_attention`（简称 SDPA），调用时再分发到具体内核。本机的实际情况：

| 后端 | 本机可用 | 说明 |
|---|---|---|
| `FLASH_ATTENTION` | ✗ | 本机的 PyTorch 没有编译这个后端（`is_flash_attention_available()` 返回 `False`，强制指定会报 `No available kernel`） |
| `CUDNN_ATTENTION` | ✓（bf16） | 本机唯一真正可用的融合内核 |
| `EFFICIENT_ATTENTION` | 部分 | 只在 `n_kv_head == n_head` 时可用；k/v 头数少于 q 头数（GQA）时不支持 |
| `MATH` | ✓ | 按定义逐步算的参考实现，又慢又吃显存 |

判定办法：这台机器的 profiler（CUPTI 初始化失败）与 `nvidia-smi`（NVML 报错）都拿不到内核名，所以只能「强制只允许某一个后端，看谁不报错」。`bench.py` 就是这么做的，逐档结果记在 JSON 的 `usable_backends` 里。

**先把话说清楚**：这里**没有**实现 FlashAttention 内核本身。用的是 PyTorch 提供的融合注意力入口，在本机落到的是 cuDNN 内核。两者思路一致（都不落地完整分数矩阵），但**不是同一个东西** —— 所以全文一律写「融合注意力（SDPA）」，不写「FlashAttention」。

**实测**（batch=8、bf16、4 层 / 4 头 / 256 维，整模型一次前向；前三列是各档位的耗时，最后一列是 T=2048 的峰值显存）：

| n_kv_head | 实现 | T=128 | T=512 | T=2048 | 峰值显存（T=2048） |
|---|---|---|---|---|---|
| 4 | 基线（完整矩阵） | 3.73 ms | 4.63 ms | 53.91 ms | 604.3 MB |
| 4 | **融合（SDPA）** | 3.40 ms | 3.19 ms | **19.28 ms** | **144.3 MB** |
| 4 | chunked（分块在线 softmax） | 6.26 ms | 14.31 ms | 131.66 ms | 185.3 MB |
| 1 | 基线（完整矩阵） | 3.33 ms | 4.45 ms | 51.09 ms | 607.6 MB |
| 1 | **融合（SDPA）** | 3.09 ms | 3.03 ms | **13.88 ms** | **143.6 MB** |
| 1 | chunked（分块在线 softmax） | 6.12 ms | 13.65 ms | 129.37 ms | 188.6 MB |

T=2048 一档：**3.68×** 提速（n_kv_head=1），峰值显存 **607.6 → 143.6 MB**。T 越大差距越大 —— T=128 时两者都在固定开销主导的区间里，看不出区别。

**一个容易踩的坑：不指定后端时，PyTorch 会挑中 MATH。** 同一组输入（batch=8、n_head=4、n_kv_head=1、T=2048、bf16），直接调用默认入口得到 **21.73 ms / 1257.3 MB**；显式指定融合内核后是 **1.292 ms / 41.25 MB**，差 **16.8×**。所以 `_attn_sdpa` 是显式列出融合内核的（`sdpa_kernel` 上下文），而不是直接调用默认入口。

**顺带补上了改进 C 的一个缺口。** `enable_gqa=True` 让内核直接使用未展开的 k/v，替掉了原来的 `repeat_interleave`。改进 C 当时只兑现了「缓存容量变小」，注意力计算里那次复制并没有省掉；现在省掉了。

**另外两个实测确认过的坑**（都写进了 `model.py` 的注释）：

- **别用 `is_causal=True`。** SDPA 的 `is_causal` 是「左上对齐」的：`Tq=1、Tk=64` 时它只让当前 query 看到第 0 个 key，而基线实现允许它看到全部 64 个 —— 两者输出相差 **3.2**，不是舍入误差。显式传掩码才能与基线逐元素一致。
- **掩码的数据类型要和 q 一致。** q 是 bf16 而掩码是 float32 时，会**不报错地**算出错误结果。传 bool 张量最省事，也绕开了这个坑。

**数值等价性**（`T=64`、batch=2，与基线实现逐元素比）：

| dtype | n_kv_head | 实现 | max\|Δ\| | cosine | argmax 一致 |
|---|---|---|---|---|---|
| float32 | 4 | sdpa | 8.3e-07 | 1.00000 | ✓ |
| float32 | 4 | chunked | 6.0e-07 | 1.00000 | ✓ |
| float32 | 1 | sdpa | 6.0e-07 | 1.00000 | ✓ |
| float32 | 1 | chunked | 6.1e-07 | 1.00000 | ✓ |
| bfloat16 | 4 | sdpa | 1.2e-02 | 0.99997 | ✓ |
| bfloat16 | 4 | chunked | 1.6e-02 | 0.99997 | ✓ |
| bfloat16 | 1 | sdpa | 1.3e-02 | 0.99997 | ✓ |
| bfloat16 | 1 | chunked | 1.2e-02 | 0.99997 | ✓ |

bf16 那一档的差值看起来比 fp32 大三个数量级，但 bf16 只有 8 位尾数（相对精度约 0.4%），分数矩阵的量级在 ±10 附近，所以 `1.2e-02` 完全落在它的表示精度之内；`argmax` 全部一致，cosine 与 1 的差在 3e-05 以内。

**噪声地板。** 同一档位（n_kv_head=1、T=2048）在实验首尾各测一次：baseline 51.14 → 51.94 ms（相差 1.56%），sdpa 13.38 → 14.40 ms（7.69%）。所以这张表里只有远大于百分之几的差距才值得当真 —— T=2048 的 3.68× 远远超过它，而 T=128 档的 1.08× 就应当在噪声量级上看待。

**第三种实现 `chunked` 是干什么的。** 它是「分块 + 在线 softmax」：把 key 沿长度切块，逐块累加，同时维护 running max 与 running sum，用重缩放系数把上一轮的累加结果折算到新的 max 上 —— 算法结构和 FlashAttention 相同。区别是它**不是**一个融合内核，只是纯 PyTorch 的逐块算子组合。留它是为了回答一个具体问题：**「不落地完整矩阵」这件事本身值多少？** 结论是显存确实降下来了（T=2048：604 → 185 MB），但墙钟比基线还慢（53.91 → 131.66 ms），因为每个分块都要单独启动一次算子。**换句话说，融合内核的收益来自「一个内核里把整件事做完」，而不是来自分块这个想法本身。**

**局限**：①拿到的是 cuDNN 内核，不是 FlashAttention；②fp32 下可用的融合内核很少（`n_kv_head == n_head` 时只剩 `EFFICIENT` 一档，GQA 时融合内核一个都用不了、只剩 `MATH` 参考实现），所以上面的提速只在 bf16 下成立。

### 改进 E —— 分页 KV cache

改进 B 的缓存是「每条序列一整块连续张量，用 `torch.cat` 往里接」。这样有两处浪费：

1. **预留式分配**。想支持最长 512 个 token，就得先按 512 预留；一条只生成 33 个 token 的请求，剩下 479 个位置白占着。
2. **重复存储**。多条请求共享同一段前缀（同一个系统提示词、同一份文档）时，各自重算、各存一份。

分页的做法是把缓存切成固定大小的块（这里 16 个 token 一块），物理块统一放在一个池子里，每条序列只持有一张「块表」指向自己用到的块。于是上面两个问题变成：按需分配块，以及**多条序列指向同一批物理块**（用引用计数管理）。

主指标是**容量与共享**，不是速度。

**容量**（4 层 / 4 头 / 头维 64 / fp16，块大小 16 个 token，最长 512）：

| 请求数 | 实际 token 总数 | 预留式基线 | 分页实占 | 预留式的浪费率 | 分页的浪费率 |
|---|---|---|---|---|---|
| 8 | 254 | 16.00 MB | **1.188 MB** | 93.8% | **16.45%** |
| 32 | 899 | 64.00 MB | **4.375 MB** | 94.5% | **19.73%** |

分页剩下的那部分浪费全部来自「最后一块没写满」，实测在 16%～20% 之间。

**前缀共享**（8 条序列共享同一段 64 字符 = 4 个整块）：

| | 缓存字节 | prefill 次数 |
|---|---|---|
| 各持一份 | 2.000 MB | 8 |
| **分页 + fork** | **0.250 MB（−87.5%）** | **1** |

只共享写满的整块；前缀尾部不满一块的部分各序列各持一份（**没有**实现写时复制）。

**墙钟：这里拿不到收益，照实写。** 单条序列生成 32 个字符，连续缓存 128.68 ms，分页 **141.61 ms（1.10×，更慢）**。原因是这个纯 Python 实现每步多了一次查表与拷贝。真实系统（如 vLLM）会把这些搬进 GPU 内核并配合连续批处理，那时才谈得上吞吐 —— 在这个实现里谈速度是不诚实的，所以本文只把它当作「容量与共享」的概念验证。

**正确性**：分页只是把同样内容的 k/v 换了个存放位置，注意力算法一个字没改，所以两条路径生成的 token 必须**逐字相同**。固定种子生成 16 个字符，实测 `token_match = True`。

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

# 给一个开头，让模型接着往下写。模型是字符级的，开头请用英文字母 / 数字 / 标点
python sample.py --prompt "Shakespeare"
python sample.py --run_name base         # 不给开头时，用内置的剧本片段
```

## 七、复现全部对照实验

```bash
# 改进 A —— RMSNorm
python train.py --run_name base      --norm layernorm
python train.py --run_name rmsnorm   --norm rmsnorm

# 改进 B / C / D / E —— 推理侧（五组基准，一次跑完）
python bench.py

# 只跑其中几组，不覆盖已有 JSON（1 加速比 / 2 forward 耗时 / 3 GQA 显存 / 4 注意力实现 / 5 分页缓存）
python bench.py 4 5

# 改进 C —— 训练侧质量对比
python train.py --run_name gqa2 --n_kv_head 2
python train.py --run_name gqa1 --n_kv_head 1
```

训练脚本写 `results/<run_name>_metrics.json`（参数量、耗时、峰值显存、完整 loss 曲线），`bench.py` 写 `results/bench_*.json`。表格里的数字都可以重新跑出来。

> 改进 D 的开关（`GPTConfig.attn`）只用在推理侧的前向对照上，没有接进 `train.py` —— 训练仍是 `baseline` 路径，所以它不可能悄悄改变训练结果。

### 仓库结构

```
mygpt/
├── model.py          模型：GPTConfig / LayerNorm / RMSNorm / CausalSelfAttention（三种注意力实现）/ MLP / Block / GPT
├── paged_kv.py       分页 KV cache：块池 + 块表 + 引用计数 + 前缀共享
├── train.py          训练循环 + 指标记录
├── sample.py         加载权重生成文本（同时是 KV cache 的等价性检查）
├── bench.py          五组推理基准，产出 README 中全部推理数据
├── data/tinyshakespeare/input.txt
├── results/          原始实测 JSON（随仓库提交，可逐项核对）
└── requirements.txt
```

`out/`（模型权重，每个约 13 MB）被 `.gitignore` 排除；`results/`（几 KB 的 JSON）**有意提交**。

## 八、局限性

- **这不是一个能用的语言模型。** 3.2M 参数、字符级、1.1 MB 文本、3000 步 —— 它是用来做测量的载体。生成的文本只是「长得像英文」的乱码（能正确输出说话人名字和换行结构）。
- **单次运行、单一随机种子、单台笔记本 GPU。** val loss 上 0.005 量级的差异是**噪声**，我照实记录，但不据此下结论。
- **30.5× 不是一个通用常数。** 它来自「无缓存一侧是 O(T²)、有缓存一侧是 O(T)」在本机 batch=32 上的交点；换模型规模或换硬件，这个数字会变。
- **架构是 GPT-2 风格，不是 Llama 风格。** 没有 RoPE、没有 SwiGLU、没有梯度检查点。五项改进是「点状」引入的，不代表完整复现了现代 LLM 的全部工程实践。
- **注意力走的是 PyTorch 的融合入口，不是 FlashAttention 内核。** 本机的 PyTorch 没有编译 `FLASH_ATTENTION` 后端（`is_flash_attention_available()` 返回 `False`），实际可用的是 cuDNN 内核 —— 所以全文只写「融合注意力（SDPA）」。另外 fp32 下这个融合内核基本不可用（GQA 时只剩 `MATH`），提速只在 bf16 下成立。`chunked` 那一条只是把 FlashAttention 的算法结构用纯 PyTorch 复现出来当对照，不是内核。
- **分页 KV cache 是 Python 层的概念验证，不等于 paged attention。** 只实现了块池、块表、引用计数与前缀共享；注意力计算时仍要把用到的块 `gather` 成连续张量，也没有写时复制、没有请求调度器。它换来的是容量与共享，墙钟反而更慢（1.10×）。
- **默认的 KV cache 仍是最朴素的那一版**（逐步 `torch.cat`、无预分配）。分页缓存是并行存在的另一条实现路径（`paged_kv.py`），没有替换掉默认路径。
- **同一配置的重复测量本身就有波动**，实测在 1.6%～7.7% 之间，所以只有远大于这个量级的差距才当作结论。
- **计时取自会降频的笔记本。** 每次测量都用 `torch.cuda.synchronize()` 包围并对重复运行取最小值，但它仍然是墙钟数据，不宜当作精确基准。

## 九、致谢与参考资料

**代码与数据来源**

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) —— 架构、权重初始化方案、残差缩放均沿用其思路。本仓库与它的差别有五处：注意力拆成三个投影（GQA 必需）、归一化层可切换、生成时带 KV cache（上游 nanoGPT 未实现）、注意力实现可在三种路径间切换、另有一条分页 KV cache 实现。
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
| FlashAttention（分块 + 在线 softmax 的原始论文） | Dao et al., 2022 · [2205.14135](https://arxiv.org/abs/2205.14135) |
| PagedAttention / vLLM | Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 · [2309.06180](https://arxiv.org/abs/2309.06180) |

## License

[MIT](LICENSE)

---
---

# mygpt (English)

**A character-level GPT (~3.2M params) built from scratch — no `transformers`, no `torchtext` — with five switchable inference optimisations, each backed by an actual before/after measurement.**

I built this to understand the *internals* of modern LLM inference (KV cache, GQA, RMSNorm) rather than take the claims on faith, and then to push the same treatment onto two things inference engines do (fused attention, paged KV cache). I could not find a small enough repo where you can flip one switch at a time and see both sides of the numbers.

```bash
pip install -r requirements.txt
python train.py --run_name base      # ~75 s on a laptop GPU
python sample.py --prompt "Shakespeare"   # continue writing from your own prompt
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
- **D · Fused attention (SDPA)** — the baseline materialises the whole `Tq × Tk` score matrix; `F.scaled_dot_product_attention` does not. At T=2048 / batch 8 / bf16: **3.68×** faster, peak memory 607.6 → **143.6 MB**. Two things worth flagging. (1) This is PyTorch's fused entry point, **not** the FlashAttention kernel: this machine's PyTorch is not built with the `FLASH_ATTENTION` backend (`is_flash_attention_available()` is `False`), so the fused kernel that actually runs is cuDNN. (2) The default dispatcher silently picks `MATH` for grouped-query shapes — 21.73 ms / 1257.3 MB — so the backend list is pinned explicitly: 1.292 ms / 41.25 MB, **16.8×**. A third path, `chunked`, reimplements the FlashAttention algorithm structure (tiling + online softmax) in plain PyTorch: it does cut memory (604 → 185 MB) but is *slower* than the baseline (53.91 → 131.66 ms). That is the point of including it — the win comes from doing everything inside one kernel, not from tiling by itself.
- **E · Paged KV cache** — fixed-size physical blocks in a shared pool; each sequence holds only a block table. Serving 8 requests that actually use 254 tokens: reserved-allocation baseline 16.00 MB vs paged **1.188 MB** (waste 93.8% → **16.45%**). Eight sequences sharing one 64-token prefix: 2.000 → **0.250 MB (−87.5%)**, with one prefill instead of eight. Correctness first again: paged and `torch.cat` paths emit **byte-identical tokens**. **Wall clock is 1.10× slower** (128.68 → 141.61 ms) — this is a pure-Python stand-in for the idea, so it trades speed for capacity, and it is reported that way.

**The result I did not expect.** My first benchmark gave **0.98×** — the cache was *slower*. Instead of retuning until the number looked good, I measured where the time actually goes: one forward pass takes 10.24 ms at T=1 and 11.59 ms at T=512. A 512× increase in sequence length barely moves the time, so the cost is **fixed overhead** (Python + kernel launches) — and at batch 1 there is almost no GPU work to remove. Raise the batch and the cache starts paying off: 30.5× at batch 32. *KV cache removes arithmetic, not overhead.*

**Caveats.** 3.2M params, 1.1 MB of text — a measurement vehicle, not a usable language model. Single seed, single run, one thermally-throttling laptop; val-loss deltas of ~0.005 are noise, and back-to-back runs of the same configuration vary by 1.6–7.7%, so only gaps well beyond that are treated as results. The 30.5× is not a universal constant. The architecture is GPT-2 style (no RoPE / SwiGLU) and the default KV cache is the naive `torch.cat` version. Attention runs through PyTorch's fused entry point, **not** FlashAttention itself — and only in bf16 (in fp32 the fused backends are unavailable for grouped-query shapes). The paged cache is a Python-level proof of concept: no copy-on-write, no request scheduler, and attention still gathers blocks back into a contiguous tensor. Note also that in this small setting GQA *lowers* val loss too — that is a small-model artefact and must not be read as "GQA is generally better".

**References.** Architecture follows [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT); TinyShakespeare comes from [char-rnn](https://github.com/karpathy/char-rnn). RMSNorm — [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467); MQA — [Shazeer, 2019](https://arxiv.org/abs/1911.02150); GQA — [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245); FlashAttention — [Dao et al., 2022](https://arxiv.org/abs/2205.14135); PagedAttention / vLLM — [Kwon et al., 2023](https://arxiv.org/abs/2309.06180).

**License:** [MIT](LICENSE)

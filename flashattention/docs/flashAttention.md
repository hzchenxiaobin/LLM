# 深入浅出理解 Flash Attention

Flash Attention 是由 Tri Dao 等人在 2022 年提出的一种计算自注意力（Self-Attention）的算法。可以说，它是近年来大模型（LLM）能够支持超长上下文（Context Window，如 32k、128k 甚至 1M）的最核心基石之一。

理解 Flash Attention，最关键的是要明白：它不是通过改变数学公式来近似计算，而是通过「硬件感知（Hardware-aware）」来优化计算过程，实现了一字不差的精确计算（Exact Attention），同时大幅提升了速度并降低了显存占用。

## 1. 痛点：标准 Attention 为什么慢？

在标准的 Transformer 中，Attention 的计算公式为：

$$\mathbf{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

假设序列长度为 $N$，特征维度为 $d$。标准计算过程包含以下几步：

1. 计算 $S = QK^T$（复杂度 $O(N^2)$）
2. 计算 $P = \text{softmax}(S)$（复杂度 $O(N^2)$）
3. 计算 $O = PV$（复杂度 $O(N^2)$）

真正的问题不在于计算量（FLOPs），而在于「内存墙（Memory Wall）」。

GPU 的存储分为两级：

- **HBM（高带宽内存 / 显存）**：容量大（如 80GB），但读写速度相对较慢。
- **SRAM（片上缓存）**：容量极小（如 20MB），但在计算核心旁边，读写速度极快。

### 图示：标准 Attention 的内存噩梦

在标准 Attention 中，巨大的 $O(N^2)$ 中间矩阵 $S$ 和 $P$ 必须被来回搬运：

```text
[ HBM (慢速, 容量大) ]                      [ SRAM (快速, 容量小) ]
   Q, K ----------------- 读取 ------------> 计算 S = Q*K^T
   S <------------------- 写入 -------------
   S -------------------- 读取 ------------> 计算 P = Softmax(S)
   P <------------------- 写入 -------------
   P, V ----------------- 读取 ------------> 计算 O = P*V
   O <------------------- 写入 -------------
```

每一次中间结果的读写，都是 $O(N^2)$ 的数据量。GPU 的计算单元实际上在「干等」数据的搬运。

**通俗比喻**：想象你在一个很小的厨房（SRAM）做饭，但所有的食材都在地下室的大仓库（HBM）里。标准做法是：你把所有的肉（Q 和 K）搬到厨房切好，做成半成品（S 矩阵），然后把半成品全部搬回地下室。接着为了调味（Softmax），你又把半成品全搬上来，调完味（P 矩阵）再搬回去。最后再搬上来加蔬菜（V）炒熟（O）。绝大部分时间都浪费在了上下楼梯搬东西上。

## 2. Flash Attention (V1) 的三大核心创新

Flash Attention 的目标是：尽可能把数据留在 SRAM 中计算完毕，避免中间结果（S 和 P）在 HBM 和 SRAM 之间来回搬运。

### 图示：Flash Attention 的优雅流程

Flash Attention 将庞大的矩阵切分成小块（Blocks），在 SRAM 中完成全部计算，最后只把结果 $O$ 写回：

```text
[ HBM ]                                     [ SRAM ]
 Q,K,V 块 ---- 分块读取 (Tiling) ----------> | 1. 计算局部 S_block
                                            | 2. 更新 Online Softmax
                                            | 3. 计算局部 O_block
   O_最终 <---- 仅写入最终结果 O_block -------+ 
```

### 核心一：分块计算 (Tiling)

既然 $N \times N$ 的矩阵太大，SRAM 装不下，那就切块（Block）。

Flash Attention 将 $Q$、$K$、$V$ 矩阵切成一个个小块，每次只把一小块数据从 HBM 载入 SRAM，在 SRAM 内完成矩阵乘法和 Softmax 操作，最后把最终结果 $O$ 的一小块写回 HBM。

显存占用从 $O(N^2)$ 直接降到了 $O(N)$。

### 核心二：在线 Softmax (Online Softmax)

分块计算遇到了一个巨大的数学障碍：Softmax 是全局操作。

如果不知道全局信息，就没法算出正确的概率。Online Softmax 技巧允许我们在遍历数据块时，维护并更新两个局部变量：局部最大值 $m$ 和局部指数和 $l$。

当读入新的数据块时，用新的局部最大值去「修正」之前计算过的结果。这样即使每次只看一部分数据，最后得到的结果也和全局 Softmax 算出来的一模一样。

### 核心三：反向传播的重计算 (Recomputation)

为了节省显存，前向传播时我们根本没有保存庞大的中间矩阵（S 和 P）。

反向传播怎么算梯度？重计算。反向传播时，利用保留的 $Q$、$K$、$V$ 重新分块计算一次局部的 Softmax 结果，然后再算梯度。因为避免了 HBM 的海量数据读取，即便是重新计算一遍，总时间也比去 HBM 里读数据快得多。

## 3. 伪代码对比：体会 SRAM 中的魔法

### 传统 Attention 伪代码

```python
def standard_attention(Q, K, V):
    S = Q @ K.transpose() # 写入 HBM
    P = softmax(S)        # 从 HBM 读 S，将 P 写入 HBM
    O = P @ V             # 从 HBM 读 P 和 V，将 O 写入 HBM
    return O, S, P        # 保存 S 和 P 用于反向传播计算梯度
```

### Flash Attention V1 简化伪代码

```python
def flash_attention_v1(Q, K, V, block_size_M, block_size_N):
    O = zeros(N, d)
    l = zeros(N, 1) # 全局指数和
    m = fill(N, 1, -inf) # 全局最大值

    # 外层循环：遍历 K, V 
    for j in range(0, N, block_size_N):
        K_j = K[j : j + block_size_N]
        V_j = V[j : j + block_size_N]

        # 内层循环：遍历 Q 
        for i in range(0, N, block_size_M):
            Q_i = Q[i : i + block_size_M]
            O_i, l_i, m_i = O[i...], l[i...], m[i...]

            # 1. 局部矩阵乘法 (SRAM)
            S_ij = Q_i @ K_j.transpose()

            # 2. Online Softmax 更新 (SRAM)
            m_ij = max(S_ij, axis=1)
            m_new = max(m_i, m_ij)   # 修正后的全局最大值
            
            P_ij = exp(S_ij - m_new)
            l_new = l_i * exp(m_i - m_new) + sum(P_ij, axis=1)

            # 3. 更新局部输出 (SRAM) -> 注意这里有繁琐的缩放操作
            O_i = (O_i * l_i * exp(m_i - m_new) + P_ij @ V_j) / l_new

            # 写回 HBM
            m[i...], l[i...], O[i...] = m_new, l_new, O_i
    return O
```

## 4. 进阶：Flash Attention 2 到底优化了什么？

如果说 Flash Attention V1 的核心是**「打破内存墙」，把瓶颈从访存变成了计算**；那么 Flash Attention-2 的核心就是**「极致压榨计算力」**。

在 V1 时代，虽然速度快了，但人们发现 A100 GPU 上的 Tensor Core（专门做矩阵乘法的硬件）利用率只有 25%～40%。V2 的目标就是让 Tensor Core 跑满，最终将利用率提升到了惊人的 70%～73%，整体速度又翻了约一倍。

它做了三大核心升级：

### 升级一：减少非矩阵乘法开销 (Fewer non-matmul FLOPs)

仔细看上面 V1 的伪代码中 `O_i` 的更新公式：`(O_i * l_i * exp(...) + P_ij @ V_j) / l_new`。

每一次内层循环，都要进行大量的除法、乘法和指数运算。这些**非矩阵乘法（Non-matmul）**运算不能由 Tensor Core 执行，只能由普通的 CUDA Core 执行，速度慢，严重拖慢了整体节奏。

**V2 的解法**：通过代数变换，推迟缩放（Rescaling）。在内层循环中，它只做加法和矩阵乘法，把所有需要除以分母（指数和）的操作，全部推迟到整行块计算完毕后，在最外层循环仅仅执行一次。这让 Tensor Core 得以火力全开。

### 升级二：序列维度的并行化 (Sequence Parallelism)

在 V1 中，并行化的单位是 Batch（批次）和 Head（注意力头）。系统把不同的 Batch 和 Head 分配给 GPU 上不同的计算单元（SM，流多处理器）。

**痛点**：如今处理超长文本（Long Context）时，Batch Size 往往是 1，Head 数量也是固定的（比如 32）。而一块 A100 GPU 有 108 个计算单元，这就导致大量计算单元处于闲置状态（即「饿肚子」）。

**V2 的解法**：在 Sequence Length（序列长度）维度上也进行切分并行。即使 Batch = 1，长达 100k 的序列也会被切成许多小块，均匀分发给所有的 GPU 计算单元。这极大提升了长文本场景下的 GPU 占用率。

### 升级三：优化循环顺序与共享内存同步 (Better Work Partitioning)

在一个 GPU 计算单元内部，有多个更小的执行线程组（称为 Warp）。它们共用一块 SRAM（Shared Memory）。

**V1 的循环结构**：外循环遍历 K、V，内循环遍历 Q。这意味着在计算一小块数据时，不同的 Warp 之间需要频繁地把局部的 $l_i$ 和 $m_i$ 写入 SRAM 进行同步和共享。

**V2 的循环结构颠倒**：外循环遍历 Q，内循环遍历 K、V。

这样调整后，系统可以将一个巨大的 Q 块固定分配给特定的 Warp。这个 Warp 只需要自己闷头遍历 K 和 V 就能完成整行的 Attention 计算。Warp 之间不再需要频繁通信，大幅减少了 SRAM 的读写冲突（Shared Memory Traffic）。

### 图示：V1 与 V2 的循环顺序对比

```text
FlashAttention-1                          FlashAttention-2
─────────────────                         ─────────────────
for j:  K_j, V_j 块                         for i:  Q_i 块
    for i:  Q_i 块      ← 内层频繁换 Q           for j:  K_j, V_j 块  ← 内层扫完整条 KV
        Online softmax                         Online softmax
        (Warp 间要同步 m, l, O)                  (每个 Warp 独占一行块的 m, l, O)
```

### 图示：FlashAttention-2 的数据流（固定 Q 块、顺序扫 K/V）

同一 Q 块驻留在 SRAM/寄存器中，沿序列维依次消费各 K、V 块；整行 Online Softmax 完成后，仅对输出做一次按行除法（归一化），减少内层循环里的标量除法。

```text
[ HBM ]                                              [ SRAM / 寄存器 ]
                                                     
  Q_i 块 ---- 一次加载，外层固定 ------------------>  Q_i 常驻
                                                      |
  K_j,V_j ---- j=0,1,2,... 内层依次加载 ----------->  S_ij = Q_i K_j^T
                                                      |    Online: 更新 m, l
                                                      |    累加 O_i（先不除 l）
                                                      v
  O[i:...] <--- 内层 j 结束后写回 ------------------  O_i ← O_i / l_i  （每行仅一次除法）
```

### 伪代码：FlashAttention-2（循环交换 + 行末归一化）

下面与 V1 使用相同的 Online Softmax 数学，但**外层遍历 Q 块、内层遍历 K/V 块**；**把对输出的除法推迟到该 Q 块处理完所有 K/V 之后**，从而减少内层循环中的非矩阵乘法开销。

```python
def flash_attention_v2(Q, K, V, Br, Bc):
    """
    Q, K, V: [N, d]
    Br: Q 方向块大小（行块）；Bc: K/V 方向块大小（列块）
    """
    N, d = Q.shape
    O = zeros(N, d)

    # 外层：Q 块 —— 与 V1 相反；便于 Warp 绑定行块、减少同步
    for i in range(0, N, Br):
        Qi = Q[i : i + Br]                    # [Br, d]
        Oi = zeros(Br, d)
        li = zeros(Br)
        mi = full(Br, -inf)

        # 内层：K,V 块 —— 顺序扫过整段序列
        for j in range(0, N, Bc):
            Kj = K[j : j + Bc]
            Vj = V[j : j + Bc]

            S_ij = Qi @ Kj.T                  # [Br, Bc]  GEMM，走 Tensor Core
            m_row = max(S_ij, axis=1)         # 本块行内 max
            m_new = maximum(mi, m_row)      # 逐元素 max，与历史全局 max 合并

            # 数值稳定的 exp；P_ij 尚未除以最终 softmax 分母
            P_ij = exp(S_ij - m_new[:, None])
            li = li * exp(mi - m_new) + sum(P_ij, axis=1)
            Oi = Oi * exp(mi - m_new)[:, None] + P_ij @ Vj

            mi = m_new

        # 该 Q 块对应的所有 K/V 块处理完毕后再归一化（减少内层除法）
        O[i : i + Br, :] = Oi / li[:, None]

    return O
```

说明：`maximum`、`exp`、`sum` 等为按行向量运算；真实实现还会融合掩码、变长序列、`scale = 1/sqrt(d)` 等，且 `li` 会加极小量避免除零。序列维并行体现在：不同的 `i` 区间（不同 Q 块）可派发到不同 SM/线程块同时执行。

## 5. 总结：性能演进的阶梯

理解 Flash Attention 及其后续版本，本质上是理解算法与底层硬件（计算机体系结构）的深度绑定：

- **Standard Attention**：算法上最直接，但败给了硬件的「内存墙」（显存读写太慢）。
- **Flash Attention V1**：用 Tiling 和 Online Softmax 拯救了内存读写，把 IO 复杂度降到最低，解决了长文本显存爆炸的问题。
- **Flash Attention V2**：发现内存瓶颈解决后，计算单元的利用率还不高，于是重构循环、减少除法、增加序列并行，把 Tensor Core 的算力榨干。
- **Flash Attention V3 (2024)**：进一步针对英伟达 Hopper 架构（如 H100 GPU）的专属硬件特性（如 TMA 张量内存加速器、WGMMA）进行极致的汇编级优化，速度再次翻倍。

**核心启发**：在现代深度学习中，减少计算量（FLOPs）固然重要，但如何让数据在硬件存储层级中高效流动，往往更能带来数十倍的性能飞跃。

# CUDA Top-K 算子性能优化教程（面向 RTX 5090）

> 本文档介绍如何在 NVIDIA RTX 5090 (Blackwell 架构) 上高效实现 1D Top-K 算子。

## 目录

1. [背景与硬件认知](#0-背景与硬件认知)
2. [Step 1: 朴素版本 (单线程串行)](#step-1-朴素版本-单线程串行)
3. [Step 2: 进阶版本 (Block级协作 + Shared Memory)](#step-2-进阶版本-block级协作--shared-memory)
4. [Step 3: 工业级利器 (Warp级协作 + Warp Primitives)](#step-3-工业级利器-warp级协作--warp-primitives)
5. [Step 4: RTX 5090 专属极致优化](#step-4-rtx-5090--blackwell-专属极致优化)
6. [总结与路线图](#总结优化路径路线图)
7. [推荐学习资源](#推荐的下一步学习)

---

## 0. 背景与硬件认知

### RTX 5090 (Blackwell 架构) 特点

| 特性 | 说明 |
|------|------|
| **海量 SM** | 计算能力极强，但 Top-K 通常是**访存密集型 (Memory-bound)**，计算易被访存延迟掩盖 |
| **极高显存带宽** | 通常超过 1.5 TB/s |
| **庞大 L2 Cache** | 非常适合在 Cache 中复用数据 |

### 问题定义：1D Top-K

假设输入为一维数组 `[N]`，需要在其中找出最大的 `K` 个元素（只输出值，不输出索引）。输出为 `[K]` 的值数组。

**典型场景**：LLM 推理中的词表采样
- `N ≈ 32000 ~ 128000`（词表大小）
- `K ≈ 10 ~ 50`（采样数量）
- **注意**：本版本只输出 Top-K 值，不输出对应索引

---

## Step 1: 朴素版本 (单线程串行)

### 核心思想

使用单个线程串行处理整个数组，在寄存器中维护大小为 `K` 的数组，遍历 `N` 个元素执行插入排序。

### CUDA 代码实现

```cuda
// V1: 单线程串行插入排序 (1D输入，无索引输出)
__global__ void topk_v1_kernel(const float* input, float* out_vals, int N, int K) {
    // 只使用线程0处理
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // 初始化 Top-K 数组 (存储在寄存器中)
    float top_vals[32];  // 假设 K 最大不超过 32
    for (int i = 0; i < K; ++i) {
        top_vals[i] = -INFINITY;
    }

    // 串行遍历 N 个元素
    for (int i = 0; i < N; ++i) {
        float val = input[i];
        // 如果比当前 Top-K 中的最小值大，则插入
        if (val > top_vals[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && val > top_vals[pos - 1]) {
                top_vals[pos] = top_vals[pos - 1];
                pos--;
            }
            top_vals[pos] = val;
        }
    }

    // 写回 Global Memory
    for (int i = 0; i < K; ++i) {
        out_vals[i] = top_vals[i];
    }
}
```

### 性能痛点分析

- **寄存器溢出 (Register Spilling)**
  - 当 `K` 稍大（如 64）时，`top_vals` 消耗大量寄存器
  - 编译器会将它们溢出到 Local Memory（即 Global Memory 的一部分）
  - **后果**：性能暴跌，RTX 5090 沦为计算器

- **算力闲置**
  - 当 `N` 很大时，单个线程串行处理太慢
  - 无法充分利用 RTX 5090 庞大的 SM

---

## Step 2: 进阶版本 (Block级协作 + Shared Memory)

### 核心思想

用一个 **Block** 内的所有线程协作处理整个数组。

每个线程负责读取不同部分（Grid-Stride Loop），维护自己的 Local Top-K，最后通过 **Shared Memory** 进行 Block 级别规约 (Reduction)。

### 并行策略

假设 Block 大小为 256：

1. **线程 0** 读取第 0, 256, 512... 个元素
2. **线程 1** 读取第 1, 257, 513... 个元素

这样保证了**合并访存 (Coalesced Access)**，极大利用 RTX 5090 的带宽。

3. 每个线程得到一个 Local Top-K，256 个线程将结果存入 Shared Memory
4. 在 Shared Memory 中进行归并排序或规约，得到最终的 Global Top-K

### 性能提升点

- ✅ **完美合并访存**：内存读取带宽利用率接近 100%
- ✅ **分摊寄存器压力**：每个线程只处理 `N / 256` 个元素

### 遗留问题

- ⚠️ Shared Memory 的 **Bank Conflict**
- ⚠️ Block 级别的 `__syncthreads()` 同步开销

---

## Step 3: 工业级利器 (Warp级协作 + Warp Primitives)

### 核心思想

在实际场景中（如 Transformer 推理），`K` 通常不大（`K = 1 ~ 32`）。

**关键洞察**：Warp（32 个线程）是 GPU 调度的最小单位，Warp 内线程同步是**隐式且极快**的。

**业界最常用策略**：一个 **Warp** 处理整个数组，使用 CUDA 的 Warp 级洗牌指令（`__shfl_down_sync`, `__shfl_sync`）直接在寄存器层面交换数据，**彻底绕开 Shared Memory**！

### 核心操作：Warp 规约 (Warp Reduce)

```cuda
// Warp 级别找最大值示例
__inline__ __device__ void warp_reduce_max(float& val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        if (other_val > val) {
            val = other_val;
        }
    }
}
```

### Warp-per-Array Top-K 流程

1. 使用 1 个 Warp（32 threads）处理整个数组
2. 每个线程串行/循环读取数据（每次读 32 个连续元素，完美合并访存），维护大小为 `K` 的寄存器数组（Local Top-K）
3. 数据读完后，Warp 内 32 个线程共有 `32 × K` 个候选值
4. 使用基于 `__shfl_sync` 的归并网络，选出最终的 `K` 个最大值

### 优点

- ✅ **无 `__syncthreads()`**：完全利用寄存器带宽，极低延迟
- ✅ **TensorRT / FasterTransformer 默认实现**：工业级验证

---

## Step 4: Radix Select 优化策略

V4版本采用**Radix-Style**的选择策略，是V3 Warp-per-Array的扩展版本。

### 核心思想

借鉴**QuickSelect**和**Radix Sort**的思想，通过数值的二进制表示逐层筛选：

1. **粗筛选 (Coarse Selection)**：每个线程先独立收集一个较大的候选集
2. **分层归约 (Hierarchical Reduction)**：通过Warp/Block级别的规约，逐步精确到Top-K

### CUDA代码亮点

```cuda
// 两阶段算法
// 阶段1：每个线程粗筛（Local Top-K）
for (int i = lane_id; i < N; i += WARP_SIZE) {
    radix_insert_topk(local_vals, K, input[i]);
}

// 阶段2：Warp/Block级别精排
for (int k = 0; k < K; ++k) {
    // 使用warp shuffle找出当前全局最大值
    warp_reduce_max(...);
    // 获胜线程推进指针，下一轮找次大值
}
```

### 与V3的区别

| 特性 | V3 Warp-per-Array | V4 Radix Select |
|------|-------------------|-----------------|
| **适用K** | K ≤ 32 | K ≤ 32 (或更大) |
| **存储** | 纯寄存器 | 寄存器 + Shared Memory |
| **同步** | 无 `__syncthreads()` | 需要Block同步 |
| **算法** | 插入排序 + Warp归约 | 粗筛 + 分层选择 |
| **优势** | 极致低延迟 | 更灵活，支持更大K值 |

---

## Step 5: RTX 5090 / Blackwell 专属极致优化

当你实现到 Warp-per-Array，已超过 90% 的初学者。要完全榨干 RTX 5090，考虑以下前沿优化：

### 1. 向量化访存 (Vectorized Memory Access)

不要一次读一个 `float`，使用 `float4`（128-bit 访存）一次性读取 4 个元素，减少内存事务数量。

```cuda
const float4* vec_input = reinterpret_cast<const float4*>(input);
float4 data = vec_input[tid];  // 一次取 4 个 float
```

### 2. 半精度与张量类型 (BF16 / FP8)

RTX 5090 对 FP8 和 BF16 有极其恐怖的吞吐量。将输入转为 `__nv_bfloat162`（包含两个 bf16 的向量），**显存带宽瓶颈直接减半**。

### 3. 异步内存拷贝 (TMA / cp.async)

Hopper 和 Blackwell 架构支持 **TMA (Tensor Memory Accelerator)**，允许 SM 绕过寄存器直接将数据从 Global Memory 搬运到 Shared Memory。

当 `K` 较大（如 `K = 1024`），Warp-per-Array 寄存器不够用时，必须回退到 Shared Memory。此时使用 `cuda::memcpy_async` 可以将计算和下一轮访存**流水线化**，掩盖延迟。

---

## 总结：优化路径路线图

### 1. 根据 K 的大小选择策略

| K 的范围 | 推荐策略 | 实现版本 |
|----------|----------|----------|
| `K ≤ 32` | **Warp-per-Array + Warp Shuffle Primitives** ⭐ 最优！ | V3/V4 |
| `32 < K ≤ 256` | **Radix Select + Block Shared Memory** ⭐ 推荐 | V4 |
| `K > 256` | 多 Pass 的 Radix Sort（如 CUB 库的 `cub::DeviceRadixSort`） | CUB |

### 版本对比速查

| 版本 | 核心思想 | 特点 | 适用场景 |
|------|----------|------|----------|
| **V1** | 单线程串行 + 插入排序 | 实现简单，但性能差 | 学习/调试 |
| **V2** | Block级协作 + Shared Memory | 合并访存好，但有同步开销 | 通用场景 |
| **V3** | Warp级协作 + Warp Primitives | 无同步，极致性能 | **K ≤ 32，生产首选** |
| **V4** | Radix Select + 分层归约 | 支持大K，稳定性好 | **K > 32，大K场景** |

### 2. 通用优化原则

- **优化访存**：确保所有 Global Memory 读写必须是**合并的 (Coalesced)**，引入 `float4`
- **减少分支**：插入排序或归并时，尽量使用无分支代码或 `#pragma unroll`，防止指令流水线被打断

---

## Kernel 调用方式（当前版本）

所有版本均已简化为 **1D输入，无Batch轴，只输出值** 的形式：

```cuda
// V1: 单线程串行
topk_v1_kernel<<<1, 1>>>(input, out_vals, N, K);

// V2: Block级并行
topk_v2_kernel<<<1, 256>>>(input, out_vals, N, K);

// V3: Warp级并行
topk_v3_kernel<<<1, 32>>>(input, out_vals, N, K);

// V4: 小数据量(N <= 1024)
topk_v4_warp_kernel<<<1, 32>>>(input, out_vals, N, K);

// V4: 大数据量(N > 1024)
int block_size = 256;
int num_warps = block_size / 32;
size_t shared_mem = num_warps * K * sizeof(float);
topk_v4_kernel<<<1, block_size, shared_mem>>>(input, out_vals, N, K);
```

---

## 推荐的下一步学习

1. **NVIDIA CUB 库**
   - 查阅 `BlockRadixSort` 源码实现

2. **FasterTransformer / vLLM**
   - 搜索 `topk_kernels.cu` 源码
   - 学习 Beam Search 中的纯粹 Warp-level 优化艺术

---

*文档版本: 2.0 | 适用架构: NVIDIA Blackwell (RTX 5090)*
*更新说明: 已修改为1D输入，无Batch轴，只输出值（无索引）*

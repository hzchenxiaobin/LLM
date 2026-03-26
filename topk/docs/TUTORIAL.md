# CUDA Top-K 算子性能优化教程（面向 RTX 5090）

> 本文档介绍如何在 NVIDIA RTX 5090 (Blackwell 架构) 上高效实现 Batched Top-K 算子。

## 目录

1. [背景与硬件认知](#0-背景与硬件认知)
2. [Step 1: 朴素版本 (Thread-per-Row)](#step-1-朴素版本-thread-per-row)
3. [Step 2: 进阶版本 (Block-per-Row + Shared Memory)](#step-2-进阶版本-block-per-row--shared-memory)
4. [Step 3: 工业级利器 (Warp-per-Row + Warp Primitives)](#step-3-工业级利器-warp-per-row--warp-primitives)
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

### 问题定义：Batched Top-K

假设输入矩阵维度为 `[BatchSize, N]`，需要在每一行（长度为 `N`）中找出最大的 `K` 个元素及其索引。输出为 `[BatchSize, K]` 的值矩阵和索引矩阵。

**典型场景**：LLM 推理中的词表采样
- `N ≈ 32000 ~ 128000`（词表大小）
- `K ≈ 10 ~ 50`（采样数量）

---

## Step 1: 朴素版本 (Thread-per-Row)

### 核心思想

一个线程处理一行。每个线程在寄存器或局部内存中维护大小为 `K` 的数组，遍历 `N` 个元素，执行插入排序。

### CUDA 代码实现

```cuda
// V1: Thread-per-Row 插入排序
template <typename T>
__global__ void topk_v1_kernel(const T* input, T* out_vals, int* out_inds, 
                               int Batch, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Batch) return;

    const T* row_input = input + row * N;
    T* row_out_vals = out_vals + row * K;
    int* row_out_inds = out_inds + row * K;

    // 初始化 Top-K 数组 (存储在寄存器或 Local Memory)
    T top_vals[128];  // 假设 K 最大不超过 128
    int top_inds[128];
    for (int i = 0; i < K; ++i) {
        top_vals[i] = -INFINITY;
        top_inds[i] = -1;
    }

    // 遍历该行的 N 个元素
    for (int i = 0; i < N; ++i) {
        T val = row_input[i];
        // 如果比当前 Top-K 中的最小值大，则插入
        if (val > top_vals[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && val > top_vals[pos - 1]) {
                top_vals[pos] = top_vals[pos - 1];
                top_inds[pos] = top_inds[pos - 1];
                pos--;
            }
            top_vals[pos] = val;
            top_inds[pos] = i;
        }
    }

    // 写回 Global Memory
    for (int i = 0; i < K; ++i) {
        row_out_vals[i] = top_vals[i];
        row_out_inds[i] = top_inds[i];
    }
}
```

### 性能痛点分析

- **寄存器溢出 (Register Spilling)**
  - 当 `K` 稍大（如 64）时，`top_vals` 和 `top_inds` 消耗大量寄存器
  - 编译器会将它们溢出到 Local Memory（即 Global Memory 的一部分）
  - **后果**：性能暴跌，RTX 5090 沦为计算器

- **非合并访存 (Uncoalesced Memory Access)**
  - 连续线程读取同一 `i` 时，访问的是跨行数据 `row_input[i]`
  - 这种读法对显存带宽极不友好

- **算力闲置**
  - 当 `N` 很大时，单个线程串行处理太慢
  - 无法充分利用 RTX 5090 庞大的 SM

---

## Step 2: 进阶版本 (Block-per-Row + Shared Memory)

### 核心思想

既然一个线程处理一行太慢，就用一个 **Block** 处理一行（或几行）。

每个线程负责读取该行不同部分（Grid-Stride Loop），维护自己的 Local Top-K，最后通过 **Shared Memory** 进行 Block 级别规约 (Reduction)。

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

## Step 3: 工业级利器 (Warp-per-Row + Warp Primitives)

### 核心思想

在实际场景中（如 Transformer 推理），`K` 通常不大（`K = 1 ~ 32`）。

**关键洞察**：Warp（32 个线程）是 GPU 调度的最小单位，Warp 内线程同步是**隐式且极快**的。

**业界最常用策略**：一个 **Warp** 处理一行，使用 CUDA 的 Warp 级洗牌指令（`__shfl_down_sync`, `__shfl_sync`）直接在寄存器层面交换数据，**彻底绕开 Shared Memory**！

### 核心操作：Warp 规约 (Warp Reduce)

```cuda
// Warp 级别找最大值示例
__inline__ __device__ void warp_reduce_max(float& val, int& idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}
```

### Warp-per-Row Top-K 流程

1. 分配一个 Warp（32 threads）处理一行
2. 每个线程串行/循环读取数据（每次读 32 个连续元素，完美合并访存），维护大小为 `K` 的寄存器数组（Local Top-K）
3. 数据读完后，Warp 内 32 个线程共有 `32 × K` 个候选值
4. 使用基于 `__shfl_sync` 的归并网络（Bitonic Merge Network）或多次 Warp Reduce，选出最终的 `K` 个最大值

### 优点

- ✅ **无 `__syncthreads()`**：完全利用寄存器带宽，极低延迟
- ✅ **TensorRT / FasterTransformer 默认实现**：工业级验证

---

## Step 4: RTX 5090 / Blackwell 专属极致优化

当你实现到 Warp-per-Row，已超过 90% 的初学者。要完全榨干 RTX 5090，考虑以下前沿优化：

### 1. 向量化访存 (Vectorized Memory Access)

不要一次读一个 `float`，使用 `float4`（128-bit 访存）一次性读取 4 个元素，减少内存事务数量。

```cuda
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 data = vec_input[tid];  // 一次取 4 个 float
```

### 2. 半精度与张量类型 (BF16 / FP8)

RTX 5090 对 FP8 和 BF16 有极其恐怖的吞吐量。将输入转为 `__nv_bfloat162`（包含两个 bf16 的向量），**显存带宽瓶颈直接减半**。

### 3. 异步内存拷贝 (TMA / cp.async)

Hopper 和 Blackwell 架构支持 **TMA (Tensor Memory Accelerator)**，允许 SM 绕过寄存器直接将数据从 Global Memory 搬运到 Shared Memory。

当 `K` 较大（如 `K = 1024`），Warp-per-Row 寄存器不够用时，必须回退到 Shared Memory。此时使用 `cuda::memcpy_async` 可以将计算和下一轮访存**流水线化**，掩盖延迟。

---

## 总结：优化路径路线图

### 1. 根据 K 的大小选择策略

| K 的范围 | 推荐策略 |
|----------|----------|
| `K ≤ 32` | **Warp-per-Row + Warp Shuffle Primitives** ⭐ 最优！ |
| `32 < K ≤ 256` | Block-per-Row + Shared Memory + Bitonic Sort |
| `K > 256` | 多 Pass 的 Radix Sort（如 CUB 库的 `cub::DeviceRadixSort`） |

### 2. 通用优化原则

- **优化访存**：确保所有 Global Memory 读写必须是**合并的 (Coalesced)**，引入 `float4`
- **减少分支**：插入排序或归并时，尽量使用无分支代码或 `#pragma unroll`，防止指令流水线被打断

---

## 推荐的下一步学习

1. **NVIDIA CUB 库**
   - 查阅 `BlockRadixSort` 源码实现

2. **FasterTransformer / vLLM**
   - 搜索 `topk_kernels.cu` 源码
   - 学习 Beam Search 中的纯粹 Warp-level 优化艺术

---

*文档版本: 1.0 | 适用架构: NVIDIA Blackwell (RTX 5090)*

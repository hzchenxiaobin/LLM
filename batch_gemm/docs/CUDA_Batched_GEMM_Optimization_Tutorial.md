# CUDA Batched GEMM (批量矩阵乘法) 性能优化步步通

**目标平台**: NVIDIA RTX 5090 (Blackwell 架构)

**任务**: 计算 Strided Batched GEMM: $C[b] = A[b] \times B[b]$

**假设矩阵维度**: $A$ 为 $[B, M, K]$， $B$ 为 $[B, K, N]$，$C$ 为 $[B, M, N]$。均为行优先 (Row-Major) 存储。

## 优化阶梯一览

- **V0**: 朴素实现 (Naive) —— 了解基本逻辑
- **V1**: 共享内存分块 (Shared Memory Tiling) —— 解决访存瓶颈（Global Memory Bound）
- **V2**: 寄存器分块 (Register Tiling / 2D Tiling) —— 提升计算访存比，隐藏延迟
- **V3**: 向量化访存 (Vectorized Loads) & 解决 Bank Conflict —— 榨干显存带宽
- **V4**: Tensor Cores (RTX 5090 的杀手锏) —— 降维打击，拥抱硬件矩阵运算单元

## Step 0: 问题的基础映射

对于 Batched GEMM，最简单的方法是将批次维度 (Batch) 映射到 Grid 的 Z 维度 (`blockIdx.z`)。而矩阵的行和列映射到 X 和 Y 维度。

### 内存偏移计算

- $A$ 的起始指针：`A + batch_idx * M * K`
- $B$ 的起始指针：`B + batch_idx * K * N`

## V0: 朴素实现 (Naive Batched GEMM)

这是最符合直觉的写法。每个线程 (Thread) 负责计算输出矩阵 $C$ 中的一个元素。

```cuda
__global__ void bmm_naive(const float* A, const float* B, float* C, 
                          int B_size, int M, int N, int K) {
    // 1. 获取当前属于哪一个 Batch
    int batch_idx = blockIdx.z;
    if (batch_idx >= B_size) return;

    // 2. 计算当前线程负责 C 矩阵的行号和列号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // 3. 计算各个矩阵的 Batch 偏移量
        const float* A_batch = A + batch_idx * M * K;
        const float* B_batch = B + batch_idx * K * N;
        float* C_batch = C + batch_idx * M * N;

        // 4. 计算点积
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A_batch[row * K + k] * B_batch[k * N + col];
        }
        C_batch[row * N + col] = sum;
    }
}
```

### 痛点分析

虽然 RTX 5090 显存带宽极大，但这种写法的计算访存比 (Compute-to-Memory Ratio) 极低。为了计算 1 个乘加指令 (FMA)，需要从全局内存 (Global Memory) 读 2 个 float。A 和 B 矩阵的数据被重复读取了无数次，这会让你的 5090 处于"饿死"状态。

## V1: 共享内存分块 (Shared Memory Block Tiling)

**核心思想**: 类似于将大块蛋糕切成小块。我们将 $A$ 和 $B$ 切成 $TILE\_SIZE \times TILE\_SIZE$ 的小块，先从极其缓慢的全局内存加载到速度极快的共享内存 (Shared Memory) 中，然后在一个 Block 内的线程共享这些数据。

```cuda
#define TILE_SIZE 32

__global__ void bmm_shared_memory(const float* A, const float* B, float* C,
                                  int B_size, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= B_size) return;

    // 分配共享内存
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;

    float sum = 0.0f;

    // 在 K 维度上滑动分块
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // 1. 协同加载数据到共享内存 (注意边界检查)
        int k_A = t * TILE_SIZE + threadIdx.x;
        int k_B = t * TILE_SIZE + threadIdx.y;

        if (row < M && k_A < K) 
            sA[threadIdx.y][threadIdx.x] = A_batch[row * K + k_A];
        else 
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (k_B < K && col < N) 
            sB[threadIdx.y][threadIdx.x] = B_batch[k_B * N + col];
        else 
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        // 2. 线程同步，确保大家都加载完了
        __syncthreads();

        // 3. 使用共享内存中的数据进行计算
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // 4. 线程同步，确保大家算完了，准备加载下一个 Tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}
```

### 性能飞跃

全局内存的读取次数减少了约 TILE_SIZE (32) 倍！性能会有质的飞跃。

## V2: 寄存器分块 (Register Tiling / 2D Tiling)

### V1 的局限

在 V1 中，每个线程依然只计算 $C$ 的一个元素。这意味着每次乘加计算，线程都要从共享内存读 2 个数。虽然 Shared Memory 很快，但仍然受限于带宽，且无法掩盖指令延迟。

**核心思想**: 让每个线程负责计算一个小的矩阵块 (比如 $8 \times 8$)。把需要频繁访问的数据放在寄存器 (Registers) 中。寄存器是 GPU 上最快的存储器。

```cuda
// 概念伪代码：
// Block 大小管辖 128x128 的 C 矩阵区域。
// 每个线程管辖 8x8 的 C 矩阵区域 (Thread_Tile)。
// 这样一来，一个线程只需从共享内存读取 8 个 A 和 8 个 B，
// 就可以执行 8x8 = 64 次乘加运算 (FMA)！计算访存比提升了 4 倍！

// 线程内部的累加器全部是物理寄存器
float c_reg[8][8] = {0.0f};
float a_reg[8];
float b_reg[8];

for (int k = 0; k < K; k += BK) {
    // 1. Block 协同加载数据到 Shared Memory...
    __syncthreads();
    
    // 2. 从 Shared Memory 读入寄存器，在寄存器中计算
    for (int step = 0; step < BK; ++step) {
        // 读到寄存器
        #pragma unroll
        for(int i=0; i<8; ++i) a_reg[i] = sA[...];
        #pragma unroll
        for(int j=0; j<8; ++j) b_reg[j] = sB[...];
        
        // 寄存器级别的外积 (Outer Product)
        #pragma unroll
        for(int i=0; i<8; ++i) {
            #pragma unroll
            for(int j=0; j<8; ++j) {
                c_reg[i][j] += a_reg[i] * b_reg[j];
            }
        }
    }
    __syncthreads();
}
// 最终将 c_reg 写回 Global Memory 的 C 矩阵中
```

> 注：这里通常还会涉及到解决 Shared Memory Bank Conflict（例如通过把 sA 的列数设为奇数如 BK+1）的优化。

## V3: 向量化访存 (Vectorized Memory Access)

RTX 5090 拥有巨大的单次内存抓取能力。使用 float 每次只读取 4 Bytes。如果内存是连续的，我们可以使用内置类型 `float4`，一条指令读取 16 Bytes (128 bits)。这能最大化填满内存总线。

```cuda
// 核心操作是将 float* 转换为 float4* 进行加载
const float4* A_vec = reinterpret_cast<const float4*>(A_batch);
// 读取时：
float4 a_vals = A_vec[index]; 
// a_vals.x, a_vals.y, a_vals.z, a_vals.w 就包含了 4 个连续的浮点数
```

配合 V2 的寄存器分块，这构成了传统的基于 CUDA Cores 的极限优化 (SGEMM)。

## V4: Tensor Cores (RTX 5090 性能密码)

前面所有的优化（V1-V3）都在使用 CUDA Cores (FMA 单元)。但是，RTX 5090 (Blackwell) 的绝大部分算力都在 Tensor Cores 上！如果不使用 Tensor Cores，你的 5090 连 10% 的功力都没发挥出来。

NVIDIA 提供了 WMMA (Warp-level Matrix Multiply Accumulate) API 来调用 Tensor Cores。对于 RTX 5090，建议使用 TF32 (TensorFloat-32) 或者 FP16/BF16，因为 Tensor Core 原生不支持纯 FP32 乘法，但 TF32 允许你输入 FP32 格式的数据，硬件会自动截断尾数，利用 Tensor Core 加速，并累加为 FP32。

```cuda
#include <mma.h>
using namespace nvcuda;

// 使用 16x16x8 的 Warp 形状 (针对 TF32)
// 注意：以下是针对单个 Warp 的简述，实际使用需要结合 Block Tiling 组织多个 Warp

__global__ void bmm_tensor_core_tf32(const float* A, const float* B, float* C,
                                     int B_size, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= B_size) return;

    // 定义 WMMA 片段 (Fragments)
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;

    // 假设恰好可以被 16/8 整除，简化说明
    int row_offset = blockIdx.y * 16;
    int col_offset = blockIdx.x * 16;

    for (int k = 0; k < K; k += 8) {
        // 将 Global/Shared Memory 加载到 Tensor Core 寄存器
        wmma::load_matrix_sync(frag_a, A_batch + row_offset * K + k, K);
        wmma::load_matrix_sync(frag_b, B_batch + k * N + col_offset, N);

        // 执行 Tensor Core 矩阵乘法 (mma.sync)
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    // 将结果写回
    wmma::store_matrix_sync(C_batch + row_offset * N + col_offset, frag_c, N, wmma::mem_row_major);
}
```

## 面向 RTX 5090 的进阶方向 (Next Steps)

有了 5090 之后，使用 WMMA 只是起步，为了榨干架构红利，你还需要去了解：

- **PTX mma.sync / wgmma指令**: 绕过高级 API，直接手写 PTX 汇编调用异步 Tensor Core (Hopper/Blackwell 特性)
- **TMA (Tensor Memory Accelerator)**: Blackwell 架构的杀手锏。不需要线程执行 load 指令，直接配置 TMA 描述符，硬件会异步将 Global Memory 的 2D 矩阵块搬运到 Shared Memory，甚至解除了传统的访存限制！
- **Double Buffering / Pipeline**: 使用 `cuda::pipeline` 隐藏全局内存到共享内存的传输延迟
- **CUTLASS 库**: 工业界不会真的从零手写极致优化的 GEMM，而是使用 NVIDIA 的 CUTLASS 模板库。学会阅读 CUTLASS 3.x/4.x (支持 Hopper/Blackwell) 源码是终极目标

## 总结与建议

作为初学者，建议你不要一开始就死磕 Tensor Cores。

1. 先把 V1 (Shared Memory) 和 V2 (Register Tiling) 完全写出来，并通过验证
2. 试着通过 Nsight Compute (ncu) 分析器，观察 Memory Bandwidth 和 Compute Throughput 的柱状图
3. 当你用传统的 CUDA Cores 把算力跑到理论极限的 70% 左右后，再换用 wmma API 开启 Tensor Core。你会亲眼看到 TFLOPS 出现一个数量级的爆炸式增长！

祝你在 RTX 5090 上炼丹愉快！

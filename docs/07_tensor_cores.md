# 第三部分 Step 4：Tensor Core 优化

> **学习目标**：掌握 Tensor Core 编程，实现数量级性能提升
> 
> **预期性能**：200+ TFLOPS (FP16)，相比 CUDA Core 提升 4-6 倍

---

## 目录

1. [Tensor Core 概述](#1-tensor-core-概述)
2. [WMMA API 基础](#2-wmma-api-基础)
3. [Tensor Core GEMM 实现](#3-tensor-core-gemm-实现)
4. [性能分析与优化](#4-性能分析与优化)

---

## 1. Tensor Core 概述

### 1.1 什么是 Tensor Core？

```
Tensor Core vs CUDA Core 对比：

CUDA Core（标量运算）：
┌─────────┐
│ FP32 FMA│  1 次乘加 = 2 FLOPS
│ 1×1×1   │
└─────────┘

Tensor Core（矩阵运算）：
┌─────────────────────────┐
│  8×4×16 FP16 矩阵乘加   │
│  D = A × B + C          │
│  一次操作 = 128 FMA     │
│         = 256 FLOPS     │
└─────────────────────────┘

性能对比（RTX 5090）：
CUDA Core FP32:     104.9 TFLOPS
Tensor Core FP16:   835.3 TFLOPS  (8×)
Tensor Core FP8:   1670.6 TFLOPS  (16×)
```

### 1.2 Tensor Core 架构（第 5 代）

```
第 5 代 Tensor Core 支持的精度：

┌────────┬────────┬────────┬────────┐
│  FP16  │  BF16  │  FP8   │  FP4   │
│16-bit  │16-bit  │8-bit   │4-bit   │
│基础    │替代FP16│精度换速│极致压缩│
└────────┴────────┴────────┴────────┘

推荐：
- 训练：FP16/BF16
- 推理：FP8/FP4（如果精度允许）
```

---

## 2. WMMA API 基础

### 2.1 Fragment（矩阵片段）

```cuda
#include <mma.h>
using namespace nvcuda;

// 定义矩阵片段
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// Fragment 参数说明：
// 1. matrix_a/matrix_b/accumulator: 矩阵类型
// 2. 16, 16, 16: M, N, K 维度（WMMA 支持的尺寸）
// 3. half/float: 数据类型
// 4. row_major/col_major: 内存布局
```

### 2.2 核心 API

| API | 功能 | 说明 |
|:---|:---|:---|
| `fill_fragment(frag, val)` | 初始化片段 | 清零累加器 |
| `load_matrix_sync(frag, ptr, ldm)` | 加载矩阵 | 从全局/共享内存加载 |
| `mma_sync(c, a, b, c)` | 矩阵乘加 | 核心计算指令 |
| `store_matrix_sync(ptr, frag, ldm, layout)` | 存储结果 | 写回全局内存 |

---

## 3. Tensor Core GEMM 实现

### 3.1 基础 WMMA GEMM

```cuda
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm(half *A, half *B, float *C, int M, int N, int K) {
    // 定义矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 计算当前 Warp 负责的 16×16 C 块坐标
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // 计算 warp 在矩阵中的起始位置
    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;
    
    // 检查边界
    if (row >= M || col >= N) return;

    // K 维度循环
    for (int i = 0; i < K; i += WMMA_K) {
        // 加载 A 和 B 的 16×16 块到 Fragment
        wmma::load_matrix_sync(a_frag, A + row * K + i, K);
        wmma::load_matrix_sync(b_frag, B + i * N + col, N);

        // 执行 Tensor Core 矩阵乘法
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 写回结果
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

// 启动函数
void run_wmma_gemm(half *A, half *B, float *C, int M, int N, int K) {
    // 每个 Warp 计算 16×16
    // Block: 32 线程 = 1 Warp
    dim3 block(32, 1);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    
    wmma_gemm<<<grid, block>>>(A, B, C, M, N, K);
}
```

### 3.2 混合精度 GEMM

```cuda
// TF32 混合精度（输入 FP32，内部 TF32，累加 FP32）
__global__ void tf32_gemm(const float *A, const float *B, float *C, 
                          int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    for (int k = 0; k < K; k += 8) {
        wmma::load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN * 16, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N, 
                            wmma::mem_row_major);
}
```

### 3.3 结合 Shared Memory 优化

```cuda
// 更高效的实现：使用 Shared Memory + Tensor Core
#define BLOCK_SIZE 128
#define WMMA_SIZE 16

__global__ void wmma_shared_gemm(const half *A, const half *B, float *C,
                                  int M, int N, int K) {
    // 共享内存存储 A、B 块
    __shared__ half sA[BLOCK_SIZE][WMMA_SIZE];
    __shared__ half sB[WMMA_SIZE][BLOCK_SIZE];
    
    // WMMA Fragment
    wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // 计算 Warp 在 Block 中的位置
    int warpId = threadIdx.x / 32;
    int warpM = warpId / 2;  // 8 Warps, 4×2 分布
    int warpN = warpId % 2;
    
    int blockRow = blockIdx.y * BLOCK_SIZE;
    int blockCol = blockIdx.x * BLOCK_SIZE;
    
    // 主循环
    for (int k = 0; k < K; k += WMMA_SIZE) {
        // 协作加载 A、B 到共享内存
        // ... (加载代码)
        
        __syncthreads();
        
        // 每个 Warp 处理 16×16 块
        wmma::load_matrix_sync(a_frag, sA[warpM * WMMA_SIZE], WMMA_SIZE);
        wmma::load_matrix_sync(b_frag, sB[warpN * WMMA_SIZE], BLOCK_SIZE);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        __syncthreads();
    }
    
    // 写回结果
    int cRow = blockRow + warpM * WMMA_SIZE;
    int cCol = blockCol + warpN * WMMA_SIZE;
    wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
}
```

---

## 4. 性能分析与优化

### 4.1 性能对比

| 实现 | 精度 | 预期性能 | 适用场景 |
|:---|:---:|:---:|:---|
| CUDA Core (Register) | FP32 | 40-50 TFLOPS | 通用计算 |
| WMMA (基础) | FP16 | 150-200 TFLOPS | 深度学习训练 |
| WMMA + Shared | FP16 | 200-300 TFLOPS | 极致性能 |
| WMMA (FP8) | FP8 | 400-500 TFLOPS | 推理加速 |
| cuBLAS | FP16 | 600-800 TFLOPS | 生产环境 |

### 4.2 优化建议

```
Tensor Core 优化要点：

1. 内存对齐：
   - 矩阵地址需 128-byte 对齐
   - 使用 cudaMalloc 自动对齐

2. 数据预取：
   - 使用 cp.async 异步加载
   - 双缓冲隐藏延迟

3. Warp 分布：
   - 一个 Block 内 4-8 个 Warp
   - 每个 Warp 独立计算 16×16 块

4. 精度选择：
   - 训练：FP16 + FP32 累加
   - 推理：FP8/FP4（检查精度）
```

### 4.3 实际性能估算

```
RTX 5090 Tensor Core 峰值：
FP16: 835.3 TFLOPS

实际可达到（考虑效率）：
- 基础 WMMA: ~25% 峰值 = ~200 TFLOPS
- Shared + WMMA: ~35% 峰值 = ~290 TFLOPS
- 极致优化: ~50% 峰值 = ~400 TFLOPS (FP8)

与 cuBLAS 差距原因：
- cuBLASS 使用 PTX mma.sync 指令
- 更精细的 Warp 级调度
- 汇编级优化
```

---

## 5. 下一步学习

掌握 Tensor Core 后，可以进一步学习：
- [08_performance_analysis.md](08_performance_analysis.md) - Roofline 模型与性能分析
- [09_profiling_tools.md](09_profiling_tools.md) - Nsight Compute 使用
- CUTLASS 框架 - 工业级 Tensor Core 算子实现

---

*文档更新时间：2026年3月*

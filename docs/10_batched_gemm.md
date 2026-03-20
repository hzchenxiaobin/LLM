# 扩展专题：批量矩阵乘法 (Batched GEMM) 优化

> **学习目标**：掌握批量矩阵乘法的优化技巧，适用于多矩阵场景

---

## 目录

1. [Batched GEMM 基础](#1-batched-gemm-基础)
2. [实现策略](#2-实现策略)
3. [优化技巧](#3-优化技巧)

---

## 1. Batched GEMM 基础

### 1.1 问题定义

批量矩阵乘法计算：

$$C[b] = A[b] \times B[b], \quad b = 0, 1, ..., B_{size}-1$$

其中：
- $A[B_{size}][M][K]$：批量 A 矩阵
- $B[B_{size}][K][N]$：批量 B 矩阵
- $C[B_{size}][M][N]$：批量 C 矩阵

### 1.2 内存布局

```
Strided Batched Layout:

A: [Batch, M, K]
偏移计算: A + batch_idx * M * K

Batch 0: ┌─────────┐
         │ M × K   │
         └─────────┘
Batch 1: ┌─────────┐
         │ M × K   │
         └─────────┘
...

总内存: B_size × M × K × 4 bytes
```

---

## 2. 实现策略

### 2.1 维度映射策略

```
Grid/Block 维度分配：

Grid:  (gridDim.x, gridDim.y, gridDim.z)
           ↓           ↓           ↓
          N 维        M 维      Batch 维

最佳实践：
- blockIdx.z → Batch 索引
- blockIdx.x → N 维度
- blockIdx.y → M 维度
```

### 2.2 基础实现

```cuda
__global__ void bmm_naive(const float* A, const float* B, float* C,
                          int B_size, int M, int N, int K) {
    // 1. 获取当前 Batch
    int batch_idx = blockIdx.z;
    if (batch_idx >= B_size) return;

    // 2. 计算 C 的坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 3. 计算 Batch 偏移
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;

    // 4. 计算点积
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A_batch[row * K + k] * B_batch[k * N + col];
        }
        C_batch[row * N + col] = sum;
    }
}

// 启动配置
void run_bmm_naive(int B_size, int M, int N, int K, ...) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32, B_size);
    bmm_naive<<<grid, block>>>(A, B, C, B_size, M, N, K);
}
```

### 2.3 Shared Memory 优化版本

```cuda
#define TILE_SIZE 32

__global__ void bmm_shared(const float* A, const float* B, float* C,
                           int B_size, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= B_size) return;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
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

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C_batch[row * N + col] = sum;
}
```

---

## 3. 优化技巧

### 3.1 小矩阵优化

```
问题：当 M, N, K 较小时，Block 利用率低

解决方案：
1. 一个 Block 处理多个 Batch
2. 合并小矩阵到大矩阵计算

示例：
Batch=1000, M=N=K=32
原始: 1000 Blocks, 每个计算 32×32
优化: 250 Blocks, 每个处理 4 Batch
```

### 3.2 异步批量计算

```cuda
// 使用 CUDA Streams 并行计算多个 Batch
cudaStream_t streams[4];
for (int i = 0; i < 4; ++i) cudaStreamCreate(&streams[i]);

// 将 Batch 分组到不同 Stream
for (int b = 0; b < B_size; ++b) {
    int stream_id = b % 4;
    bmm_kernel<<<grid, block, 0, streams[stream_id]>>>(
        A + b * M * K, B + b * K * N, C + b * M * N, ...
    );
}
```

### 3.3 使用 cuBLAS

```cuda
// 生产环境推荐使用 cuBLAS
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// Strided Batched GEMM
// C[i] = alpha * A[i] * B[i] + beta * C[i]
cublasSgemmStridedBatched(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
    M, N, K,                   // 维度
    &alpha,                    // alpha
    A, K, M * K,               // A, lda, strideA
    B, N, K * N,               // B, ldb, strideB
    &beta,                     // beta
    C, N, M * N,               // C, ldc, strideC
    B_size                     // batch count
);
```

---

## 4. 性能对比

| 实现 | 小矩阵 (32×32) | 中等矩阵 (256×256) | 大矩阵 (1024×1024) |
|:---|:---:|:---:|:---:|
| Naive | 基准 | 基准 | 基准 |
| Shared | 2× | 10× | 30× |
| cuBLAS | 5× | 20× | 50× |

---

*详细教程请参考 batch_gemm/docs/ 目录*

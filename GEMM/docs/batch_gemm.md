# Batch GEMM 详解

## 1. 什么是 Batch GEMM

Batch GEMM（Batch General Matrix Multiplication）是指**批量矩阵乘法**，即一次性执行多个独立的矩阵乘法运算。

### 数学定义

给定 `batch_count` 个矩阵三元组 `(A[i], B[i], C[i])`，Batch GEMM 计算：

```
C[i] = alpha * A[i] × B[i] + beta * C[i],  i = 0, 1, ..., batch_count-1
```

其中：
- `A[i]` 是第 i 个 M×K 矩阵
- `B[i]` 是第 i 个 K×N 矩阵
- `C[i]` 是第 i 个 M×N 矩阵
- `alpha` 和 `beta` 是标量系数（所有 batch 共享）

### 与普通 GEMM 的区别

| 特性 | 单矩阵 GEMM | Batch GEMM |
|------|------------|-----------|
| 输入 | 3 个矩阵 (A, B, C) | 3 组矩阵 (A[i], B[i], C[i]) |
| 计算 | 1 次矩阵乘法 | batch_count 次独立矩阵乘法 |
| GPU 利用率 | 小矩阵时低 | 通过并行提高利用率 |
| 启动开销 | 每次调用都有 kernel 启动开销 | 单次 kernel 启动 |

## 2. 实现架构

### 2.1 线程网格设计

Batch GEMM 使用**三维线程网格**：

```cuda
dim3 block(32, 32);        // 每个 block 1024 线程，处理 32×32 输出 tile
dim3 grid(
    (N + 31) / 32,         // X 维度：N 方向的分块数
    (M + 31) / 32,         // Y 维度：M 方向的分块数
    batch_count            // Z 维度：batch 数量
);
```

**线程索引映射：**
- `threadIdx.x / blockIdx.x` → N 维度（列）
- `threadIdx.y / blockIdx.y` → M 维度（行）
- `blockIdx.z` → Batch 维度

```1:27:src/batch_gemm/sgemm_batch_naive.cu
```

### 2.2 数据访问模式

每个线程通过 stride 计算出当前 batch 的矩阵基地址：

```
A_batch = A + batch * strideA
B_batch = B + batch * strideB
C_batch = C + batch * strideC
```

## 3. 内存布局

### 3.1 Strided 布局（通用版本）

支持灵活的内存排列，每个矩阵之间可以有任意间隔（stride）：

```
A: [矩阵0] [padding] [矩阵1] [padding] [矩阵2] ...
     ↑           ↑
   base      base+strideA
```

**适用场景：**
- PyTorch/TensorFlow 的 Tensor（带 stride）
- 需要对齐的内存布局
- 非连续的 batch 数据

### 3.2 连续布局（Contiguous 版本）

所有矩阵在内存中紧密排列：

```
A: [矩阵0][矩阵1][矩阵2]...[矩阵batch_count-1]
```

此时 stride 由矩阵大小决定：
- `strideA = M * K`
- `strideB = N * K`
- `strideC = M * N`

```42:81:src/batch_gemm/sgemm_batch_naive.cu
```

## 4. 使用示例

### 4.1 连续存储的 Batch GEMM

```cpp
#include "batch_gemm/batch_gemm_kernels.h"

// 参数
int M = 512, N = 512, K = 512;
int batch_count = 32;
float alpha = 1.0f, beta = 0.0f;

// 分配设备内存 (所有 batch 连续存储)
size_t bytes_per_matrix = M * K * sizeof(float);
size_t total_bytes = bytes_per_matrix * batch_count;

float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, total_bytes);
cudaMalloc(&d_B, total_bytes);
cudaMalloc(&d_C, total_bytes);

// 初始化数据...

// 执行 Batch GEMM
run_sgemm_batch_contiguous(
    M, N, K,
    alpha, d_A, d_B, beta, d_C,
    batch_count
);
```

### 4.2 带 Stride 的 Batch GEMM

```cpp
// 假设矩阵之间有 64 字节对齐
long long int strideA = ((M * K + 15) / 16) * 16;  // 对齐到 64 字节 (16个float)
long long int strideB = ((N * K + 15) / 16) * 16;
long long int strideC = ((M * N + 15) / 16) * 16;

run_sgemm_batch_naive(
    M, N, K,
    alpha, d_A, d_B, beta, d_C,
    batch_count, strideA, strideB, strideC
);
```

## 5. 性能考虑

### 5.1 为什么 Batch GEMM 更快

| 因素 | 单矩阵（循环调用） | Batch GEMM |
|------|------------------|-----------|
| Kernel 启动开销 | batch_count 次 | 1 次 |
| GPU 占用率 | 小矩阵时 SM 闲置 | 所有 SM 并行处理不同 batch |
| 调度效率 | CPU-GPU 频繁同步 | 单次调度，更多 warps 可切换 |
| 内存合并访问 | 可能不连续 | 同一 batch 内访问连续 |

### 5.2 优化建议

1. **选择合适的 block 大小**
   - 对于小矩阵，减小 block 尺寸（如 16×16）以增加并行度
   - 对于大矩阵，保持 32×32 或更大 tile

2. **内存对齐**
   - 确保 stride 是 128/256 字节的倍数
   - 使用 `cudaMallocPitch` 分配对齐内存

3. **Batch 大小选择**
   - 尽量使 `batch_count` 是 SM 数量的倍数（如 80 的倍数 on Ampere）
   - 大 batch 可以隐藏内存延迟

4. **数据预取**
   - 对于大规模 batch，考虑使用 CUDA Streams 并行处理

## 6. 与其他实现的对比

### cuBLAS 的 `cublasSgemmStridedBatched`

```cpp
cublasSgemmStridedBatched(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    d_B, N, strideB,   // B 的 leading dim 和 stride
    d_A, K, strideA,   // A 的 leading dim 和 stride
    &beta,
    d_C, N, strideC,   // C 的 leading dim 和 stride
    batch_count
);
```

**注意：** cuBLAS 使用列主序（Column-Major），我们的实现使用行主序（Row-Major），因此参数顺序相反。

### CUTLASS Batch GEMM

CUTLASS 提供高度优化的 batch GEMM，支持：
- 灵活的 stride 模式
- 张量核心加速
- 自定义 epilogue（融合激活函数）

## 7. 进一步优化的方向

### 7.1 共享内存优化版

- 使用共享内存缓存 A/B tile
- 减少全局内存访问次数
- 参考 `sgemm_shared.cu` 的优化策略

### 7.2 寄存器分块优化版

- 每个线程计算 8×8 或更大的输出 tile
- 增加计算强度（Arithmetic Intensity）
- 参考 `sgemm_register_v2.cu` 的向量化加载

### 7.3 张量核心（Tensor Core）版

- 使用 WMMA API 调用 Tensor Core
- FP16 输入 + FP32 累加
- 需要矩阵维度是 8/16 的倍数

## 8. 总结

Batch GEMM 是深度学习中的核心算子，广泛应用于：
- **Transformer** 中的多头注意力（Multi-Head Attention）
- **CNN** 中的分组卷积
- **RNN/LSTM** 的批量序列处理

我们的朴素实现展示了 Batch GEMM 的基本原理：
1. 使用 3D 线程网格扩展并行维度
2. 通过 stride 灵活支持不同内存布局
3. 保持与单矩阵 GEMM 相同的计算逻辑

作为基准实现，它可以验证更复杂优化的正确性，并帮助理解 Batch GEMM 的核心机制。

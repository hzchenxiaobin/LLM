# 第三部分 Step 0：朴素 GEMM 实现

> **学习目标**：实现基础的矩阵乘法，理解其性能瓶颈，为后续优化奠定基础
> 
> **预期性能**：~7 TFLOPS（仅为理论峰值的 ~7%）

---

## 目录

1. [问题定义](#1-问题定义)
2. [朴素实现](#2-朴素实现)
3. [性能瓶颈分析](#3-性能瓶颈分析)
4. [Arithmetic Intensity 计算](#4-arithmetic-intensity-计算)
5. [优化方向展望](#5-优化方向展望)

---

## 1. 问题定义

### 1.1 SGEMM 数学定义

矩阵乘法（Single-precision General Matrix Multiply）：

$$C = \alpha \cdot A \times B + \beta \cdot C$$

其中：
- $A$ 是 $M \times K$ 矩阵
- $B$ 是 $K \times N$ 矩阵  
- $C$ 是 $M \times N$ 矩阵
- $\alpha$, $\beta$ 是标量系数

### 1.2 计算复杂度

```
浮点运算次数 (FLOPs):
- 每个 C 元素需要 K 次乘法和 K-1 次加法
- 总计算量 = M × N × (2K - 1) ≈ 2MNK

示例 (M=N=K=4096):
FLOPs = 2 × 4096³ ≈ 137.4 GFLOPs
```

---

## 2. 朴素实现

### 2.1 线程映射策略

**核心思想**：每个线程计算 C 的一个元素

```
线程分配策略：
┌─────────────────────────────────────────────────────────┐
│  Grid: (ceil(N/32), ceil(M/32))                          │
│  Block: (32, 32) = 1024 线程                             │
│                                                          │
│  Thread (tx, ty) 负责计算 C[row][col]:                   │
│    col = blockIdx.x × 32 + tx                            │
│    row = blockIdx.y × 32 + ty                            │
│                                                          │
│  ┌────────────────────────────────────────┐              │
│  │  Block(0,0)   Block(1,0)   Block(2,0) │              │
│  │  ┌─────┐     ┌─────┐     ┌─────┐     │              │
│  │  │T0,0 │     │     │     │     │     │              │
│  │  │T1,0 │     │     │     │     │     │              │
│  │  └─────┘     └─────┘     └─────┘     │              │
│  │  Block(0,1)   Block(1,1)   Block(2,1) │              │
│  └────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 CUDA Kernel 代码

```cuda
// sgemm_naive.cu
__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha,
                                 const float *A, const float *B,
                                 float beta, float *C) {
    // 计算当前线程负责的矩阵 C 的行号和列号
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N 维度
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M 维度

    // 边界检查
    if (row < M && col < N) {
        float tmp = 0.0f;
        
        // 计算点积：C[row][col] = A[row][*] · B[*][col]
        for (int k = 0; k < K; ++k) {
            // A 行主序索引：row * K + k
            // B 行主序索引：k * N + col
            tmp += A[row * K + k] * B[k * N + col];
        }
        
        // 应用 alpha 和 beta 系数
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// 启动函数
void run_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                     const float *B, float beta, float *C) {
    dim3 block(32, 32);  // 1024 线程 / Block
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    sgemm_naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
```

### 2.3 代码逐行解析

| 行 | 代码 | 说明 |
|:---:|:---|:---|
| 3-4 | `int col = ...` | 计算线程负责的全局列坐标 |
| 5-6 | `int row = ...` | 计算线程负责的全局行坐标 |
| 9 | `if (row < M && ...)` | 边界检查，处理非 32 整数倍矩阵 |
| 12 | `for (int k = 0; ...)` | 沿 K 维度遍历计算点积 |
| 14 | `A[row * K + k]` | A 的行主序索引，连续访问 |
| 14 | `B[k * N + col]` | B 的行主序索引，跨步访问 |

---

## 3. 性能瓶颈分析

### 3.1 内存访问模式

```
A 矩阵访问（固定 row，变化 k）：

Row 固定：row × K + 0, row × K + 1, row × K + 2, ...
         │            │            │
         ▼            ▼            ▼
    ┌────────┬────────┬────────┬────────┐
    │ [row,0]│ [row,1]│ [row,2]│ [row,3]│ ...
    └────────┴────────┴────────┴────────┘
         ↑            ↑            ↑
      Thread 0     Thread 1     Thread 2  ... 连续访问（合并）✓

B 矩阵访问（固定 col，变化 k）：

Col 固定：0×N+col, 1×N+col, 2×N+col, ...
         │          │          │
         ▼          ▼          ▼
    ┌────────┐
    │ [0,col]│
    ├────────┤
    │ [1,col]│
    ├────────┤
    │ [2,col]│
    └────────┘
       ↑
    间隔 N×4 bytes！非连续访问 ✗
```

### 3.2 内存访问统计

**全局内存访问分析**：

| 操作 | 访问类型 | 访问次数 | 问题 |
|:---|:---|:---:|:---|
| 读取 A | 全局内存 | M × K × N | 每次计算都读全局内存 |
| 读取 B | 全局内存 | K × N × M | 跨步访问，非合并 |
| 读取 C | 全局内存 | M × N | beta 系数需要 |
| 写入 C | 全局内存 | M × N | 结果写回 |

**关键问题**：
1. **极低的计算访存比**：1 FMA / 2 次全局内存读取
2. **B 矩阵非合并访问**：严重浪费显存带宽
3. **无数据复用**：每个数据只被使用一次

### 3.3 性能估算

```
假设 M = N = K = 4096：

计算量: 2 × 4096³ ≈ 137.4 GFLOPs

内存访问量:
- 读取 A: 4096³ × 4 bytes = 274.9 GB（最坏情况）
- 读取 B: 4096³ × 4 bytes = 274.9 GB
- 读写 C: 2 × 4096² × 4 bytes = 134.2 MB

Arithmetic Intensity:
AI = FLOPs / Bytes = 137.4e9 / (274.9e9 × 2)
   ≈ 0.25 FLOPs/byte（实际约 0.5，考虑部分缓存）

Roofline 分析:
- Ridge Point = 58.5
- AI = 0.5 << 58.5 → 严重内存受限

理论性能:
Performance = Memory Bandwidth × AI
            = 1,792 GB/s × 0.5
            = 896 GFLOPS = 0.896 TFLOPS

实际可达: ~7 TFLOPS（考虑缓存部分命中）
利用率: 7 / 104.9 ≈ 6.7%
```

---

## 4. Arithmetic Intensity 计算

### 4.1 理论推导

**朴素实现的内存访问模型**：

```
理想情况（无缓存）:
- 每个线程计算 1 个 C 元素
- 需要读取 A 的 1 行（K 个元素）
- 需要读取 B 的 1 列（K 个元素）
- 但这些数据不被其他线程复用

总内存访问:
Bytes = M × K × 4 (A) + K × N × 4 (B) + 2 × M × N × 4 (C)
      ≈ 4 × (M × K + K × N + 2 × M × N)

当 M = N = K 时:
Bytes ≈ 4 × (M² + M² + 2M²) = 16M²

FLOPs = 2 × M³

AI = FLOPs / Bytes = 2M³ / 16M² = M / 8

当 M = 4096 时:
AI = 4096 / 8 = 512？不对，实际需要考虑无缓存情况...
```

**修正分析（考虑 B 的重复访问）**：

```
实际上，在朴素实现中：
- A 的每行被 N 个线程读取（每个 C 行元素都要用）
- B 的每列被 M 个线程读取（每个 C 列元素都要用）
- 在无缓存情况下，这是 O(M×K×N) 的内存访问

保守估算（假设 L2 缓存无效）：
Memory = M × K × N × 4 × 2（A 和 B）
       ≈ 2 × M × K × N × 4 bytes

AI = 2 × M × N × K / (2 × M × K × N × 4)
   = 0.25 FLOPs/byte

考虑部分缓存（约 2× 提升）:
AI ≈ 0.5 FLOPs/byte
```

### 4.2 对比总结

| Kernel 类型 | Arithmetic Intensity | 主要瓶颈 | 理论利用率 |
|:---:|:---:|:---:|:---:|
| **Naive** | ~0.5 | 全局内存带宽 | < 1% |
| **Shared Memory** | ~683 | 共享内存带宽 | ~9% |
| **Register Tiling** | ~683 | 计算单元 | ~30-50% |
| **Tensor Core** | ~100+ | Tensor Core 利用率 | ~60-80% |

---

## 5. 优化方向展望

### 5.1 核心问题与解决方案

| 问题 | 影响 | 解决方案 | 预期提升 |
|:---|:---|:---|:---:|
| **频繁全局内存访问** | 内存带宽瓶颈 | Shared Memory Tiling | 100×+ |
| **B 矩阵非合并访问** | 带宽利用率低 | 数据重排/转置 | 2-5× |
| **无数据复用** | 重复读取 | Tiling + 缓存 | 30×+ |
| **低计算/访存比** | AI 过低 | Register Tiling | 10×+ |

### 5.2 优化路线图

```
优化阶段路线图：

Stage 0: Naive (当前)
├── 性能: ~7 TFLOPS
├── AI: ~0.5
└── 瓶颈: 全局内存带宽

     ↓ 应用 Shared Memory Tiling

Stage 1: Shared Memory
├── 性能: ~9 TFLOPS
├── AI: ~683
└── 瓶颈: 共享内存带宽

     ↓ 应用 Register Tiling

Stage 2: Register Tiling
├── 性能: ~30-50 TFLOPS
├── AI: ~683
└── 瓶颈: 计算单元利用率

     ↓ 应用 Tensor Cores

Stage 3: Tensor Core
├── 性能: 200+ TFLOPS (FP16)
├── AI: ~100+
└── 瓶颈: Tensor Core 调度
```

### 5.3 下一步学习

下一章节将学习 **Shared Memory Tiling**，这是 GEMM 优化的第一关键步骤：

- 核心思想：将 A 和 B 分块加载到共享内存
- 数据复用：每个元素复用 BLOCK_SIZE 次
- 预期效果：AI 从 0.5 提升到 683，性能提升 100×+

---

## 6. 课后练习

### 练习 1：索引计算

给定矩阵维度 M=1024, N=1024, K=1024，Block(16, 16)：
- 计算 Thread(5, 3) in Block(2, 1) 负责的 C 元素坐标
- 计算它需要读取的 A 和 B 元素范围

### 练习 2：内存访问分析

分析以下代码的内存访问模式：
```cuda
float sum = 0;
for (int k = 0; k < K; ++k) {
    sum += A[k * M + row] * B[k * N + col];  // A 按列访问
}
```
- A 和 B 的访问模式分别是什么？
- 相比原始版本有何差异？

### 练习 3：性能估算

假设 RTX 5090 参数：
- 显存带宽：1,792 GB/s
- FP32 峰值：104.9 TFLOPS

计算朴素 GEMM (M=N=K=2048) 的理论性能上限。

---

## 7. 参考代码完整版

```cuda
// sgemm_naive.cu - 完整注释版
#include "common.h"
#include "gemm_kernels.h"

/**
 * 朴素 GEMM Kernel
 * 
 * 特点：
 * - 每个线程计算 C 的一个元素
 * - 直接从全局内存读取 A 和 B
 * - 简单但效率极低
 */
__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha,
                                   const float *A, const float *B,
                                   float beta, float *C) {
    // ========== 索引计算 ==========
    // 计算当前线程负责的全局坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N 维度
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M 维度
    
    // ========== 边界检查 ==========
    // 处理矩阵维度非 Block 大小整数倍的情况
    if (row >= M || col >= N) {
        return;
    }
    
    // ========== 核心计算 ==========
    // 计算 C[row][col] = dot(A[row][*], B[*][col])
    float tmp = 0.0f;
    for (int k = 0; k < K; ++k) {
        // A 行主序：连续访问（合并）
        float a = A[row * K + k];
        
        // B 行主序：跨步访问（非合并）
        float b = B[k * N + col];
        
        tmp += a * b;
    }
    
    // ========== 结果写回 ==========
    // C = alpha * AB + beta * C
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
}

/**
 * 启动函数
 */
void run_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                     const float *B, float beta, float *C) {
    // Block 配置：32×32 = 1024 线程
    dim3 block(32, 32);
    
    // Grid 配置：向上取整确保覆盖整个矩阵
    dim3 grid((N + block.x - 1) / block.x, 
              (M + block.y - 1) / block.y);
    
    // 启动 Kernel
    sgemm_naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    
    // 错误检查
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

---

*下一步学习：[04_shared_memory_tiling.md](04_shared_memory_tiling.md) - 掌握 Shared Memory 分块优化*

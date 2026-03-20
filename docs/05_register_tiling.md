# 第三部分 Step 2：Register Tiling 优化

> **学习目标**：掌握寄存器分块技术，进一步提升计算访存比，释放 GPU 计算潜力
> 
> **预期性能**：30-50 TFLOPS（相比 Shared Memory 提升 3-5 倍）

---

## 目录

1. [核心思想](#1-核心思想)
2. [双层分块策略](#2-双层分块策略)
3. [Kernel 实现](#3-kernel-实现)
4. [外积计算模式](#4-外积计算模式)
5. [性能分析](#5-性能分析)
6. [优化技巧](#6-优化技巧)

---

## 1. 核心思想

### 1.1 Shared Memory Kernel 的局限

```
Shared Memory Kernel 问题分析：

┌─────────────────────────────────────────────────────────┐
│  每个线程工作量：                                        │
│  ┌─────────┐                                            │
│  │Thread   │ 计算 1 个 C 元素                            │
│  │(tx, ty) │ = 32 次乘加 (K=32)                          │
│  └─────────┘                                            │
│                                                          │
│  内存访问：                                              │
│  ├─ 每轮 k: 读取 sA[ty][k] + sB[k][tx] = 2 次            │
│  └─ 总计: 2 × 32 = 64 次共享内存访问                      │
│                                                          │
│  计算/访存比：32 / 64 = 0.5                               │
│                                                          │
│  瓶颈：频繁共享内存访问，无法充分利用计算单元             │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Register Tiling 的核心突破

```
核心思想：让每个线程计算多个 C 元素，使用寄存器缓存

┌─────────────────────────────────────────────────────────┐
│  每个线程工作量（8×8 分块）：                            │
│                                                          │
│  ┌─────────────────────────────┐                       │
│  │                             │                       │
│  │  ┌─────┬─────┬─────┬─────┐  │                       │
│  │  │ C00 │ C01 │ C02 │ ... │  │  8×8 = 64 个元素     │
│  │  ├─────┼─────┼─────┼─────┤  │                       │
│  │  │ C10 │ C11 │ C12 │ ... │  │                       │
│  │  ├─────┼─────┼─────┼─────┤  │                       │
│  │  │ ... │ ... │ ... │ ... │  │                       │
│  │  └─────┴─────┴─────┴─────┘  │                       │
│  │                             │                       │
│  └─────────────────────────────┘                       │
│                                                          │
│  内存访问：                                              │
│  ├─ 每轮 k: 读取 sA[ty*8:ty*8+7][k] + sB[k][tx*8:tx*8+7] │
│  │          = 8 + 8 = 16 次                               │
│  └─ 总计: 16 × 8 = 128 次                               │
│                                                          │
│  计算: 64 × 8 = 512 次 FMA                               │
│                                                          │
│  计算/访存比：512 / 128 = 4.0  (8× 提升！)                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 双层分块策略

### 2.1 分块层级结构

```
双层分块策略图解：

┌─────────────────────────────────────────────────────────────────┐
│                     矩阵 C (4096×4096)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Block Tile (128×128)                  │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┐              │   │
│  │  │Thread   │Thread   │Thread   │Thread   │              │   │
│  │  │Tile     │Tile     │Tile     │Tile     │ ... (16×)    │   │
│  │  │(8×8)    │(8×8)    │(8×8)    │(8×8)    │              │   │
│  │  │64元素   │64元素   │64元素   │64元素   │              │   │
│  │  └─────────┴─────────┴─────────┴─────────┘              │   │
│  │  ├─────────┼─────────┼─────────┼─────────┤              │   │
│  │  │ ... (16 rows)                              │         │   │
│  │  └───────────────────────────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ ... (32×32 Blocks)                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

分块参数：
┌───────────┬───────┬────────────────────────────────────┐
│ 参数      │ 值    │ 说明                               │
├───────────┼───────┼────────────────────────────────────┤
│ BM        │ 128   │ Block 在 M 维度负责 128 行         │
│ BN        │ 128   │ Block 在 N 维度负责 128 列         │
│ BK        │ 8     │ K 维度步长为 8                     │
│ TM        │ 8     │ Thread 在 M 维度负责 8 行          │
│ TN        │ 8     │ Thread 在 N 维度负责 8 列          │
│ Block     │ 16×16 │ 256 线程 / Block                   │
└───────────┴───────┴────────────────────────────────────┘

验证计算：
- Block 负责区域: 128 × 128 = 16,384 元素
- 每线程计算量: 8 × 8 = 64 元素
- 总线程数: 16 × 16 = 256
- 验证: 256 × 64 = 16,384 ✓
```

### 2.2 为什么减少线程数反而更好？

| 配置 | Shared Kernel | Register Kernel | 优势 |
|:---|:---:|:---:|:---|
| **Block 大小** | 32×32 = 1024 | 16×16 = 256 | 每线程工作量更大 |
| **每线程计算** | 1 元素 | 64 元素 | 计算/开销比 64× |
| **共享内存** | sA[32][32] | sA[128][8] | K 维度更小，更易缓存 |
| **寄存器使用** | 1 个 tmp | 80 个变量 | 更多数据进寄存器 |

**关键洞察**：
- 1024 线程：每个线程工作量太少，启动开销占比高
- 256 线程：每个线程工作量充足，计算开销占比高
- 256 线程分摊共享内存压力 → 每线程带宽更高
- 更多寄存器分配给每个线程 → 更少 spilling

---

## 3. Kernel 实现

### 3.1 完整代码

```cuda
// sgemm_register.cu
#include "common.h"
#include "gemm_kernels.h"

// 分块参数定义
#define BM 128  // Block 在 M 维度负责 128 行
#define BN 128  // Block 在 N 维度负责 128 列
#define BK 8    // K 维度步长为 8（关键：小步长让更多数据进寄存器）
#define TM 8    // Thread 在 M 维度负责 8 行
#define TN 8    // Thread 在 N 维度负责 8 列（共 64 元素）

__global__ void sgemm_register_kernel(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    // ========== 1. 索引计算 ==========
    int bx = blockIdx.x;           // Block 在 N 维度的索引
    int by = blockIdx.y;           // Block 在 M 维度的索引
    int tx = threadIdx.x;          // Thread 在 Block 内的列索引（0~15）
    int ty = threadIdx.y;          // Thread 在 Block 内的行索引（0~15）
    int tid = ty * 16 + tx;        // Thread 的一维 ID（0~255）
    
    // ========== 2. 内存申请 ==========
    // 共享内存：sA[128][8], sB[8][128]，共 8KB
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    
    // 寄存器数组：64 个累加器 + 16 个缓存，共 80 个 float
    float accum[TM][TN] = {0.0f};  // 64 个累加器，初始化为 0
    
    // 当前线程负责的 C 子矩阵起始坐标
    int row_start = by * BM + ty * TM;  // 全局行坐标
    int col_start = bx * BN + tx * TN;  // 全局列坐标
    
    // 协作加载坐标（每线程负责 4 个元素）
    int load_a_row = tid / BK;         // A 的行坐标
    int load_a_col = tid % BK;         // A 的列坐标
    int load_b_row = tid / BN;         // B 的行坐标
    int load_b_col = tid % BN;         // B 的列坐标
    
    // ========== 3. 主循环：沿 K 维度滑动 ==========
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;    // 当前 K 块的起始位置
        
        // --- 协作加载 A 到共享内存（256 线程 × 4 元素 = 1024 元素）---
        #pragma unroll
        for (int i = 0; i < BM * BK / 256; ++i) {
            int row = load_a_row + i * 32;   // 256/8 = 32
            int col = load_a_col;
            int global_row = by * BM + row;
            int global_col = k_offset + col;
            if (global_row < M && global_col < K)
                sA[row][col] = A[global_row * K + global_col];
            else
                sA[row][col] = 0.0f;
        }
        
        // --- 协作加载 B 到共享内存（256 线程 × 4 元素 = 1024 元素）---
        #pragma unroll
        for (int i = 0; i < BK * BN / 256; ++i) {
            int row = load_b_row + i * 2;    // 256/128 = 2
            int col = load_b_col;
            int global_row = k_offset + row;
            int global_col = bx * BN + col;
            if (global_row < K && global_col < N)
                sB[row][col] = B[global_row * N + global_col];
            else
                sB[row][col] = 0.0f;
        }
        
        __syncthreads();  // 同步：等待所有线程完成加载
        
        // --- 寄存器分块计算（核心优化）---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 1. 从共享内存加载到寄存器（16 次读取）
            float frag_a[TM];
            float frag_b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];
            
            // 2. 外积计算（64 次 FMA，全部从寄存器读取）
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    accum[i][j] += frag_a[i] * frag_b[j];
        }
        
        __syncthreads();  // 同步：等待所有线程完成计算
    }
    
    // ========== 4. 写回结果 ==========
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = row_start + i;
            int col = col_start + j;
            if (row < M && col < N)
                C[row * N + col] = alpha * accum[i][j] + beta * C[row * N + col];
        }
}

// 启动函数
void run_sgemm_register(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
    dim3 block(16, 16);                           // 256 线程 / Block
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // 向上取整
    sgemm_register_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
```

### 3.2 关键代码解析

| 段落 | 核心代码 | 优化原理 |
|:---|:---|:---|
| **分块参数** | `#define TM 8, TN 8` | 每线程计算 8×8 子矩阵 |
| **寄存器数组** | `float accum[8][8]` | 64 个累加器在寄存器中 |
| **协作加载** | `tid = ty * 16 + tx` | 256 线程分工加载 1024 元素 |
| **寄存器缓存** | `frag_a[i] = sA[...]` | 从共享内存批量加载到寄存器 |
| **外积计算** | `accum[i][j] += frag_a[i] * frag_b[j]` | 64 次 FMA 从寄存器读取 |
| **循环展开** | `#pragma unroll` | 消除循环分支开销 |

---

## 4. 外积计算模式

### 4.1 点积 vs 外积

```
点积模式（Shared Kernel）：
C[ty][tx] = Σ sA[ty][k] × sB[k][tx]

每轮 k 迭代：
- 读取 1 个 sA + 1 个 sB = 2 次
- 计算 1 次 FMA
- 计算/访存比 = 0.5

外积模式（Register Kernel）：
accum[i][j] = Σ frag_a[i] × frag_b[j]

每轮 k 迭代：
- 读取 8 个 frag_a + 8 个 frag_b = 16 次
- 计算 8×8 = 64 次 FMA
- 计算/访存比 = 4.0

提升：8×！
```

### 4.2 外积数学原理

```
矩阵乘法的另一种视角：

C = A × B = Σ(k) A[:,k] × B[k,:]

其中 A[:,k] 是 A 的第 k 列，B[k,:] 是 B 的第 k 行

外积：A[:,k] (M×1) × B[k,:] (1×N) = M×N 矩阵

累加所有 k 的外积得到 C

在 Register Kernel 中：
- frag_a[TM] = A 的 TM 个元素（行切片）
- frag_b[TN] = B 的 TN 个元素（列切片）
- frag_a[i] × frag_b[j] 是外积的一个元素
- accum[i][j] 累加所有 k 的外积贡献
```

### 4.3 指令级并行 (ILP)

```
外积计算的指令并行性：

for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
        accum[i][j] += frag_a[i] * frag_b[j];

64 次 FMA 指令相互独立：
- accum[0][0] 不依赖 accum[0][1]
- 所有 64 个 FMA 可以同时发射
- GPU 可以流水线执行，提高计算单元利用率

对比点积模式：
- tmp += sA[ty][k] * sB[k][tx]
- 每次迭代只有 1 个 FMA
- 指令级并行性低
```

---

## 5. 性能分析

### 5.1 内存访问对比

| 内存层级 | Shared Kernel | Register Kernel | 提升 |
|:---|:---:|:---:|:---:|
| **全局内存** | 2048 次 (1024×2) | 512 次 (协作加载) | **4×** |
| **共享内存读取** | 64 次 (32×2) | 16 次 (8+8) | **4×** |
| **寄存器访问** | 1 次 (tmp) | 80 次 (8+8+64) | **更高的寄存器利用率** |
| **FMA 计算** | 32 次 | 64 次 | **2×** |
| **计算/内存比** | 0.5 | 4.0 | **8×** |

### 5.2 为什么 BK=8 是关键？

```
BK 选择的影响：

BK = 8:
- sA[128][8], sB[8][128]
- frag_a[8], frag_b[8] 可以完全驻留寄存器
- 共享内存读取: 16 次/轮
- K 迭代: K/8 次

BK = 32（Shared Kernel）:
- sA[32][32], sB[32][32]
- 无法全部缓存到寄存器
- 共享内存读取: 64 次/轮
- K 迭代: K/32 次

权衡：
- BK 越小 → 更多数据进寄存器 → 更高计算/访存比
- 但 K 迭代次数增加 → 更多同步
- 实际测试：BK=8 是 Sweet Spot
```

### 5.3 性能预测

```
基于 4096×4096×4096 矩阵的预测：

┌────────────────┬─────────────┬──────────┬────────────┐
│ Kernel         │ 预计 TFLOPS │ 利用率   │ vs Shared  │
├────────────────┼─────────────┼──────────┼────────────┤
│ Naive          │ 7.5         │ 7.1%     │ -          │
│ Shared         │ 9.1         │ 8.7%     │ 基准       │
│ Register       │ 30-50       │ 30-50%   │ 3-5×       │
│ cuBLAS         │ 66.7        │ 63.6%    │ 7×         │
└────────────────┴─────────────┴──────────┴────────────┘

为什么 Register 无法达到 100%？
1. 仍未使用 Tensor Core（可达 200+ TFLOPS）
2. 同步次数增加：K/8 = 512 次 vs 128 次
3. 全局内存带宽仍是上限
4. Occupancy 限制：256 线程/Block
```

### 5.4 Occupancy 分析

```
Register Kernel 资源使用：

寄存器：
- accum[8][8]: 64 个
- frag_a[8]: 8 个
- frag_b[8]: 8 个
- 其他变量: ~10 个
- 总计: ~90 个/线程

计算：
- 每 Warp: 90 × 32 = 2,880 个 = 11.25 KB
- 每 Block (8 Warps): 8 × 11.25 = 90 KB
- 每 SM (256 KB): 可运行 256/11.25 ≈ 22 Warps
- 实际: 22 / 8 = 2.75 个 Block
- Occupancy: 22 / 64 ≈ 34%

优化建议：
- 可以接受 25-50% Occupancy（GEMM 是计算密集型）
- 更高 Occupancy 不一定更好
- 更多寄存器 → 更高 ILP → 更好性能
```

---

## 6. 优化技巧

### 6.1 循环展开

```cuda
// 编译器展开循环，消除分支开销
#pragma unroll
for (int k = 0; k < BK; ++k) {
    // ...
}

// 等价于：
// k=0 的代码
// k=1 的代码
// ...
// k=7 的代码

优势：
1. 消除循环变量递增开销
2. 消除分支预测失败
3. 允许更好的指令调度
```

### 6.2 边界检查优化

```cuda
// 原始边界检查（在内层循环）
for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
        if (row < M && col < N)
            C[...] = ...;

// 优化：将边界检查移到外层
if (row_start + TM <= M && col_start + TN <= N) {
    // 无边界检查的快速路径
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[...] = ...;  // 无需检查
} else {
    // 有边界检查的慢速路径
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            if (row < M && col < N)
                C[...] = ...;
}
```

### 6.3 进一步优化方向

| 优化方向 | 难度 | 预期提升 | 说明 |
|:---|:---:|:---:|:---|
| **输出向量化** | 低 | 10% | 使用 float4 写回 |
| **双缓冲** | 中 | 30% | 重叠计算与数据传输 |
| **异步拷贝** | 中 | 20% | 使用 cp.async |
| **Warp 级优化** | 高 | 2× | 使用 mma.sync |
| **Tensor Core** | 高 | 4× | 使用 WMMA API |

---

## 7. 课后练习

### 练习 1：分块验证

验证分块参数的正确性：
- BM=128, BN=128, TM=8, TN=8, Block=16×16
- 证明：Block 负责区域 = 总线程计算量

### 练习 2：外积计算

给定：
- frag_a = [1, 2, 3, 4]
- frag_b = [5, 6, 7, 8]

计算外积矩阵 accum[4][4]。

### 练习 3：性能估算

给定 RTX 5090 参数：
- FP32 峰值：104.9 TFLOPS
- Register Kernel 计算/访存比：4.0

估算理论性能上限。

---

*下一步学习：[06_vectorization_and_bank_conflict.md](06_vectorization_and_bank_conflict.md) - 向量化加载与 Bank Conflict 消除*

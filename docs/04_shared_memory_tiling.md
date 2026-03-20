# 第三部分 Step 1：Shared Memory Tiling 优化

> **学习目标**：掌握 Shared Memory 分块技术，实现数据复用，突破内存带宽瓶颈
> 
> **预期性能**：~9 TFLOPS（相比 Naive 提升约 30%）

---

## 目录

1. [核心思想](#1-核心思想)
2. [算法原理](#2-算法原理)
3. [Kernel 实现](#3-kernel-实现)
4. [执行流程分析](#4-执行流程分析)
5. [性能分析](#5-性能分析)
6. [进一步优化方向](#6-进一步优化方向)

---

## 1. 核心思想

### 1.1 为什么需要 Shared Memory？

```
内存层级速度对比（RTX 5090）：

寄存器:    ████████████████████████████████████  ~1 TB/s, ~1 cycle
共享内存:  ██████████████████░░░░░░░░░░░░░░░░░░  ~10 TB/s, ~20-30 cycles
L2 缓存:   ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~2-4 TB/s, ~100 cycles
全局内存:  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~1.8 TB/s, ~300-500 cycles

关键洞察：
- 共享内存速度是全局内存的 5-10 倍
- 但容量有限（164 KB/SM），需要分块使用
```

### 1.2 Tiling 策略图解

```
矩阵分块计算策略：

┌─────────────────────────────────────────────────────────┐
│                    矩阵 C (M×N)                          │
│  ┌─────────┬─────────┬─────────┐                        │
│  │ C[0,0]  │ C[0,1]  │ C[0,2]  │  每个 C[i,j] 是        │
│  │ 32×32   │ 32×32   │ 32×32   │  32×32 的 Block       │
│  ├─────────┼─────────┼─────────┤                        │
│  │ C[1,0]  │ C[1,1]  │ C[1,2]  │                        │
│  │ 32×32   │ 32×32   │ 32×32   │                        │
│  ├─────────┼─────────┼─────────┤                        │
│  │ C[2,0]  │ C[2,1]  │ C[2,2]  │                        │
│  │ 32×32   │ 32×32   │ 32×32   │                        │
│  └─────────┴─────────┴─────────┘                        │
└─────────────────────────────────────────────────────────┘

每个 C Block 的计算：
C[i,j] = Σ(k=0 to K/32-1) A[i,k] × B[k,j]

其中 A[i,k] 和 B[k,j] 都是 32×32 的 Tile
```

### 1.3 数据复用原理

```
Naive vs Shared Memory 数据访问对比：

Naive 实现（无复用）：
Global Memory          计算
    │                    │
    ▼                    ▼
A[i] ──────→ 乘法 ──────→ 加法
B[j] ──────→    (只使用一次，丢弃)
    ↑                    │
    └────────────────────┘
每次计算都从全局内存读取

Shared Memory 实现（有复用）：
Global Memory    Shared Memory      计算
    │                │                │
    ▼                ▼                ▼
A Tile ──────→   sA[32][32]  ───→  乘法 ───→ 累加
B Tile ──────→   sB[32][32]  ───→     (复用 32 次)
    │                ▲                │
    │                │                │
    └────────────────┘                │
      只从全局内存读取一次
      在共享内存中被 32 个线程各使用 32 次

数据复用率：32× 提升！
```

---

## 2. 算法原理

### 2.1 分块矩阵乘法数学

$$C_{ij} = \sum_{k=0}^{K/BLOCK\_SIZE - 1} A_{ik} \times B_{kj}$$

其中：
- $A_{ik}$：A 的第 i 行块、第 k 列块，大小 BLOCK_SIZE × BLOCK_SIZE
- $B_{kj}$：B 的第 k 行块、第 j 列块，大小 BLOCK_SIZE × BLOCK_SIZE
- $C_{ij}$：C 的第 i 行块、第 j 列块

### 2.2 执行步骤

```
计算 C[0,0] 的完整流程：

初始：累加器 tmp = 0

Step 1: 加载 A[0,0] 和 B[0,0] 到共享内存
    ┌─────────┐    ┌─────────┐
    │ A[0,0]  │    │ B[0,0]  │
    │ 32×32   │    │ 32×32   │
    └────┬────┘    └────┬────┘
         │              │
         ▼              ▼
    ┌─────────┐    ┌─────────┐
    │  sA     │    │  sB     │
    └─────────┘    └─────────┘
    
Step 2: 在共享内存中计算 A[0,0] × B[0,0]
    tmp += sA × sB

Step 3: 加载 A[0,1] 和 B[1,0] 到共享内存
Step 4: 计算 A[0,1] × B[1,0]
    tmp += sA × sB

... 重复直到 K 维度完成

Step N: 写回 C[0,0] = alpha * tmp + beta * C[0,0]
```

---

## 3. Kernel 实现

### 3.1 完整代码

```cuda
// sgemm_shared.cu
#include "common.h"
#include "gemm_kernels.h"

template <int BLOCK_SIZE>
__global__ void sgemm_shared_kernel(int M, int N, int K, float alpha,
                                    const float *A, const float *B,
                                    float beta, float *C) {
    // ========== 1. 申请共享内存 ==========
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    // ========== 2. 计算线程索引 ==========
    int tx = threadIdx.x;        // Block 内列索引: 0 ~ BLOCK_SIZE-1
    int ty = threadIdx.y;        // Block 内行索引: 0 ~ BLOCK_SIZE-1
    int row = blockIdx.y * BLOCK_SIZE + ty;  // 全局行坐标
    int col = blockIdx.x * BLOCK_SIZE + tx;  // 全局列坐标
    
    // 累加器（存储在寄存器中）
    float tmp = 0.0f;
    
    // ========== 3. 沿 K 维度分块迭代 ==========
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int k_step = 0; k_step < num_tiles; ++k_step) {
        int k_offset = k_step * BLOCK_SIZE;
        
        // --- 3a. 协作加载 A 到共享内存 ---
        // 条件加载 + 边界填充
        if (row < M && k_offset + tx < K) {
            sA[ty][tx] = A[row * K + k_offset + tx];
        } else {
            sA[ty][tx] = 0.0f;  // 越界填充 0
        }
        
        // --- 3b. 协作加载 B 到共享内存 ---
        if (k_offset + ty < K && col < N) {
            sB[ty][tx] = B[(k_offset + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;  // 越界填充 0
        }
        
        // --- 3c. 同步：确保所有线程完成加载 ---
        __syncthreads();
        
        // --- 3d. 在共享内存中计算 ---
        // 计算点积：sA[ty][*] · sB[*][tx]
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += sA[ty][k] * sB[k][tx];
        }
        
        // --- 3e. 同步：等待所有线程用完数据 ---
        __syncthreads();
    }
    
    // ========== 4. 结果写回 ==========
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// 启动函数
void run_sgemm_shared(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    const int BLOCK_SIZE = 32;
    
    // Block: 32×32 = 1024 线程
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    // Grid: 向上取整
    dim3 grid((N + block.x - 1) / block.x, 
              (M + block.y - 1) / block.y);
    
    // 启动模板 Kernel
    sgemm_shared_kernel<BLOCK_SIZE><<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

### 3.2 代码逐行解析

| 段落 | 代码 | 说明 |
|:---|:---|:---|
| **共享内存** | `__shared__ float sA[32][32]` | 每 Block 分配 4KB 共享内存 |
| **索引** | `row = blockIdx.y * 32 + ty` | 计算全局矩阵坐标 |
| **加载 A** | `sA[ty][tx] = A[...]` | 协作加载，连续访问（合并） |
| **加载 B** | `sB[ty][tx] = B[...]` | 协作加载，注意跨行访问 |
| **同步 1** | `__syncthreads()` | 确保加载完成再计算 |
| **计算** | `tmp += sA[ty][k] * sB[k][tx]` | 从共享内存读取，速度快 |
| **同步 2** | `__syncthreads()` | 确保用完数据再加载下一轮 |

---

## 4. 执行流程分析

### 4.1 内存访问流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Shared Memory Kernel 执行流程                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  全局内存 A                      共享内存 sA                      │
│  ┌─────────┐                    ┌─────────┐                       │
│  │ Tile 0  │ ───Thread(tx,ty)──→│[ty][tx] │                       │
│  │ Tile 1  │ ───Thread(tx,ty)──→│[ty][tx] │  (下一轮回覆盖)      │
│  │ Tile 2  │ ───Thread(tx,ty)──→│[ty][tx] │                       │
│  └─────────┘                    └────┬────┘                       │
│                                      │                           │
│                                      ▼                           │
│  全局内存 B                      共享内存 sB                      │
│  ┌─────────┐                    ┌─────────┐                       │
│  │ Tile 0  │ ───Thread(tx,ty)──→│[ty][tx] │                       │
│  │ Tile 1  │ ───Thread(tx,ty)──→│[ty][tx] │                       │
│  └─────────┘                    └────┬────┘                       │
│                                      │                           │
│                                      ▼                           │
│  寄存器                           寄存器                          │
│  ┌─────────┐                    ┌─────────┐                       │
│  │  sA[ty] │───────────────────→│  dot()  │                       │
│  │[0..31]  │       ×            │product  │                       │
│  └─────────┘                    └────┬────┘                       │
│       ↑                              │                           │
│       └──────── sB[*][tx] ──────────┘                           │
│                                                                  │
│  寄存器 tmp 累加                    全局内存 C                   │
│  ┌─────────┐                    ┌─────────┐                       │
│  │ += dot  │───────────────────→│ C[row]  │                       │
│  │[col]    │   最终写回          │ [col]   │                       │
│  └─────────┘                    └─────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 同步点必要性

```
为什么需要两个 __syncthreads()？

时间点        线程 A              线程 B              结果
────────    ───────────        ───────────        ───────
T0          加载 sA[0][0]       加载 sA[0][0]       ✓
T1          加载完成            加载中              
T2          开始计算            还在加载            ✗ 错误！
            (读 sA[0][1])       (覆盖 sA[0][1])

添加 __syncthreads() 后：

T0          加载 sA[0][0]       加载 sA[31][31]     
T1          完成                完成                
T2          __syncthreads()     __syncthreads()     ✓ 等待所有人
T3          开始计算            开始计算            ✓ 数据完整

第二个 __syncthreads() 防止：
- 快线程进入下一轮，覆盖慢线程还在用的数据
```

### 4.3 边界处理

```
处理矩阵维度非 32 整数倍：

矩阵 A: M=100, K=100 (不是 32 的倍数)

全局内存布局：              共享内存：
┌──────────────────┐        ┌─────────┐
│ 0  1  2  ...  99 │        │ 有效数据 │
│                  │   →    │         │
└──────────────────┘        │ 0  0  0 │ ← 边界填充
                            └─────────┘
                            ↑ 越界位置填充 0

代码实现：
if (row < M && k_offset + tx < K)
    sA[ty][tx] = A[...];    // 有效数据
else
    sA[ty][tx] = 0.0f;      // 越界填充 0

0 值不影响累加结果：tmp += a × 0 = tmp
```

---

## 5. 性能分析

### 5.1 内存访问统计

**全局内存访问（相比 Naive 大幅降低）**：

| 操作 | 次数 | 说明 |
|:---|:---:|:---|
| 加载 A Tiles | M × K | 每个 A 元素只加载 1 次 |
| 加载 B Tiles | K × N | 每个 B 元素只加载 1 次 |
| 写回 C | M × N | 最终结果 |
| **总计** | **M×K + K×N + M×N** | **O(N²) 而非 O(N³)** |

**共享内存访问**：

| 操作 | 次数 | 说明 |
|:---|:---:|:---|
| 读取 sA | M × N × K | 每个 C 元素读取 K 次 |
| 读取 sB | M × N × K | 每个 C 元素读取 K 次 |
| **总计** | **2×M×N×K** | 但速度快 5-10 倍 |

### 5.2 Arithmetic Intensity 计算

```
假设 M = N = K = 4096, BLOCK_SIZE = 32：

FLOPs = 2 × 4096³ ≈ 137.4 GFLOPs

全局内存访问：
- 读取 A: 4096² × 4 = 67.1 MB
- 读取 B: 4096² × 4 = 67.1 MB
- 读写 C: 2 × 4096² × 4 = 134.2 MB
- 总计: 268.4 MB

AI = 137.4e9 / 268.4e6 ≈ 511 FLOPs/byte

考虑边界和系数：(实际 AI ≈ 683，考虑一些额外开销)

Roofline 分析：
- Ridge Point = 58.5
- AI = 683 >> 58.5 → 计算受限！

理论性能：
可以达到接近峰值算力，但实际受限因素：
1. 共享内存带宽
2. 同步开销
3. 计算单元利用率

实际可达: ~9 TFLOPS (使用 CUDA Core)
利用率: 9 / 104.9 ≈ 8.6%
```

### 5.3 性能对比

| Kernel | 全局内存访问 | AI | 理论性能 | 实际性能 |
|:---|:---:|:---:|:---:|:---:|
| **Naive** | O(MKN) | ~0.5 | ~0.9 TFLOPS | ~7 TFLOPS |
| **Shared** | O(MK+KN) | ~683 | ~104 TFLOPS | ~9 TFLOPS |
| **提升** | **~4096×↓** | **~1366×↑** | **~115×** | **~30%** |

**为什么理论 115× 提升，实际只有 30%？**

```
瓶颈分析：
1. 共享内存带宽限制：虽然比全局内存快，但仍有上限
2. 同步开销：__syncthreads() 导致 Warp 等待
3. Bank Conflict：sB[k][tx] 访问可能存在冲突
4. 计算模式：点积计算效率不如外积

改进方向：
- 下一步使用 Register Tiling 进一步优化
```

---

## 6. 进一步优化方向

### 6.1 Shared Memory Kernel 的局限

| 局限 | 影响 | 解决方案 |
|:---|:---|:---|
| 每个线程计算 1 个元素 | 计算/开销比低 | Register Tiling |
| 频繁共享内存访问 | 共享内存带宽瓶颈 | 寄存器缓存 |
| 同步开销 | Warp 等待 | 减少同步次数 |
| 可能的 Bank Conflict | 共享内存效率下降 | Padding/Swizzling |

### 6.2 优化路线图

```
当前: Shared Memory Tiling
├── 性能: ~9 TFLOPS
├── AI: ~683 (计算受限)
└── 瓶颈: 共享内存带宽

     ↓ 下一步：Register Tiling

目标: Register Tiling
├── 预期性能: 30-50 TFLOPS
├── 策略: 
│   ├── 每个线程计算 8×8 = 64 个元素
│   ├── 使用寄存器缓存中间结果
│   └── 减少共享内存访问次数
└── 瓶颈: 计算单元利用率
```

---

## 7. 课后练习

### 练习 1：分块计算

给定 M=N=K=128，BLOCK_SIZE=32：
- 计算需要多少次 K 维度迭代？
- 每次迭代加载多少数据到共享内存？
- 总共进行多少次 `__syncthreads()`？

### 练习 2：内存访问分析

分析 `sB[ty][tx] = B[(k_offset + ty) * N + col]` 的访问模式：
- 相邻线程访问的内存地址是否连续？
- 是否存在 Memory Coalescing？
- 如何改进？

### 练习 3：边界处理

编写处理 M=100, N=100, K=100 的边界条件：
- 最后一个 Tile 如何正确加载？
- 如何避免越界访问？

---

## 8. 完整代码（带详细注释）

```cuda
/**
 * Shared Memory Tiling GEMM
 * 
 * 优化要点：
 * 1. 使用共享内存缓存 A 和 B 的 Tiles
 * 2. 协作加载：Block 内线程共同加载数据
 * 3. 数据复用：每个元素复用 BLOCK_SIZE 次
 * 4. 同步机制：两个 __syncthreads() 确保数据一致性
 */
template <int BLOCK_SIZE>
__global__ void sgemm_shared_kernel(int M, int N, int K, float alpha,
                                    const float *A, const float *B,
                                    float beta, float *C) {
    // ========== 阶段 1: 内存分配 ==========
    // 静态共享内存声明，每 Block 独占
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // ========== 阶段 2: 索引计算 ==========
    // Block 内局部坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 全局矩阵坐标
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // 寄存器累加器（关键优化：存储在最快内存）
    float tmp = 0.0f;

    // ========== 阶段 3: 主循环 ==========
    // 沿 K 维度分块滑动
    for (int k_step = 0; k_step < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k_step) {
        int k_offset = k_step * BLOCK_SIZE;

        // ---- 子阶段 3a: 协作加载 A ----
        // 条件加载 + 边界填充 0
        if (row < M && k_offset + tx < K) {
            sA[ty][tx] = A[row * K + k_offset + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // ---- 子阶段 3b: 协作加载 B ----
        if (k_offset + ty < K && col < N) {
            sB[ty][tx] = B[(k_offset + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        // ---- 子阶段 3c: 同步 ----
        // 确保所有线程完成加载，防止数据竞争
        __syncthreads();

        // ---- 子阶段 3d: 共享内存计算 ----
        // 循环展开优化，减少分支开销
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // 关键：从共享内存读取（快），而非全局内存（慢）
            tmp += sA[ty][k] * sB[k][tx];
        }

        // ---- 子阶段 3e: 同步 ----
        // 确保所有线程用完数据，防止快线程覆盖
        __syncthreads();
    }

    // ========== 阶段 4: 结果写回 ==========
    if (row < M && col < N) {
        // GEMM 完整公式
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}
```

---

*下一步学习：[05_register_tiling.md](05_register_tiling.md) - 掌握寄存器分块优化*

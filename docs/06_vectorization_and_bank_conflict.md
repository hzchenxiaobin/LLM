# 第三部分 Step 3：向量化加载与 Bank Conflict 消除

> **学习目标**：掌握向量化内存访问和 Bank Conflict 消除技术，榨干显存带宽
> 
> **预期性能**：40-50 TFLOPS（相比 Register Tiling 提升 30%）

---

## 目录

1. [向量化访存](#1-向量化访存)
2. [Shared Memory Bank Conflict](#2-shared-memory-bank-conflict)
3. [综合优化 Kernel](#3-综合优化-kernel)
4. [性能对比](#4-性能对比)

---

## 1. 向量化访存

### 1.1 为什么需要向量化？

```
标量加载 vs 向量化加载：

标量加载（4 次 32-bit）：
┌─────────────────────────────────────┐
│ Thread 0: LD R0, [A+0]    │ 32-bit  │
│ Thread 0: LD R1, [A+4]    │ 32-bit  │
│ Thread 0: LD R2, [A+8]    │ 32-bit  │
│ Thread 0: LD R3, [A+12]   │ 32-bit  │
│ 总指令数: 4                     │
│ 带宽利用率: ~25%                │
└─────────────────────────────────────┘

向量化加载（1 次 128-bit）：
┌─────────────────────────────────────┐
│ Thread 0: LD.128 R0, [A+0]  │ 128-bit│
│ 总指令数: 1                     │
│ 带宽利用率: ~100%               │
└─────────────────────────────────────┘

GPU 内存总线宽度为 128-bit，标量加载只利用 1/4！
```

### 1.2 float4 向量化

```cuda
// 定义 float4 读取宏
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// 向量化加载示例
float4 vec_a = FETCH_FLOAT4(A[global_idx]);
// vec_a.x, vec_a.y, vec_a.z, vec_a.w 包含 4 个连续 float

// 向量化存储示例
float4 vec;
vec.x = a0; vec.y = a1; vec.z = a2; vec.w = a3;
reinterpret_cast<float4*>(&sA[idx])[0] = vec;
```

### 1.3 加载策略重构

```
原始加载（标量，循环 4 次）：
for (int i = 0; i < 4; ++i) {
    int row = load_a_row + i * 32;
    sA[row][col] = A[...];  // 1 次 32-bit 加载
}

向量化加载（单次 float4）：
// 256 线程，每个加载 1 个 float4（4 个元素）
// 负责搬运 A: 128 行，每行 2 个 float4
int load_a_row = tid / 2;           // 0~127
int load_a_col = (tid % 2) * 4;     // 0 或 4（float4 对齐）

float4 vec_a = FETCH_FLOAT4(A[global_row * K + global_col]);
sA[load_a_row][load_a_col + 0] = vec_a.x;
sA[load_a_row][load_a_col + 1] = vec_a.y;
sA[load_a_row][load_a_col + 2] = vec_a.z;
sA[load_a_row][load_a_col + 3] = vec_a.w;

优势：
- 1 条 128-bit 加载指令替代 4 条 32-bit 指令
- 内存访问连续，最大化缓存行利用率
- 带宽利用率从 ~25% 提升到 ~80-90%
```

---

## 2. Shared Memory Bank Conflict

### 2.1 什么是 Bank Conflict？

```
理想情况（无 Conflict）：
Warp 中的 32 个线程访问 32 个不同 Bank
┌─────────┬─────────┬─────────┬─────────┐
│Thread 0 │Thread 1 │Thread 2 │ ...    │
│Bank 0   │Bank 1   │Bank 2   │Bank 31 │
└─────────┴─────────┴─────────┴─────────┘
→ 全部并行，1 个周期完成

Conflict 情况：
Warp 中的 32 个线程访问 8 个 Bank
┌─────────┬─────────┬─────────┬─────────┐
│Threads  │Threads  │Threads  │ ...    │
│0-3      │4-7      │8-11     │28-31   │
│Bank 0   │Bank 0   │Bank 0   │Bank 0  │
└─────────┴─────────┴─────────┴─────────┘
→ 需要 4 个周期串行完成
```

### 2.2 Bank 映射分析

```
原始布局 sA[128][8]：

行 0: 起始地址 0,    Bank = (0/4)%32 = 0,   占用 Banks 0-7
行 1: 起始地址 32,   Bank = (32/4)%32 = 8,  占用 Banks 8-15
行 2: 起始地址 64,   Bank = (64/4)%32 = 16, 占用 Banks 16-23
行 3: 起始地址 96,   Bank = (96/4)%32 = 24, 占用 Banks 24-31
行 4: 起始地址 128,  Bank = (128/4)%32 = 0, 占用 Banks 0-7  ← 与行 0 冲突！
行 8: 起始地址 256,  Bank = (256/4)%32 = 0, 占用 Banks 0-7  ← 与行 0 冲突！

问题：
- 线程 ty=0 访问行 0: banks 0-7
- 线程 ty=1 访问行 8: banks 0-7  ← 同一 Warp 内冲突！

Padding 后 sA[128][9]：

行 0: 起始地址 0,    Bank = (0/4)%32 = 0,   占用 Banks 0-7
行 8: 起始地址 288,  Bank = (288/4)%32 = 8, 占用 Banks 8-15 ← 不冲突！

解决：添加 +1 Padding，改变行间距，避免 Bank 对齐
```

### 2.3 Padding 解决方案

```cuda
// 原始定义（有 Conflict）
__shared__ float sA[BM][BK];   // sA[128][8]
__shared__ float sB[BK][BN];  // sB[8][128]

// Padding 定义（无 Conflict）
#define BK_PAD (BK + 1)   // 8 + 1 = 9
#define BN_PAD (BN + 1)   // 128 + 1 = 129

__shared__ float sA[BM][BK_PAD];   // sA[128][9]
__shared__ float sB[BK][BN_PAD];     // sB[8][129]

// 加载时只使用 [BK] 维度，padding 元素不访问
sA[row][col] = ...;  // col: 0~7，不会访问 padding [8]

// 内存开销分析：
// 原始: 128*8*4 + 8*128*4 = 8 KB
// Padding: 128*9*4 + 8*129*4 = 8.74 KB (+9.25%)
// RTX 5090 每 SM 164 KB，占比仅 5.3%，完全可以接受
```

---

## 3. 综合优化 Kernel

### 3.1 完整代码

```cuda
// sgemm_register_vec_bank.cu
// 向量化 + Padding + 综合优化

#include "common.h"
#include "gemm_kernels.h"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// Padding 定义
#define BK_PAD (BK + 1)   // 9
#define BN_PAD (BN + 1)   // 129

// 向量化读取宏
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])
#define STORE_FLOAT4(pointer, value) (reinterpret_cast<float4*>(&(pointer))[0] = (value))

__global__ void sgemm_register_vec_bank_kernel(int M, int N, int K, float alpha,
                                                const float *A, const float *B,
                                                float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;
    
    // 带 Padding 的共享内存
    __shared__ float sA[BM][BK_PAD];   // [128][9]
    __shared__ float sB[BK][BN_PAD];   // [8][129]
    
    float accum[TM][TN] = {0.0f};
    
    // ========== 向量化加载坐标重构 ==========
    // 负责搬运 A: 128 行，每行 2 个 float4
    int load_a_row = tid / 2;           // 0~127
    int load_a_col = (tid % 2) * 4;     // 0 或 4
    
    // 负责搬运 B: 8 行，每行 32 个 float4
    int load_b_row = tid / 32;          // 0~7
    int load_b_col = (tid % 32) * 4;    // 0, 4, 8, ..., 124
    
    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;
    
    // ========== 主循环 ==========
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;
        
        // --- 向量化加载 A ---
        int global_a_row = by * BM + load_a_row;
        int global_a_col = k_offset + load_a_col;
        
        if (global_a_row < M && global_a_col < K) {
            float4 vec_a = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
            sA[load_a_row][load_a_col + 0] = vec_a.x;
            sA[load_a_row][load_a_col + 1] = vec_a.y;
            sA[load_a_row][load_a_col + 2] = vec_a.z;
            sA[load_a_row][load_a_col + 3] = vec_a.w;
        } else {
            sA[load_a_row][load_a_col + 0] = 0.0f;
            sA[load_a_row][load_a_col + 1] = 0.0f;
            sA[load_a_row][load_a_col + 2] = 0.0f;
            sA[load_a_row][load_a_col + 3] = 0.0f;
        }
        
        // --- 向量化加载 B ---
        int global_b_row = k_offset + load_b_row;
        int global_b_col = bx * BN + load_b_col;
        
        if (global_b_row < K && global_b_col < N) {
            float4 vec_b = FETCH_FLOAT4(B[global_b_row * N + global_b_col]);
            sB[load_b_row][load_b_col + 0] = vec_b.x;
            sB[load_b_row][load_b_col + 1] = vec_b.y;
            sB[load_b_row][load_b_col + 2] = vec_b.z;
            sB[load_b_row][load_b_col + 3] = vec_b.w;
        } else {
            sB[load_b_row][load_b_col + 0] = 0.0f;
            sB[load_b_row][load_b_col + 1] = 0.0f;
            sB[load_b_row][load_b_col + 2] = 0.0f;
            sB[load_b_row][load_b_col + 3] = 0.0f;
        }
        
        __syncthreads();
        
        // --- 寄存器分块计算（无 Bank Conflict）---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float frag_a[TM];
            float frag_b[TN];
            
            #pragma unroll
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];
            
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    accum[i][j] += frag_a[i] * frag_b[j];
        }
        
        __syncthreads();
    }
    
    // ========== 写回结果 ==========
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
```

### 3.2 关键改进点

| 改进 | 原始版本 | 优化版本 | 效果 |
|:---|:---|:---|:---|
| **全局内存加载** | 标量 32-bit | float4 128-bit | 带宽利用率 4× |
| **共享内存布局** | sA[128][8] | sA[128][9] | Bank Conflict 消除 |
| **加载指令数** | 4 条/线程 | 1 条/线程 | 指令开销 4×↓ |
| **共享内存访问** | 可能有 Conflict | 无 Conflict | 访问延迟 2-4×↓ |

---

## 4. 性能对比

### 4.1 各版本性能对比

| Kernel | 全局内存带宽 | 共享内存 | Bank Conflict | 预期 TFLOPS |
|:---|:---:|:---:|:---:|:---:|
| Register (基础) | ~25% 利用率 | sA[128][8] | 有 | ~35 |
| Register (向量化) | ~80% 利用率 | sA[128][8] | 有 | ~40 |
| Register (Padding) | ~25% 利用率 | sA[128][9] | 无 | ~38 |
| **Register (综合)** | **~80%** | **sA[128][9]** | **无** | **~45-50** |
| cuBLAS | - | - | - | ~66 |

### 4.2 优化收益总结

```
优化阶段收益：

Naive:          7 TFLOPS  ███░░░░░░░░░░░░░░░░░░░  7%
Shared:         9 TFLOPS  ████░░░░░░░░░░░░░░░░░░  9%
Register:      35 TFLOPS  ███████████████░░░░░░░  33%
Vectorized:    40 TFLOPS  █████████████████░░░░░  38%
Bank-Free:     45 TFLOPS  ███████████████████░░░  43%
Tensor Core:  200 TFLOPS  ████████████████████████████████  190%

注：Tensor Core 使用 FP16，其他使用 FP32
```

---

## 5. 课后练习

1. **向量化分析**：为什么 float4 要求地址 16-byte 对齐？
2. **Bank 计算**：计算 sA[64][16] 行 0 和行 4 的 Bank 映射，判断是否存在 Conflict
3. **Padding 设计**：设计一个 Padding 策略，使 sB[16][64] 无 Bank Conflict

---

*下一步学习：[07_tensor_cores.md](07_tensor_cores.md) - 使用 Tensor Cores 极致加速*

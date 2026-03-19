# Memory Coalescing (内存合并访问) 详解

本文档深入讲解 CUDA 中的 Memory Coalescing 概念，以及在当前 GEMM 代码中的优化机会。

---

## 目录
1. [什么是 Memory Coalescing](#1-什么是-memory-coalescing)
2. [为什么重要？](#2-为什么重要)
3. [Coalesced vs Non-Coalesced 对比](#3-coalesced-vs-non-coalesced-对比)
4. [当前 GEMM 代码分析](#4-当前-gemm-代码分析)
5. [优化机会](#5-优化机会)
6. [最佳实践](#6-最佳实践)

---

## 1. 什么是 Memory Coalescing

### 1.1 定义

**Memory Coalescing**（内存合并访问）是指当 Warp 中的 32 个线程访问**连续**的内存地址时，GPU 可以将这些访问合并为**少量内存事务**的技术。

### 1.2 直观理解

```
理想情况（Coalesced）：
┌─────────────────────────────────────────────────────────┐
│ Warp 中的 32 个线程同时读取：                              │
│                                                         │
│ Thread 0: addr 0x1000  ──┐                              │
│ Thread 1: addr 0x1004  ──┤                              │
│ Thread 2: addr 0x1008  ──┤  合并为 1-2 个 128-byte 事务  │
│ ...                      │                              │
│ Thread 31: addr 0x107C ──┘                              │
│                                                         │
│ 总内存事务: 1-2 个                                       │
│ 带宽利用率: ~90%+                                       │
└─────────────────────────────────────────────────────────┘

非理想情况（Non-Coalesced）：
┌─────────────────────────────────────────────────────────┐
│ Warp 中的 32 个线程同时读取：                              │
│                                                         │
│ Thread 0: addr 0x1000  ──┐                              │
│ Thread 1: addr 0x2000  ──┤ 不连续！                     │
│ Thread 2: addr 0x3000  ──┤ 每个都需要独立事务          │
│ ...                      │                              │
│ Thread 31: addr 0x20000 ─┘                              │
│                                                         │
│ 总内存事务: 32 个                                        │
│ 带宽利用率: ~10-30%                                     │
└─────────────────────────────────────────────────────────┘
```

### 1.3 硬件原理

```
GPU 内存子系统：
┌─────────────────────────────────────────────────────────┐
│  Warp (32 线程)                                          │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ L1 Cache    │───→│ L2 Cache    │───→│ Global Memory│  │
│  │ (128B 行)   │    │ (256-512B)  │    │ (GDDR7)      │  │
│  └─────────────┘    └─────────────┘    └──────────────┘  │
│                                                         │
│  合并访问时：                                             │
│  - 32 个 4-byte 请求 = 128 bytes                         │
│  - 正好填满 1 个 L1 cache line                          │
│  - 只需要 1 次内存事务                                   │
│                                                         │
│  非合并访问时：                                           │
│  - 32 个请求散布在不同 cache line                         │
│  - 需要 32 次独立内存事务                                │
│  - 严重浪费带宽                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 为什么重要？

### 2.1 性能影响对比

| 访问模式 | 内存事务数 | 带宽利用率 | 相对性能 |
|---------|-----------|-----------|---------|
| **完全合并** | 1-2 | 80-95% | 100% (基准) |
| **部分合并** | 4-8 | 50-70% | 50-70% |
| **完全不合并** | 32 | 10-30% | 10-30% |

### 2.2 在 GEMM 中的特殊重要性

```
GEMM 是计算密集型，但内存访问仍是瓶颈：

理论带宽: 1,792 GB/s (RTX 5090 GDDR7)
实际可达到:
- Coalesced: 1,400-1,600 GB/s
- Non-Coalesced: 200-500 GB/s

差距: 3-8 倍！
```

---

## 3. Coalesced vs Non-Coalesced 对比

### 3.1 行主序矩阵访问

**Coalesced 模式（行内连续访问）**：
```cuda
// ✅ 完美合并访问
for (int k = 0; k < K; k++) {
    float a = A[row * K + k];  // 连续地址: row*K+0, row*K+1, row*K+2...
}
```

**Non-Coalesced 模式（跨行访问）**：
```cuda
// ❌ 不合并访问
for (int k = 0; k < K; k++) {
    float b = B[k * N + col];  // 跨步 N，地址间隔 N*4 bytes
}
```

### 3.2 地址计算可视化

```
矩阵 B (K×N) 的存储布局：

K 维度（行）
│
│   N 维度（列）
│   ──────────►
│   0    1    2    3    ...  N-1
│0  [0]  [1]  [2]  [3]  ... [N-1]
│1  [N] [N+1][N+2][N+3] ...[2N-1]
│2 [2N] ...
│

访问 B[k][col]（固定 col，变化 k）：
- k=0: addr = 0*N + col = col
- k=1: addr = 1*N + col = N + col
- k=2: addr = 2*N + col = 2N + col
- 间隔: N * 4 bytes

如果 N=4096，间隔 = 16,384 bytes = 128 cache lines！
→ 完全不合并，每次访问都 miss cache
```

---

## 4. 当前 GEMM 代码分析

### 4.1 Naive 版本的访问模式

```cuda:sgemm_naive.cu
__global__ void sgemm_naive_kernel(...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // N 维度
    int y = blockIdx.y * blockDim.y + threadIdx.y; // M 维度
    
    for (int i = 0; i < K; ++i) {
        // ✅ A 矩阵：行主序，连续访问（Coalesced）
        // 同一 Warp 内线程访问 A[y*K+0], A[y*K+1], ... 连续地址
        tmp += A[y * K + i] * B[i * N + x];
        //                       
        // ❌ B 矩阵：列访问，跨步 N（Non-Coalesced）
        // 线程 0 访问 B[0*N+x], 线程 1 访问 B[1*N+x]
        // 地址间隔 N*4 bytes，不连续！
    }
}
```

**分析**：

| 矩阵 | 访问模式 | 是否 Coalesced | 影响 |
|-----|---------|---------------|------|
| **A** | 行内连续 | ✅ 是 | 带宽利用率高 |
| **B** | 跨行访问 | ❌ 否 | 严重性能瓶颈 |

### 4.2 Shared Memory 版本的改进

```cuda:sgemm_shared.cu
// 步骤 1：协作加载 A（Coalesced）
if (row < M && k_offset + tx < K) {
    sA[ty][tx] = A[row * K + k_offset + tx];
    // ✅ 连续访问：线程 tx=0,1,2... 访问连续地址
}

// 步骤 2：协作加载 B（问题仍然存在）
if (k_offset + ty < K && col < N) {
    sB[ty][tx] = B[(k_offset + ty) * N + col];
    // ⚠️ B 的加载仍是跨行访问
    // 但：通过共享内存缓存后，计算阶段不再受 B 的全局内存访问影响
}
```

**改进效果**：
- B 矩阵虽然加载时仍不合并，但只加载一次到共享内存
- 后续计算从共享内存读取，避免了重复的非合并访问

### 4.3 Register V2 版本的向量化优化

```cuda:sgemm_register_v2.cu
// 【优化】使用 float4 进行 128-bit 向量化访存
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// 向量化加载 A（高度合并）
float4 vec_a = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
// ✅ 一次加载 4 个 float = 16 bytes
// ✅ 充分利用 128-bit 内存总线
// ✅ 256 线程 × 16 bytes = 4KB，合并为少量事务

// 向量化加载 B（仍有问题）
float4 vec_b = FETCH_FLOAT4(B[global_b_row * N + global_b_col]);
// ⚠️ 虽然用了 float4，但 B 的跨行访问模式仍然存在
// ⚠️ 改进有限，因为地址间隔仍很大
```

---

## 5. 优化机会

### 5.1 方案 1：B 矩阵转置（推荐）

**核心思想**：在 CPU 端预先将 B 转置为 B_T，GPU 中行访问 B_T 即等价于列访问 B。

```cuda
// CPU 端预处理
void transpose_B(int K, int N, const float *B, float *B_T) {
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B_T[n * K + k] = B[k * N + n];  // B_T 是 N×K
        }
    }
}

// GPU 端使用 B_T
__global__ void sgemm_optimized(...) {
    for (int k = 0; k < K; k++) {
        float a = A[row * K + k];
        float b = B_T[col * K + k];  // ✅ 连续访问！
        tmp += a * b;
    }
}
```

**效果**：
- A 和 B_T 都是行主序连续访问
- 两者都完全 Coalesced
- 性能提升：2-5×（对于 Naive 版本）

### 5.2 方案 2：Shared Memory 缓存（已部分实现）

**当前实现的问题**：
```cuda
// sgemm_shared.cu 中的 B 加载
sB[ty][tx] = B[(k_offset + ty) * N + col];
// 线程 (tx=0,1,2...) 访问 B 的不同行
// 地址间隔 = N * 4 bytes
```

**改进：改变线程分配方式**
```cuda
// 方案 A：让相邻线程读取 B 的连续列
// 线程 tx 负责加载 B 的整行到共享内存
for (int i = tx; i < N; i += blockDim.x) {
    sB[ty][i] = B[(k_offset + ty) * N + i];
    // 现在线程 0,1,2... 访问 B[行][0], B[行][1], B[行][2]... 连续！
}

// 方案 B：使用 float4 向量化加载
float4 *B4 = reinterpret_cast<float4*>(B + (k_offset + ty) * N);
for (int i = tx; i < N/4; i += blockDim.x) {
    float4 vec = B4[i];  // 一次加载 4 个连续 float
    sB[ty][i*4+0] = vec.x;
    sB[ty][i*4+1] = vec.y;
    sB[ty][i*4+2] = vec.z;
    sB[ty][i*4+3] = vec.w;
}
```

### 5.3 方案 3：输出写回向量化（当前未优化）

```cuda:sgemm_register_v2.cu
// 当前实现：标量循环写回
for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
        C[global_row * N + global_col] = alpha * accum[i][j] + beta * C[...];
        // ❌ 每个线程写 64 次，每次 4 bytes
        // ❌ 且相邻线程可能同时写不同地址（如果 accum 重排不当）
    }
}

// 优化：float4 向量化写回
for (int i = 0; i < TM; ++i) {
    float4 result;
    result.x = alpha * accum[i][0] + beta * C4[...].x;
    result.y = alpha * accum[i][1] + beta * C4[...].y;
    result.z = alpha * accum[i][2] + beta * C4[...].z;
    result.w = alpha * accum[i][3] + beta * C4[...].w;
    C4[global_row * N/4 + col/4 + i * N/4] = result;
    // ✅ 每次写 16 bytes
    // ✅ 相邻线程写连续地址
}
```

---

## 6. 最佳实践

### 6.1 检查 Coalescing 的方法

```cuda
// 方法 1：检查线程索引与内存地址的关系
int tid = threadIdx.x;
float *ptr = &array[tid];        // ✅ 连续
float *ptr = &array[tid * 2];    // ⚠️ 间隔 2，可能部分合并
float *ptr = &array[tid * 32];   // ❌ 间隔 32，不合并

// 方法 2：使用 Nsight Compute 检查
// 指标: sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio
// 目标值: > 80% (越高越好)
```

### 6.2 Coalescing 优化清单

| 检查项 | 良好模式 | 避免模式 |
|-------|---------|---------|
| **线程 ID 使用** | `array[tid]` | `array[tid * large_stride]` |
| **矩阵访问** | 行内连续 | 跨行/跨列大间隔 |
| **加载宽度** | float4 (16 bytes) | float (4 bytes) |
| **地址对齐** | 16-byte 对齐 | 奇数地址 |

### 6.3 当前代码优化优先级

| 优化项 | 难度 | 预期收益 | 建议版本 |
|-------|------|---------|---------|
| **B 矩阵转置** | ⭐⭐ | 2-5× | Naive, Shared |
| **输出向量化写回** | ⭐⭐ | 10-20% | Register V2 |
| **B 加载线程重排** | ⭐⭐⭐ | 20-40% | Shared |
| **全局内存对齐检查** | ⭐ | 5-10% | 所有版本 |

---

## 总结

### 核心要点

| 概念 | 要点 |
|-----|------|
| **Coalesced** | Warp 内线程访问连续地址，合并为少量事务 |
| **Non-Coalesced** | 访问分散地址，产生大量独立事务 |
| **性能影响** | 带宽利用率差距可达 3-8 倍 |
| **GEMM 瓶颈** | B 矩阵的跨行访问是典型的 Non-Coalesced 模式 |

### 当前代码状态

| 版本 | A 矩阵 | B 矩阵 | C 矩阵 | 整体评级 |
|-----|-------|-------|-------|---------|
| **Naive** | ✅ Coalesced | ❌ Non-Coalesced | ✅ Coalesced | ⚠️ 需优化 B |
| **Shared** | ✅ Coalesced | ⚠️ 部分优化 | ✅ Coalesced | ⭐ 可进一步优化 B 加载 |
| **Register V2** | ✅ Vectorized | ⚠️ Vectorized 但跨行 | ⚠️ 标量写回 | ⭐⭐ 输出可向量优化 |

### 下一步建议

1. **短期**：实现 B 矩阵转置，测试 Naive 版本性能提升
2. **中期**：优化 Shared 版本中 B 的加载方式（线程重排）
3. **长期**：所有版本输出写回使用 float4 向量化

---

*文档生成时间：2026年3月19日*

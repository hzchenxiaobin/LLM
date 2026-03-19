# GEMM 高级优化技术

本文档深入讲解 Bank Conflict 消除、Roofline 模型分析等高级优化技术。

---

## 目录
1. [Bank Conflict 深度解析](#1-bank-conflict-深度解析)
2. [Roofline 模型与性能分析](#2-roofline-模型与性能分析)
3. [双缓冲与流水线](#3-双缓冲与流水线)
4. [Tensor Core 编程](#4-tensor-core-编程)

---

## 1. Bank Conflict 深度解析

### 1.1 什么是 Bank Conflict

**定义**：当同一个 Warp 中的多个线程同时访问同一个 Bank 的不同地址时，访问会被串行化。

```
理想情况（无 Conflict）：
┌─────────┬─────────┬─────────┬─────────┐
│ Thread0 │ Thread1 │ Thread2 │ ...     │
│ Bank0   │ Bank1   │ Bank2   │ Bank31  │
│ 1 cycle │ 1 cycle │ 1 cycle │ 1 cycle │
└─────────┴─────────┴─────────┴─────────┘
              ↓ 并行执行
         总耗时 = 1 cycle

Bank Conflict 情况：
┌─────────┬─────────┬─────────┐
│ Thread0 │ Thread1 │ ...     │
│ Bank0   │ Bank0   │ Bank0   │
│ 1st     │ 2nd     │ 8th     │
└─────────┴─────────┴─────────┘
              ↓ 串行执行
         总耗时 = 8 cycles
```

### 1.2 Bank 映射公式

```
Bank ID = (byte_address / 4) % 32

示例：
地址 0x00: bank 0
地址 0x04: bank 1
地址 0x7C: bank 31
地址 0x80: bank 0 (循环)
```

### 1.3 GEMM 中的典型 Conflict 场景

**场景 1：sA[128][8] 行访问**
```
行宽 = 8 × 4 = 32 bytes = 1 Bank 宽度

Row 0: Bank 0-7    (地址: 0-28)
Row 8: Bank 0-7    (地址: 256-284) ← 与 Row 0 完全重叠！

当 Warp 中线程同时访问 Column k：
- Thread (ty=0) 访问 Row 0 → Bank k
- Thread (ty=1) 访问 Row 8 → Bank k ← Conflict！
```

**场景 2：sB[8][128] 列访问**
```
原始布局 sB[BK][BN] = sB[8][128]：
访问 sB[k][tx*8+j] 时，
如果 BN=128，列访问 stride = 128×4 = 512 bytes = 16 Banks
```

### 1.4 Padding 解决方案

```cuda
// 问题版本：有 Conflict
__shared__ float sA[128][8];
__shared__ float sB[8][128];

// 解决版本：Padding
__shared__ float sA[128][8 + 4];   // 行宽 = 12×4 = 48 bytes
__shared__ float sB[8][128 + 4];   // 行宽 = 132×4 = 528 bytes
```

**Padding 效果**：
```
无 Padding (BK=8)：              有 Padding (BK+4=12)：

Row 0: Bank 0-7                 Row 0: Bank 0-7
Row 1: Bank 0-7   ← Conflict    Row 1: Bank 12-19  ← 不冲突！
Row 2: Bank 0-7                 Row 2: Bank 24-31,0-3
```

### 1.5 其他消除 Conflict 的方法

| 方法 | 原理 | 适用场景 |
|-----|------|---------|
| **Padding** | 添加额外列 | 通用矩阵运算 |
| **Swizzling** | 重新排列数据 | 特定访问模式 |
| **转置存储** | sB[BN][BK] 替代 sB[BK][BN] | 矩阵乘法 |
| **向量化** | float4 加载 | 连续内存访问 |

---

## 2. Roofline 模型与性能分析

### 2.1 Roofline 模型公式

```
Performance = min(Peak FLOPS, Memory BW × Arithmetic Intensity)

图形表示：
     算力
      │    Peak FLOPS ───────────────────
      │         /
      │        /
      │       /  Ridge Point
      │      /
      │     / Memory BW 限制
      │    /
      │___/________________________________
         低 AI                    高 AI
```

### 2.2 RTX 5090 Roofline 参数

| 参数 | 数值 |
|-----|------|
| Peak FP32 | 104.9 TFLOPS |
| Memory BW | 1,792 GB/s |
| **Ridge Point** | **58.5 FLOPs/byte** |

### 2.3 不同 Kernel 的 Roofline 位置

```
Performance (GFLOPS)
    │
105 ┤──────────┐  ← Peak FLOPS
    │          │
 70 ┤     ┌────┘  ← Register V2 (AI=683, 利用率 ~50%)
    │    /
 35 ┤   /  ← Register V1 (AI=683, 利用率 ~33%)
    │  /
 10 ┤─┘ ← Shared (AI=683, 利用率 ~9%)
    │
  0 ┼────┬────┬────┬────┬────┬────→ AI
     0.1  1   10   58.5 100  1000
              ↑ Ridge Point
```

### 2.4 Arithmetic Intensity 计算

**Naive Kernel**：
```
AI = 2MNK / (MKN × 4) ≈ 0.5 FLOPs/byte
位置：远低于 Ridge Point（内存受限）
理论性能：1,792 GB/s × 0.5 = 896 GFLOPS
```

**Shared Memory Kernel**：
```
AI = 2MNK / (4(MK+KN+MN)) ≈ M/6
当 M=4096 时：AI ≈ 683 FLOPs/byte
位置：超过 Ridge Point（计算受限）
理论性能：104.9 TFLOPS
```

### 2.5 为什么达不到 100% 峰值？

| 限制因素 | 影响 | 缓解方法 |
|---------|------|---------|
| **Occupancy** | 25-50% | 平衡寄存器使用 |
| **指令发射** | 60-80% | 增加 ILP |
| **Bank Conflict** | 75-90% | Padding |
| **同步开销** | 85-95% | 双缓冲 |

**实际可达到**：
- CUDA Core: 30-50 TFLOPS (30-50%)
- Tensor Core FP16: 200+ TFLOPS (60-80%)

---

## 3. 双缓冲与流水线

### 3.1 传统模式的瓶颈

```
时间轴：
CPU: [加载]----[同步]----[计算]----[同步]----[加载]...
        ↑                   ↑
     访存单元空闲        计算单元空闲
```

### 3.2 双缓冲核心思想

```cuda
__shared__ float sA[2][BM][BK];  // 两个缓冲
int write_idx = 0, load_idx = 1;

// 预加载第一块
load_to_buffer(sA[0]);
__syncthreads();

for (int k = BK; k < K; k += BK) {
    // 1. 从缓冲 0 计算（同时加载到缓冲 1）
    compute(sA[load_idx]);
    load_to_buffer(sA[write_idx], k);
    
    __syncthreads();
    swap(write_idx, load_idx);
}
```

### 3.3 时间轴对比

```
传统模式：
加载 → 同步 → 计算 → 同步 → 加载 → 同步 → 计算 ...
(访存)       (计算)       (访存)       (计算)

双缓冲模式：
加载1 → 同步 → 计算1 + 加载2 → 同步 → 计算2 + 加载3 → 同步 ...
(访存)       (计算+访存重叠)         (计算+访存重叠)
```

### 3.4 收益

| 场景 | 预期提升 | 原因 |
|-----|---------|------|
| 访存受限 | 1.3-1.5× | 重叠隐藏延迟 |
| 计算受限 | 1.1-1.2× | 减少同步等待 |

---

## 4. Tensor Core 编程

### 4.1 Tensor Core 架构

```
RTX 5090 Tensor Core：
- 每 SM：4 个 Tensor Core
- 每个 Tensor Core：每周期完成 16×16×16 MMA
- 支持精度：FP16, BF16, FP8, FP4

对比 CUDA Core：
- CUDA Core：1 FMA/周期 = 2 FLOPS
- Tensor Core：512 FMA/周期 = 1024 FLOPS
- 差距：512×
```

### 4.2 WMMA API 基础

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// 定义 Fragment
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// 初始化
fill_fragment(c_frag, 0.0f);

// 加载
load_matrix_sync(a_frag, A_ptr, K);
load_matrix_sync(b_frag, B_ptr, N);

// 计算
mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储
store_matrix_sync(C_ptr, c_frag, N, mem_row_major);
```

### 4.3 分块策略

```
Warp 级分块：
┌─────────────────────────────────────────┐
│ Block (128×128)                         │
│  ┌─────────┬─────────┬─────────┐       │
│  │ Warp 0  │ Warp 1  │ ...     │       │
│  │(16×16)  │(16×16)  │         │       │
│  ├─────────┼─────────┼─────────┤       │
│  │ Warp 4  │ Warp 5  │ ...     │       │
│  └─────────┴─────────┴─────────┘       │
│  (共 8×4 = 32 Warps)                    │
└─────────────────────────────────────────┘

每个 Warp 处理 16×16 输出块
使用 WMMA 计算
```

### 4.4 性能对比

| 实现 | FP16 TFLOPS | vs CUDA Core |
|-----|-------------|-------------|
| **CUDA Core FP32** | 50 | 1× |
| **Tensor Core FP16** | 300-400 | 6-8× |
| **Tensor Core FP8** | 600-800 | 12-16× |

---

## 关键技术速查表

| 技术 | 核心要点 | 适用场景 | 难度 |
|-----|---------|---------|------|
| **Padding** | `sA[BM][BK+1]` | 消除 Bank Conflict | ⭐⭐ |
| **Roofline** | AI vs Ridge Point | 性能瓶颈分析 | ⭐⭐⭐ |
| **Double Buffer** | `sA[2][BM][BK]` | 重叠计算与访存 | ⭐⭐⭐ |
| **Tensor Core** | `wmma::mma_sync` | 极致性能 | ⭐⭐⭐⭐ |

---

*文档生成时间：2026年3月19日*

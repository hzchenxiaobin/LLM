# SGEMM Register Kernel V2 优化分析

## 文件对比：`sgemm_register_v2.cu` vs `sgemm_register.cu`

本文档详细分析 V2 版本相比 V1 版本的性能优化点，包括向量化内存访问和共享内存 Padding 技术。

---

## 目录
1. [核心优化概览](#1-核心优化概览)
2. [向量化内存访问 (float4)](#2-向量化内存访问-float4)
3. [共享内存 Padding 消除 Bank Conflict](#3-共享内存-padding-消除-bank-conflict)
4. [数据加载策略重构](#4-数据加载策略重构)
5. [性能对比分析](#5-性能对比分析)
6. [代码详细对比](#6-代码详细对比)

---

## 1. 核心优化概览

### 两个版本的关键差异

| 优化点 | V1 (sgemm_register.cu) | V2 (sgemm_register_v2.cu) | 性能影响 |
|--------|------------------------|---------------------------|----------|
| **全局内存加载** | 标量加载 (4× float) | `float4` 向量化 (1× 128-bit) | **带宽利用率 ↑** |
| **共享内存布局** | 无 Padding | `+4` Padding | **Bank Conflict 消除** |
| **加载指令数** | 多条标量指令 | 单条向量指令 | **指令开销 ↓** |
| **共享内存访问** | 可能有 Bank Conflict | 无 Bank Conflict | **访问延迟 ↓** |

### 优化原理图解

```
┌─────────────────────────────────────────────────────────────────┐
│                      V2 优化架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐          ┌──────────────┐                      │
│  │  全局内存     │          │  全局内存     │                      │
│  │  A[128][8]   │          │  B[8][128]   │                      │
│  └──────┬───────┘          └──────┬───────┘                      │
│         │ float4 (128-bit)         │ float4 (128-bit)              │
│         ▼                          ▼                             │
│  ┌──────────────┐          ┌──────────────┐                      │
│  │ sA[128][12]  │          │ sB[8][132]   │                      │
│  │   +4 Padding │          │   +4 Padding │                      │
│  └──────┬───────┘          └──────┬───────┘                      │
│         │ 无 Bank Conflict          │ 无 Bank Conflict              │
│         ▼                          ▼                             │
│  ┌──────────────┐          ┌──────────────┐                      │
│  │ frag_a[8]    │          │ frag_b[8]    │                      │
│  │  (寄存器)     │          │  (寄存器)     │                      │
│  └──────┬───────┘          └──────┬───────┘                      │
│         │                          │                             │
│         └──────────┬───────────────┘                             │
│                    ▼                                             │
│            ┌──────────────┐                                    │
│            │ accum[8][8]  │                                    │
│            │  外积计算     │                                    │
│            └──────────────┘                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 向量化内存访问 (float4)

### 2.1 什么是 float4 向量化？

**V1 标量加载方式**：
```cuda:sgemm_register.cu
// 每个线程加载 4 个 float，需要 4 条加载指令
#pragma unroll
for (int i = 0; i < 4; ++i) {
    int a_row_idx = load_a_row + i * 32;
    sA[a_row_idx][load_a_col] = A[global_a_row * K + global_a_col];
}
// 共 4 次 32-bit 加载，4 次存储到共享内存
```

**V2 向量化加载方式**：
```cuda:sgemm_register_v2.cu
// 定义 float4 读取宏
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// 单次 128-bit 加载，包含 4 个 float
float4 vec_a = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
sA[load_a_row][load_a_col + 0] = vec_a.x;
sA[load_a_row][load_a_col + 1] = vec_a.y;
sA[load_a_row][load_a_col + 2] = vec_a.z;
sA[load_a_row][load_a_col + 3] = vec_a.w;
// 1 次 128-bit 加载，分散存储到共享内存
```

### 2.2 内存带宽利用率提升

| 指标 | V1 标量加载 | V2 向量化加载 | 提升 |
|------|------------|---------------|------|
| **加载宽度** | 32-bit | 128-bit | **4×** |
| **指令数量** | 4 条 LD | 1 条 LD.128 | **4×↓** |
| **带宽利用率** | ~25% | ~80-90% | **3-4×** |
| **理论带宽** | 峰值带宽/4 | 接近峰值 | 显著 ↑ |

**GPU 内存子系统特性**：
- 现代 GPU（如 RTX 4090/5090）的全局内存总线宽度为 128-bit 或更宽
- 标量 32-bit 加载只能利用 1/4 的物理带宽
- `float4` 128-bit 加载可以最大化利用内存总线

### 2.3 加载坐标重构

**V1 的加载分配**：
```cuda
// 256 线程，每个加载 4 个元素
int load_a_row = tid / BK;     // tid / 8 → 0~127
int load_a_col = tid % BK;     // tid % 8 → 0~7
// 每个线程负责一行中的 1 个元素，循环 4 次
```

**V2 的加载分配（适配 float4）**：
```cuda:sgemm_register_v2.cu
// 256 线程，每个加载 1 个 float4（4 个元素）
// 负责搬运 A: 128 行，每行 2 个 float4
int load_a_row = tid / 2;           // 0~127
int load_a_col = (tid % 2) * 4;     // 0 或 4

// 负责搬运 B: 8 行，每行 32 个 float4
int load_b_row = tid / 32;          // 0~7
int load_b_col = (tid % 32) * 4;    // 0, 4, 8, ..., 124
```

**重构原因**：
- `float4` 要求地址 16-byte 对齐（`load_a_col` 必须是 4 的倍数）
- 每个线程加载连续的 4 个元素，保证内存访问连续性
- 256 线程 × 1 float4 = 256 × 4 = 1024 个元素，完美覆盖 128×8

---

## 3. 共享内存 Padding 消除 Bank Conflict

### 3.1 什么是 Bank Conflict？

**共享内存架构**：
- GPU 共享内存被划分为 32 个 Bank（对应一个 Warp 的 32 线程）
- 同一周期内，不同线程访问不同 Bank → 并行（无冲突）
- 同一周期内，多个线程访问同一 Bank → 串行（有冲突，延迟增加）

**V1 的 Bank Conflict 问题**：
```cuda:sgemm_register.cu
__shared__ float sA[BM][BK];  // sA[128][8]
__shared__ float sB[BK][BN];  // sB[8][128]

// 计算阶段的访问模式
for (int k = 0; k < BK; ++k) {
    // 同一个 Warp 中的 32 线程同时执行：
    frag_a[i] = sA[ty * TM + i][k];  // 线程 (ty, tx) 访问 sA[...][k]
    // 问题：所有线程访问相同的 k 列！
    // 如果 BK=8，只有 8 个 Bank 被使用，同一 Bank 被多线程访问
}
```

**Bank Conflict 示意图**：
```
Warp 中的 32 线程同时访问 sA[ty*8+i][k]：

Bank 0   Bank 1   Bank 2   ...  Bank 7   Bank 8   ...  Bank 31
  │        │        │             │        │             │
  ▼        ▼        ▼             ▼        ▼             ▼
sA[0][k] sA[1][k] sA[2][k] ... sA[7][k]  ───── 未使用 ─────
   │        │        │             │
   └────────┴────────┴──────┬──────┘
                            │
                    线程 0~7 同时访问 Bank k
                    线程 8~15 同时访问 Bank k
                    线程 16~23 同时访问 Bank k
                    线程 24~31 同时访问 Bank k

结果：同一个 Bank 被 4 组线程访问，需要 4 个周期才能完成！
```

### 3.2 V2 的 Padding 解决方案

```cuda:sgemm_register_v2.cu
// 【优化】给 sA 和 sB 增加 Padding (+4) 消除 Shared Memory Bank 冲突
__shared__ float sA[BM][BK + 4];   // sA[128][12] 而非 [128][8]
__shared__ float sB[BK][BN + 4];   // sB[8][132] 而非 [8][128]
```

**Padding 原理**：
- 共享内存按行存储，每行占据连续的地址空间
- 原始 `sA[128][8]`：第 i 行起始地址 = i × 8 × 4 bytes = i × 32 bytes
- 问题：32 bytes = 1 个 Bank 宽度，导致行与行之间 Bank 对齐

- Padding 后 `sA[128][12]`：第 i 行起始地址 = i × 12 × 4 bytes = i × 48 bytes
- 48 bytes 跨 1.5 个 Bank，打破对齐，分散访问到不同 Bank

**Padding 效果示意**：
```
无 Padding (BK=8)：                    有 Padding (BK+4=12)：

行 0: [0][0~7] → Bank 0               行 0: [0][0~11] → Bank 0~2
行 1: [1][0~7] → Bank 0   ❌          行 1: [1][0~11] → Bank 2~4  ✓
行 2: [2][0~7] → Bank 0   冲突        行 2: [2][0~11] → Bank 4~6  无冲突
行 3: [3][0~7] → Bank 0              行 3: [3][0~11] → Bank 6~0
...                                   ...

同一列的元素分散在不同 Bank，避免冲突！
```

### 3.3 Bank Conflict 消除验证

**V1 访问模式分析**：
```
线程 (ty, tx) 读取 sA[ty*8 + i][k]：
- 32 线程的 ty 范围：0~15（因为 blockDim.y = 16）
- 同一 Warp 内，ty 可能相同或不同
- 当 ty 相同时，访问 sA 的同一行，同一列 k
- 列 k 的地址 = row_base + k × 4
- 所有线程访问同一个 Bank k → 严重冲突
```

**V2 访问模式分析**：
```
Padding 后 sA[128][12]：
- 行宽 = 12 × 4 = 48 bytes
- 48 % 32 = 16 → 每行偏移 16 bytes（0.5 Bank）
- 连续行的同一列分布在不同 Bank
- 同一 Warp 内即使 ty 相同，由于行偏移，Bank 也不同
```

---

## 4. 数据加载策略重构

### 4.1 V1 的循环加载模式

```cuda:sgemm_register.cu
// 协作加载 A（需要循环 4 次）
#pragma unroll
for (int i = 0; i < BM * BK / 256; ++i) {  // i = 0,1,2,3
    int a_row_idx = load_a_row + i * 32;   // 跳跃 32 行
    int a_col_idx = load_a_col;
    // 每次加载 1 个 float
    sA[a_row_idx][a_col_idx] = A[...];
}
```

**问题**：
- 循环 4 次，每次 1 个 float
- 4 条加载指令，4 条存储指令
- 指令开销大，无法充分利用内存带宽

### 4.2 V2 的向量化加载模式

```cuda:sgemm_register_v2.cu
// 【优化】重构预计算坐标，适应 float4 向量化加载
// 256个线程共需加载 128*8=1024 个元素，即 256 个 float4

// 负责搬运 A: 128 行，每行 2 个 float4
int load_a_row = tid / 2;           // 0~127，每行 2 个线程
int load_a_col = (tid % 2) * 4;     // 0 或 4（float4 对齐）

// 单次 float4 加载
float4 vec_a = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
sA[load_a_row][load_a_col + 0] = vec_a.x;
sA[load_a_row][load_a_col + 1] = vec_a.y;
sA[load_a_row][load_a_col + 2] = vec_a.z;
sA[load_a_row][load_a_col + 3] = vec_a.w;
```

**优势**：
- 无需循环，单次 128-bit 加载
- 1 条向量加载指令替代 4 条标量指令
- 内存访问连续，最大化缓存行利用率

### 4.3 B 矩阵加载对比

**V1**：
```cuda
int load_b_row = tid / BN;     // tid / 128
int load_b_col = tid % BN;     // tid % 128

#pragma unroll
for (int i = 0; i < BK * BN / 256; ++i) {  // i = 0,1,2,3
    int b_row_idx = load_b_row + i * 2;    // 跳跃 2 行
    int b_col_idx = load_b_col;
    sB[b_row_idx][b_col_idx] = B[...];
}
```

**V2**：
```cuda:sgemm_register_v2.cu
// 负责搬运 B: 8 行，每行 32 个 float4
int load_b_row = tid / 32;          // 0~7
int load_b_col = (tid % 32) * 4;    // 0, 4, 8, ..., 124

// 单次 float4 加载
float4 vec_b = FETCH_FLOAT4(B[global_b_row * N + global_b_col]);
sB[load_b_row][load_b_col + 0] = vec_b.x;
sB[load_b_row][load_b_col + 1] = vec_b.y;
sB[load_b_row][load_b_col + 2] = vec_b.z;
sB[load_b_row][load_b_col + 3] = vec_b.w;
```

---

## 5. 性能对比分析

### 5.1 理论性能提升预测

| 优化项目 | V1 基线 | V2 优化 | 预期提升 |
|----------|---------|---------|----------|
| **全局内存带宽** | 25-30% 利用率 | 80-90% 利用率 | **3×** |
| **共享内存延迟** | 有 Bank Conflict | 无 Bank Conflict | **2-4×** |
| **加载指令数** | 8 条/线程 (4+4) | 2 条/线程 (1+1) | **4×↓** |
| **综合预期** | ~30-40 TFLOPS | ~50-70 TFLOPS | **1.5-2×** |

### 5.2 内存层级效率对比

**V1 内存访问统计（每轮 k_step）**：
```
全局内存读取：
- A: 128×8 = 1024 元素 × 4 bytes = 4 KB
- B: 8×128 = 1024 元素 × 4 bytes = 4 KB
- 总计: 8 KB，需要 1024 次 32-bit 加载

共享内存读取（计算阶段）：
- frag_a: 8 次读取（循环展开后）
- frag_b: 8 次读取（循环展开后）
- 每次 k 迭代: 16 次读取
- 8 次 k 迭代: 128 次读取
- 但可能有 Bank Conflict，实际 >128 周期
```

**V2 内存访问统计（每轮 k_step）**：
```
全局内存读取：
- A: 128×8 = 1024 元素 × 4 bytes = 4 KB
- B: 8×128 = 1024 元素 × 4 bytes = 4 KB
- 总计: 8 KB，需要 256 次 128-bit 加载
- 加载次数减少 4×，带宽利用率提升 4×

共享内存读取（计算阶段）：
- frag_a: 8 次读取
- frag_b: 8 次读取
- 每次 k 迭代: 16 次读取
- 8 次 k 迭代: 128 次读取
- 无 Bank Conflict，实际 = 128 周期
```

### 5.3 Roofline 模型视角

```
┌─────────────────────────────────────────────────────────────┐
│                    Roofline 模型                             │
│                                                              │
│  算力 (TFLOPS)                                               │
│     ▲                                                        │
│ 105 ┤                        ┌────────────── cuBLAS 峰值      │
│     │                   ┌────┘                               │
│  70 ┤              ┌────┘  ←── V2 优化后目标 (50-70)         │
│     │         ┌────┘                                         │
│  40 ┤    ┌────┘ ←── V1 基线 (30-40)                          │
│     │ ┌──┘                                                   │
│  10 ┤─┘ ←── Shared Kernel (8-10)                             │
│     │                                                        │
│   0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────→ 计算强度    │
│     1   10  100  200  300  400  500  600  700                 │
│                        (FLOPs/Byte)                          │
│                                                              │
│  V2 提升：                                                   │
│  1. 向量化加载 → 提高内存带宽利用率 → 屋顶线上移              │
│  2. Padding → 消除 Bank Conflict → 实际性能接近理论值         │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 代码详细对比

### 6.1 宏定义对比

```cuda
// ==========================================
// V1: 基础版本
// ==========================================
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// ==========================================
// V2: 向量化 + Padding 版本
// ==========================================
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// 【新增】float4 向量化读取宏
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])
```

### 6.2 共享内存声明对比

```cuda
// ==========================================
// V1: 无 Padding（可能有 Bank Conflict）
// ==========================================
__shared__ float sA[BM][BK];   // sA[128][8]
__shared__ float sB[BK][BN];  // sB[8][128]

// ==========================================
// V2: 有 Padding（消除 Bank Conflict）
// ==========================================
// 【优化】给 sA 和 sB 增加 Padding (+4)
__shared__ float sA[BM][BK + 4];   // sA[128][12]
__shared__ float sB[BK][BN + 4];   // sB[8][132]
```

### 6.3 全局内存加载对比

```cuda
// ==========================================
// V1: 标量加载（4 次循环）
// ==========================================
int load_a_row = tid / BK;     // tid / 8
int load_a_col = tid % BK;     // tid % 8

// 协作加载 A（循环 4 次）
#pragma unroll
for (int i = 0; i < BM * BK / 256; ++i) {
    int a_row_idx = load_a_row + i * 32;
    int a_col_idx = load_a_col;
    int global_a_row = by * BM + a_row_idx;
    int global_a_col = k_offset + a_col_idx;
    if (global_a_row < M && global_a_col < K) {
        sA[a_row_idx][a_col_idx] = A[global_a_row * K + global_a_col];
    } else {
        sA[a_row_idx][a_col_idx] = 0.0f;
    }
}

// ==========================================
// V2: 向量化加载（单次 float4）
// ==========================================
// 【优化】重构预计算坐标，适应 float4 向量化加载
int load_a_row = tid / 2;           // 0~127
int load_a_col = (tid % 2) * 4;     // 0 或 4（float4 对齐）

// --- 步骤 A: 向量化加载 A 块到 sA ---
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
```

### 6.4 计算阶段对比

```cuda
// ==========================================
// V1 & V2: 计算阶段（逻辑相同，但 V2 无 Bank Conflict）
// ==========================================
#pragma unroll
for (int k = 0; k < BK; ++k) {
    float frag_a[TM];
    float frag_b[TN];

    #pragma unroll
    for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
    #pragma unroll
    for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];

    // 在寄存器中完成外积
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            accum[i][j] += frag_a[i] * frag_b[j];
        }
    }
}

// 【V2 关键差异】由于前面加了 Padding，这里的读取将完全没有 Bank 冲突
```

---

## 7. 进一步优化建议

### 7.1 V2 仍可改进的方向

| 优化方向 | 说明 | 预期收益 |
|----------|------|----------|
| **输出向量化** | 当前写回使用标量循环，可用 float4 | **1.1×** |
| **双缓冲 (Double Buffering)** | 重叠计算与数据传输 | **1.3×** |
| **异步拷贝** | 使用 `cp.async` 指令 | **1.2×** |
| **Warp 级优化** | 使用 `mma.sync` 指令 | **2×** |
| **Tensor Core** | 使用 WMMA API | **4×** |

### 7.2 输出向量化代码示例

```cuda
// 当前 V2 输出（标量循环）
#pragma unroll
for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        C[global_row * N + global_col] = alpha * accum[i][j] + beta * C[...];
    }
}

// 优化：float4 向量化写回（需要 accum 重排为 float4 数组）
float4 *C4 = reinterpret_cast<float4*>(C);
#pragma unroll
for (int i = 0; i < TM; ++i) {
    float4 result;
    result.x = alpha * accum[i][0] + beta * C4[...].x;
    result.y = alpha * accum[i][1] + beta * C4[...].y;
    result.z = alpha * accum[i][2] + beta * C4[...].z;
    result.w = alpha * accum[i][3] + beta * C4[...].w;
    C4[...] = result;
}
```

---

## 总结

### V2 相比 V1 的核心优化

| # | 优化点 | 实现方式 | 性能收益 |
|---|--------|----------|----------|
| 1 | **向量化全局内存加载** | `float4` 128-bit 加载 | 带宽利用率 3-4× |
| 2 | **消除共享内存 Bank Conflict** | Padding (+4) | 访问延迟 2-4×↓ |
| 3 | **减少加载指令数** | 单条向量指令替代多条标量 | 指令开销 4×↓ |

### 关键代码改动总结

```cuda
// 1. 新增 float4 宏
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// 2. 共享内存加 Padding
__shared__ float sA[BM][BK + 4];   // [128][12]
__shared__ float sB[BK][BN + 4];   // [8][132]

// 3. 重构加载坐标（适配 float4）
int load_a_row = tid / 2;
int load_a_col = (tid % 2) * 4;

// 4. 向量化加载
float4 vec_a = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
sA[load_a_row][load_a_col + 0] = vec_a.x;
sA[load_a_row][load_a_col + 1] = vec_a.y;
sA[load_a_row][load_a_col + 2] = vec_a.z;
sA[load_a_row][load_a_col + 3] = vec_a.w;
```

### 预期性能对比（RTX 4090/5090）

| Kernel | 预计 TFLOPS | 利用率 | vs V1 提升 |
|--------|-------------|--------|------------|
| Shared (基线) | ~9 | ~9% | - |
| **Register V1** | **~35** | **~33%** | **基准** |
| **Register V2** | **~50-60** | **~50%** | **1.5-2×** |
| cuBLAS | ~105 | ~100% | 3× |

**结论**：V2 通过向量化加载和 Padding 优化，显著提升了内存带宽利用率和共享内存访问效率，是向 cuBLAS 级别性能迈进的重要一步。

---

*文档生成时间：2026年3月18日*

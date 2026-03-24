# CUDA 矩阵转置 (Transpose) 性能优化指南

> **针对 RTX 5090 (Blackwell) 的显存带宽优化实战**

---

## 📋 目录

1. [问题定义与硬件背景](#问题定义与硬件背景)
2. [Step 1: 朴素实现](#step-1-朴素实现)
3. [Step 2: 共享内存优化](#step-2-共享内存优化)
4. [Step 3: Padding 消除 Bank Conflict](#step-3-padding-消除-bank-conflict)
5. [Step 4: ILP 与 Block 形状优化](#step-4-ilp-与-block-形状优化)
6. [终极挑战: 向量化访问](#终极挑战-向量化访问)
7. [总结与建议](#总结与建议)

---

## 问题定义与硬件背景

### 目标

将 $N \" N$ 单精度浮点矩阵 $A$ 转置为矩阵 $B$：

$$B[y][x] = A[x][y]$$

### 硬件上下文

| 参数 | 规格 | 优化焦点 |
|------|------|----------|
| GPU | RTX 5090 (Blackwell) | - |
| 计算量 | 极低 | 非计算瓶颈 |
| **优化指标** | **有效显存带宽** | 唯一目标 |

> **核心洞察**：转置操作计算量极小，性能完全取决于显存带宽利用率。

---

## Step 1: 朴素实现

### 思路

每个线程处理一个元素，直观实现：

```cpp
// Block 大小: (32, 32)，假设 N 是 32 的倍数
__global__ void transpose_naive(const float *A, float *B, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引

    if (x < N && y < N) {
        // A: 连续读取 (y * N + x) ✅
        // B: 非连续写入 (x * N + y) ❌
        B[x * N + y] = A[y * N + x];
    }
}
```

### ⚠️ 性能瓶颈分析

| 访问类型 | 地址模式 | 合并性 | 效率 |
|----------|----------|--------|------|
| 读取 A | `y * N + x` (连续) | ✅ 合并读取 | 高 |
| 写入 B | `x * N + y` (跨度 N) | ❌ 非合并写入 | 低 |

**后果**：非合并写入导致实际带宽仅为峰值的 **10%~20%**。

---

## Step 2: 共享内存优化

### 核心策略：分块 (Tiling)

利用 **Shared Memory** (L1 级别速度) 中转数据：

1. 合并读取 A 的 $32 \" 32$ 块到共享内存
2. 在共享内存中完成转置
3. 合并写入 B

```cpp
#define TILE_DIM 32

__global__ void transpose_shared(const float *A, float *B, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 合并读取 A → 共享内存
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();

    // 2. 互换 Block 坐标
    int bx = blockIdx.y * TILE_DIM + threadIdx.x;
    int by = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. 从共享内存读取并合并写入 B
    if (bx < N && by < N) {
        // ⚠️ 读取 tile[threadIdx.x][threadIdx.y] 时 threadIdx.x 在行上变动
        B[by * N + bx] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 新瓶颈：Bank Conflict

```
┌─────────────────────────────────────────────────────┐
│  Bank Conflict 示意图                                │
├─────────────────────────────────────────────────────┤
│  线程 0: 读取 tile[0][0] → Bank 0                   │
│  线程 1: 读取 tile[1][0] → Bank 0  (冲突!)          │
│  线程 2: 读取 tile[2][0] → Bank 0  (冲突!)          │
│  ...                                                │
│  线程 31: 读取 tile[31][0] → Bank 0 (冲突!)         │
│                                                     │
│  结果: 32 路 Bank Conflict，硬件串行化访问          │
└─────────────────────────────────────────────────────┘
```

---

## Step 3: Padding 消除 Bank Conflict

### 解决方案

声明共享内存时，**列宽 +1**：

```cpp
#define TILE_DIM 32

__global__ void transpose_shared_pad(const float *A, float *B, int N) {
    // ✅ 关键修改：TILE_DIM + 1
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();

    int bx = blockIdx.y * TILE_DIM + threadIdx.x;
    int by = blockIdx.x * TILE_DIM + threadIdx.y;

    if (bx < N && by < N) {
        B[by * N + bx] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 为什么 +1 有效？

| 配置 | 列宽 | 线程访问索引 | 对 32 取模 | Bank 分布 |
|------|------|--------------|-----------|------------|
| 原始 | 32 | 0, 32, 64... | 0, 0, 0... | 全部 Bank 0 ❌ |
| Padding | 33 | 0, 33, 66, 99... | 0, 1, 2, 3... | 分散到各 Bank ✅ |

> **效果**：简单修改带来 **30%+** 性能提升！

---

## Step 4: ILP 与 Block 形状优化

### 问题

RTX 5090 上 $32 \" 32$ 线程块不足以掩盖全局内存延迟。

### 方案：提高指令级并行 (ILP)

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| 数据块 | $32 \" 32$ | $32 \" 32$ (不变) |
| 线程块 | $32 \" 32$ = 1024 线程 | $32 \" 8$ = 256 线程 |
| 每线程处理 | 1 元素 | 4 元素 (循环) |

```cpp
#define TILE_DIM    32
#define BLOCK_ROWS   8  // 32 / 8 = 4，每线程处理 4 个元素

__global__ void transpose_optimized(const float *A, float *B, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 每个线程读取 4 个元素
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * N + x];
        }
    }
    __syncthreads();

    // 2. 互换 Block 坐标
    int bx = blockIdx.y * TILE_DIM + threadIdx.x;
    int by = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. 每个线程写入 4 个元素
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (bx < N && (by + j) < N) {
            B[(by + j) * N + bx] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

### Host 启动配置

```cpp
dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM, 1);
dim3 block(TILE_DIM, BLOCK_ROWS, 1);  // 32x8 = 256 线程
transpose_optimized<<<grid, block>>>(d_A, d_B, N);
```

### 为什么更快？

| 优化点 | 效果 |
|--------|------|
| **Memory-level Parallelism** | 线程在等待时可发出下一数据读取指令 |
| **更高 Occupancy** | 256 线程/Block → 更多 Active Blocks/SM |
| **延迟隐藏** | 更好地掩盖全局内存延迟 |

---

## 终极挑战: 向量化访问

### 目标

达到理论带宽的 **90%+**。

### 方案：float4 向量化

- 每次内存指令搬运 **16 字节** (4 个 float)
- 减少 Memory Transactions 数量
- 最大化总线利用率

> ⚠️ **警告**：涉及复杂的索引计算重构，建议：
> 1. 先熟练掌握 Step 4
> 2. 使用 NVIDIA Nsight Compute (`ncu`) 分析
> 3. 再尝试实现

---

## 总结与建议

### 优化路径回顾

```
┌─────────────────────────────────────────────────────────────┐
│  性能演进路径                                                │
├─────────────────────────────────────────────────────────────┤
│  Step 1: 朴素实现        ──►  非合并写入 (10~20% 带宽)     │
│       ↓                                                      │
│  Step 2: 共享内存        ──►  合并读写，Bank Conflict       │
│       ↓                                                      │
│  Step 3: Padding         ──►  消除 Bank Conflict (+30%)   │
│       ↓                                                      │
│  Step 4: ILP 优化        ──►  隐藏延迟，提高并行度          │
│       ↓                                                      │
│  终极: float4 向量化     ──►  90%+ 理论带宽                 │
└─────────────────────────────────────────────────────────────┘
```

### RTX 5090 测试建议

| 建议 | 说明 |
|------|------|
| **矩阵尺寸** | $N \" 8192$ 或更大，避免 Launch Overhead 成为瓶颈 |
| **分析工具** | `ncu --set full ./program` |
| **关键指标** | `gld_throughput` (加载吞吐量), `gst_throughput` (存储吞吐量) |
| **检查项** | Bank Conflicts 数量 |

---

**祝你榨干 RTX 5090 的每一字节带宽！** 🚀

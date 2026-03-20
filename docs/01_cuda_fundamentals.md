# 第一部分：CUDA 编程基础

> **学习目标**：理解 CUDA 线程层级结构和内存体系，为后续 GEMM 优化奠定基础

---

## 目录

1. [CUDA 线程层级](#1-cuda-线程层级)
2. [GPU 内存体系](#2-gpu-内存体系)
3. [线程同步机制](#3-线程同步机制)
4. [实战示例](#4-实战示例)

---

## 1. CUDA 线程层级

### 1.1 四层结构概览

CUDA 采用四级线程层级，从大到小依次为：**Grid → Block → Warp → Thread**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Grid (整个 GPU)                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────────┐ │
│  │   Block (0, 0)      │  │   Block (1, 0)      │  │   Block (2, 0)    │ │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌─────────────┐  │ │
│  │  │  Warp 0       │  │  │  │  Warp 0       │  │  │  │  Warp 0     │  │ │
│  │  │  (线程0-31)    │  │  │  │  (线程0-31)    │  │  │  │ (线程0-31)   │  │ │
│  │  ├───────────────┤  │  │  ├───────────────┤  │  │  ├─────────────┤  │ │
│  │  │  Warp 1       │  │  │  │  Warp 1       │  │  │  │  Warp 1     │  │ │
│  │  │  (线程32-63)   │  │  │  │  (线程32-63)   │  │  │  │(线程32-63)  │  │ │
│  │  └───────────────┘  │  │  └───────────────┘  │  │  └─────────────┘  │ │
│  └─────────────────────┘  └─────────────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

线程层级关系：
- 1 个 Grid 包含多个 Block
- 1 个 Block 包含多个 Warp（最多 32 个，即 1024 线程）
- 1 个 Warp 包含 32 个 Thread（固定）
```

### 1.2 各层级详解

#### Grid（网格）

| 属性 | 说明 |
|:---|:---|
| **定义** | 一次 Kernel 启动的所有线程集合 |
| **维度** | 1D、2D 或 3D（通过 `dim3 gridDim` 定义） |
| **范围** | 可跨多个 SM（Streaming Multiprocessor） |
| **关键变量** | `gridDim.x/y/z`, `blockIdx.x/y/z` |

```cuda
// 示例：2D Grid 配置
dim3 grid((N + 31) / 32, (M + 31) / 32);
// grid.x = 列方向的 Block 数
// grid.y = 行方向的 Block 数
```

#### Block（线程块）

| 属性 | 说明 |
|:---|:---|
| **定义** | 一组线程的集合，调度到单个 SM 执行 |
| **最大线程数** | 1024 线程/Block |
| **关键特性** | 同 Block 内线程可同步、可共享内存 |
| **关键变量** | `blockDim.x/y/z`, `threadIdx.x/y/z` |

```cuda
// 示例：2D Block 配置
dim3 block(32, 32);  // 32×32 = 1024 线程
// blockDim.x = 32, blockDim.y = 32
// threadIdx.x = 0~31, threadIdx.y = 0~31
```

#### Warp（线程束）

| 属性 | 说明 |
|:---|:---|
| **定义** | 32 个线程组成的执行单元，GPU 调度的最小单位 |
| **执行模式** | SIMT（Single Instruction Multiple Threads） |
| **关键概念** | Warp 内线程执行相同指令，分支会导致 Divergence |
| **Warp ID** | `threadIdx.x / 32` |
| **Lane ID** | `threadIdx.x % 32` |

**Warp Divergence（线程分歧）**：

```cuda
// 坏的例子：Warp 内线程走不同分支
if (threadIdx.x % 2 == 0) {
    // 线程 0, 2, 4... 执行
} else {
    // 线程 1, 3, 5... 执行
}
// 结果：Warp 被拆分为两部分串行执行，效率减半

// 好的例子：Warp 级别对齐分支
if ((threadIdx.x / 32) % 2 == 0) {
    // 整个 Warp 0, 2, 4... 执行
} else {
    // 整个 Warp 1, 3, 5... 执行
}
// 结果：无 Divergence，完全并行
```

### 1.3 线程索引计算

#### 全局线程 ID 计算

```cuda
__global__ void kernel() {
    // 1. 获取 Block 在 Grid 中的索引（2D）
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    
    // 2. 获取 Thread 在 Block 中的索引（2D）
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    
    // 3. 计算全局线程 ID
    int globalId = blockId * (blockDim.x * blockDim.y) + threadId;
    
    // 4. Warp 级别索引
    int warpId = threadId / 32;        // Warp 编号
    int laneId = threadId % 32;        // Warp 内的线程编号
}
```

#### 矩阵坐标映射示例

```cuda
// 矩阵 C 的坐标映射（M×N 矩阵）
__global__ void matrix_kernel(int M, int N) {
    // 列索引（x 方向）
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 行索引（y 方向）
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查
    if (row < M && col < N) {
        // 处理 C[row][col]
        int idx = row * N + col;  // 行主序索引
    }
}
```

---

## 2. GPU 内存体系

### 2.1 内存层级金字塔

GPU 内存按照速度和容量分为多个层级：

```
速度（快 → 慢）    容量（小 → 大）
    │                 │
    ▼                 ▼
┌─────────┐       ┌─────────┐
│ 寄存器   │       │  ~256 B │ 每线程
│Register │       │  per    │
│  ~1 TB/s │       │ thread  │
└────┬────┘       └────┬────┘
     │                 │
┌────▼────┐       ┌────▼────┐
│ 共享内存 │       │  ~48 KB │ 每 Block
│ Shared  │       │  per    │
│ ~10 TB/s│       │ block   │
└────┬────┘       └────┬────┘
     │                 │
┌────▼────┐       ┌────▼────┐
│ L1/L2   │       │  72 MB  │ L2 缓存
│ 缓存     │       │ (总计)  │
└────┬────┘       └────┬────┘
     │                 │
┌────▼────┐       ┌────▼────┐
│ 全局内存 │       │  32 GB  │ 显存
│ Global  │       │ (GDDR7) │
│ 1.8 TB/s│       │         │
└─────────┘       └─────────┘
```

### 2.2 各类内存详解

#### 寄存器（Register）

| 属性 | 说明 |
|:---|:---|
| **速度** | 最快（~1 cycle 延迟） |
| **容量** | 每线程最多 255 个 32-bit 寄存器 |
| **作用域** | 线程私有 |
| **典型用途** | 局部变量、累加器、临时缓存 |

```cuda
__global__ void kernel() {
    float tmp = 0.0f;           // 存储在寄存器
    float array[8];             // 如果大小合适，也存寄存器
    
    // 大量寄存器使用示例（GEMM 累加器）
    float accum[8][8];          // 64 个寄存器变量
}
```

**寄存器溢出（Register Spilling）警告**：

```cuda
// 危险：数组太大会溢出到本地内存（L1/L2）
float large[256];  // 可能导致 spilling，性能急剧下降

// 安全：控制数组大小
float small[64];   // 通常安全
```

#### 共享内存（Shared Memory）

| 属性 | RTX 5090 规格 |
|:---|:---|
| **物理位置** | SM 内部 SRAM |
| **容量** | 164 KB/SM（可配置） |
| **带宽** | ~10 TB/SM |
| **延迟** | ~20-30 cycles |
| **作用域** | Block 内所有线程共享 |
| **生命周期** | Block 执行期间 |

**共享内存分配方式**：

```cuda
// 方式 1：静态分配
__shared__ float sA[128][8];   // 编译期确定大小

// 方式 2：动态分配
extern __shared__ float smem[];  // 运行时指定大小
// 启动时：kernel<<<grid, block, smem_size>>>(...)
```

**共享内存 Bank 结构**：

```
共享内存分为 32 个 Bank（对应一个 Warp）

Bank 0  Bank 1  Bank 2  ...  Bank 31
  │       │       │             │
  ▼       ▼       ▼             ▼
┌────┐  ┌────┐  ┌────┐       ┌────┐
│0x00│  │0x04│  │0x08│  ...  │0x7C│  ← 地址映射
├────┤  ├────┤  ├────┤       ├────┤
│0x80│  │0x84│  │0x88│  ...  │0xFC│  ← 循环
└────┘  └────┘  └────┘       └────┘

地址到 Bank 映射：Bank = (地址 / 4) % 32
```

#### 全局内存（Global Memory）

| 属性 | RTX 5090 规格 |
|:---|:---|
| **物理位置** | GPU 芯片外（GDDR7） |
| **容量** | 32 GB |
| **带宽** | 1,792 GB/s |
| **延迟** | ~300-500 cycles |
| **作用域** | 整个 Grid 可见 |
| **生命周期** | 程序运行期间 |

**关键优化：Memory Coalescing（合并访问）**

```cuda
// 好的例子：连续线程访问连续地址（合并访问）
int tid = threadIdx.x;
float val = A[tid];        // 线程 0 访问 A[0]，线程 1 访问 A[1]...
// Warp 内 32 个线程的访问合并为 1-2 个内存事务

// 坏的例子：跨步访问（非合并）
float val = A[tid * 32];   // 线程间隔 32 个元素
// 需要 32 个独立内存事务，带宽利用率极低
```

### 2.3 内存访问模式对比

| 访问模式 | 图示 | 性能 | 适用场景 |
|:---|:---|:---:|:---|
| **连续访问** | Thread i → A[i] | 最佳 | 行优先矩阵遍历 |
| **跨步访问** | Thread i → A[i×stride] | 差 | 列优先矩阵遍历 |
| **随机访问** | Thread i → A[random] | 最差 | 哈希表、图遍历 |
| **广播访问** | 所有 Thread → A[0] | 好 | 读取常量 |

---

## 3. 线程同步机制

### 3.1 同步类型

| 同步方式 | 作用范围 | API | 用途 |
|:---|:---|:---|:---|
| **Block 内同步** | 单个 Block 内所有线程 | `__syncthreads()` | 共享内存数据交换 |
| **Warp 内同步** | 单个 Warp 内线程 | `__syncwarp()` | Warp Shuffle 操作 |
| **全局同步** | 整个 Grid | 不支持直接同步 | 需多次 Kernel 启动 |
| **内存屏障** | 特定内存空间 | `__threadfence()` | 全局内存可见性 |

### 3.2 __syncthreads() 详解

**使用场景**：

```cuda
__global__ void shared_mem_kernel(float *A, float *C) {
    __shared__ float sA[128];
    
    // 阶段 1：协作加载数据到共享内存
    sA[threadIdx.x] = A[blockIdx.x * 128 + threadIdx.x];
    
    // 同步点 1：确保所有线程完成加载
    __syncthreads();
    
    // 阶段 2：使用共享内存计算
    // 此时可以安全读取其他线程加载的数据
    float sum = 0;
    for (int i = 0; i < 128; ++i) {
        sum += sA[i];  // 所有线程都完成加载后执行
    }
    
    // 同步点 2：防止覆盖下一轮数据
    __syncthreads();
    
    // 阶段 3：加载下一轮数据
    // ...
}
```

**注意事项**：

```cuda
// 危险：条件分支中的 __syncthreads()
if (threadIdx.x < 16) {
    __syncthreads();  // ❌ 错误！只有部分线程执行同步
}

// 正确：所有线程都执行同步
__syncthreads();      // ✓ 正确
if (threadIdx.x < 16) {
    // 业务逻辑
}
```

---

## 4. 实战示例

### 4.1 向量加法（入门示例）

```cuda
// 向量加法：C[i] = A[i] + B[i]
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 启动配置
void run_vectorAdd(const float *A, const float *B, float *C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}
```

### 4.2 矩阵转置（内存访问模式示例）

```cuda
// 矩阵转置：使用共享内存优化非合并访问
__global__ void matrixTranspose(float *out, const float *in, int M, int N) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // 读取：行优先（合并访问）
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    
    __syncthreads();
    
    // 计算转置后的坐标
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // 写入：行优先，但逻辑上是转置
    if (x < M && y < N) {
        out[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 4.3 GEMM 线程映射（预告）

```cuda
// GEMM 基础线程映射
__global__ void gemm_mapping(int M, int N, int K) {
    // 每个线程负责 C 的一个元素
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N 维度
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M 维度
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

---

## 5. 关键概念总结

### 5.1 线程层级速查表

| 概念 | 大小 | 同步方式 | 内存访问 | 硬件映射 |
|:---|:---:|:---|:---|:---|
| **Thread** | 1 | 无 | 寄存器、局部内存 | CUDA Core |
| **Warp** | 32 | 隐式同步 | Warp Shuffle | SM 执行单元 |
| **Block** | ≤1024 | `__syncthreads()` | 共享内存 | 单个 SM |
| **Grid** | 无限制 | 无直接同步 | 全局内存 | 跨多个 SM |

### 5.2 内存层级速查表

| 内存类型 | 作用域 | 生命周期 | 速度 | 容量 |
|:---|:---|:---|:---:|:---:|
| **Register** | 线程私有 | Kernel | 最快 | 255/线程 |
| **Shared** | Block | Kernel | 很快 | 164KB/SM |
| **L1/L2** | 所有线程 | 动态 | 中等 | 128B/4MB |
| **Global** | 所有线程 | 程序 | 慢 | 32GB |
| **Constant** | 所有线程 | 程序 | 快（缓存） | 64KB |

---

## 6. 课后练习

1. **索引计算练习**：
   - 给定 `grid(4, 4)`, `block(16, 16)`，计算 Thread(5, 3) in Block(1, 2) 的全局 ID

2. **内存访问分析**：
   - 分析 `A[row * N + col]` 和 `B[col * K + row]` 的访问模式差异

3. **同步点设计**：
   - 为什么 Shared Memory Kernel 需要两个 `__syncthreads()`？

---

*下一步学习：[02_hardware_architecture.md](02_hardware_architecture.md) - 深入了解 RTX 5090 硬件架构*

# CUDA 中 Grid、Block 和 Warp 的关系

## 结构层次图

```
┌─────────────────────────────────────────────────────────┐
│                    Grid (整个 GPU)                       │
│   ┌─────────────────┐  ┌─────────────────┐              │
│   │   Block (0,0)   │  │   Block (1,0)   │   ...        │
│   │  ┌───────────┐  │  │  ┌───────────┐  │              │
│   │  │ Warp 0    │  │  │  │ Warp 0    │  │              │
│   │  │ (线程0-31) │  │  │  │ (线程0-31) │  │              │
│   │  ├───────────┤  │  │  ├───────────┤  │              │
│   │  │ Warp 1    │  │  │  │ Warp 1    │  │              │
│   │  │(线程32-63)│  │  │  │(线程32-63)│  │              │
│   │  └───────────┘  │  │  └───────────┘  │              │
│   └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

## 1. SM (Streaming Multiprocessor) - 流式多处理器

### 1.1 什么是 SM？

**SM 是 GPU 的核心计算单元**，是实际执行线程的硬件模块。

```
GPU 架构层次：
┌────────────────────────────────────────┐
│              GPU 芯片                   │
│  ┌────────────────────────────────┐   │
│  │         SM 0 (流处理器)         │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐     │   │
│  │  │Warp0│ │Warp1│ │Warp2│ ... │   │
│  │  │32线程│ │32线程│ │32线程│     │   │
│  │  └─────┘ └─────┘ └─────┘     │   │
│  │  ┌─────────────────────┐     │   │
│  │  │  共享内存 (164KB)   │     │   │
│  │  │  寄存器文件 (256KB) │     │   │
│  │  └─────────────────────┘     │   │
│  └────────────────────────────────┘   │
│  ┌────────────────────────────────┐   │
│  │         SM 1                   │   │
│  └────────────────────────────────┘   │
│              ...                      │
│  ┌────────────────────────────────┐   │
│  │         SM 169                 │   │
│  └────────────────────────────────┘   │
└────────────────────────────────────────┘
         ↑ 以 RTX 5090 为例 (170 SM)
```

### 1.2 SM 的核心资源

| 资源 | RTX 5090 配置 | 作用 |
|------|--------------|------|
| **CUDA Core** | 128 个/SM | 执行 FP32/FP64 运算 |
| **Tensor Core** | 4 个/SM | 执行矩阵乘法加速 |
| **寄存器文件** | 256 KB/SM | 存储线程私有数据 |
| **共享内存** | 164 KB/SM | Block 内线程共享数据 |
| **Warp 调度器** | 4 个/SM | 调度 Warp 执行 |
| **L1 缓存** | 与共享内存共享 | 数据缓存 |

### 1.3 Block 与 SM 的关系

**关键概念**：
- **Block 被分配到 SM 上执行**：一个 Block 只能在一个 SM 上运行
- **一个 SM 可以同时执行多个 Block**：只要资源足够
- **Block 之间无法直接通信**：即使它们在同一 SM 上

```
SM 调度示例：

SM 0:
  ┌───────────────┐ ┌───────────────┐
  │  Block (0,0)  │ │  Block (1,0)  │
  │  1024 线程    │ │  1024 线程    │
  │  使用 8KB SMEM│ │  使用 8KB SMEM│
  │  使用 90KB Reg│ │  使用 90KB Reg│
  └───────────────┘ └───────────────┘
        ↑ 时间片轮转执行
        
SM 1:
  ┌───────────────┐
  │  Block (0,1)  │
  │  256 线程     │
  │  使用 48KB SMEM│
  └───────────────┘
```

### 1.4 Occupancy (占用率)

**定义**：实际运行的 Warp 数 / SM 支持的最大 Warp 数

```
计算公式：
Occupancy = 活跃 Warp 数 / 最大 Warp 数 (通常为 32-64)

影响因素：
- 寄存器使用：每线程使用的寄存器越多，Occupancy 越低
- 共享内存使用：每 Block 使用的共享内存越多，Occupancy 越低
- Block 大小：线程数太少无法填满 Warp
```

**示例计算**：
```cuda
// sgemm_register.cu 配置
// 每线程使用 90 个寄存器
// 每 Block 256 线程 = 8 Warps
// 每 SM 可同时运行 256KB / (90×32×4B) ≈ 22 Warps
// 每个 Block 8 Warps
// 因此每 SM 可同时运行 22/8 ≈ 2-3 个 Block
// Occupancy = 22/64 ≈ 34%
```

---

## 2. Grid（网格）

- **定义**：一个完整的 kernel 启动所对应的所有线程的总体
- **范围**：可以跨越多个 Streaming Multiprocessor (SM)
- **大小**：由 `dim3 gridDim` 定义，可以是 1D、2D 或 3D
- **与 SM 的关系**：Grid 中的所有 Block 被分发到多个 SM 上并行执行
- **示例**：
  ```cuda
  dim3 grid((N + 31) / 32, (M + 31) / 32);  // 2D Grid
  // Grid 中的 Block 会被 GPU 调度器自动分配到空闲的 SM 上
  ```

---

## 3. Block（线程块）

### 3.1 基本定义

- **定义**：一组线程的集合，是 GPU 调度的基本单位
- **特点**：
  - 同一块内的线程可以**同步**（`__syncthreads()`）
  - 同一块内的线程可以**共享共享内存**（Shared Memory）
  - 块与块之间**不能**直接通信或同步
- **大小**：最多 1024 个线程
- **示例**：
  ```cuda
  dim3 block(32, 32);  // 每个 block 32×32 = 1024 个线程
  ```

### 3.2 Block 与 SM 资源限制

| 资源 | 每 SM 总量 | 使用示例 | 最大 Block 数 |
|------|-----------|---------|--------------|
| **寄存器** | 256 KB | 90 线程 × 90 寄存器 × 4B = 32KB | 8 个 Block |
| **共享内存** | 164 KB | 每 Block 8 KB | 20 个 Block |
| **线程数** | 2048 | 每 Block 256 线程 | 8 个 Block |
| **Warp 数** | 32-64 | 每 Block 8 Warps | 4-8 个 Block |

**关键限制**：实际并发 Block 数受限于最先耗尽的资源。

---

## 4. Warp（线程束）

### 4.1 基本定义

- **定义**：32 个线程组成的一组，是 GPU 执行的最小单位
- **特点**：
  - 一个 Warp 内的 32 个线程执行**相同的指令**（SIMD）
  - 如果 Warp 内线程走不同分支，会导致**线程分歧**（divergence）
  - **Warp 调度器**负责调度 Warp 执行（每 SM 有 4 个调度器）

### 4.2 Warp 与 SM 的关系

```
SM 内部结构：

┌─────────────────────────────────────────┐
│                SM 0                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │Warp调度器0│ │Warp调度器1│ │Warp调度器2│ │
│  │Warp调度器3│ │         │ │         │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ │
│       │           │           │        │
│  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ │
│  │CUDA Core│ │CUDA Core│ │CUDA Core│ │
│  │TensorCor│ │TensorCor│ │TensorCor│ │
│  └─────────┘ └─────────┘ └─────────┘ │
│                                        │
│  每周期每个 Warp 调度器可以发射 1 条指令 │
│  4 个调度器每周期最多发射 4 条 Warp 指令 │
└─────────────────────────────────────────┘
```

**执行流程**：
1. Block 被分配到 SM
2. Block 内的 32 个线程组成一个或多个 Warp
3. Warp 调度器选择就绪的 Warp 执行
4. 遇到 `__syncthreads()` 时，Warp 可能被挂起等待

---

## 5. 完整层次结构总结

### 5.1 各层级对比

| 层级 | 大小 | 同步方式 | 内存访问 | 硬件映射 |
|------|------|----------|----------|----------|
| **Warp** | 32 线程 | 隐式同步 | 寄存器 | SM 内执行单元 |
| **Block** | 最多 1024 线程 | `__syncthreads()` | 共享内存 | 分配到单个 SM |
| **Grid** | 无限（受硬件限制） | 无法直接同步 | 全局内存 | 跨多个 SM |
| **SM** | 32-64 Warp | Warp 调度器 | L1/共享内存 | GPU 核心单元 |

### 5.2 完整的调用层次

```
Kernel 启动流程：

1. 主机端调用 kernel
   kernel<<<grid, block>>>(...)
              ↓
2. Grid 创建
   - 创建 gridDim.x × gridDim.y 个 Block
              ↓
3. Block 分配到 SM
   - GPU 调度器将 Block 分配到空闲 SM
   - 每个 Block 独占一个 SM 的资源
              ↓
4. Block 内部分解为 Warp
   - 每 32 线程组成一个 Warp
   - Warp 被分配到 Warp 调度器
              ↓
5. Warp 执行
   - Warp 调度器选择就绪的 Warp
   - 发射指令到执行单元
   - 重复直到所有 Warp 完成
              ↓
6. Block 完成
   - 所有 Warp 完成后，Block 释放 SM 资源
   - 调度下一个 Block
```

### 5.3 关键硬件限制（以 RTX 5090 为例）

| 限制项 | 数值 | 影响 |
|--------|------|------|
| **SM 数量** | 170 | 最大并行度 |
| **每 SM 最大线程** | 2048 | 限制 Block 并发数 |
| **每 SM 最大寄存器** | 256 KB (65,536 个) | 限制每线程寄存器使用 |
| **每 SM 最大共享内存** | 164 KB | 限制每 Block 共享内存使用 |
| **每 Block 最大线程** | 1024 | 限制 Block 大小 |
| **每线程最大寄存器** | 255 | 限制局部变量数量 |
| **Warp 大小** | 32 线程 | 最小执行单位 |

### 5.4 性能优化的核心原则

1. **最大化 Occupancy**：平衡寄存器和共享内存使用
2. **减少 Bank Conflict**：设计良好的共享内存访问模式
3. **避免 Warp Divergence**：确保 Warp 内线程走相同分支
4. **内存访问合并**：确保连续的线程访问连续的内存地址
5. **利用 Tensor Core**：对于矩阵乘法，使用专用硬件加速

---

## 6. 在代码中的体现

```cuda
__global__ void kernel() {
    // Grid 级别
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    // Block 级别
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;

    // 全局线程 ID
    int globalId = blockId * (blockDim.x * blockDim.y) + threadId;

    // Warp 隐式：threadId / 32 即为 warp 编号
    int warpId = threadId / 32;
}
```

---

## 6. 在代码中的体现

### 6.1 获取硬件信息的代码

```cuda
// 获取当前 GPU 的 SM 数量和计算能力
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("SM 数量: %d\\n", prop.multiProcessorCount);
printf("计算能力: %d.%d\\n", prop.major, prop.minor);
printf("每 SM 最大共享内存: %d KB\\n", prop.sharedMemPerMultiprocessor / 1024);
printf("每 SM 最大寄存器: %d KB\\n", prop.regsPerMultiprocessor * 4 / 1024);

// RTX 5090 输出示例：
// SM 数量: 170
// 计算能力: 10.0
// 每 SM 最大共享内存: 164 KB
// 每 SM 最大寄存器: 256 KB
```

### 6.2 完整的线程层次访问示例

```cuda
__global__ void kernel() {
    // ========== SM 级别 ==========
    // 无法直接获取 SM ID，但可以通过 block 间接推断
    int smId;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smId));  // PTX 内联汇编
    
    // ========== Grid 级别 ==========
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int totalBlocks = gridDim.x * gridDim.y;
    
    // ========== Block 级别 ==========
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int totalThreads = blockDim.x * blockDim.y;
    
    // ========== Warp 级别 ==========
    int warpId = threadId / 32;           // Warp 编号
    int laneId = threadId % 32;           // Warp 内线程编号 (0-31)
    
    // ========== 全局计算 ==========
    int globalThreadId = blockId * totalThreads + threadId;
    
    // 示例：打印线程信息
    if (threadId == 0 && blockId == 0) {
        printf("SM %d, Block (%d,%d), Warp %d, 总线程数: %d\\n",
               smId, blockIdx.x, blockIdx.y, warpId, totalThreads);
    }
}
```

### 6.3 GEMM 中的应用示例

以朴素 GEMM 为例（sgemm_naive.cu）：

```cuda
__global__ void sgemm_naive_kernel(int M, int N, int K, ...) {
    // 每个线程计算 C 的一个元素
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 全局列号
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 全局行号
    
    if (x < N && y < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[y * K + k] * B[k * N + x];
        }
        C[y * N + x] = alpha * sum + beta * C[y * N + x];
    }
}

void run_sgemm_naive(int M, int N, int K, ...) {
    dim3 block(32, 32);  // Block: 32×32 = 1024 线程
    // Grid: 根据矩阵大小计算
    dim3 grid((N + block.x - 1) / block.x, 
              (M + block.y - 1) / block.y);
    sgemm_naive_kernel<<<grid, block>>>(...);
}
```

**执行分析**：
- 假设 M=N=4096，则 grid = (128, 128) = 16,384 个 Block
- 这些 Block 会被分配到 170 个 SM 上并行执行
- 每个 SM 同时运行多个 Block（取决于资源限制）
- 总线程数：16,384 × 1,024 = 16,777,216 线程
- 需要 16,777,216 / 32 = 524,288 个 Warp

### 6.4 Occupancy 计算示例

```cuda
// 计算当前 kernel 的 Occupancy
void calculateOccupancy() {
    int blockSize = 256;  // 每 Block 256 线程
    int regsPerThread = 90;  // 每线程 90 个寄存器
    int sharedMemPerBlock = 8192;  // 每 Block 8 KB 共享内存
    
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, sgemm_register_kernel, blockSize, sharedMemPerBlock);
    
    printf("每 SM 最大活跃 Block 数: %d\\n", numBlocks);
    printf("Occupancy: %.1f%%\\n", 
           (numBlocks * blockSize / 2048.0) * 100);
}
```

---

## 7. 总结

CUDA 的线程层次结构是一个从硬件到软件的完整映射：

| 概念 | 硬件对应 | 软件接口 | 关键限制 |
|------|---------|---------|---------|
| **SM** | GPU 核心计算单元 | 无直接 API | 170 个 (RTX 5090) |
| **Grid** | 跨 SM 的线程集合 | `blockIdx`, `gridDim` | 无硬性限制 |
| **Block** | SM 内执行的线程组 | `threadIdx`, `blockDim` | 1024 线程/Block |
| **Warp** | SM 内执行的单位 | `threadIdx/32` | 32 线程/Warp |

**优化的核心**：理解这些层次关系，合理分配资源（寄存器、共享内存），最大化硬件利用率（Occupancy），最终实现高性能 GPU 程序。

---

*文档更新时间：2026年3月17日*

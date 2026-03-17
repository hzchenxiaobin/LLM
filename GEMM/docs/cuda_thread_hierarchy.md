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

## 1. Grid（网格）

- **定义**：一个完整的 kernel 启动所对应的所有线程的总体
- **范围**：可以跨越多个 Streaming Multiprocessor (SM)
- **大小**：由 `dim3 gridDim` 定义，可以是 1D、2D 或 3D
- **示例**：
  ```cuda
  dim3 grid((N + 31) / 32, (M + 31) / 32);  // 2D Grid
  ```

---

## 2. Block（线程块）

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

---

## 3. Warp（线程束）

- **定义**：32 个线程组成的一组，是 GPU 执行的最小单位
- **特点**：
  - 一个 Warp 内的 32 个线程执行**相同的指令**（SIMD）
  - 如果 Warp 内线程走不同分支，会导致**线程分歧**（divergence）
  - Warp 调度器负责调度 Warp 执行

---

## 三者关系总结

| 层级 | 大小 | 同步方式 | 内存访问 |
|------|------|----------|----------|
| **Warp** | 32 线程 | 隐式同步 | 寄存器 |
| **Block** | 最多 1024 线程 | `__syncthreads()` | 共享内存 |
| **Grid** | 无限（受硬件限制） | 无法直接同步 | 全局内存 |

---

## 在代码中的体现

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

## 在 GEMM 中的应用示例

以朴素 GEMM 为例（sgemm_naive.cu）：

```cuda
__global__ void sgemm_naive_kernel(int M, int N, int K, ...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局列号
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 计算全局行号
    // ...
}

void run_sgemm_naive(int M, int N, int K, ...) {
    dim3 block(32, 32);  // Block: 32×32 线程
    dim3 grid((N + block.x - 1) / block.x, 
              (M + block.y - 1) / block.y);  // Grid 大小根据矩阵计算
    sgemm_naive_kernel<<<grid, block>>>(...);
}
```

这种结构让每个线程负责计算输出矩阵 C 中的一个元素，实现大规模并行。

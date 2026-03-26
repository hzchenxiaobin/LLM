# LLM - CUDA 高性能计算学习仓库

本仓库用于系统学习大规模语言模型 (LLM) 相关的底层高性能计算优化技术，涵盖从基础算子（GEMM、Softmax、Reduction、Scan、Transpose、TopK）到复杂算子（FlashAttention）的完整优化路径，深入理解 GPU 并行计算的核心原理。

## 仓库结构

```
LLM/
├── README.md              # 本文件
├── .gitignore
├── GEMM/                  # CUDA SGEMM 优化教程 (7个版本)
│   ├── README.md          # GEMM 详细教程文档
│   ├── Makefile
│   ├── src/               # 源代码目录
│   │   └── gemm/          # GEMM 实现目录
│   │       ├── sgemm_naive.cu              # V0: 朴素实现
│   │       ├── sgemm_shared.cu             # V1: 共享内存分块
│   │       ├── sgemm_register.cu           # V2: 寄存器分块
│   │       ├── sgemm_register_vectorized.cu    # V3: 向量化加载
│   │       ├── sgemm_register_bank_conflict.cu # V4: Bank Conflict消除
│   │       ├── sgemm_register_vec_bank.cu      # V5: 向量化+双缓冲
│   │       └── sgemm_cublas.cu             # V6: cuBLAS参考
│   └── docs/              # 教程文档
│
├── scan/                  # CUDA Scan (前缀和) 优化教程 (4个版本)
│   ├── src/scan/
│   │   ├── v1_hillis_steele.cu     # V1: Hillis-Steele朴素扫描
│   │   ├── v2_blelloch.cu          # V2: Blelloch工作高效扫描
│   │   ├── v3_bank_free.cu         # V3: 消除Bank Conflict
│   │   └── v4_warp_primitive.cu    # V4: Warp Primitives优化
│   └── docs/              # 详细教程文档
│
├── reduction/             # CUDA Reduction 归约优化教程 (6个版本)
│   └── kernels/reduce.cu  # V1-V6递进优化实现
│
├── transpose/             # CUDA Transpose 转置优化教程 (5个版本)
│   └── src/transpose/
│       ├── v1_naive.cu          # V1: 朴素实现
│       ├── v2_shared_memory.cu  # V2: 共享内存分块
│       ├── v3_shared_pad.cu     # V3: Padding消除Bank Conflict
│       ├── v4_optimized.cu      # V4: ILP优化
│       └── v5_vectorized.cu     # V5: float4向量化
│
├── softmax/                 # CUDA Softmax 优化教程 (5个版本)
│   └── src/softmax/
│       ├── v1_naive.cu          # V1: 3个独立Kernel
│       ├── v2_shared_memory.cu  # V2: Kernel融合+Shared Memory
│       ├── v3_warp_reduction.cu # V3: Warp级归约
│       ├── v4_vectorized.cu     # V4: 向量化访存
│       └── v5_online.cu         # V5: Online Softmax算法
│
├── topk/                    # CUDA TopK 优化教程 (4个版本)
│   ├── src/
│   │   ├── topk_v1_thread_per_row.cu      # V1: Thread-per-Row
│   │   ├── topk_v2_block_shared_memory.cu  # V2: Block-per-Row
│   │   ├── topk_v3_warp_shuffle.cu         # V3: Warp Shuffle
│   │   └── topk_v4_radix_select.cu         # V4: Radix Select
│   └── docs/
│       ├── TUTORIAL.md                     # 完整教程
│       ├── topk_v1_visualization.html      # V1可视化
│       ├── topk_v2_visualization.html      # V2可视化
│       ├── topk_v3_visualization.html      # V3可视化
│       └── topk_v4_visualization.html      # V4可视化
│
├── flashattention/          # FlashAttention 优化教程 (6个版本)
│   └── src/flashattention/
│       ├── v1_naive.cu          # V1: Online Softmax算法
│       ├── v2_shared_kv.cu      # V2: Shared Memory KV分块
│       ├── v3_q_tiling.cu       # V3: 双缓冲+Q分块
│       ├── v4_vectorized.cu     # V4: 向量化+Bank-Free
│       ├── v5_fa2.cu            # V5: FlashAttention-2风格
│       └── v6_tensor_core.cu    # V6: Tensor Core优化
│
├── docs/                  # 【体系化】GEMM教程（推荐）
│   ├── 01_cuda_fundamentals.md
│   ├── 02_hardware_architecture.md
│   ├── 03_gemm_naive.md
│   ├── 04_shared_memory_tiling.md
│   ├── 05_register_tiling.md
│   ├── 06_vectorization_and_bank_conflict.md
│   ├── 07_tensor_cores.md
│   ├── 08_performance_analysis.md
│   ├── 09_profiling_tools.md
│   └── 10_batched_gemm.md
│
└── batch_gemm/            # 批量矩阵乘法实现
    └── docs/
```

## 学习目标

本仓库涵盖 LLM 核心算子的完整优化路径，从朴素实现到生产级性能。

### 1. GEMM 矩阵乘法 (7个版本)

| 版本 | 核心技术 | 关键优化 | 性能 (RTX 4090) |
|------|----------|----------|----------------|
| V0 | Naive | 全局内存直接访问 | 7-8 TFLOPS (8%) |
| V1 | Shared Memory | 分块缓存到 Shared Memory | 9-10 TFLOPS (11%) |
| V2 | Register Tiling | 2D寄存器分块 + 外积累加 | 30-40 TFLOPS (36%) |
| V3 | Vectorized | float4 向量化加载 | 35-45 TFLOPS (42%) |
| V4 | Bank Conflict Free | Padding消除Bank Conflict | 35-45 TFLOPS (42%) |
| V5 | Vec+Bank+Double Buffer | 双缓冲隐藏延迟 | 40-50 TFLOPS (48%) |
| V6 | cuBLAS | NVIDIA官方实现 | 58-65 TFLOPS (70-79%) |

### 2. Softmax (5个版本)

| 版本 | 核心技术 | 优化要点 |
|------|----------|----------|
| V1 | 3-Separate Kernels | 朴素实现，6次全局内存访问/元素 |
| V2 | Kernel Fusion + Shared Memory | 融合为1个Kernel，Block级归约 |
| V3 | Warp-Level Reduction | Warp Shuffle替代Shared Memory，无同步开销 |
| V4 | Vectorized Memory Access | float4 (128-bit) 访存，4x带宽利用 |
| V5 | Online Softmax | 单遍算法：`m_new = max(m_old, x)`，FlashAttention基础 |

### 3. Reduction 归约 (6个版本)

| 版本 | 核心技术 | 解决的问题 | 关键改进 |
|------|----------|------------|----------|
| V1 | Interleaved Addressing | 基础树状归约 | Warp Divergence严重 |
| V2 | Strided Indexing | 连续线程访问连续数据 | 消除Warp Divergence |
| V3 | Sequential Addressing | 反向步长索引 | 消除Bank Conflict |
| V4 | First Add During Load | 加载时预计算 | 减少Block数量，提高指令吞吐 |
| V5 | Warp Shuffle | 寄存器级归约 | 消除最后阶段同步开销 |
| V6 | Vectorized + Grid-Stride | 极致带宽利用 | float4 + Warp Shuffle |

### 4. Scan 前缀和 (4个版本)

| 版本 | 算法 | 复杂度 | 关键优化 |
|------|------|--------|----------|
| V1 | Hillis-Steele | O(N log N) | 双缓冲技术 |
| V2 | Blelloch | O(N) | Up-sweep + Down-sweep树，工作高效 |
| V3 | Bank-Free Blelloch | O(N) | Padding: `offset = n >> 5` |
| V4 | Warp Primitives | O(N) | `__shfl_up_sync`寄存器级扫描 |

### 5. Transpose 矩阵转置 (5个版本)

| 版本 | 核心技术 | 解决的问题 | 性能 |
|------|----------|------------|------|
| V1 | Naive | 基准 | 10-20%带宽 |
| V2 | Shared Memory Tiling | 非合并写 | Bank Conflict出现 |
| V3 | Padding (TILE_DIM+1) | Bank Conflict | +30%性能 |
| V4 | ILP Optimization | 每个线程处理4元素 | 更好延迟隐藏 |
| V5 | float4 Vectorization | 128-bit内存事务 | 90%+理论带宽 |

### 6. TopK (4个版本)

| 版本 | 核心技术 | 适用K值 | 关键特性 |
|------|----------|---------|----------|
| V1 | Thread-per-Row + 插入排序 | K ≤ 32 | 简单，但寄存器易溢出 |
| V2 | Block-per-Row + Shared Memory | 通用 | 合并访存好，有同步开销 |
| V3 | Warp-per-Row + Warp Primitives | **K ≤ 32** | 无`__syncthreads()`，**生产首选** |
| V4 | Radix Select + 分层归约 | **K > 32** | Float键值转换，两阶段筛选，大K场景 |

### 7. FlashAttention (6个版本)

| 版本 | 核心创新 | 内存策略 | 并行策略 |
|------|----------|----------|----------|
| V1 | Online Softmax算法 | 无Shared Memory | 串行KV循环 |
| V2 | Shared Memory KV分块 | 2×Bc×d共享缓冲 | 协作分块加载 |
| V3 | 双缓冲 + Q分块 | 4×Bc×d缓冲 | 计算与加载重叠 |
| V4 | 向量化 + Bank-Free | Padding共享内存 | float4全局加载 |
| V5 | FlashAttention-2风格 | Warp级KV分割 | Warp级并行 |
| V6 | Tensor Core (WMMA) | FP16片段 | MMA加速 |

## 通用优化技术总结

### 内存优化技术栈

```
┌─────────────────────────────────────────────────────────┐
│ Level 1: 全局内存优化                                    │
├─────────────────────────────────────────────────────────┤
│ • 合并访问 (Coalesced Access) - 线程按顺序访问连续地址    │
│ • 向量化加载 (Vectorized) - float4/float2, 128-bit事务   │
│ • Grid-Stride Loop - 固定线程数处理任意规模数据          │
├─────────────────────────────────────────────────────────┤
│ Level 2: Shared Memory优化                              │
├─────────────────────────────────────────────────────────┤
│ • 分块缓存 (Tiling) - 减少全局内存访问次数               │
│ • Padding技术 - tile[TILE_DIM][TILE_DIM+1]消除Bank Conflict│
│ • 双缓冲 (Double Buffer) - 计算与数据加载重叠            │
├─────────────────────────────────────────────────────────┤
│ Level 3: 寄存器优化                                      │
├─────────────────────────────────────────────────────────┤
│ • 寄存器分块 - 2D分块(TM×TN)，外积累加                  │
│ • ILP优化 - 每个线程处理多个独立数据元素                 │
│ • 最大化寄存器利用率 - 减少spill到Local Memory          │
└─────────────────────────────────────────────────────────┘
```

### 同步优化技术栈

| 技术 | 替代方案 | 优势 | 适用场景 |
|------|----------|------|----------|
| **Warp Shuffle** | `__syncthreads()` | 寄存器级速度，无内存访问 | Warp内线程通信 |
| **`__shfl_down_sync`** | Shared Memory Reduce | 5轮比较找32线程最大值 | Warp级归约 |
| **`__shfl_up_sync`** | - | 前缀和计算 | Warp级Scan |
| **`__match_any_sync`** | 复杂条件判断 | 快速匹配相同值线程 | TopK等选举场景 |
| **`#pragma unroll`** | 运行时循环 | 消除分支，指令调度优化 | 已知边界循环 |

### 算法优化技术栈

| 技术 | 算子应用 | 数学原理 | 效果 |
|------|----------|----------|------|
| **Online Algorithm** | Softmax, FlashAttention | 单遍更新: `m_new = max(m_old, x)` | O(N)空间→O(1)空间 |
| **Work-Efficient** | Scan (Blelloch) | 树状归约: Up-sweep + Down-sweep | O(N log N)→O(N)工作量 |
| **Radix Select** | TopK | 位分析逐层筛选 | 避免全排序，O(N)选择 |
| **Kernel Fusion** | Softmax | 合并多个Kernel为1个 | 减少全局内存往返 |
| **Tiling + Accumulation** | GEMM | 外积: A_tile × B_tile | 最大化数据复用 |

## 快速开始

### 环境要求

- NVIDIA GPU (Compute Capability 7.0+, 推荐 8.0+)
- CUDA Toolkit 11.0+ (RTX 4090/5090 需要 12.0+)
- make

### 编译与运行各算子

```bash
# GEMM
cd GEMM && make && ./build/benchmark_gemm

# Scan
cd scan && make && ./build/benchmark

# Reduction
cd reduction && make && ./build/benchmark

# Transpose
cd transpose && make && ./build/benchmark

# Softmax
cd softmax && make && ./build/benchmark

# TopK
cd topk && make && ./build/topk_benchmark

# FlashAttention
cd flashattention && make && ./build/benchmark_flashattention
```

## 学习路径

### 阶段 1: CUDA 基础与硬件架构

| 文档 | 内容描述 |
|------|----------|
| `[docs/01_cuda_fundamentals.md](docs/01_cuda_fundamentals.md)` | CUDA线程层级与内存体系 |
| `[docs/02_hardware_architecture.md](docs/02_hardware_architecture.md)` | RTX 5090硬件架构详解 |

### 阶段 2: GEMM 矩阵乘法优化进阶


| 阶段  | 内容                 | 文档                                                                                    | 代码                             |
| --- | ------------------ | ------------------------------------------------------------------------------------- | ------------------------------ |
| 2.1 | 朴素实现               | `[03_gemm_naive.md](docs/03_gemm_naive.md)`                                           | `sgemm_naive.cu`               |
| 2.2 | 共享内存分块             | `[04_shared_memory_tiling.md](docs/04_shared_memory_tiling.md)`                       | `sgemm_shared.cu`              |
| 2.3 | 寄存器分块              | `[05_register_tiling.md](docs/05_register_tiling.md)`                                 | `sgemm_register.cu`            |
| 2.4 | 向量化与 Bank Conflict | `[06_vectorization_and_bank_conflict.md](docs/06_vectorization_and_bank_conflict.md)` | `sgemm_register_vec_bank.cu`   |
| 2.5 | Tensor Core 优化     | `[07_tensor_cores.md](docs/07_tensor_cores.md)`                                       | 参考代码                           |
| 2.6 | 性能分析               | `[08_performance_analysis.md](docs/08_performance_analysis.md)`                       | `scripts/generate_roofline.py` |
| 2.7 | 调试工具               | `[09_profiling_tools.md](docs/09_profiling_tools.md)`                                 | Nsight Compute                 |
| 2.8 | 批量 GEMM            | `[10_batched_gemm.md](docs/10_batched_gemm.md)`                                       | `batch_gemm/`                  |


### 阶段 3: 进阶算子 (Scan / Transpose / TopK)

**Scan 前缀和** - 参照 `scan/docs/`:


| 版本  | 技术要点               | 解决的问题            | 核心改进                    | 文档                                                                             |
| --- | ------------------ | ---------------- | ----------------------- | ------------------------------------------------------------------------------ |
| V1  | Hillis-Steele      | 朴素并行扫描           | 双缓冲技术，O(N log N) 工作量    | `[V1_HILLIS_STEELE_EXPLAINED.md](scan/docs/V1_HILLIS_STEELE_EXPLAINED.md)`     |
| V2  | Blelloch           | 工作高效扫描           | 树状归约，O(N) 工作量           | `[V2_BLELLOCH_EXPLAINED.md](scan/docs/V2_BLELLOCH_EXPLAINED.md)`               |
| V3  | Bank Conflict Free | 消除 Bank Conflict | Padding 技术，提升内存带宽       | `[V3_BANK_FREE_EXPLAINED.md](scan/docs/V3_BANK_FREE_EXPLAINED.md)`             |
| V4  | Warp Primitives    | 寄存器级扫描           | `__shfl_up_sync`，接近硬件极限 | `[V4_WARP_PRIMITIVES_EXPLAINED.md](scan/docs/V4_WARP_PRIMITIVES_EXPLAINED.md)` |


```bash
cd scan && make && ./build/benchmark
```

**Transpose 矩阵转置** - 参照 `transpose/docs/`:

| 版本 | 文档 | 代码 |
|------|------|------|
| V1-V5 总览 | `transpose/docs/TUTORIAL.md` | `transpose/src/transpose/` |
| V1 朴素 | `V1_NAIVE_EXPLAINED.md` | `v1_naive.cu` |
| V2 共享内存 | `V2_SHARED_MEMORY_EXPLAINED.md` | `v2_shared_memory.cu` |
| V3 Padding | `V3_SHARED_PAD_EXPLAINED.md` | `v3_shared_pad.cu` |
| V4 ILP | `V4_OPTIMIZED_EXPLAINED.md` | `v4_optimized.cu` |
| V5 向量化 | `V5_VECTORIZED_EXPLAINED.md` | `v5_vectorized.cu` |

```bash
cd transpose && make && ./build/benchmark
```

**TopK** - 参照 `topk/docs/`:

| 版本 | 文档 | 代码 | 适用场景 |
|------|------|------|----------|
| V1 | `topk_v1_thread_per_row_详解.md` | `topk_v1_thread_per_row.cu` | 学习/调试 |
| V2 | `topk_v2_block_shared_memory_详解.md` | `topk_v2_block_shared_memory.cu` | 通用 |
| V3 | `topk_v3_warp_shuffle_详解.md` + `.html` | `topk_v3_warp_shuffle.cu` | **K ≤ 32，生产首选** |
| V4 | `topk_v4_radix_select_详解.md` + `.html` | `topk_v4_radix_select.cu` | **K > 32，大K场景** |

```bash
cd topk && make && ./build/topk_benchmark
```

### 阶段 4: 复杂算子 (FlashAttention)

参照 `flashattention/docs/` 完成 6 步进阶优化：

| 版本 | 文档 | 代码 | 核心创新 |
|------|------|------|----------|
| V1 | `V1_NAIVE_EXPLAINED.md` | `v1_naive.cu` | Online Softmax算法，O(N) HBM访问 |
| V2 | `V2_SHARED_KV_EXPLAINED.md` | `v2_shared_kv.cu` | Shared Memory KV分块，2×Bc×d缓冲 |
| V3 | `V3_DOUBLE_BUFFER_EXPLAINED.md` | `v3_q_tiling.cu` | 双缓冲+Q分块，计算与加载重叠 |
| V4 | `V4_VECTORIZED_EXPLAINED.md` | `v4_vectorized.cu` | float4向量化+Bank-Free |
| V5 | `V5_FA2_EXPLAINED.md` | `v5_fa2.cu` | FlashAttention-2风格，Warp级KV并行 |
| V6 | `V6_TENSOR_CORE_EXPLAINED.md` | `v6_tensor_core.cu` | Tensor Core (WMMA)加速 |

```bash
cd flashattention && make && ./build/benchmark_flashattention
```

**Reduction 归约** - 参照 `reduction/docs/`:

| 版本 | 文档 | 代码 |
|------|------|------|
| V1-V6 总览 | `reduction/docs/reduce_v1_to_v6_explanation.md` | `reduction/kernels/reduce.cu` |
| V5 Warp Shuffle | `reduce_v5_warp_shuffle_explanation.md` | - |

```bash
cd reduction && make && ./build/benchmark
```

## 性能基准 (RTX 4090)

### GEMM 矩阵乘法 (理论峰值: 82.58 TFLOPS)

| 实现 | 性能 (TFLOPs) | 利用率 | 核心优化技术 |
|------|--------------|--------|-------------|
| cuBLAS | 58-65 | 70-79% | NVIDIA官方优化 |
| Register Vec+Bank (V5) | 40-50 | 48-61% | 双缓冲 + 向量化 + Bank-free |
| Register Vectorized (V3) | 35-45 | 42-55% | float4向量化加载 |
| Register Tiling (V2) | 30-40 | 36-48% | 寄存器分块 + 双层Tiling |
| Shared Memory (V1) | 9-10 | 11-12% | Shared Memory Tiling |
| Naive (V0) | 7-8 | 8-10% | 无优化 |

### Softmax (Batch=128, N=32000)

| 实现 | 带宽 (GB/s) | 相对速度 | 核心优化技术 |
|------|-------------|----------|-------------|
| V5 Online | ~450 | 4.5× | Online算法单遍完成 |
| V4 Vectorized | ~400 | 4.0× | float4向量化 |
| V3 Warp Reduce | ~350 | 3.5× | Warp Shuffle归约 |
| V2 Shared Memory | ~180 | 1.8× | Kernel融合 |
| V1 Naive | ~100 | 1.0× | 3个独立Kernel |

### Reduction 归约 (N=1M)

| 实现 | 带宽 (GB/s) | 相对速度 | 核心优化技术 |
|------|-------------|----------|-------------|
| V6 Vectorized | ~480 | 6.0× | float4 + Warp Shuffle |
| V5 Warp Shuffle | ~420 | 5.25× | 寄存器级最终归约 |
| V4 First Add Load | ~350 | 4.4× | 加载时预计算 |
| V3 Sequential | ~280 | 3.5× | 消除Bank Conflict |
| V2 Strided | ~180 | 2.25× | 消除Warp Divergence |
| V1 Interleaved | ~80 | 1.0× | 朴素实现 |

### Scan 前缀和 (N=1024)

| 实现 | 带宽 (GB/s) | 相对速度 | 核心优化技术 |
|------|-------------|----------|-------------|
| V4 Warp Primitives | ~204 | 3.0× | Warp Shuffle寄存器级扫描 |
| V3 Bank-Free | ~128 | 1.88× | Padding消除Bank Conflict |
| V2 Blelloch | ~85 | 1.25× | 工作高效树状归约 |
| V1 Hillis-Steele | ~68 | 1.0× | 双缓冲朴素扫描 |

### Transpose 转置 (4096×4096)

| 实现 | 带宽 (GB/s) | 利用率 | 核心优化技术 |
|------|-------------|--------|-------------|
| V5 Vectorized | ~520 | 90%+ | float4 + Padding |
| V4 ILP | ~450 | 78% | 每线程4元素 |
| V3 Padding | ~400 | 70% | TILE_DIM+1消除Conflict |
| V2 Shared | ~280 | 49% | 基础Shared Memory |
| V1 Naive | ~80 | 14% | 非合并访问 |

### TopK (Batch=128, N=32000, K=16)

| 实现 | 延迟 (ms) | 带宽 (GB/s) | 核心优化技术 |
|------|-----------|-------------|-------------|
| V3 Warp Shuffle | ~0.8 | ~380 | 无同步Warp级归约 |
| V4 Radix Select | ~1.0 | ~300 | 分层筛选，适合大K |
| V2 Block Shared | ~1.5 | ~200 | Block级归约 |
| V1 Thread-per-Row | ~5.0 | ~60 | 寄存器溢出严重 |

### FlashAttention (Batch=4, Heads=12, Seq=1024, Dim=64)

| 实现 | 性能 (TFLOPS) | HBM访问 | 核心优化技术 |
|------|--------------|---------|-------------|
| V6 Tensor Core | ~12 | O(N) | WMMA + Online Softmax |
| V5 FA2 Style | ~8 | O(N) | Warp级KV并行 |
| V4 Vectorized | ~5 | O(N) | float4 + Bank-Free |
| V3 Double Buffer | ~4 | O(N) | 双缓冲重叠 |
| V2 Shared KV | ~2.5 | O(N) | Shared Memory分块 |
| V1 Naive | ~1 | O(N²) | 无优化 |


## 核心概念速查

### 线程组织

```
Grid (整个问题空间)
  └── Block (协作线程组，最多 1024 线程)
        └── Warp (32 线程，调度基本单元)
              └── Thread (执行单元)
```

### 内存层次 (延迟从低到高)

```
寄存器 (~1 cycle) → 共享内存 (~20 cycles) → L2缓存 (~200 cycles) → 全局内存 (~400 cycles)
```

### 关键优化原则

1. **最大化并行性** - 足够的线程数隐藏延迟
2. **减少全局内存访问** - 使用共享内存缓存数据
3. **合并内存访问** - 线程按顺序访问连续地址
4. **避免 Bank Conflict** - 共享内存地址分布到不同 bank
5. **提高指令级并行** - 每个线程处理多个数据元素
6. **利用专用硬件** - Tensor Core 用于矩阵计算

## 文档索引

### GEMM 核心技术文档


| 文档                                                                                         | 内容描述                  | 推荐阶段 |
| ------------------------------------------------------------------------------------------ | --------------------- | ---- |
| `[docs/01_cuda_fundamentals.md](docs/01_cuda_fundamentals.md)`                             | CUDA 线程层级与内存体系        | 1    |
| `[docs/02_hardware_architecture.md](docs/02_hardware_architecture.md)`                     | RTX 5090 硬件架构详解       | 1    |
| `[docs/03_gemm_naive.md](docs/03_gemm_naive.md)`                                           | 朴素实现与瓶颈分析             | 2    |
| `[docs/04_shared_memory_tiling.md](docs/04_shared_memory_tiling.md)`                       | Shared Memory Tiling  | 3    |
| `[docs/05_register_tiling.md](docs/05_register_tiling.md)`                                 | 寄存器分块与外积计算            | 4    |
| `[docs/06_vectorization_and_bank_conflict.md](docs/06_vectorization_and_bank_conflict.md)` | 向量化与 Bank Conflict 消除 | 5    |
| `[docs/07_tensor_cores.md](docs/07_tensor_cores.md)`                                       | Tensor Core 优化        | 6    |
| `[docs/08_performance_analysis.md](docs/08_performance_analysis.md)`                       | Roofline 模型分析         | 4    |
| `[docs/09_profiling_tools.md](docs/09_profiling_tools.md)`                                 | Nsight Compute 实战     | 7    |
| `[docs/10_batched_gemm.md](docs/10_batched_gemm.md)`                                       | 批量矩阵乘法                | 扩展   |


### Scan (前缀和) 核心技术文档


| 文档                                                                                       | 内容描述                       | 推荐阶段 |
| ---------------------------------------------------------------------------------------- | -------------------------- | ---- |
| `[scan/docs/V1_HILLIS_STEELE_EXPLAINED.md](scan/docs/V1_HILLIS_STEELE_EXPLAINED.md)`     | Hillis-Steele 朴素扫描与双缓冲     | 1    |
| `[scan/docs/V2_BLELLOCH_EXPLAINED.md](scan/docs/V2_BLELLOCH_EXPLAINED.md)`               | Blelloch 工作高效扫描算法          | 2    |
| `[scan/docs/V3_BANK_FREE_EXPLAINED.md](scan/docs/V3_BANK_FREE_EXPLAINED.md)`             | Padding 技术消除 Bank Conflict | 3    |
| `[scan/docs/V4_WARP_PRIMITIVES_EXPLAINED.md](scan/docs/V4_WARP_PRIMITIVES_EXPLAINED.md)` | Warp Primitives 寄存器级优化     | 4    |
| `[scan/docs/TUTORIAL.md](scan/docs/TUTORIAL.md)`                                         | 算法原理概述与优化路径                | 总览   |


### Softmax 核心技术文档

| 文档 | 内容描述 | 阶段 |
|------|----------|------|
| `[softmax/docs/TUTORIAL.md](softmax/docs/TUTORIAL.md)` | 完整优化教程与5个版本详解 | 总览 |
| `[softmax/docs/softmax_v1_explanation.md](softmax/docs/softmax_v1_explanation.md)` | V1-V5 技术要点详解 | 详细 |

### Transpose 核心技术文档

| 文档 | 内容描述 | 阶段 |
|------|----------|------|
| `[transpose/docs/TUTORIAL.md](transpose/docs/TUTORIAL.md)` | 完整优化路径与5个版本详解 | 总览 |
| `[transpose/docs/V1_NAIVE_EXPLAINED.md](transpose/docs/V1_NAIVE_EXPLAINED.md)` | V1 朴素实现 | 1 |
| `[transpose/docs/V2_SHARED_MEMORY_EXPLAINED.md](transpose/docs/V2_SHARED_MEMORY_EXPLAINED.md)` | V2 共享内存分块 | 2 |
| `[transpose/docs/V3_SHARED_PAD_EXPLAINED.md](transpose/docs/V3_SHARED_PAD_EXPLAINED.md)` | V3 Padding优化 | 3 |
| `[transpose/docs/V4_OPTIMIZED_EXPLAINED.md](transpose/docs/V4_OPTIMIZED_EXPLAINED.md)` | V4 ILP优化 | 4 |
| `[transpose/docs/V5_VECTORIZED_EXPLAINED.md](transpose/docs/V5_VECTORIZED_EXPLAINED.md)` | V5 向量化优化 | 5 |

### TopK 核心技术文档

| 文档 | 内容描述 | 阶段 |
|------|----------|------|
| `[topk/docs/TUTORIAL.md](topk/docs/TUTORIAL.md)` | 完整优化路径与4个版本详解 | 总览 |
| `[topk/docs/topk_v1_thread_per_row_详解.md](topk/docs/topk_v1_thread_per_row_详解.md)` | V1 Thread-per-Row | 1 |
| `[topk/docs/topk_v2_block_shared_memory_详解.md](topk/docs/topk_v2_block_shared_memory_详解.md)` | V2 Block Shared Memory | 2 |
| `[topk/docs/topk_v3_warp_shuffle_详解.md](topk/docs/topk_v3_warp_shuffle_详解.md)` | V3 Warp Primitives | 3 |
| `[topk/docs/topk_v4_radix_select_详解.md](topk/docs/topk_v4_radix_select_详解.md)` | V4 Radix Select | 4 |
| `[topk/docs/topk_v1_visualization.html](topk/docs/topk_v1_visualization.html)` | V1 交互式可视化 | 可视化 |
| `[topk/docs/topk_v2_visualization.html](topk/docs/topk_v2_visualization.html)` | V2 交互式可视化 | 可视化 |
| `[topk/docs/topk_v3_visualization.html](topk/docs/topk_v3_visualization.html)` | V3 交互式可视化 | 可视化 |
| `[topk/docs/topk_v4_visualization.html](topk/docs/topk_v4_visualization.html)` | V4 交互式可视化 | 可视化 |

### FlashAttention 核心技术文档

| 文档 | 内容描述 | 阶段 |
|------|----------|------|
| `[flashattention/docs/V1_NAIVE_EXPLAINED.md](flashattention/docs/V1_NAIVE_EXPLAINED.md)` | V1 Online Softmax算法 | 1 |
| `[flashattention/docs/V2_SHARED_KV_EXPLAINED.md](flashattention/docs/V2_SHARED_KV_EXPLAINED.md)` | V2 Shared Memory KV分块 | 2 |
| `[flashattention/docs/V3_DOUBLE_BUFFER_EXPLAINED.md](flashattention/docs/V3_DOUBLE_BUFFER_EXPLAINED.md)` | V3 双缓冲优化 | 3 |
| `[flashattention/docs/V4_VECTORIZED_EXPLAINED.md](flashattention/docs/V4_VECTORIZED_EXPLAINED.md)` | V4 向量化优化 | 4 |
| `[flashattention/docs/V5_FA2_EXPLAINED.md](flashattention/docs/V5_FA2_EXPLAINED.md)` | V5 FlashAttention-2风格 | 5 |
| `[flashattention/docs/V6_TENSOR_CORE_EXPLAINED.md](flashattention/docs/V6_TENSOR_CORE_EXPLAINED.md)` | V6 Tensor Core优化 | 6 |

### Reduction 核心技术文档

| 文档 | 内容描述 | 阶段 |
|------|----------|------|
| `[reduction/docs/reduce_v1_to_v6_explanation.md](reduction/docs/reduce_v1_to_v6_explanation.md)` | V1-V6 完整优化路径详解 | 总览 |
| `[reduction/docs/reduce_v5_warp_shuffle_explanation.md](reduction/docs/reduce_v5_warp_shuffle_explanation.md)` | V5 Warp Shuffle详解 | 5 |

### 历史文档（已整合）

`GEMM/docs/` 目录下的旧文档内容已整合到新版教程，仅供参考。详见 `[GEMM/docs/README.md](GEMM/docs/README.md)`。

### 练习题与面试题


| 文档                                                        | 类型  | 说明              |
| --------------------------------------------------------- | --- | --------------- |
| `GEMM/exercises/occupancy_calculation_exercises.md`       | 计算题 | Occupancy 计算练习  |
| `GEMM/exercises/gemm_basic_interview_questions.md`        | 面试题 | 基础概念面试题         |
| `GEMM/exercises/gemm_optimization_interview_questions.md` | 面试题 | 优化技术面试题         |
| `GEMM/exercises/gemm_tensor_core_interview_questions.md`  | 面试题 | Tensor Core 面试题 |
| `GEMM/exercises/gemm_practice_problems.md`                | 编程题 | 编程实践题           |


## 参考资料

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS - NVIDIA 官方高效 GEMM 实现](https://github.com/NVIDIA/cutlass)
- [Roofline Model Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/roofline.pdf)
- [NVIDIA Ampere/Blackwell 架构白皮书](https://www.nvidia.com/en-us/data-center/ampere-architecture/)

## 贡献

欢迎提交 Issue 和 PR 来完善教程或添加新的算子实现。

可能的改进方向：

- 更多数据类型支持 (FP16/BF16/INT8)
- Nsight Compute profiling 分析
- 多 GPU 支持
- 更多 LLM 算子 (Attention, LayerNorm, etc.)
- 自动调优 (Auto-Tuning) 功能

## License

MIT License

---

**作者**: hzchenxiaobin  
**创建时间**: 2026年3月
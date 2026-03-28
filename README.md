# LLM CUDA 高性能计算学习仓库

本仓库面向 CUDA 开发者，逐步讲解和实现大规模语言模型（LLM）中的关键算子优化路径。覆盖从基础算子：GEMM、Scan、Reduction、Transpose、Softmax、TopK，到高级算子：FlashAttention 的多版本演进与性能优化。

## 目录结构概览

- `GEMM/`：SGEMM 优化教程，7 个版本，含代码、基准、文档
- `scan/`：前缀和 Scan 4 个版本，演进自 Hillis-Steele 到 Warp Primitive
- `reduction/`：归约算子 6 个优化版本，按访问模式和并行归约递进
- `transpose/`：矩阵转置 5 个版本，含 Shared Memory、Padding、ILP、Vectorize
- `softmax/`：Softmax 5 个版本，从多 Kernel 到 Online Softmax
- `topk/`：TopK 4 个版本，含排序、Warp Shuffle 与 Radix Select
- `flashattention/`：FlashAttention 6 个版本，含 Shared KV、双缓冲、Tensor Core
- `batch_gemm/`：批量 GEMM 实现与分析
- `docs/`：体系化教程，从 CUDA 基础到 GEMM、性能分析、剖析方法

## 快速开始

### 系统要求

- NVIDIA GPU（建议架构 Ampere 及以上，Compute Capability 7.0+）
- CUDA Toolkit 11.0+（推荐 12.0+ 以支持最新硬件）
- `make`
- C++ 编译器（支持 `nvcc`）

### 编译与运行示例

```bash
cd GEMM && make && ./build/benchmark_gemm
cd ../scan && make && ./build/benchmark
cd ../reduction && make && ./build/benchmark
cd ../transpose && make && ./build/benchmark
cd ../softmax && make && ./build/benchmark
cd ../topk && make && ./build/topk_benchmark
cd ../flashattention && make && ./build/benchmark_flashattention
```

## 各模块重点

### GEMM：矩阵乘法优化（7 版本）

- V0：朴素全局内存版本（`sgemm_naive.cu`）
- V1：Shared Memory 分块（`sgemm_shared.cu`）
- V2：寄存器分块（`sgemm_register.cu`）
- V3：向量化（`float4`）访问（`sgemm_register_vectorized.cu`）
- V4：Bank Conflict 消除（`sgemm_register_bank_conflict.cu`）
- V5：向量化 + Bank 消除（`sgemm_register_vec_bank.cu`）
- V6：cuBLAS 参考实现（`sgemm_cublas.cu`）

### Scan：前缀和算法（4 版本）

- V1：Hillis-Steele
- V2：Blelloch
- V3：Bank Free（Bank Conflict 减少）
- V4：Warp Primitive（`__shfl_up_sync`）

### Reduction：归约运算（6 版本）

- V1：Interleaved Addressing（`reduce_v1_interleaved.cu`）
- V2：Strided Indexing（`reduce_v2_strided.cu`）
- V3：Sequential Addressing（`reduce_v3_sequential.cu`）
- V4：加载时提前归约（`reduce_v4_first_add.cu`）
- V5：Warp Shuffle（寄存器归约）（`reduce_v5_warp_shuffle.cu`）
- V6：向量化 + Grid-Stride（`reduce_v6_vectorized.cu`）

### Transpose：矩阵转置（5 版本）

- V1：朴素
- V2：Shared Memory Tiling
- V3：Padding（`TILE_DIM+1`）
- V4：ILP（每线程多元素）
- V5：Vectorized `float4`（`sgemm_register_vec_bank.cu`）

### Softmax：归一化（5 版本）

- V1：三阶段 Kernel
- V2：Kernel 融合 + SharedMemory
- V3：Warp 级归约
- V4：向量化访问
- V5：Online Softmax（FlashAttention 核心）

### TopK：TopK 选择（4 版本）

- V1：线程按行
- V2：Block 共享内存
- V3：Warp Shuffle，无 `__syncthreads()`
- V4：Radix Select，适合大 K

### FlashAttention：自注意力（6 版本）

- V1：基础 Online Softmax
- V2：Shared KV 矩阵分块
- V3：Q 分块 + 双缓冲
- V4：向量化 + Bank-Free
- V5：FlashAttention-2 风格
- V6：Tensor Core WMMA 加速

## 学习路线推荐

1. `docs/01_cuda_fundamentals.md`：CUDA 线程和内存模型
2. `docs/02_hardware_architecture.md`：GPU 架构理解
3. `GEMM` 系列：算子优化全链路
4. `Scan`/`Reduction`/`Transpose`：内存与同步优化技巧
5. `Softmax`/`TopK`：LLM 核心算子
6. `FlashAttention`：高级 LLM 内核

## 贡献与问题

欢迎提 Issue、PR 共同完善：

- 优化算法实现
- 不同架构性能对比
- 添加更多可视化与性能分析

---

### 研发作者

`LLM` 仓库维护者

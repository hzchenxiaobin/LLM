# CUDA GEMM 算子性能优化教程 (面向 RTX 5090)

> **注意**：本目录下的详细文档已整合到新的体系化教程中。请参考以下学习路径。

---

## 新版体系化教程

已创建完整的结构化教程，位于 `/docs/` 目录：

```
/docs/
├── README.md                           # 教程入口和导航
├──
├── 第一部分：CUDA 基础与硬件架构
│   ├── 01_cuda_fundamentals.md         # CUDA 线程层级与内存体系
│   └── 02_hardware_architecture.md     # RTX 5090 硬件详解
│
├── 第二部分：GEMM 优化阶梯
│   ├── 03_gemm_naive.md                # Step 0: 朴素实现
│   ├── 04_shared_memory_tiling.md      # Step 1: 共享内存分块
│   └── 05_register_tiling.md           # Step 2: 寄存器分块
│
├── 第三部分：进阶优化技术
│   ├── 06_vectorization_and_bank_conflict.md  # 向量化与 Bank Conflict
│   └── 07_tensor_cores.md              # Tensor Core 优化
│
├── 第四部分：性能分析与调试
│   ├── 08_performance_analysis.md    # Roofline 模型分析
│   └── 09_profiling_tools.md           # Nsight Compute 实战
│
└── 扩展专题
    └── 10_batched_gemm.md              # 批量矩阵乘法
```

---

## 学习路径推荐

### 初学者（2-3 周）

1. [01_cuda_fundamentals.md](/docs/01_cuda_fundamentals.md) - CUDA 基础
2. [02_hardware_architecture.md](/docs/02_hardware_architecture.md) - 硬件架构
3. [03_gemm_naive.md](/docs/03_gemm_naive.md) - 朴素实现
4. [04_shared_memory_tiling.md](/docs/04_shared_memory_tiling.md) - Shared Memory
5. [05_register_tiling.md](/docs/05_register_tiling.md) - Register Tiling

### 进阶优化（1-2 周）

6. [06_vectorization_and_bank_conflict.md](/docs/06_vectorization_and_bank_conflict.md) - 向量化与 Bank Conflict
7. [07_tensor_cores.md](/docs/07_tensor_cores.md) - Tensor Cores
8. [08_performance_analysis.md](/docs/08_performance_analysis.md) - 性能分析
9. [09_profiling_tools.md](/docs/09_profiling_tools.md) - Profiling 工具

---

## 本目录遗留文档

以下文档为历史版本，内容已整合到新版教程：

| 本文档 | 对应新版教程 | 说明 |
|:---|:---|:---|
| `cuda_thread_hierarchy.md` | [01_cuda_fundamentals.md](/docs/01_cuda_fundamentals.md) | CUDA 线程层级 |
| `rtx5090_hardware_constraints.md` | [02_hardware_architecture.md](/docs/02_hardware_architecture.md) | 硬件约束 |
| `sgemm_shared_kernel_explained.md` | [04_shared_memory_tiling.md](/docs/04_shared_memory_tiling.md) | Shared Kernel |
| `sgemm_register_code_explanation.md` | [05_register_tiling.md](/docs/05_register_tiling.md) | Register Kernel 代码 |
| `sgemm_register_analysis.md` | [05_register_tiling.md](/docs/05_register_tiling.md) | 性能分析 |
| `sgemm_register_v2_optimization.md` | [06_vectorization_and_bank_conflict.md](/docs/06_vectorization_and_bank_conflict.md) | V2 优化 |
| `bank_conflict_analysis.md` | [06_vectorization_and_bank_conflict.md](/docs/06_vectorization_and_bank_conflict.md) | Bank Conflict |
| `roofline_analysis.md` | [08_performance_analysis.md](/docs/08_performance_analysis.md) | Roofline 模型 |
| `04_memory_coalescing.md` | [01_cuda_fundamentals.md](/docs/01_cuda_fundamentals.md) | 内存合并访问 |

---

## 配套代码

```bash
# 编译所有版本
cd GEMM
make clean && make

# 运行性能测试
./build/benchmark_gemm
```

---

*新版教程更新日期：2026年3月*

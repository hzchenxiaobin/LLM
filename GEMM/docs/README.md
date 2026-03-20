# GEMM 文档目录（历史版本）

> **注意**：此目录下的详细教程已整合到新的体系化教程中，推荐访问 [`/docs/`](/docs/) 目录获取更好的学习体验。

---

## 新版体系化教程（推荐）

```
/docs/
├── README.md                           # 教程入口
├──
├── 第一部分：CUDA 基础与硬件架构
│   ├── 01_cuda_fundamentals.md         # CUDA 线程层级与内存体系
│   └── 02_hardware_architecture.md     # RTX 5090 硬件详解
│
├── 第二部分：GEMM 优化阶梯
│   ├── 03_gemm_naive.md                # 朴素实现
│   ├── 04_shared_memory_tiling.md      # Shared Memory 分块
│   └── 05_register_tiling.md           # 寄存器分块
│
├── 第三部分：进阶优化技术
│   ├── 06_vectorization_and_bank_conflict.md  # 向量化与 Bank Conflict
│   └── 07_tensor_cores.md              # Tensor Core 优化
│
├── 第四部分：性能分析与调试
│   ├── 08_performance_analysis.md        # Roofline 模型
│   └── 09_profiling_tools.md             # Nsight Compute 实战
│
└── 扩展专题
    └── 10_batched_gemm.md              # 批量矩阵乘法
```

---

## 本目录历史文档映射

| 本目录文档 | 对应新版教程 | 说明 |
|:---|:---|:---|
| `cuda_thread_hierarchy.md` | [01_cuda_fundamentals.md](/docs/01_cuda_fundamentals.md) | CUDA 基础 |
| `rtx5090_hardware_constraints.md` | [02_hardware_architecture.md](/docs/02_hardware_architecture.md) | 硬件架构 |
| `sgemm_shared_kernel_explained.md` | [04_shared_memory_tiling.md](/docs/04_shared_memory_tiling.md) | Shared Kernel |
| `sgemm_register_code_explanation.md` | [05_register_tiling.md](/docs/05_register_tiling.md) | Register Kernel |
| `sgemm_register_analysis.md` | [05_register_tiling.md](/docs/05_register_tiling.md) | 性能分析 |
| `sgemm_register_v2_optimization.md` | [06_vectorization_and_bank_conflict.md](/docs/06_vectorization_and_bank_conflict.md) | V2 优化 |
| `bank_conflict_analysis.md` | [06_vectorization_and_bank_conflict.md](/docs/06_vectorization_and_bank_conflict.md) | Bank Conflict |
| `roofline_analysis.md` | [08_performance_analysis.md](/docs/08_performance_analysis.md) | Roofline 模型 |
| `04_memory_coalescing.md` | [01_cuda_fundamentals.md](/docs/01_cuda_fundamentals.md) | 内存合并访问 |

---

## 学习建议

**强烈建议**使用新版体系化教程：
1. 循序渐进的学习路径
2. 统一的讲解风格
3. 完整的代码示例
4. 课后练习与性能分析

从 [`/docs/README.md`](/docs/README.md) 开始学习！

---

*更新日期：2026年3月*

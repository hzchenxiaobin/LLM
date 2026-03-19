# LLM - CUDA 高性能计算学习仓库

本仓库用于记录和学习大规模语言模型 (LLM) 相关的底层高性能计算优化技术，从矩阵乘法 (GEMM) 入手，逐步深入到更复杂的算子优化。

## 📂 仓库结构

```
LLM/
├── README.md              # 本文件
├── .gitignore
└── GEMM/                  # CUDA SGEMM 优化教程
    ├── README.md          # 详细教程文档
    ├── Makefile
    ├── src/               # 源代码目录
    │   ├── main.cu                 # 基准测试框架
    │   ├── common.h                # 公共头文件
    │   ├── gemm_kernels.h          # Kernel 声明
    │   ├── sgemm_naive.cu          # 朴素实现
    │   ├── sgemm_shared.cu         # 共享内存优化
    │   ├── sgemm_register.cu       # 寄存器分块优化 (V1)
    │   ├── sgemm_register_v2.cu    # 向量化优化 (V2)
    │   ├── sgemm_register_v3.cu    # 双缓冲优化 (V3)
    │   ├── sgemm_register_bank_conflict.cu # Bank Conflict 消除优化
    │   └── sgemm_cublas.cu         # cuBLAS 参考实现
    ├── docs/              # 详细文档
    │   ├── README.md                          # 优化技术完整教程
    │   ├── cuda_thread_hierarchy.md           # CUDA 线程层次详解
    │   ├── roofline_analysis.md               # Roofline 性能模型
    │   ├── rtx5090_hardware_constraints.md      # RTX 5090 硬件约束
    │   ├── sgemm_register_analysis.md          # 寄存器优化性能分析
    │   ├── sgemm_register_code_explanation.md  # 寄存器 Kernel 逐行解读
    │   ├── sgemm_register_v2_optimization.md   # V2 向量化优化详解
    │   ├── bank_conflict_analysis.md           # Bank Conflict 深度解析
    │   └── sgemm_shared_kernel_explained.md    # 共享内存 Kernel 详解
    ├── images/            # 图解和图表
    └── scripts/           # 可视化脚本
```

## 🎯 学习目标

1. **GEMM 优化** - 理解 GPU 并行计算的核心优化技术
   - 从朴素实现 → 共享内存分块 → 寄存器分块
   - 向量化加载 (float4) 提高带宽利用率
   - **Bank Conflict 消除** - 理解 GPU 共享内存架构
   - **双缓冲 (Double Buffering)** - 隐藏数据加载延迟
   - 理解 Roofline 性能模型
   - 学习 CUDA 编程最佳实践

2. **未来扩展** - 更多 LLM 核心算子
   - Flash Attention
   - Layer Normalization
   - Softmax
   - 量化算子 (INT8/FP16)
   - Tensor Core (WMMA) 优化

## 🚀 快速开始

### 环境要求
- NVIDIA GPU (Compute Capability 7.0+, 推荐 8.0+)
- CUDA Toolkit 11.0+ (RTX 5090 需要 12.8+)
- make

### 进入 GEMM 项目并编译

```bash
cd GEMM
make
```

### 运行基准测试

```bash
./benchmark_gemm
```

## 📚 学习资源

### 入门教程
- [GEMM 详细教程](./GEMM/README.md) - 完整的优化教程文档，从 Naive 到 Tensor Cores
- [CUDA 线程层次详解](./GEMM/docs/cuda_thread_hierarchy.md) - 理解 GPU 并行架构基础

### 性能优化专题
- [Roofline 性能分析](./GEMM/docs/roofline_analysis.md) - 系统性性能分析方法论
- [RTX 5090 硬件约束详解](./GEMM/docs/rtx5090_hardware_constraints.md) - 硬件架构与优化策略
- [寄存器 Kernel 逐行解读](./GEMM/docs/sgemm_register_code_explanation.md) - 深入理解寄存器分块实现

### 进阶优化
- [Bank Conflict 深度解析](./GEMM/docs/bank_conflict_analysis.md) - **⭐ 新** 从 RTX 5090 硬件视角详解 Bank Conflict 及 Padding 解决方案
- [V2 向量化优化详解](./GEMM/docs/sgemm_register_v2_optimization.md) - float4 协作加载技术
- [寄存器优化性能分析](./GEMM/docs/sgemm_register_analysis.md) - 各版本性能对比分析
- [共享内存 Kernel 详解](./GEMM/docs/sgemm_shared_kernel_explained.md) - Tiling 技术基础

## 📊 性能基准

在 RTX 5090 (104.9 TFLOPS 理论峰值) 上的测试结果：

| 实现 | 性能 (TFLOPs) | 峰值利用率 | 优化技术 |
|------|--------------|-----------|---------|
| cuBLAS | 66.7 | 63.6% | NVIDIA 官方优化 |
| **Register V3** (Double Buffering) | 40-60 | 38-57% | 双缓冲 + 软件流水线 |
| **Register V2** (Vectorized) | 35-55 | 35-55% | float4 向量化加载 |
| **Register Bank Conflict** | 35-55 | 35-55% | Shared Memory Padding |
| Register V1 | 30-50 | 30-50% | 寄存器分块 + 双层 Tiling |
| Shared Memory | 9.1 | 8.7% | Shared Memory Tiling |
| Naive | 7.5 | 7.1% | 无优化 |

### 关键优化技术进展

| 版本 | 新增技术 | 解决的问题 | 性能提升 |
|------|---------|-----------|---------|
| V1 | 寄存器分块 | 减少共享内存访问 | ~5× vs Naive |
| V2 | float4 向量化 | 提高带宽利用率 | ~1.2× vs V1 |
| **Bank Conflict** | Shared Memory Padding | 消除 Bank Conflict | 更稳定的性能 |
| V3 | 双缓冲 | 计算与访存重叠 | ~1.2× vs V2 |

## 🤝 贡献

欢迎提交 Issue 和 PR 来完善教程或添加新的算子实现。

## 📄 License

MIT License

---

**作者**: hzchenxiaobin  
**创建时间**: 2026年3月

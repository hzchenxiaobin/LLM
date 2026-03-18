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
    ├── main.cu            # 基准测试框架
    ├── common.h           # 公共头文件
    ├── gemm_kernels.h     # Kernel 声明
    ├── sgemm_naive.cu     # 朴素实现
    ├── sgemm_shared.cu    # 共享内存优化
    ├── sgemm_register.cu  # 寄存器分块优化
    ├── sgemm_cublas.cu    # cuBLAS 参考实现
    ├── docs/              # 详细文档
    │   ├── roofline_analysis.md
    │   ├── sgemm_shared_kernel_explained.md
    │   ├── sgemm_register_analysis.md
    │   ├── sgemm_register_code_explanation.md
    │   └── cuda_thread_hierarchy.md
    ├── images/            # 图解和图表
    └── scripts/           # 可视化脚本
```

## 🎯 学习目标

1. **GEMM 优化** - 理解 GPU 并行计算的核心优化技术
   - 从朴素实现 → 共享内存分块 → 寄存器分块
   - 理解 Roofline 性能模型
   - 学习 CUDA 编程最佳实践

2. **未来扩展** - 更多 LLM 核心算子
   - Flash Attention
   - Layer Normalization
   - Softmax
   - 量化算子 (INT8/FP16)

## 🚀 快速开始

### 环境要求
- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 11.0+
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

- [GEMM 详细教程](./GEMM/README.md) - 完整的优化教程文档
- [Roofline 性能分析](./GEMM/docs/roofline_analysis.md)
- [CUDA 线程层次详解](./GEMM/docs/cuda_thread_hierarchy.md)

## 📊 性能基准

在 RTX 5090 上的测试结果：

| 实现 | 性能 (TFLOPs) | 峰值利用率 |
|------|--------------|-----------|
| cuBLAS | 66.7 | 63.6% |
| Register Tiling | 30-50 | 30-50% |
| Shared Memory | 9.1 | 8.7% |
| Naive | 7.5 | 7.1% |

## 🤝 贡献

欢迎提交 Issue 和 PR 来完善教程或添加新的算子实现。

## 📄 License

MIT License

---

**作者**: hzchenxiaobin  
**创建时间**: 2026年3月

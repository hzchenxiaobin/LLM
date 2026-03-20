# LLM - CUDA 高性能计算学习仓库

本仓库用于记录和学习大规模语言模型 (LLM) 相关的底层高性能计算优化技术，从矩阵乘法 (GEMM) 入手，逐步深入到更复杂的算子优化。

## 仓库结构

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
    │   ├── gemm_kernels.h          # 统一头文件入口
    │   └── gemm/                   # 单矩阵 GEMM 实现目录
    │       ├── gemm_kernels.h      # Kernel 函数声明
    │       ├── sgemm_naive.cu      # 朴素实现
    │       ├── sgemm_shared.cu     # 共享内存优化
    │       ├── sgemm_register.cu   # 寄存器分块优化 (V1)
    │       ├── sgemm_register_v2.cu  # 向量化优化 (V2)
    │       ├── sgemm_register_v3.cu  # 双缓冲优化 (V3)
    │       ├── sgemm_register_bank_conflict.cu # Bank Conflict 消除优化
    │       ├── sgemm_wmma.cu       # Tensor Core WMMA 实现
    │       ├── sgemm_wmma_v2.cu    # Tensor Core WMMA 优化版
    │       ├── sgemm_cutlass.cu    # CUTLASS 实现 (可选)
    │       ├── sgemm_cute.cu       # CuTe 实现 (可选)
    │       └── sgemm_cublas.cu     # cuBLAS 参考实现
    ├── docs/              # 详细文档
    │   ├── README.md                          # 优化技术完整教程
    │   ├── cuda_thread_hierarchy.md           # CUDA 线程层次详解
    │   ├── roofline_analysis.md               # Roofline 性能模型
    │   ├── rtx5090_hardware_constraints.md    # RTX 5090 硬件约束
    │   ├── sgemm_register_analysis.md         # 寄存器优化性能分析
    │   ├── sgemm_register_code_explanation.md # 寄存器 Kernel 逐行解读
    │   ├── sgemm_register_v2_optimization.md  # V2 向量化优化详解
    │   ├── bank_conflict_analysis.md          # Bank Conflict 深度解析
    │   └── sgemm_shared_kernel_explained.md    # 共享内存 Kernel 详解
    ├── images/            # 图解和图表
    └── scripts/           # 可视化脚本
```

## 学习目标

1. **GEMM 优化** - 理解 GPU 并行计算的核心优化技术
   - 从朴素实现 → 共享内存分块 → 寄存器分块 → Tensor Core
   - 向量化加载 (float4) 提高带宽利用率
   - **Bank Conflict 消除** - 理解 GPU 共享内存架构
   - **双缓冲 (Double Buffering)** - 隐藏数据加载延迟
   - **Tensor Core (WMMA)** - 利用专用矩阵计算单元
   - 理解 Roofline 性能模型
   - 学习 CUDA 编程最佳实践

2. **未来扩展** - 更多 LLM 核心算子
   - Flash Attention
   - Layer Normalization
   - Softmax
   - 量化算子 (INT8/FP16)
   - MoE (Mixture of Experts) 路由优化

## 快速开始

### 环境要求
- NVIDIA GPU (Compute Capability 7.0+, 推荐 8.0+)
- CUDA Toolkit 11.0+ (RTX 4090/5090 需要 12.0+)
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

### 可选：启用 CUTLASS 支持

如需测试 CUTLASS/CuTe 实现，先克隆 CUTLASS：

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
make clean && make
```

## 学习资源

### 入门教程
- [GEMM 详细教程](./GEMM/README.md) - 完整的优化教程文档，从 Naive 到 Tensor Cores
- [CUDA 线程层次详解](./GEMM/docs/cuda_thread_hierarchy.md) - 理解 GPU 并行架构基础

### 性能优化专题
- [Roofline 性能分析](./GEMM/docs/roofline_analysis.md) - 系统性性能分析方法论
- [RTX 5090 硬件约束详解](./GEMM/docs/rtx5090_hardware_constraints.md) - 硬件架构与优化策略
- [寄存器 Kernel 逐行解读](./GEMM/docs/sgemm_register_code_explanation.md) - 深入理解寄存器分块实现

### 进阶优化
- [Bank Conflict 深度解析](./GEMM/docs/bank_conflict_analysis.md) - 从 RTX 5090 硬件视角详解 Bank Conflict 及 Padding 解决方案
- [V2 向量化优化详解](./GEMM/docs/sgemm_register_v2_optimization.md) - float4 协作加载技术
- [寄存器优化性能分析](./GEMM/docs/sgemm_register_analysis.md) - 各版本性能对比分析
- [共享内存 Kernel 详解](./GEMM/docs/sgemm_shared_kernel_explained.md) - Tiling 技术基础

## 性能基准

在 RTX 4090 (82.58 TFLOPS 理论峰值) 上的测试结果：

| 实现 | 性能 (TFLOPs) | 峰值利用率 | 核心优化技术 |
|------|--------------|-----------|-------------|
| cuBLAS | 58-65 | 70-79% | NVIDIA 官方优化 |
| **WMMA V2** | 45-55 | 55-67% | Tensor Core + 优化调度 |
| **Register V3** (Double Buffering) | 40-50 | 48-61% | 双缓冲 + 软件流水线 |
| **Register V2** (Vectorized) | 35-45 | 42-55% | float4 向量化加载 |
| **Register Bank Conflict** | 35-45 | 42-55% | Shared Memory Padding |
| Register V1 | 30-40 | 36-48% | 寄存器分块 + 双层 Tiling |
| Shared Memory | 9-10 | 11-12% | Shared Memory Tiling |
| Naive | 7-8 | 8-10% | 无优化 |

### 关键优化技术进展

| 版本 | 新增技术 | 解决的问题 | 性能提升 |
|------|---------|-----------|---------|
| V1 | 寄存器分块 | 减少共享内存访问 | ~5× vs Naive |
| V2 | float4 向量化 | 提高带宽利用率 | ~1.2× vs V1 |
| Bank Conflict | Shared Memory Padding | 消除 Bank Conflict | 更稳定的性能 |
| V3 | 双缓冲 | 计算与访存重叠 | ~1.2× vs V2 |
| WMMA | Tensor Core | 专用矩阵计算单元 | ~1.5× vs V3 |

### 优化技术效果对比

| 技术 | Arithmetic Intensity | 带宽需求 | 计算密度 |
|------|---------------------|---------|---------|
| Naive | 0.5 FLOP/byte | 高 | 低 |
| Shared Memory | 16 FLOP/byte | 中 | 中 |
| Register V1 | 128 FLOP/byte | 低 | 高 |
| Register V2/V3 | 256+ FLOP/byte | 极低 | 极高 |
| WMMA | 512+ FLOP/byte | 极低 | 极高 |

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

## 贡献

欢迎提交 Issue 和 PR 来完善教程或添加新的算子实现。

可能的改进方向：
- 更多数据类型支持 (FP16/BF16/INT8)
- Nsight Compute profiling 分析
- 多 GPU 支持
- 更多 LLM 算子 (Attention, LayerNorm, etc.)

## License

MIT License

---

**作者**: hzchenxiaobin  
**创建时间**: 2026年3月

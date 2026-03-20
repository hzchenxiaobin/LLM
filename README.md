# LLM - CUDA 高性能计算学习仓库

本仓库用于系统学习大规模语言模型 (LLM) 相关的底层高性能计算优化技术，从基础算子优化入手，深入理解 GPU 并行计算的核心原理。

## 仓库结构

```
LLM/
├── README.md              # 本文件
├── .gitignore
├── GEMM/                  # CUDA SGEMM 优化教程
│   ├── README.md          # GEMM 详细教程文档
│   ├── Makefile           # 编译脚本
│   ├── src/               # 源代码目录
│   │   ├── main.cu                 # 基准测试主程序
│   │   ├── common.h                # 公共头文件（CUDA 错误检查）
│   │   ├── gemm_kernels.h          # Kernel 函数统一入口
│   │   └── gemm/                   # GEMM 实现目录
│   │       ├── gemm_kernels.h      # Kernel 函数声明
│   │       ├── sgemm_naive.cu      # 朴素实现
│   │       ├── sgemm_shared.cu     # 共享内存分块优化
│   │       ├── sgemm_register.cu   # 寄存器分块优化 (V1)
│   │       ├── sgemm_register_vectorized.cu   # 向量化加载优化 (V2)
│   │       ├── sgemm_register_bank_conflict.cu  # Bank Conflict 消除优化
│   │       ├── sgemm_register_vec_bank.cu   # 向量化 + Bank Conflict + 双缓冲 (V3)
│   │       └── sgemm_cublas.cu     # cuBLAS 参考实现
│   ├── docs/              # 详细技术文档
│   │   ├── README.md                          # 文档入口与教程汇总
│   │   ├── cuda_thread_hierarchy.md           # CUDA 线程层次与 SM 架构
│   │   ├── roofline_analysis.md               # Roofline 性能模型
│   │   ├── rtx5090_hardware_constraints.md    # GPU 硬件约束分析
│   │   ├── sgemm_shared_kernel_explained.md   # 共享内存 Kernel 详解
│   │   ├── sgemm_register_code_explanation.md # 寄存器 Kernel 逐行解读
│   │   ├── sgemm_register_analysis.md         # 寄存器优化性能分析
│   │   ├── sgemm_register_v2_optimization.md  # 向量化优化详解
│   │   ├── bank_conflict_analysis.md          # Bank Conflict 深度解析
│   │   └── 04_memory_coalescing.md            # 内存合并访问优化
│   ├── exercises/         # 练习与面试题
│   │   ├── occupancy_calculation_exercises.md        # Occupancy 计算练习题
│   │   ├── gemm_basic_interview_questions.md         # 基础概念面试题
│   │   ├── gemm_optimization_interview_questions.md  # 优化技术面试题
│   │   ├── gemm_tensor_core_interview_questions.md   # Tensor Core 面试题
│   │   └── gemm_practice_problems.md                 # 编程实践题
│   └── scripts/           # 可视化脚本
│       ├── generate_roofline.py         # 生成 Roofline 性能模型图
│       ├── visualize_shared_gemm.py     # Shared GEMM 执行流程可视化
│       └── visualize_register_gemm.py   # Register GEMM 执行流程可视化
└── reduction/             # CUDA Reduction 归约算子优化教程
    └── README.md          # Reduction 优化完整教程（6 步进阶）
```

## 学习目标

### 1. GEMM 优化
理解 GPU 并行计算的核心优化技术，从朴素实现到 CUDA Core 极致优化：

- **朴素实现 → 共享内存分块 → 寄存器分块** 的递进优化路径
- **向量化加载 (float4)** 提高带宽利用率
- **Bank Conflict 消除** 理解 GPU 共享内存架构
- **双缓冲 (Double Buffering)** 隐藏数据加载延迟
- **Roofline 性能模型** 系统性性能分析方法
- **Warp Shuffle** 寄存器级线程通信

### 2. Reduction 归约优化
掌握访存密集型算子的优化技术：

- **消除 Warp Divergence** 分支发散优化
- **解决 Bank Conflict** 共享内存冲突消除
- **提高指令吞吐** First Add During Load 技巧
- **Warp Shuffle** 寄存器级归约
- **向量化访存** 极致带宽利用

### 3. 未来扩展
- **Flash Attention** - 高效的注意力机制实现
- **Layer Normalization** - 层归一化算子优化
- **Softmax** - Softmax 算子高性能实现
- **量化算子** - INT8/FP16/BF16 低精度计算
- **MoE 路由** - Mixture of Experts 路由优化

## 快速开始

### 环境要求
- NVIDIA GPU (Compute Capability 7.0+, 推荐 8.0+)
- CUDA Toolkit 11.0+ (RTX 4090/5090 需要 12.0+)
- make

### GEMM 项目编译与运行

```bash
cd GEMM
make
./build/benchmark_gemm
```

### Reduction 项目编译与运行

```bash
cd reduction
# 参照 reduction/README.md 中的完整代码进行编译运行
# 包含 6 个递进版本的归约优化实现
```

Reduction 目录包含可直接编译运行的 CUDA 归约算子示例代码。

## 学习路径

### 阶段 1: 基础概念
- 阅读 `GEMM/docs/cuda_thread_hierarchy.md` 了解 CUDA 线程组织
- 理解 `dim3`, `blockIdx`, `threadIdx`, SM 架构

### 阶段 2: GEMM 矩阵乘法优化进阶

| 阶段 | 内容 | 文档 | 代码 |
|:---|:---|:---|:---|
| 2.1 | 朴素实现 | - | `sgemm_naive.cu` |
| 2.2 | 共享内存分块 | `sgemm_shared_kernel_explained.md` | `sgemm_shared.cu` |
| 2.3 | Roofline 分析 | `roofline_analysis.md` | `scripts/generate_roofline.py` |
| 2.4 | 寄存器分块 V1 | `sgemm_register_code_explanation.md` | `sgemm_register.cu` |
| 2.5 | 向量化优化 V2 | `sgemm_register_v2_optimization.md` | `sgemm_register_vectorized.cu` |
| 2.6 | Bank Conflict 消除 | `bank_conflict_analysis.md` | `sgemm_register_bank_conflict.cu` |
| 2.7 | 双缓冲优化 V3 | `sgemm_register_analysis.md` | `sgemm_register_vec_bank.cu` |
| 2.8 | 硬件深入 | `rtx5090_hardware_constraints.md` | `exercises/occupancy_calculation_exercises.md` |

### 阶段 3: Reduction 归约优化进阶

参照 `reduction/README.md` 完成 6 步进阶优化：

| 版本 | 技术要点 | 解决的问题 | 核心改进 |
|:---|:---|:---|:---|
| V1 | Interleaved Addressing | 基础树状归约 | Shared Memory 基础使用 |
| V2 | Strided Index | Warp Divergence | 连续线程访问连续数据 |
| V3 | Sequential Addressing | Bank Conflict | 反向步长消除 Bank Conflict |
| V4 | First Add During Load | 提高指令吞吐 | 加载时预计算，减少 Block 数 |
| V5 | Warp Shuffle | 消除最后阶段同步 | 寄存器级线程通信 |
| V6 | Vectorized Memory Access | 极致带宽利用 | float4 向量化加载 |

## 性能基准

### GEMM - RTX 4090 (82.58 TFLOPS 理论峰值)

| 实现 | 性能 (TFLOPs) | 峰值利用率 | 核心优化技术 |
|:---|:---:|:---:|:---|
| cuBLAS | 58-65 | 70-79% | NVIDIA 官方优化 |
| Register Vec+Bank (V3) | 40-50 | 48-61% | 双缓冲 + 向量化 + Bank-free |
| Register Vectorized (V2) | 35-45 | 42-55% | float4 向量化加载 |
| Register Bank Conflict | 35-45 | 42-55% | Shared Memory Padding |
| Register Tiling (V1) | 30-40 | 36-48% | 寄存器分块 + 双层 Tiling |
| Shared Memory | 9-10 | 11-12% | Shared Memory Tiling |
| Naive | 7-8 | 8-10% | 无优化 |

### 关键优化技术进展

| 版本 | 新增技术 | 解决的问题 | 性能提升 |
|:---|:---|:---|:---:|
| Shared | 共享内存 Tiling | 减少全局内存访问 | ~1.3× vs Naive |
| Register | 寄存器分块 | 减少共享内存访问 | ~4× vs Shared |
| Vectorized | float4 向量化 | 提高带宽利用率 | ~1.2× vs Register |
| Bank Conflict | Shared Memory Padding | 消除 Bank Conflict | 更稳定的性能 |
| Vec+Bank | 双缓冲 | 计算与访存重叠 | ~1.2× vs Vectorized |

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

| 文档 | 内容描述 | 推荐阶段 |
|:---|:---|:---:|
| `docs/cuda_thread_hierarchy.md` | CUDA 线程层次与 SM 架构 | 1 |
| `docs/roofline_analysis.md` | Roofline 模型与 Arithmetic Intensity | 2 |
| `docs/sgemm_shared_kernel_explained.md` | Shared Memory Kernel 详解 | 3 |
| `docs/sgemm_register_code_explanation.md` | Register Kernel 逐行解读 | 4 |
| `docs/sgemm_register_analysis.md` | 各版本性能对比分析 | 4 |
| `docs/sgemm_register_v2_optimization.md` | 向量化优化详解 | 5 |
| `docs/bank_conflict_analysis.md` | Bank Conflict 深度解析 | 5 |
| `docs/rtx5090_hardware_constraints.md` | GPU 硬件约束详解 | 6 |
| `docs/04_memory_coalescing.md` | 内存合并访问优化 | 补充 |

### 练习题与面试题

| 文档 | 类型 | 说明 |
|:---|:---|:---|
| `exercises/occupancy_calculation_exercises.md` | 计算题 | Occupancy 计算练习 |
| `exercises/gemm_basic_interview_questions.md` | 面试题 | 基础概念面试题 |
| `exercises/gemm_optimization_interview_questions.md` | 面试题 | 优化技术面试题 |
| `exercises/gemm_tensor_core_interview_questions.md` | 面试题 | Tensor Core 面试题 |
| `exercises/gemm_practice_problems.md` | 编程题 | 编程实践题 |

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

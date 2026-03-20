# CUDA SGEMM 优化教程

一个从朴素实现到 CUDA Core 极致优化的完整 CUDA 矩阵乘法学习项目，通过 7 个递进式算子实现，展示 GPU 编程的核心优化技术。

## 目录

- [项目简介](#项目简介)
- [实现列表](#实现列表)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [性能对比](#性能对比)
- [学习路径](#学习路径)
- [文档索引](#文档索引)
- [贡献与许可](#贡献与许可)

## 项目简介

### 什么是 SGEMM？

**SGEMM** (Single-precision General Matrix Multiply) 是单精度浮点矩阵乘法的基础运算：

```
C = alpha × A × B + beta × C
```

其中：
- **A**: M×K 矩阵
- **B**: K×N 矩阵
- **C**: M×N 矩阵
- **alpha, beta**: 缩放系数

### 项目特点

- **渐进式学习**: 从 7 TFLOPS 的朴素实现到 50+ TFLOPS 的 CUDA Core 极致优化
- **完整文档**: 10+ 篇技术文档深入解析每个优化技术
- **实战导向**: 包含练习题、面试题和可视化工具
- **硬件适配**: 支持 Volta/Turing/Ampere/Hopper/Blackwell 架构

## 实现列表

| 序号 | 算子 | 文件 | 核心优化技术 | 预期性能 |
|:---:|:---|:---|:---|:---:|
| 0 | **cuBLAS** | `sgemm_cublas.cu` | NVIDIA 官方高度优化库 | 60-70 TFLOPS |
| 1 | **Naive** | `sgemm_naive.cu` | 基础并行实现 | ~7 TFLOPS |
| 2 | **Shared Memory** | `sgemm_shared.cu` | 共享内存分块 (Tiling) | ~9 TFLOPS |
| 3 | **Register Tiling** | `sgemm_register.cu` | 寄存器分块 + 双层 Tiling | ~30-40 TFLOPS |
| 4 | **Register Vectorized** | `sgemm_register_vectorized.cu` | 向量化加载 (float4) | ~35-45 TFLOPS |
| 5 | **Register Bank Conflict** | `sgemm_register_bank_conflict.cu` | Padding 消除 Bank Conflict | ~35-45 TFLOPS |
| 6 | **Register VecBank** | `sgemm_register_vec_bank.cu` | 向量化 + Padding 综合优化 | ~40-50 TFLOPS |

### 关键优化技术详解

| 技术 | 解决的问题 | 效果 |
|:---|:---|:---|
| **共享内存 Tiling** | 减少全局内存访问 | 计算强度从 0.5 → 683 |
| **寄存器分块** | 减少共享内存访问 | 每轮计算 64 FMA |
| **外积计算** | 提高指令级并行 (ILP) | 8× 计算强度提升 |
| **向量化加载** | 提高带宽利用率 | 128-bit 协作加载 |
| **Shared Memory Padding** | 消除 Bank Conflict | 冲突-free 访存 |
| **双缓冲** | 隐藏数据加载延迟 | 计算与访存重叠 |

## 项目结构

```
GEMM/
├── README.md                    # 项目说明（本文件）
├── Makefile                     # 编译脚本
│
├── src/                         # 源代码
│   ├── main.cu                  # 测试框架主程序
│   ├── common.h                 # 公共头文件（CUDA 错误检查）
│   ├── gemm_kernels.h           # Kernel 函数统一入口
│   └── gemm/                    # GEMM 实现目录
│       ├── gemm_kernels.h       # Kernel 声明
│       ├── sgemm_cublas.cu      # cuBLAS 参考实现
│       ├── sgemm_naive.cu       # 朴素实现
│       ├── sgemm_shared.cu      # 共享内存优化
│       ├── sgemm_register.cu    # 寄存器分块 (V1)
│       ├── sgemm_register_vectorized.cu # 向量化优化 (V2)
│       ├── sgemm_register_bank_conflict.cu # Bank Conflict 消除
│       └── sgemm_register_vec_bank.cu # 向量化 + Padding 综合优化 (V3)
│
├── docs/                        # 技术文档
│   ├── README.md                # 文档入口
│   ├── roofline_analysis.md     # Roofline 模型性能分析
│   ├── cuda_thread_hierarchy.md # CUDA 线程层次与 SM 架构
│   ├── sgemm_shared_kernel_explained.md     # Shared Kernel 详解
│   ├── sgemm_register_code_explanation.md   # Register Kernel 逐行解读
│   ├── sgemm_register_analysis.md           # Register 性能对比
│   ├── sgemm_register_v2_optimization.md    # V2 向量化优化详解
│   ├── bank_conflict_analysis.md          # Bank Conflict 深度解析
│   ├── rtx5090_hardware_constraints.md      # GPU 硬件约束分析
│   └── 04_memory_coalescing.md            # 内存合并访问
│
├── exercises/                   # 练习与面试题
│   ├── occupancy_calculation_exercises.md        # Occupancy 计算练习题
│   ├── gemm_basic_interview_questions.md         # 基础面试题
│   ├── gemm_optimization_interview_questions.md  # 优化面试题
│   ├── gemm_tensor_core_interview_questions.md   # Tensor Core 面试题
│   └── gemm_practice_problems.md                 # 编程练习题
│
└── scripts/                     # 辅助脚本
    ├── generate_roofline.py     # 生成 Roofline 图
    ├── visualize_shared_gemm.py # Shared GEMM 可视化
    └── visualize_register_gemm.py # Register GEMM 可视化
```

## 快速开始

### 环境要求

| 组件 | 版本要求 |
|:---|:---|
| NVIDIA GPU | Compute Capability 7.0+ (推荐 RTX 3060/4090) |
| CUDA Toolkit | 11.0+ |
| nvcc 编译器 | 随 CUDA Toolkit |
| make | 系统自带 |

### 1. 环境检查

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 信息
nvidia-smi
```

### 2. 编译项目

```bash
# 编译所有算子
make

# 生成可执行文件: build/benchmark_gemm
```

### 3. 运行测试

```bash
# 运行基准测试
./build/benchmark_gemm
```

**示例输出:**
```
========================================================
检测到显卡设备: NVIDIA GeForce RTX 4090 (Compute 8.9)
SM 数量: 128, 核心频率: 2520 MHz
理论 FP32 CUDA Core 峰值算力: 82.58 TFLOPs
========================================================

开始 GEMM 性能基准测试...
矩阵尺寸: M=4096, N=4096, K=4096
--------------------------------------------------------
正在运行: cuBLAS SGEMM (Reference / Upper Bound)
  [正确性检查] 跳过 (作为标准参考答案).
  [性能统计] 平均耗时: 2.345 ms
  [算力吞吐] 实际算力: 58.500 TFLOPs (达理论峰值的 70.83%)
...
```

### 4. 自定义矩阵尺寸

编辑 `src/main.cu` 修改测试参数：

```cpp
int M = 4096;  // 修改为需要的大小
int N = 4096;
int K = 4096;
```

重新编译运行：
```bash
make clean && make && ./build/benchmark_gemm
```

## 性能对比

### RTX 4090 (82.58 TFLOPS 理论峰值) 典型测试结果

| 算子 | 性能 (TFLOPS) | 利用率 | vs Naive | vs cuBLAS |
|:---|:---:|:---:|:---:|:---:|
| **cuBLAS** | 58-65 | 70-79% | 8-9× | 100% |
| **Register VecBank (V3)** | 40-50 | 48-61% | 5-6× | 69-77% |
| **Register Vectorized (V2)** | 35-45 | 42-55% | 4-5× | 60-69% |
| **Register Bank Conflict** | 35-45 | 42-55% | 4-5× | 60-69% |
| **Register Tiling (V1)** | 30-40 | 36-48% | 4-5× | 52-62% |
| **Shared Memory** | 9-10 | 11-12% | 1.3× | 15-17% |
| **Naive** | 7-8 | 8-10% | 1× | 12-14% |

### 性能可视化

```bash
# 生成 Roofline 性能模型图
python3 scripts/generate_roofline.py

# 生成 GEMM 执行流程可视化
python3 scripts/visualize_shared_gemm.py
python3 scripts/visualize_register_gemm.py
```

## 学习路径

> **新版体系化教程已移动到 `/docs/` 目录**

### 阶段 1: 基础概念
- 阅读 [`/docs/01_cuda_fundamentals.md`](/docs/01_cuda_fundamentals.md) 了解 CUDA 线程组织
- 阅读 [`/docs/02_hardware_architecture.md`](/docs/02_hardware_architecture.md) 理解 RTX 5090 硬件架构

### 阶段 2: 朴素实现
- 阅读 [`/docs/03_gemm_naive.md`](/docs/03_gemm_naive.md)
- 理解基础并行实现和性能瓶颈（频繁全局内存访问）

### 阶段 3: 共享内存优化
- 阅读 [`/docs/04_shared_memory_tiling.md`](/docs/04_shared_memory_tiling.md)
- 理解 Shared Memory Tiling 原理
- 阅读代码 `src/gemm/sgemm_shared.cu`
- 运行 `scripts/visualize_shared_gemm.py` 生成图解

### 阶段 4: 性能分析
- 阅读 [`/docs/08_performance_analysis.md`](/docs/08_performance_analysis.md)
- 运行 `scripts/generate_roofline.py` 生成图表
- 理解 Arithmetic Intensity 和 Roofline 模型

### 阶段 5: 寄存器优化
- 阅读 [`/docs/05_register_tiling.md`](/docs/05_register_tiling.md)
- 理解双层分块策略和外积计算
- 阅读代码 `src/gemm/sgemm_register.cu`

### 阶段 6: 进阶优化
- **向量化与 Bank Conflict**: [`/docs/06_vectorization_and_bank_conflict.md`](/docs/06_vectorization_and_bank_conflict.md)
  - float4 向量化加载，128-bit 协作访存
  - Shared Memory Padding，理解 `bank = (address / 4) % 32`
- **Tensor Core**: [`/docs/07_tensor_cores.md`](/docs/07_tensor_cores.md)
  - WMMA API 和 Tensor Core 编程

### 阶段 7: 性能调试
- 阅读 [`/docs/09_profiling_tools.md`](/docs/09_profiling_tools.md)
- 完成 `exercises/occupancy_calculation_exercises.md` 练习题
- 使用 Nsight Compute 进行实际性能分析

## 文档索引

### 新版体系化教程（推荐）

| 文档 | 内容描述 | 推荐阶段 |
|:---|:---|:---:|
| [`/docs/01_cuda_fundamentals.md`](/docs/01_cuda_fundamentals.md) | CUDA 线程层级与内存体系 | 1 |
| [`/docs/02_hardware_architecture.md`](/docs/02_hardware_architecture.md) | RTX 5090 硬件详解 | 1 |
| [`/docs/03_gemm_naive.md`](/docs/03_gemm_naive.md) | 朴素实现与瓶颈分析 | 2 |
| [`/docs/04_shared_memory_tiling.md`](/docs/04_shared_memory_tiling.md) | Shared Memory Tiling | 3 |
| [`/docs/05_register_tiling.md`](/docs/05_register_tiling.md) | 寄存器分块与外积计算 | 5 |
| [`/docs/06_vectorization_and_bank_conflict.md`](/docs/06_vectorization_and_bank_conflict.md) | 向量化与 Bank Conflict 消除 | 6 |
| [`/docs/07_tensor_cores.md`](/docs/07_tensor_cores.md) | Tensor Core 优化 | 6 |
| [`/docs/08_performance_analysis.md`](/docs/08_performance_analysis.md) | Roofline 模型分析 | 4 |
| [`/docs/09_profiling_tools.md`](/docs/09_profiling_tools.md) | Nsight Compute 实战 | 7 |

### 本目录遗留文档（历史版本）

`docs/` 目录下的旧文档内容已整合到新版教程，仅供参考。详见 [`docs/README.md`](docs/README.md)。

### 练习题与面试题

| 文档 | 类型 | 题目数量 |
|:---|:---|:---:|
| `exercises/occupancy_calculation_exercises.md` | Occupancy 计算 | 8 道 |
| `exercises/gemm_basic_interview_questions.md` | 基础概念面试题 | - |
| `exercises/gemm_optimization_interview_questions.md` | 优化技术面试题 | - |
| `exercises/gemm_practice_problems.md` | 编程实践题 | - |

## 常见问题

### Q: 编译失败，找不到 `nvcc`

```bash
# 添加 CUDA 到环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: 运行时找不到 libcublas.so

```bash
# 链接 cuBLAS 库路径
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: 性能与预期不符

- 检查 GPU 功耗模式：`nvidia-smi -q -d POWER`
- 确保 GPU 未被其他进程占用
- 矩阵大小建议 ≥ 2048 以发挥 GPU 并行性
- 关闭 GPU 的 Persistence Mode 可能会影响性能

## 参考资料

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS - NVIDIA 官方高效 GEMM 实现](https://github.com/NVIDIA/cutlass)
- [Roofline Model Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/roofline.pdf)
- [NVIDIA Ampere 架构白皮书](https://www.nvidia.com/en-us/data-center/ampere-architecture/)

## 贡献与许可

### 贡献方向

欢迎提交 Issue 和 PR！可能的改进：
- 支持更多数据类型（FP16、BF16、TF32）
- 添加 Nsight Compute profiling 支持
- 多 GPU 分布式实现
- 更多练习题（Warp Divergence、Coalesced Access）
- 自动调优 (Auto-Tuning) 功能

### 许可证

MIT License - 自由使用和学习

---

**项目**: CUDA GEMM 优化教程  
**创建时间**: 2026年3月

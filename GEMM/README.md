# CUDA SGEMM 优化教程项目

本项目是一个用于学习和实践 CUDA 矩阵乘法（SGEMM）优化的教程代码库，从朴素实现逐步优化到高性能实现，展示了 GPU 编程的核心优化技术。

## 📋 项目简介

### 什么是 SGEMM？

**SGEMM** (Single-precision General Matrix Multiply) 是矩阵乘法的基础运算：

```
C = alpha × A × B + beta × C
```

其中：
- A 是 M×K 矩阵
- B 是 K×N 矩阵
- C 是 M×N 矩阵
- alpha 和 beta 是缩放系数

### 本项目包含的实现

| 算子 | 文件 | 优化技术 | 预期性能 |
|------|------|---------|---------|
| **cuBLAS** | `src/sgemm_cublas.cu` | NVIDIA 官方高度优化库 | 60+ TFLOPS |
| **Naive** | `src/sgemm_naive.cu` | 无优化，直接实现 | ~7 TFLOPS |
| **Shared Memory** | `src/sgemm_shared.cu` | 共享内存分块 (Tiling) | ~9 TFLOPS |
| **Register** | `src/sgemm_register.cu` | 寄存器分块 + 双层 Tiling | ~30-50 TFLOPS |
| **Register V2** | `src/sgemm_register_v2.cu` | 向量化加载 + Padding 优化 | ~35-55 TFLOPS |

### 硬件要求

- NVIDIA GPU（支持 CUDA，推荐 Compute Capability 7.0+）
- CUDA Toolkit 11.0+
- 测试通过硬件：RTX 4090 / RTX 5090

## 📁 项目结构

```
GEMM/
├── README.md                    # 本文件
├── Makefile                     # 编译脚本
│
├── src/                         # 源代码目录
│   ├── main.cu                  # 测试框架主程序
│   ├── common.h                 # 公共头文件（CUDA 错误检查宏）
│   ├── gemm_kernels.h           # Kernel 函数声明
│   ├── sgemm_naive.cu           # 朴素实现
│   ├── sgemm_shared.cu          # 共享内存优化实现
│   ├── sgemm_register.cu        # 寄存器分块优化实现
│   ├── sgemm_register_v2.cu     # 优化版寄存器分块 (向量化 + Padding)
│   └── sgemm_cublas.cu          # cuBLAS 参考实现
│
├── docs/                        # 技术文档目录
│   ├── roofline_analysis.md            # Roofline 模型性能分析
│   ├── sgemm_shared_kernel_explained.md # Shared Kernel 详解
│   ├── sgemm_register_analysis.md       # Register Kernel 性能对比
│   ├── sgemm_register_code_explanation.md # Register Kernel 逐行解读
│   ├── rtx5090_hardware_constraints.md  # RTX 5090 硬件约束分析
│   └── cuda_thread_hierarchy.md         # CUDA 线程层次说明
│
├── images/                      # 图片目录
│   ├── roofline_plot.png               # Roofline 图
│   ├── shared_gemm_*.png               # Shared Kernel 图解
│   └── register_*.png                  # Register Kernel 图解
│
├── scripts/                     # 辅助脚本
│   ├── generate_roofline.py            # 生成 Roofline 图
│   └── visualize_shared_gemm.py      # 生成 SGEMM 可视化图
│
└── exercises/                   # 练习题目录
    └── occupancy_calculation_exercises.md  # Occupancy 计算练习题
```

## 🚀 快速开始

### 1. 环境准备

确保已安装：
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.0+)
- `nvcc` 编译器
- `make` 工具

检查 CUDA 版本：
```bash
nvcc --version
```

检查 GPU：
```bash
nvidia-smi
```

### 2. 编译项目

```bash
# 克隆/进入项目目录
cd GEMM

# 编译所有算子
make

# 编译完成后生成可执行文件：benchmark_gemm
```

编译输出：
```
nvcc -O3 -c src/main.cu -o src/main.o
nvcc -O3 -c src/sgemm_cublas.cu -o src/sgemm_cublas.o
nvcc -O3 -c src/sgemm_naive.cu -o src/sgemm_naive.o
nvcc -O3 -c src/sgemm_shared.cu -o src/sgemm_shared.o
nvcc -O3 -c src/sgemm_register.cu -o src/sgemm_register.o
nvcc -O3 -c src/sgemm_register_v2.cu -o src/sgemm_register_v2.o
nvcc -O3 src/main.o src/sgemm_cublas.o src/sgemm_naive.o src/sgemm_shared.o src/sgemm_register.o src/sgemm_register_v2.o -o benchmark_gemm -lcublas
```

### 3. 运行测试

```bash
# 运行基准测试
./benchmark_gemm
```

预期输出：
```
Starting GEMM Benchmark...
Matrix Size: M=4096, N=4096, K=4096
--------------------------------------------------------
Running: cuBLAS SGEMM (Reference / Upper Bound)
  [Performance] Avg Time: 2.061 ms
  [Performance] Throughput: 66.687 TFLOPs
--------------------------------------------------------
Running: SGEMM_Naive
  [Correctness] Pass! (Max error: 0.000)
  [Performance] Avg Time: 18.380 ms
  [Performance] Throughput: 7.477 TFLOPs
--------------------------------------------------------
Running: SGEMM_SharedMemory
  [Correctness] Pass! (Max error: 0.000)
  [Performance] Avg Time: 15.051 ms
  [Performance] Throughput: 9.132 TFLOPs
--------------------------------------------------------
Running: SGEMM_Register
  [Correctness] Pass! (Max error: 0.000)
  [Performance] Avg Time: XX.XXX ms
  [Performance] Throughput: XX.XXX TFLOPs
--------------------------------------------------------
Running: SGEMM_RegisterTiling_V2
  [Correctness] Pass! (Max error: 0.000)
  [Performance] Avg Time: XX.XXX ms
  [Performance] Throughput: XX.XXX TFLOPs
--------------------------------------------------------
```

### 4. 清理编译产物

```bash
make clean
```

## 📊 性能对比

在 RTX 5090 (104.9 TFLOPS 峰值) 上的测试结果：

| 算子 | 性能 (TFLOPS) | 利用率 | vs Naive | vs cuBLAS |
|------|---------------|--------|----------|-----------|
| **cuBLAS** | 66.7 | 63.6% | 8.9× | 100% |
| **Register V2** | 35-55 (预估) | 35-55% | 5-7× | 50-80% |
| **Register** | 30-50 (预估) | 30-50% | 4-6× | 45-75% |
| **Shared** | 9.1 | 8.7% | 1.2× | 13.7% |
| **Naive** | 7.5 | 7.1% | 1× | 11.2% |

### 关键优化技术对比

| 技术 | 解决的问题 | 效果 |
|------|-----------|------|
| **共享内存 Tiling** | 减少全局内存访问 | AI 从 0.5 → 683 |
| **寄存器分块** | 减少共享内存访问 | 每轮计算 64 FMA |
| **外积计算** | 提高指令级并行 | 8× 计算强度 |
| **向量化加载 (float4)** | 提高带宽利用率 | 128-bit 协作加载 |
| **Shared Memory Padding** | 消除 Bank Conflict | 无冲突访存 |

## 📖 学习路径

建议按以下顺序学习：

### 阶段 1: 基础概念
- 阅读 `docs/cuda_thread_hierarchy.md` 了解 CUDA 线程组织
- 理解 `dim3`, `blockIdx`, `threadIdx` 的含义
- 理解 SM (Streaming Multiprocessor) 架构

### 阶段 2: 朴素实现
- 阅读 `src/sgemm_naive.cu` 了解最基础的矩阵乘法实现
- 理解为什么性能低（频繁全局内存访问）

### 阶段 3: 共享内存优化
- 阅读 `src/sgemm_shared.cu` 和 `docs/sgemm_shared_kernel_explained.md`
- 理解 Shared Memory Tiling 原理
- 查看 `images/shared_gemm_*.png` 图解

### 阶段 4: 性能分析
- 阅读 `docs/roofline_analysis.md`
- 运行 `scripts/generate_roofline.py` 生成图表
- 理解 Arithmetic Intensity 概念

### 阶段 5: 寄存器优化
- 阅读 `src/sgemm_register.cu` 和 `docs/sgemm_register_code_explanation.md`
- 理解双层分块策略
- 查看 `images/register_*.png` 图解
- 阅读 `docs/sgemm_register_analysis.md` 了解性能对比

### 阶段 5.5: 进阶优化技巧
- 阅读 `src/sgemm_register_v2.cu` 了解进一步优化
- 学习 **float4 向量化加载**：128-bit 协作访存，提高带宽利用率
- 学习 **Shared Memory Padding**：通过 +4 Padding 消除 Bank Conflict
- 对比 V1 和 V2 版本，理解微架构优化的重要性

### 阶段 6: 硬件深入
- 阅读 `docs/rtx5090_hardware_constraints.md`
- 理解寄存器限制、共享内存限制、Bank Conflict
- 完成 `exercises/occupancy_calculation_exercises.md` 练习题

## 🔧 高级功能

### 生成可视化图表

```bash
# 生成 Roofline 图
python3 scripts/generate_roofline.py

# 生成 SGEMM 执行流程图
python3 scripts/visualize_shared_gemm.py
```

图表将保存到 `images/` 目录。

### 自定义矩阵大小

编辑 `src/main.cu` 中的矩阵大小参数：

```cpp
const int M = 4096;  // 修改为你需要的大小
const int N = 4096;
const int K = 4096;
```

重新编译运行：
```bash
make clean && make && ./benchmark_gemm
```

## 🎯 练习题

项目包含一套完整的 Occupancy 计算练习题，帮助你深入理解 GPU 资源限制：

```bash
# 查看练习题
less exercises/occupancy_calculation_exercises.md
```

**练习题内容**：
- 题目 1-3: 基础寄存器和共享内存计算
- 题目 4-5: 线程数限制和 GEMM 场景分析
- 题目 6-7: Warp 调度和动态分区
- 题目 8: 设计最优 kernel 配置

**学习目标**：
- 熟练计算 Occupancy
- 识别性能瓶颈
- 设计合理的 kernel 配置

## 📝 文档规范

### 目录结构
- `src/`：源代码文件（.cu, .h）
- `docs/`：技术文档（性能分析、代码解读、硬件约束）
- `exercises/`：练习题和计算练习
- `images/`：可视化图表（Roofline 图、执行流程图）
- `scripts/`：辅助脚本（图表生成）

### 文件命名规范
- `sgemm_<优化策略>.cu`：SGEMM 实现文件
- `<主题>_analysis.md`：性能分析文档
- `<主题>_explanation.md`：代码解读文档
- `<主题>_exercises.md`：练习题文档

### 宏定义规范
```cpp
#define BM 128  // Block M 维度大小 (Block M)
#define BN 128  // Block N 维度大小 (Block N)
#define BK 8    // Block K 维度步长 (Block K)
#define TM 8    // Thread M 维度负责大小 (Thread M)
#define TN 8    // Thread N 维度负责大小 (Thread N)
```

## 🐛 常见问题

### Q: 编译失败，找不到 `nvcc`
```bash
# 添加 CUDA 到 PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: 运行时找不到 libcublas.so
```bash
# 链接 cuBLAS 库路径
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: 性能与预期不符
- 检查 GPU 是否处于高功耗模式：`nvidia-smi -q -d POWER`
- 确保 GPU 未被其他进程占用
- 矩阵大小建议 ≥ 2048 才能发挥 GPU 并行性

## 📚 参考资料

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA 官方高效 GEMM 实现
- [Roofline Model Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/roofline.pdf)
- [NVIDIA A100/RTX 架构白皮书](https://www.nvidia.com/en-us/data-center/a100/)

## 📊 文档索引

### 分析文档
| 文档 | 内容 |
|------|------|
| `docs/roofline_analysis.md` | Roofline 模型与 Arithmetic Intensity 分析 |
| `docs/sgemm_register_analysis.md` | Register vs Shared Kernel 性能对比 |
| `docs/rtx5090_hardware_constraints.md` | RTX 5090 硬件约束详解 |

### 代码解读
| 文档 | 内容 |
|------|------|
| `docs/sgemm_shared_kernel_explained.md` | Shared Memory Kernel 详解 |
| `docs/sgemm_register_code_explanation.md` | Register Kernel 逐行解读 |
| `docs/cuda_thread_hierarchy.md` | CUDA 线程层次与 SM 架构 |

### 练习题
| 文档 | 内容 |
|------|------|
| `exercises/occupancy_calculation_exercises.md` | 8 道 Occupancy 计算练习题 |

## 🤝 贡献

欢迎提交 Issue 和 PR！

可能的改进方向：
- 添加更多优化策略（Warp Tiling、Double Buffering、Tensor Core）
- 支持更多数据类型（FP16、BF16）
- 添加性能 profiling 工具支持
- 多 GPU 支持
- 更多练习题（Bank Conflict、Warp Divergence）

## 📄 许可证

MIT License - 自由使用和学习

---

**作者**: CUDA GEMM 学习项目
**创建时间**: 2026年3月

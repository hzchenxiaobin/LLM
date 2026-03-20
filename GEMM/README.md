# CUDA SGEMM 优化教程项目

本项目是一个用于学习和实践 CUDA 矩阵乘法（SGEMM）优化的教程代码库，从朴素实现逐步优化到高性能实现，展示了 GPU 编程的核心优化技术。

## 项目简介

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
| **cuBLAS** | `src/gemm/sgemm_cublas.cu` | NVIDIA 官方高度优化库 | 60+ TFLOPS |
| **Naive** | `src/gemm/sgemm_naive.cu` | 无优化，直接实现 | ~7 TFLOPS |
| **Shared Memory** | `src/gemm/sgemm_shared.cu` | 共享内存分块 (Tiling) | ~9 TFLOPS |
| **Register** | `src/gemm/sgemm_register.cu` | 寄存器分块 + 双层 Tiling | ~30-50 TFLOPS |
| **Register V2** | `src/gemm/sgemm_register_v2.cu` | 向量化加载 + Padding 优化 | ~35-55 TFLOPS |
| **Register V3** | `src/gemm/sgemm_register_v3.cu` | 双缓冲 (Double Buffering) | ~40-60 TFLOPS |
| **Register Bank Conflict** | `src/gemm/sgemm_register_bank_conflict.cu` | Shared Memory Padding 消除 Bank Conflict | ~35-55 TFLOPS |
| **WMMA** | `src/gemm/sgemm_wmma.cu` | Tensor Core WMMA API | ~40-70 TFLOPS |
| **WMMA V2** | `src/gemm/sgemm_wmma_v2.cu` | Tensor Core 优化版本 | ~45-75 TFLOPS |
| **CUTLASS SGEMM** | `src/gemm/sgemm_cutlass.cu` | NVIDIA CUTLASS 设备级 GEMM（需本地 CUTLASS 头文件）| 视配置与 GPU 而定 |
| **CuTe SGEMM** | `src/gemm/sgemm_cute.cu` | CuTe (CUTLASS 3.x DSL) 现代张量编程实现 | 视配置与 GPU 而定 |

### 硬件要求

- NVIDIA GPU（支持 CUDA，推荐 Compute Capability 7.0+）
- CUDA Toolkit 11.0+
- 测试通过硬件：RTX 3060 / RTX 4090

## 项目结构

```
GEMM/
├── README.md                    # 本文件
├── Makefile                     # 编译脚本
│
├── src/                         # 源代码目录
│   ├── main.cu                  # 测试框架主程序
│   ├── common.h                 # 公共头文件（CUDA 错误检查宏）
│   ├── gemm_kernels.h           # 统一头文件入口
│   └── gemm/                    # 单矩阵 GEMM 实现目录
│       ├── gemm_kernels.h       # Kernel 函数声明
│       ├── sgemm_naive.cu       # 朴素实现
│       ├── sgemm_shared.cu      # 共享内存优化实现
│       ├── sgemm_register.cu    # 寄存器分块优化实现
│       ├── sgemm_register_v2.cu # 优化版寄存器分块 (向量化 + Padding)
│       ├── sgemm_register_v3.cu # 双缓冲优化版 (Double Buffering)
│       ├── sgemm_register_bank_conflict.cu  # Bank Conflict 消除优化
│       ├── sgemm_wmma.cu        # Tensor Core WMMA 实现
│       ├── sgemm_wmma_v2.cu     # Tensor Core WMMA 优化版
│       ├── sgemm_cutlass.cu     # CUTLASS SGEMM（可选）
│       ├── sgemm_cute.cu        # CuTe SGEMM（可选）
│       └── sgemm_cublas.cu      # cuBLAS 参考实现
│
├── docs/                        # 技术文档目录
│   ├── README.md                        # 项目文档入口
│   ├── roofline_analysis.md             # Roofline 模型性能分析
│   ├── sgemm_shared_kernel_explained.md # Shared Kernel 详解
│   ├── sgemm_register_analysis.md        # Register Kernel 性能对比
│   ├── sgemm_register_code_explanation.md # Register Kernel 逐行解读
│   ├── sgemm_register_v2_optimization.md # V2 向量化优化详解
│   ├── bank_conflict_analysis.md         # Bank Conflict 深度解析
│   ├── rtx5090_hardware_constraints.md   # RTX 5090 硬件约束分析
│   ├── cuda_thread_hierarchy.md          # CUDA 线程层次说明
│   ├── cutlass_build.md                  # CUTLASS 依赖与编译说明
│   └── cute_build.md                     # CuTe 依赖与编译说明
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

## 快速开始

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
# 进入项目目录
cd GEMM

# 编译所有算子
make

# 编译完成后生成可执行文件：benchmark_gemm
```

### 3. 运行测试

```bash
# 运行基准测试
./benchmark_gemm
```

预期输出：
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

### 4. 可选：启用 CUTLASS 支持

如需测试 CUTLASS/CuTe 实现，先克隆 CUTLASS：

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
make clean && make
```

### 5. 清理编译产物

```bash
make clean
```

## 性能对比

在 RTX 4090 (82.58 TFLOPS 理论峰值) 上的典型测试结果：

| 算子 | 性能 (TFLOPS) | 利用率 | vs Naive | vs cuBLAS |
|------|---------------|--------|----------|-----------|
| **cuBLAS** | 58-65 | 70-79% | 8-9× | 100% |
| **WMMA V2** | 45-55 | 55-67% | 6-7× | 78-85% |
| **Register V3** | 40-50 | 48-61% | 5-6× | 69-77% |
| **Register V2** | 35-45 | 42-55% | 4-5× | 60-69% |
| **Register Bank Conflict** | 35-45 | 42-55% | 4-5× | 60-69% |
| **Register** | 30-40 | 36-48% | 4-5× | 52-62% |
| **Shared** | 9-10 | 11-12% | 1.3× | 15-17% |
| **Naive** | 7-8 | 8-10% | 1× | 12-14% |

### 关键优化技术对比

| 技术 | 解决的问题 | 效果 |
|------|-----------|------|
| **共享内存 Tiling** | 减少全局内存访问 | AI 从 0.5 → 683 |
| **寄存器分块** | 减少共享内存访问 | 每轮计算 64 FMA |
| **外积计算** | 提高指令级并行 | 8× 计算强度 |
| **向量化加载 (float4)** | 提高带宽利用率 | 128-bit 协作加载 |
| **Shared Memory Padding** | 消除 Bank Conflict | 无冲突访存 |
| **双缓冲 (Double Buffering)** | 隐藏数据加载延迟 | 计算与访存重叠 |
| **Tensor Core WMMA** | 利用专用矩阵计算单元 | FP16/FP32 混合精度加速 |

## 学习路径

建议按以下顺序学习：

### 阶段 1: 基础概念
- 阅读 `docs/cuda_thread_hierarchy.md` 了解 CUDA 线程组织
- 理解 `dim3`, `blockIdx`, `threadIdx` 的含义
- 理解 SM (Streaming Multiprocessor) 架构

### 阶段 2: 朴素实现
- 阅读 `src/gemm/sgemm_naive.cu` 了解最基础的矩阵乘法实现
- 理解为什么性能低（频繁全局内存访问）

### 阶段 3: 共享内存优化
- 阅读 `src/gemm/sgemm_shared.cu` 和 `docs/sgemm_shared_kernel_explained.md`
- 理解 Shared Memory Tiling 原理
- 查看 `images/shared_gemm_*.png` 图解

### 阶段 4: 性能分析
- 阅读 `docs/roofline_analysis.md`
- 运行 `scripts/generate_roofline.py` 生成图表
- 理解 Arithmetic Intensity 概念

### 阶段 5: 寄存器优化
- 阅读 `src/gemm/sgemm_register.cu` 和 `docs/sgemm_register_code_explanation.md`
- 理解双层分块策略
- 查看 `images/register_*.png` 图解
- 阅读 `docs/sgemm_register_analysis.md` 了解性能对比

### 阶段 5.5: 进阶优化技巧
- 阅读 `src/gemm/sgemm_register_v2.cu` 了解向量化优化
- 学习 **float4 向量化加载**：128-bit 协作访存，提高带宽利用率
- 学习 **Shared Memory Padding**：通过 +1 Padding 消除 Bank Conflict
- 阅读 `src/gemm/sgemm_register_bank_conflict.cu` 和 `docs/bank_conflict_analysis.md`
- 理解 Bank Conflict 的硬件原理：`bank = (address / 4) % 32`

### 阶段 5.6: 双缓冲优化
- 阅读 `src/gemm/sgemm_register_v3.cu` 了解双缓冲实现
- 学习 **Double Buffering**：通过两组共享内存实现计算与访存重叠
- 理解软件流水线技术，最大化 GPU 利用率

### 阶段 6: Tensor Core
- 阅读 `src/gemm/sgemm_wmma.cu` 和 `src/gemm/sgemm_wmma_v2.cu`
- 学习 WMMA API 使用
- 理解 FP16 输入 + FP32 累加的混合精度计算
- 对比 CUDA Core 和 Tensor Core 的性能差异

### 阶段 7: 硬件深入
- 阅读 `docs/rtx5090_hardware_constraints.md`
- 理解寄存器限制、共享内存限制、Warp 调度
- 完成 `exercises/occupancy_calculation_exercises.md` 练习题

## 高级功能

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
int M = 4096;  // 修改为你需要的大小
int N = 4096;
int K = 4096;
```

重新编译运行：
```bash
make clean && make && ./benchmark_gemm
```

## 练习题

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

## 常见问题

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

## 参考资料

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA 官方高效 GEMM 实现
- [Roofline Model Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/roofline.pdf)
- [NVIDIA Ampere 架构白皮书](https://www.nvidia.com/en-us/data-center/ampere-architecture/)

## 文档索引

### 分析文档
| 文档 | 内容 |
|------|------|
| `docs/roofline_analysis.md` | Roofline 模型与 Arithmetic Intensity 分析 |
| `docs/sgemm_register_analysis.md` | Register vs Shared Kernel 性能对比 |
| `docs/rtx5090_hardware_constraints.md` | RTX 5090 硬件约束详解 |
| `docs/bank_conflict_analysis.md` | Bank Conflict 深度解析 |

### 代码解读
| 文档 | 内容 |
|------|------|
| `docs/sgemm_shared_kernel_explained.md` | Shared Memory Kernel 详解 |
| `docs/sgemm_register_code_explanation.md` | Register Kernel 逐行解读 |
| `docs/sgemm_register_v2_optimization.md` | V2 向量化优化详解 |
| `docs/cuda_thread_hierarchy.md` | CUDA 线程层次与 SM 架构 |

### 练习题
| 文档 | 内容 |
|------|------|
| `exercises/occupancy_calculation_exercises.md` | 8 道 Occupancy 计算练习题 |

## 贡献

欢迎提交 Issue 和 PR！

可能的改进方向：
- 支持更多数据类型（FP16、BF16）
- 添加性能 profiling 工具支持（Nsight Compute）
- 多 GPU 支持
- 更多练习题（Warp Divergence、Coalesced Access）

## 许可证

MIT License - 自由使用和学习

---

**作者**: CUDA GEMM 学习项目  
**创建时间**: 2026年3月

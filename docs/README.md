# CUDA GEMM 性能优化完整教程（面向 RTX 5090）

> **目标读者**：希望深入理解 CUDA 矩阵乘法优化、从入门到精通的开发者
> 
> **目标平台**：NVIDIA RTX 5090 (Blackwell 架构，Compute Capability 10.0)
> 
> **前置知识**：基础 CUDA 编程经验，了解 C/C++ 和矩阵运算

---

## 教程体系结构

本教程采用循序渐进的方式，从基础概念到高级优化技术，分为四个阶段：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CUDA GEMM 优化完整学习路径                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐  │
│  │  第一阶段        │ →  │  第二阶段        │ →  │  第三阶段            │  │
│  │  CUDA基础       │    │  GEMM优化阶梯   │    │  进阶优化技术        │  │
│  │                 │    │                 │    │                     │  │
│  │ • 线程层级       │    │ • Naive实现     │    │ • 向量化访存         │  │
│  │ • 内存体系       │    │ • Shared Memory │    │ • Bank Conflict消除 │  │
│  │ • 硬件约束       │    │ • Register Tiling│   │ • Tensor Cores      │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘  │
│           ↓                                                    ↓        │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    第四阶段：性能分析与调试                    │      │
│  │                                                               │      │
│  │ • Roofline模型分析  • Nsight Compute使用  • Occupancy优化     │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 学习路径推荐

### 🔰 初学者路径（预计 2-3 周）

如果你是 CUDA GEMM 优化的新手，按以下顺序学习：

| 顺序 | 文档 | 预计时间 | 目标 |
|:---:|:---|:---:|:---|
| 1 | [01_cuda_fundamentals.md](01_cuda_fundamentals.md) | 2-3 天 | 理解 CUDA 线程层级和内存体系 |
| 2 | [02_hardware_architecture.md](02_hardware_architecture.md) | 2-3 天 | 了解 RTX 5090 硬件约束 |
| 3 | [03_gemm_naive.md](03_gemm_naive.md) | 1-2 天 | 实现基础 GEMM，理解瓶颈 |
| 4 | [04_shared_memory_tiling.md](04_shared_memory_tiling.md) | 3-4 天 | 掌握 Shared Memory 分块技术 |
| 5 | [05_register_tiling.md](05_register_tiling.md) | 3-4 天 | 理解寄存器分块原理 |

### 🔥 进阶优化路径（预计 1-2 周）

掌握基础后，学习高级优化技术：

| 顺序 | 文档 | 预计时间 | 目标 |
|:---:|:---|:---:|:---|
| 6 | [06_vectorization_and_bank_conflict.md](06_vectorization_and_bank_conflict.md) | 3-4 天 | 向量化访存和 Bank Conflict 消除 |
| 7 | [07_tensor_cores.md](07_tensor_cores.md) | 2-3 天 | 使用 Tensor Cores 极致加速 |
| 8 | [08_performance_analysis.md](08_performance_analysis.md) | 2-3 天 | 掌握 Roofline 和 Profiling |

---

## 文档索引

### 📚 第一部分：CUDA 基础与硬件架构

| 文档 | 内容概述 | 关键知识点 |
|:---|:---|:---|
| [01_cuda_fundamentals.md](01_cuda_fundamentals.md) | CUDA 编程基础 | Grid/Block/Warp 层级、线程索引、同步机制 |
| [02_hardware_architecture.md](02_hardware_architecture.md) | RTX 5090 硬件详解 | SM 架构、寄存器文件、共享内存、Tensor Cores |

### 📚 第二部分：GEMM 优化阶梯

| 文档 | 内容概述 | 关键知识点 | 预期性能 |
|:---|:---|:---|:---:|
| [03_gemm_naive.md](03_gemm_naive.md) | 朴素实现 | 基础矩阵乘法、全局内存访问问题 | ~7 TFLOPS |
| [04_shared_memory_tiling.md](04_shared_memory_tiling.md) | 共享内存分块 | Tiling 技术、数据复用、同步机制 | ~9 TFLOPS |
| [05_register_tiling.md](05_register_tiling.md) | 寄存器分块 | 双层分块、寄存器缓存、外积计算 | ~30-50 TFLOPS |

### 📚 第三部分：进阶优化技术

| 文档 | 内容概述 | 关键知识点 | 预期性能 |
|:---|:---|:---|:---:|
| [06_vectorization_and_bank_conflict.md](06_vectorization_and_bank_conflict.md) | 向量化和 Bank Conflict 消除 | float4、Padding、Swizzling | ~40-50 TFLOPS |
| [07_tensor_cores.md](07_tensor_cores.md) | Tensor Core 优化 | WMMA API、混合精度、Warp 级优化 | 200+ TFLOPS |

### 📚 第四部分：性能分析与调试

| 文档 | 内容概述 | 关键知识点 |
|:---|:---|:---|
| [08_performance_analysis.md](08_performance_analysis.md) | 性能分析方法论 | Roofline 模型、Arithmetic Intensity、瓶颈识别 |
| [09_profiling_tools.md](09_profiling_tools.md) | 调试与 Profiling | Nsight Compute、Occupancy 分析、寄存器优化 |

### 📚 扩展专题

| 文档 | 内容概述 | 适用场景 |
|:---|:---|:---|
| [10_batched_gemm.md](10_batched_gemm.md) | 批量矩阵乘法优化 | 多矩阵批量计算 |
| [11_cutlass_integration.md](11_cutlass_integration.md) | CUTLASS 框架入门 | 工业级算子开发 |

---

## 核心概念速查

### 关键性能指标

| 指标 | 定义 | 优化目标 |
|:---|:---|:---|
| **Arithmetic Intensity** | FLOPs / 访存字节数 | > 58.5 (RTX 5090 Ridge Point) |
| **Occupancy** | 实际 Warp 数 / 最大 Warp 数 | 25%-50% (GEMM 最佳) |
| **Memory Coalescing** | 线程访问连续内存地址 | 最大化带宽利用率 |
| **Bank Conflict** | 多线程同时访问同一 Bank | 消除 (Padding/Swizzling) |

### 硬件规格 (RTX 5090)

| 参数 | 数值 | 对 GEMM 的影响 |
|:---|:---:|:---|
| FP32 峰值算力 | 104.9 TFLOPS | 优化目标上限 |
| 显存带宽 | 1,792 GB/s | 内存瓶颈约束 |
| 共享内存带宽 | ~10 TB/s/SM | 缓存优化关键 |
| Ridge Point | 58.5 FLOPs/byte | 计算/访存分界 |
| Tensor Core (FP16) | 835.3 TFLOPS | 极致性能来源 |

---

## 配套代码

本教程配套代码位于项目各目录：

```
GEMM/
├── src/gemm/
│   ├── sgemm_naive.cu              # V0: 朴素实现
│   ├── sgemm_shared.cu             # V1: Shared Memory Tiling
│   ├── sgemm_register.cu           # V2: 寄存器分块基础
│   ├── sgemm_register_v2.cu        # V2+: 向量化加载
│   ├── sgemm_register_bank_conflict.cu  # V2+: Bank Conflict 消除
│   ├── sgemm_register_vec_bank.cu  # V3: 综合优化
│   └── sgemm_cublas.cu             # 参考实现
├── docs/                           # 教程文档（本目录）
└── Makefile                        # 编译配置

batch_gemm/
└── src/                            # 批量 GEMM 实现
```

---

## 学习建议

### 1. 动手实践

每学完一个优化阶段，务必：
- 自己编写对应版本的 Kernel
- 编译运行，验证正确性
- 对比性能数据，理解优化效果

```bash
# 编译命令
cd GEMM
make clean && make

# 运行测试
./build/benchmark_gemm
```

### 2. 使用 Nsight Compute

```bash
# 性能分析
ncu -o profile_report.ncu-rep ./build/benchmark_gemm

# 关键指标
# - sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio
# - sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict
# - sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
```

### 3. 循序渐进

不要急于使用 Tensor Cores，先确保：
1. 理解为什么 Naive 实现慢
2. 掌握 Shared Memory Tiling 原理
3. 能手动实现 Register Tiling
4. 再追求极致的 Tensor Core 性能

---

## 参考资源

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

---

*教程更新日期：2026年3月*
*适用硬件：NVIDIA GeForce RTX 5090 (Blackwell 架构)*

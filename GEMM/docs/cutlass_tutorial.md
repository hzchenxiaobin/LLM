# NVIDIA CUTLASS 完整教程

## 目录

1. [CUTLASS 简介](#1-cutlass-简介)
2. [核心概念](#2-核心概念)
3. [代码实战：从 sgemm_cutlass.cu 开始](#3-代码实战从-sgemm_cutlasscu-开始)
4. [深入理解 Gemm 模板参数](#4-深入理解-gemm-模板参数)
5. [不同数据类型与布局](#5-不同数据类型与布局)
6. [性能优化技巧](#6-性能优化技巧)
7. [常见问题与调试](#7-常见问题与调试)
8. [参考资料](#8-参考资料)

---

## 1. CUTLASS 简介

### 什么是 CUTLASS？

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) 是 NVIDIA 开源的高性能线性代数库，提供了接近 cuBLAS 性能的手写 CUDA kernel 模板。

### 为什么选择 CUTLASS？

| 特性 | CUDA 手写 | CUTLASS | cuBLAS |
|-----|-----------|---------|--------|
| 性能 | 取决于优化水平 | ~95-100% cuBLAS | 100% (基准) |
| 开发时间 | 数周-数月 | 数小时 | 直接调用 |
| 灵活性 | 完全可控 | 高（模板参数） | 低（固定 API） |
| 代码量 | 数百-数千行 | 数十行 | 1 行 |
| 可维护性 | 低 | 中 | 高 |

### CUTLASS 架构层次

```
┌─────────────────────────────────────────────────────────┐
│  Application (你的代码)                                  │
├─────────────────────────────────────────────────────────┤
│  Device API (cutlass::gemm::device::Gemm)               │
│  - 高层封装，直接可用                                    │
├─────────────────────────────────────────────────────────┤
│  Kernel API (cutlass::gemm::kernel::Gemm)               │
│  - 控制 grid/block 映射                                  │
├─────────────────────────────────────────────────────────┤
│  Threadblock API (Mma, Epilogue)                        │
│  - Warp-level 和 Thread-level 原语                       │
├─────────────────────────────────────────────────────────┤
│  Core API (Copy, Mma, Reduction)                        │
│  - 底层 CUDA 原语封装                                    │
├─────────────────────────────────────────────────────────┤
│  CUDA / PTX Instructions                                │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 核心概念

### 2.1 三层并行结构

CUTLASS 将 GEMM 分解为三个层次的并行：

```
┌────────────────────────────────────────────────┐
│  Device-level (整个 GPU)                       │
│  - 多个 CTA (Cooperative Thread Array)          │
│  - 每个 CTA 计算 C 的一个大 tile               │
├────────────────────────────────────────────────┤
│  Threadblock-level (单个 SM)                     │
│  - 一个 CTA 内的多个 Warp                       │
│  - Warp 级 MMA (Matrix Multiply Accumulate)   │
├────────────────────────────────────────────────┤
│  Warp-level (32 线程)                           │
│  - Tensor Core 指令                            │
│  - WMMA 或 mma.sync                            │
└────────────────────────────────────────────────┘
```

### 2.2 数据布局 (Layout)

```cpp
// Column-Major (BLAS 标准)：按列存储
// 地址 = row + column * leading_dimension

// Row-Major (C/C++ 默认)：按行存储
// 地址 = row * leading_dimension + column
//                      K                    K
//                 ┌──────────┐       ┌──────────┐
//                 │ 0  1  2  │       │ 0  3  6  │
//              M  │ 3  4  5  │    M  │ 1  4  7  │
//                 │ 6  7  8  │       │ 2  5  8  │
//                 └──────────┘       └──────────┘
//                  Row-Major         Column-Major
```

### 2.3 关键模板类

| 类名 | 作用 |
|-----|------|
| `cutlass::gemm::device::Gemm` | 设备级 GEMM，最常用 |
| `cutlass::gemm::kernel::Gemm` | Kernel 级封装 |
| `cutlass::gemm::threadblock::Mma` | Threadblock 级 MMA |
| `cutlass::layout::RowMajor` | 行主序布局 |
| `cutlass::layout::ColumnMajor` | 列主序布局 |
| `cutlass::arch::Sm70/Sm80/Sm90` | GPU 架构标签 |

---

## 3. 代码实战：从 sgemm_cutlass.cu 开始

### 3.1 完整代码解析

```cpp
// CUTLASS SGEMM: C = alpha * A * B + beta * C
// 矩阵按行主序 (Row-Major) 存储，与项目其余实现一致：
//   A: M×K, lda = K
//   B: K×N, ldb = N
//   C: M×N, ldc = N
//
// 依赖：NVIDIA CUTLASS 头文件（见 GEMM/docs/cutlass_build.md）

#include "common.h"
#include "gemm_kernels.h"

#include <iostream>

// 核心头文件：设备级 GEMM API
#include "cutlass/gemm/device/gemm.h"

void run_sgemm_cutlass(int M, int N, int K, float alpha, const float *A, const float *B,
                       float beta, float *C) {
```

**头文件说明：**
- `"cutlass/gemm/device/gemm.h"`: 设备级 GEMM，最简单的使用方式
- 其他常用头文件：
  - `cutlass/gemm/kernel/gemm.h`: Kernel 级 API
  - `cutlass/gemm/threadblock/default_mma.h`: Threadblock 级配置

#### 步骤 1：定义 Gemm 类型

```cpp
    // FP32 + CUDA Core (SIMT)，默认 ArchTag=Sm70，与 nvcc -arch=sm_70 兼容
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,  // A: 数据类型, 布局
        float, cutlass::layout::RowMajor,  // B: 数据类型, 布局
        float, cutlass::layout::RowMajor,  // C: 数据类型, 布局
        float>;                            // 累加器数据类型
```

**模板参数解析：**

| 位置 | 参数 | 含义 | 可选值 |
|-----|------|------|--------|
| 1 | `float` | A 矩阵元素类型 | `float`, `half`, `bfloat16`, `int8`, ... |
| 2 | `RowMajor` | A 矩阵布局 | `RowMajor`, `ColumnMajor` |
| 3 | `float` | B 矩阵元素类型 | 同上 |
| 4 | `RowMajor` | B 矩阵布局 | 同上 |
| 5 | `float` | C 矩阵元素类型 | 同上 |
| 6 | `RowMajor` | C 矩阵布局 | 同上 |
| 7 | `float` | 累加器类型 | 通常与 C 相同或更高精度 |

#### 步骤 2：创建 Gemm 操作对象

```cpp
    Gemm gemm_op;
```

这一步非常简单，CUTLASS 会自动根据模板参数选择合适的 kernel 实现。

#### 步骤 3：构造参数并执行

```cpp
    typename Gemm::Arguments args(
        {M, N, K},           // 问题规模：M, N, K
        {A, K},              // A 指针和 leading dimension (lda = K)
        {B, N},              // B 指针和 leading dimension (ldb = N)
        {C, N},              // C 源指针和 leading dimension (ldc = N)
        {C, N},              // C 目标指针和 leading dimension
        {alpha, beta});      // 缩放系数

    cutlass::Status status = gemm_op(args);
```

**Arguments 详解：**

```cpp
// Arguments 构造函数签名
Arguments(
    GemmCoord problem_size,           // {M, N, K}
    TensorRef<ElementA, LayoutA> ref_A,  // {ptr, ld}
    TensorRef<ElementB, LayoutB> ref_B,  // {ptr, ld}
    TensorRef<ElementC, LayoutC> ref_C,  // {ptr, ld} - 源 C
    TensorRef<ElementC, LayoutC> ref_D,  // {ptr, ld} - 目标 D (通常就是 C)
    EpilogueOutputOp::Params epilogue    // {alpha, beta}
);
```

#### 步骤 4：错误处理

```cpp
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed with status: " << static_cast<int>(status) << std::endl;
    }
}
```

**常见错误状态：**
- `kSuccess`: 成功
- `kErrorMisalignedOperand`: 指针未对齐
- `kErrorInvalidProblem`: 问题规模无效
- `kErrorNotSupported`: 配置不支持
- `kErrorInternal`: 内部错误

### 3.2 代码执行流程图

```
┌────────────────────────────────────────────────┐
│ 1. 模板实例化                                  │
│    Gemm<float, RowMajor, ...>                  │
│    ↓ 编译期：CUTLASS 自动选择最佳配置          │
├────────────────────────────────────────────────┤
│ 2. 创建操作对象                                │
│    Gemm gemm_op;                               │
│    ↓ 初始化内部状态                             │
├────────────────────────────────────────────────┤
│ 3. 构造参数                                     │
│    Arguments args(...);                        │
│    ↓ 封装指针、步长、系数                       │
├────────────────────────────────────────────────┤
│ 4. 执行 GEMM                                   │
│    gemm_op(args);                              │
│    ↓ 启动 CUDA kernel                          │
├────────────────────────────────────────────────┤
│ 5. 返回状态                                     │
│    Status::kSuccess                            │
└────────────────────────────────────────────────┘
```

---

## 4. 深入理解 Gemm 模板参数

### 4.1 完整模板参数（高级）

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,    // A 矩阵
    ElementB, LayoutB,    // B 矩阵
    ElementC, LayoutC,    // C 矩阵
    ElementAccumulator,   // 累加器
    OperatorClass,        // 操作类型：Simt/TensorOp
    ArchTag,              // GPU 架构
    ThreadblockShape,     // CTA Tile 大小
    WarpShape,            // Warp Tile 大小
    InstructionShape,     // 指令 Tile 大小
    EpilogueOutputOp,     // 尾处理操作
    ConvertOp,            // 类型转换操作
    ReduceOp              // 归约操作
>;
```

### 4.2 示例：显式指定所有参数

```cpp
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"

using Gemm = cutlass::gemm::device::Gemm<
    float,                                      // ElementA
    cutlass::layout::RowMajor,                  // LayoutA
    float,                                      // ElementB
    cutlass::layout::RowMajor,                  // LayoutB
    float,                                      // ElementC
    cutlass::layout::RowMajor,                  // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassSimt,                 // OperatorClass: SIMT (CUDA Core)
    cutlass::arch::Sm80,                        // ArchTag: Ampere
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<1, 1, 1>,          // InstructionShape (SIMT)
    cutlass::epilogue::thread::LinearCombination<
        float, 1, float, float                  // Epilogue: C = alpha * acc + beta * C
    >
>;
```

### 4.3 ThreadblockShape 配置指南

```cpp
// ThreadblockShape< M, N, K >: 一个 CTA 处理的 tile 大小

// 小矩阵 (< 1K)：使用小 tile，增加并行度
using SmallShape = cutlass::gemm::GemmShape<64, 64, 32>;

// 中等矩阵 (1K-4K)：平衡配置
using MediumShape = cutlass::gemm::GemmShape<128, 128, 32>;

// 大矩阵 (> 4K)：使用大 tile，提高数据复用
using LargeShape = cutlass::gemm::GemmShape<256, 128, 64>;
```

**配置原则：**
- **M, N**: 越大，shared memory 用量越多，但数据复用越好
- **K**: 决定每个 tile 内的累加次数
- **总 SMEM**: `(M * K + K * N) * sizeof(Element)` 必须小于 GPU SMEM 上限

---

## 5. 不同数据类型与布局

### 5.1 混合精度 GEMM (FP16 + FP32)

```cpp
// A/B 用 FP16，累加用 FP32，输出 FP16
using GemmFP16 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,   // A: FP16
    cutlass::half_t, cutlass::layout::RowMajor,   // B: FP16
    cutlass::half_t, cutlass::layout::RowMajor,   // C: FP16
    float                                         // 累加器: FP32
>;

void run_gemm_fp16(int M, int N, int K,
                   float alpha,
                   cutlass::half_t *A, cutlass::half_t *B,
                   float beta,
                   cutlass::half_t *C) {
    GemmFP16 gemm_op;
    typename GemmFP16::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {alpha, beta}
    );
    gemm_op(args);
}
```

### 5.2 Tensor Core GEMM (WMMA)

```cpp
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"

// 使用 Tensor Core 的 FP16 GEMM
using GemmTensorOp = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,                                        // 累加器 FP32
    cutlass::arch::OpClassTensorOp,              // 使用 Tensor Core
    cutlass::arch::Sm80,                         // Ampere 架构
    cutlass::gemm::GemmShape<128, 256, 64>,     // Threadblock 配置
    cutlass::gemm::GemmShape<64, 64, 64>,         // Warp 配置
    cutlass::gemm::GemmShape<16, 8, 16>          // Tensor Core 指令形状 (mma.sync.aligned.m16n8k16)
>;
```

### 5.3 混合布局 (A: Row, B: Column)

```cpp
// A: Row-Major, B: Column-Major, C: Row-Major
// 这对应 BLAS 的 gemm(N, N, ...) 即 C = A * B
using GemmMixedLayout = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,     // A: Row
    float, cutlass::layout::ColumnMajor,   // B: Column
    float, cutlass::layout::RowMajor       // C: Row
>;

// 注意：使用 ColumnMajor 时，leading dimension 的计算方式改变
// B: K x N, Column-Major, ldb = K (而非 N)
```

---

## 6. 性能优化技巧

### 6.1 对齐要求

```cpp
// CUTLASS 要求指针对齐以获得最佳性能
// 通常要求 16 byte (128 bit) 对齐

// 分配对齐内存
cudaMalloc(&A, M * K * sizeof(float));  // 可能不对齐

// 使用对齐分配
size_t alignment = 128;  // 128 byte 对齐
cudaMalloc(&A, ((M * K * sizeof(float) + alignment - 1) / alignment) * alignment);

// 或使用 CUTLASS 的辅助函数
#include "cutlass/util/device_memory.h"
cutlass::DeviceAllocation<float> allocation(M * K);
float *A = allocation.get();
```

### 6.2 流 (Stream) 异步执行

```cpp
#include <cuda_runtime.h>

void run_gemm_async(cudaStream_t stream) {
    using Gemm = cutlass::gemm::device::Gemm<...>;
    Gemm gemm_op;

    typename Gemm::Arguments args(...);

    // 异步执行
    cutlass::Status status = gemm_op(args, stream);

    // 后续操作可以与其他 CUDA 操作并行
    // 在适当时机同步
    cudaStreamSynchronize(stream);
}
```

### 6.3 批量 GEMM (Batched GEMM)

```cpp
#include "cutlass/gemm/device/gemm_batched.h"

// 批量 GEMM: 多组小矩阵并行计算
using GemmBatched = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor
>;

void run_batched_gemm(int batch_size, int M, int N, int K,
                      float alpha,
                      float **A_array, float **B_array,
                      float beta,
                      float **C_array) {
    GemmBatched gemm_op;

    typename GemmBatched::Arguments args(
        cutlass::gemm::GemmCoord{M, N, K},  // 单个问题规模
        batch_size,                          // 批量大小
        A_array, {K, M * K},                 // A 数组和 stride
        B_array, {N, K * N},                 // B 数组和 stride
        C_array, {N, M * N},                 // C 数组和 stride
        {alpha, beta}
    );

    gemm_op(args);
}
```

### 6.4 分裂-K (Split-K) 并行

```cpp
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

// Split-K: 将 K 维度拆分，多 CTA 并行计算后归约
// 适用于 K 很大而 M,N 较小的情况
using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor
>;

void run_splitk_gemm(int M, int N, int K, int split_k_slices,
                     float alpha, float *A, float *B,
                     float beta, float *C, float *workspace) {
    GemmSplitK gemm_op;

    typename GemmSplitK::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {alpha, beta},
        split_k_slices,    // K 方向切片数
        workspace          // 临时 workspace
    );

    gemm_op(args);
}
```

---

## 7. 常见问题与调试

### 7.1 编译错误

#### 问题：模板实例化失败

```
error: no instance of constructor "cutlass::gemm::device::Gemm<...>::Arguments"
matches the argument list
```

**原因：** 模板参数不匹配，如布局与 leading dimension 不一致

**解决：**
```cpp
// 确保布局与 stride 匹配
// RowMajor: stride = K (N for B)
// ColumnMajor: stride = M (K for B)
{A, K}  // RowMajor A: M x K, stride = K
{B, N}  // RowMajor B: K x N, stride = N
```

#### 问题：C++17 标准要求

```
error: 'if constexpr' is a C++17 extension
```

**解决：** 添加 `-std=c++17`
```bash
nvcc -std=c++17 -arch=sm_70 ...
```

### 7.2 运行时错误

#### 问题：CUDA out of memory

```
CUDA error: out of memory
```

**原因：** Shared memory 超出限制

**检查：**
```cpp
// 计算所需 SMEM
size_t smem_A = M_tile * K_tile * sizeof(ElementA);
size_t smem_B = K_tile * N_tile * sizeof(ElementB);
size_t total_smem = smem_A + smem_B;

// Ampere (SM80) 每 SM 最大 164KB
assert(total_smem <= 164 * 1024);
```

#### 问题：结果不正确

**排查步骤：**
1. 检查 alpha/beta 值
2. 验证 leading dimension 与布局匹配
3. 检查指针是否对齐
4. 使用小矩阵调试

```cpp
// 调试：打印参数
std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
std::cout << "alpha=" << alpha << ", beta=" << beta << std::endl;
std::cout << "lda=" << K << ", ldb=" << N << ", ldc=" << N << std::endl;
```

### 7.3 性能问题

#### 检查清单

- [ ] 指针是否 128-bit 对齐？
- [ ] ThreadblockShape 是否适合问题规模？
- [ ] 是否使用了正确的 ArchTag？
- [ ] 是否启用了 Tensor Core（如果支持）？
- [ ] CUDA 流是否异步执行？

#### 性能分析工具

```cpp
#include <cuda_profiler_api.h>

// 启动 NVIDIA Nsight Compute 分析
cudaProfilerStart();
for (int i = 0; i < 100; ++i) {
    gemm_op(args);
}
cudaProfilerStop();
```

---

## 8. 参考资料

### 官方资源

1. [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
2. [CUTLASS 文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md)
3. [CUTLASS 示例](https://github.com/NVIDIA/cutlass/tree/main/examples)

### 关键示例代码

| 示例 | 路径 | 说明 |
|-----|------|------|
| 基础 GEMM | `examples/00_basic_gemm` | 最简单使用 |
| 自定义 Epilogue | `examples/01_epilogue` | 修改输出处理 |
| Split-K | `examples/05_splitk` | K 维度并行 |
| Batched GEMM | `examples/06_batched_gemm` | 批量矩阵乘 |
| Tensor Core | `examples/08_ampere_tensorop` | Ampere Tensor Core |

### 本项目参考实现

- `GEMM/src/sgemm_cutlass.cu`: 基础 FP32 Row-Major GEMM
- `GEMM/src/sgemm_cute.cu`: CuTe (CUTLASS 3.x) 实现
- `GEMM/Makefile`: CUTLASS 集成编译配置

---

## 总结

CUTLASS 提供了从简单到复杂的多种使用层次：

1. **入门**: 使用 `device::Gemm`，仅需 10 行代码
2. **进阶**: 显式配置 ThreadblockShape、WarpShape
3. **高级**: 自定义 Epilogue、Split-K、Batched
4. **专家**: 直接操作 Kernel API 或 Threadblock API

对于大多数应用场景，**Device API** 配合合适的模板参数已经足够获得接近 cuBLAS 的性能。

# CuTe SGEMM 编译与使用说明

## 什么是 CuTe？

**CuTe** (CUDA Tensor Extensions) 是 NVIDIA CUTLASS 3.x 引入的下一代张量编程 DSL。

与传统的 CUTLASS 2.x 模板元编程相比，CuTe 提供了：
- **更简洁的 API**：使用 `Tensor`、`Layout`、`Shape` 等概念，代码更易读
- **更强大的抽象**：张量操作可以像数组一样直接进行
- **更好的可组合性**：TiledCopy、TiledMMA 等组件可以灵活组合
- **现代 C++17**：充分利用现代 C++ 特性

## 依赖安装

CuTe 是 CUTLASS 3.x 的一部分，因此需要 CUTLASS 头文件：

```bash
cd GEMM
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

或者通过环境变量指定 CUTLASS 路径：
```bash
export CUTLASS_DIR=/path/to/cutlass
make
```

## 编译

Makefile 会自动检测 CUTLASS 是否存在：

```bash
cd GEMM
make
```

如果 CUTLASS 未找到，会打印提示信息，并编译占位实现（运行时报错）。

## CuTe 核心概念

### 1. 张量 (Tensor)

CuTe 的核心是 `Tensor`，它包含：
- **数据指针**：指向内存（global/shared/register）
- **布局 (Layout)**：描述张量的形状和步长

```cpp
// 创建一个指向 global memory 的 (M,K) 张量
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);

// 创建 shared memory 张量
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
```

### 2. 布局 (Layout)

布局描述张量的形状和步长：

```cpp
// (128, 8) 的 m-major 布局，步长为 (1, 128)
auto sA = make_layout(make_shape(Int<128>{}, Int<8>{}));

// 等价的 LayoutRight (k-major)
auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
```

### 3. 线程分区 (Partitioning)

CuTe 自动将张量分区给线程：

```cpp
// 使用线程布局 tA 将 gA 分区给各线程
Tensor tAgA = local_partition(gA, tA, threadIdx.x);

// 使用线程布局 tC 的投影将 sA 分区给各线程
Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
```

### 4. 数据移动

CuTe 提供了高级抽象的数据移动原语：

```cpp
// 从 global memory 复制到 shared memory
copy(tAgA(_,_,k_tile), tAsA);
cp_async_fence();
cp_async_wait<0>();

// 在 shared memory 上执行 gemm
gemm(tCsA, tCsB, tCrC);

// 尾部的 alpha/beta 缩放
axpby(alpha, tCrC, beta, tCgC);
```

## 代码结构

```cpp
// 1. 定义问题形状和 CTA tile 大小
auto prob_shape = make_shape(M, N, K);
auto cta_tiler = make_shape(bM, bN, bK);

// 2. 定义内存布局
auto sA = make_layout(make_shape(bM, bK));  // m-major
auto sB = make_layout(make_shape(bN, bK));  // n-major
auto sC = make_layout(make_shape(bM, bN));  // m-major

// 3. 定义线程布局
auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

// 4. 计算 smem 大小并启动 kernel
dim3 dimBlock(size(tC));
dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
gemm_cute_kernel<<<dimGrid, dimBlock, smem_size>>>(...);
```

## 性能预期

CuTe SGEMM 的性能介于手动优化版本和 CUTLASS 之间：

- **相比手写 kernel**：CuTe 代码更简洁，但性能可能略低于极致优化的手写版本
- **相比 CUTLASS**：CuTe 是 CUTLASS 3.x 的基础，两者使用相同的底层优化技术

在 RTX 4090/5090 上的典型性能：
- cuBLAS: ~60+ TFLOPS
- CuTe: ~40-50 TFLOPS (取决于 tile 配置)
- 手写 Register V3: ~40-60 TFLOPS

## 进阶：SM80 优化版本

CUTLASS 示例中提供了更优化的 `sgemm_sm80.cu`，使用了：
- **cp.async**：异步全局内存到共享内存拷贝
- **LDSM**：共享内存到寄存器的高效加载
- **WMMA/Tensor Core**：使用 Tensor Core 加速
- **Pipeline**：软件流水线隐藏延迟

这些优化也可以通过 CuTe API 实现，详见 `cutlass/examples/cute/tutorial/`。

## 参考资源

1. [CUTLASS CuTe Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial)
2. [CUTLASS 3.0 Design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_introduction.md)
3. CuTe 头文件：`cutlass/include/cute/tensor.hpp`

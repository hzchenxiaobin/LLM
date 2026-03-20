# 使用 NVIDIA CUTLASS 编译 GEMM 基准

本项目的 `run_sgemm_cutlass` 使用 [CUTLASS](https://github.com/NVIDIA/cutlass) 的 `cutlass::gemm::device::Gemm`，在 **行主序 (Row-Major)** 下计算：

\\(C = \alpha A B + \beta C\\)

- \\(A\\): \\(M \times K\\)，`lda = K`
- \\(B\\): \\(K \times N\\)，`ldb = N`
- \\(C\\): \\(M \times N\\)，`ldc = N`

## 1. 获取 CUTLASS

在 `GEMM` 目录下执行（推荐路径）：

```bash
cd GEMM
mkdir -p third_party
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

或通过环境变量指定任意已克隆路径：

```bash
export CUTLASS_DIR=/path/to/cutlass
```

## 2. 编译

Makefile 会自动检测 `$(CUTLASS_DIR)/include/cutlass/gemm/device/gemm.h`：

- **存在**：编译 `sgemm_cutlass.cu`（需要 **C++17**），并在 `main` 中启用 CUTLASS 基准项。
- **不存在**：编译 `sgemm_cutlass_disabled.cu`，基准中**不包含** CUTLASS（避免链接失败）。

```bash
cd GEMM
make
```

## 3. 架构与标准

- 默认 `nvcc -arch=sm_70`（与现有 WMMA 目标一致）；CUTLASS 默认 `ArchTag` 为 `Sm70` 的 SIMT FP32 配置。
- 若在 **Ampere/Ada** 等上追求更优性能，可将 Makefile 中 `-arch=sm_70` 改为本机对应架构（如 `sm_80`、`sm_89`），并视需要在 `sgemm_cutlass.cu` 中显式指定 `cutlass::arch::Sm80` 等（需与 CUTLASS 中该架构的默认 Gemm 配置一致）。

## 4. 参考

- CUTLASS 示例：`examples/00_basic_gemm/basic_gemm.cu`（列为 Column-Major；本项目使用 **RowMajor** 与手写 kernel 布局一致）
- 设备 API：`include/cutlass/gemm/device/gemm.h`

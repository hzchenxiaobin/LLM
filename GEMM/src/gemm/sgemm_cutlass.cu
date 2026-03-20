// CUTLASS SGEMM: C = alpha * A * B + beta * C
// 矩阵按行主序 (Row-Major) 存储，与项目其余实现一致：
//   A: M×K, lda = K
//   B: K×N, ldb = N
//   C: M×N, ldc = N
//
// 依赖：NVIDIA CUTLASS 头文件（见 GEMM/docs/cutlass_build.md）

#include "../common.h"
#include "gemm_kernels.h"

#include <iostream>

#include "cutlass/gemm/device/gemm.h"

void run_sgemm_cutlass(int M, int N, int K, float alpha, const float *A, const float *B,
                       float beta, float *C) {
    // FP32 + CUDA Core (SIMT)，默认 ArchTag=Sm70，与 nvcc -arch=sm_70 兼容
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float>;

    Gemm gemm_op;

    typename Gemm::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {alpha, beta});

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed with status: " << static_cast<int>(status) << std::endl;
    }
}

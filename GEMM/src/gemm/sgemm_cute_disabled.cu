// CuTe SGEMM 占位实现
// 当 CUTLASS/CuTe 头文件未找到时使用

#include "../common.h"
#include "gemm_kernels.h"

#include <iostream>
#include <cstdlib>

void run_sgemm_cute(int M, int N, int K, float alpha, const float *A, const float *B,
                    float beta, float *C) {
    std::cerr << "[CuTe] 错误: CuTe SGEMM 未编译，CUTLASS 头文件未找到" << std::endl;
    std::cerr << "       请克隆 CUTLASS: git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass" << std::endl;
    std::abort();
}

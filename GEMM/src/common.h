#pragma once
#include <iostream>
#include <cuda_runtime.h>

// ==========================================
// 辅助宏：CUDA 错误检查
// ==========================================
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==========================================
// 定义统一的 GEMM 算子接口
// ==========================================
// 所有被测试的 GEMM 包装函数都需要符合这个签名
typedef void (*GemmFunc)(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
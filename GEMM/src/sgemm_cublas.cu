#include "gemm_kernels.h"
#include <cuda_fp16.h>

// 定义全局的 cuBLAS 句柄
cublasHandle_t cublas_handle;

// ==========================================
// 算子实现区域
// ==========================================

// --- 算子 3: cuBLAS SGEMM (作为性能天花板和正确性基准) ---
void run_cublas(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
}

// 辅助核函数：float -> half 转换
__global__ void cublas_float_to_half_kernel(int n, const float *src, half *dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// --- 算子 3b: cuBLAS TensorCore SGEMM (FP16输入+FP32累加，与WMMA行为一致) ---
void run_cublas_tensorcore(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 分配 half 精度的设备内存
    half *d_A_half, *d_B_half;
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);

    cudaMalloc(&d_A_half, size_A);
    cudaMalloc(&d_B_half, size_B);

    // 将 float 输入转换为 half
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;

    cublas_float_to_half_kernel<<<blocks_A, threads>>>(M * K, A, d_A_half);
    cublas_float_to_half_kernel<<<blocks_B, threads>>>(K * N, B, d_B_half);
    cudaDeviceSynchronize();

    // 设置 cuBLAS 使用 TensorCore
    cublasMath_t math_mode = CUBLAS_TENSOR_OP_MATH;
    cublasSetMathMode(cublas_handle, math_mode);

    // 使用 cublasGemmEx 进行混合精度矩阵乘法
    // A/B 使用 FP16，C 使用 FP32，计算使用 TensorCore
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B_half, CUDA_R_16F, N,  // B: FP16
                 d_A_half, CUDA_R_16F, K,  // A: FP16
                 &beta,
                 C, CUDA_R_32F, N,         // C: FP32
                 CUBLAS_COMPUTE_32F,      // 计算精度: FP32
                 CUBLAS_GEMM_DEFAULT);    // 默认算法，自动使用 TensorCore

    // 恢复默认 math mode
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    // 释放临时内存
    cudaFree(d_A_half);
    cudaFree(d_B_half);
}
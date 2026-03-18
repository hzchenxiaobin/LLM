#include "gemm_kernels.h"

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
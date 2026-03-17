#include "gemm_kernels.h"

// 定义全局的 cuBLAS 句柄
cublasHandle_t cublas_handle;

// ==========================================
// 算子实现区域
// ==========================================

// --- 算子 1: Shared Memory GEMM ---
template <int BLOCK_SIZE>
__global__ void sgemm_shared_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) { tmp += A[y * K + i] * B[i * N + x]; }
        C[y * N + x] = alpha * tmp + beta * C[y * N + x];
    }
}

void run_sgemm_shared(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    sgemm_shared_kernel<BLOCK_SIZE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

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
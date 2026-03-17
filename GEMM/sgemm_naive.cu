#include "common.h"
#include "gemm_kernels.h"

// --- 算子 1: 朴素 GEMM (Naive) ---
__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // N 维度
    int y = blockIdx.y * blockDim.y + threadIdx.y; // M 维度

    if (x < N && y < M) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[y * K + i] * B[i * N + x];
        }
        C[y * N + x] = alpha * tmp + beta * C[y * N + x];
    }
}

void run_sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    sgemm_naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

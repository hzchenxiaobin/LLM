#include "gemm_kernels.h"

// --- 算子 2: Shared Memory GEMM ---
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

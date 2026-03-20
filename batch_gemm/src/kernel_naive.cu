#include "kernels.cuh"
#include <cuda_runtime.h>

// ==============================================================================
// V0: 朴素实现 (Naive)
// ==============================================================================
__global__ void bmm_naive_kernel(const float* A, const float* B, float* C, int B_size, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= B_size) return;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        const float* A_batch = A + batch_idx * M * K;
        const float* B_batch = B + batch_idx * K * N;
        float* C_batch = C + batch_idx * M * N;

        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A_batch[row * K + k] * B_batch[k * N + col];
        }
        C_batch[row * N + col] = sum;
    }
}

void run_bmm_naive(const float* A, const float* B, float* C, int B_size, int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y, 
                 B_size);
    
    bmm_naive_kernel<<<gridDim, blockDim>>>(A, B, C, B_size, M, N, K);
}
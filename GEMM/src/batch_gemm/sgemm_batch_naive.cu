#include "../common.h"
#include "batch_gemm_kernels.h"

// --- 算子: 朴素 Batch GEMM (Naive Batch) ---
// 每个线程计算 C[batch][y][x] 的一个元素
__global__ void sgemm_batch_naive_kernel(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    int batch_count, long long int strideA, long long int strideB, long long int strideC
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // N 维度
    int y = blockIdx.y * blockDim.y + threadIdx.y; // M 维度
    int batch = blockIdx.z; // Batch 维度

    if (batch < batch_count && x < N && y < M) {
        // 计算当前 batch 的矩阵起始地址
        const float *A_batch = A + batch * strideA;
        const float *B_batch = B + batch * strideB;
        float *C_batch = C + batch * strideC;

        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A_batch[y * K + i] * B_batch[i * N + x];
        }
        C_batch[y * N + x] = alpha * tmp + beta * C_batch[y * N + x];
    }
}

void run_sgemm_batch_naive(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    int batch_count, long long int strideA, long long int strideB, long long int strideC
) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);
    sgemm_batch_naive_kernel<<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C,
        batch_count, strideA, strideB, strideC
    );
}

// --- 简化版本: 连续存储的 Batch GEMM ---
// 假设所有 batch 的矩阵在内存中是连续存储的，且每个矩阵大小相同
__global__ void sgemm_batch_contiguous_kernel(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    int batch_count
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // N 维度
    int y = blockIdx.y * blockDim.y + threadIdx.y; // M 维度
    int batch = blockIdx.z; // Batch 维度

    int mn = M * N;
    int mk = M * K;
    int nk = N * K;

    if (batch < batch_count && x < N && y < M) {
        // 计算当前 batch 的矩阵起始地址 (连续存储)
        const float *A_batch = A + batch * mk;
        const float *B_batch = B + batch * nk;
        float *C_batch = C + batch * mn;

        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A_batch[y * K + i] * B_batch[i * N + x];
        }
        C_batch[y * N + x] = alpha * tmp + beta * C_batch[y * N + x];
    }
}

void run_sgemm_batch_contiguous(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    int batch_count
) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);
    sgemm_batch_contiguous_kernel<<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C, batch_count
    );
}

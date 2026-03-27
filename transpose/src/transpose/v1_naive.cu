#include "transpose_common.h"

// v1: 朴素实现
// 每个线程处理一个元素
// 读取A是合并访问，写入B是非合并访问
// A: M x N matrix, B: N x M matrix
__global__ void transpose_naive_kernel(const float *A, float *B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引 (0 to N-1)
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引 (0 to M-1)

    if (x < N && y < M) {
        // A: 连续读取 (y * N + x) - 合并访问
        // B: 非连续写入 (x * M + y) - 非合并访问
        B[x * M + y] = A[y * N + x];
    }
}

void transpose_naive(const float *d_A, float *d_B, int M, int N, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, M, N);
}

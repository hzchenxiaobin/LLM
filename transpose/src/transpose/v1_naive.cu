#include "transpose_common.h"

// v1: 朴素实现
// 每个线程处理一个元素
// 读取A是合并访问，写入B是非合并访问
__global__ void transpose_naive_kernel(const float *A, float *B, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引

    if (x < N && y < N) {
        // A: 连续读取 (y * N + x) - 合并访问
        // B: 非连续写入 (x * N + y) - 非合并访问
        B[x * N + y] = A[y * N + x];
    }
}

void transpose_naive(const float *d_A, float *d_B, int N, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, N);
}

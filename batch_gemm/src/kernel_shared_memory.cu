#include "kernels.cuh"
#include <cuda_runtime.h>

// ==============================================================================
// V1: 共享内存分块优化 (Shared Memory Tiling)
// ==============================================================================
#define TILE_SIZE 32

__global__ void bmm_shared_memory_kernel(const float* A, const float* B, float* C, int B_size, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= B_size) return;

    // 申请共享内存
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 沿着 K 维度滑动分块
    for (int t = 0; t < num_tiles; ++t) {
        // 1. 协同加载数据到共享内存
        int k_A = t * TILE_SIZE + threadIdx.x;
        int k_B = t * TILE_SIZE + threadIdx.y;

        // 边界检查并加载 A
        if (row < M && k_A < K) sA[threadIdx.y][threadIdx.x] = A_batch[row * K + k_A];
        else sA[threadIdx.y][threadIdx.x] = 0.0f;

        // 边界检查并加载 B
        if (k_B < K && col < N) sB[threadIdx.y][threadIdx.x] = B_batch[k_B * N + col];
        else sB[threadIdx.y][threadIdx.x] = 0.0f;

        // 2. 线程同步，等待所有线程加载完毕
        __syncthreads();

        // 3. 计算当前分块的局部点积
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // 4. 线程同步，防止有线程跑得快，提前把下一个 Tile 的数据覆盖进了共享内存
        __syncthreads();
    }

    // 写回 Global Memory
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

void run_bmm_v1_shared_memory(const float* A, const float* B, float* C, int B_size, int M, int N, int K) {
    // V1 使用 TILE_SIZE 作为 block 维度
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y, 
                 B_size);
    
    bmm_shared_memory_kernel<<<gridDim, blockDim>>>(A, B, C, B_size, M, N, K);
}
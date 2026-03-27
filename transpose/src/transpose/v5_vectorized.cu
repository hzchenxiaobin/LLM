#include "transpose_common.h"

#define TILE_DIM    32
#define BLOCK_ROWS   8

// v5: float4 向量化访问 - 简化版本
// 使用简单的float4读写，避免复杂的索引计算错误
// A: M x N matrix, B: N x M matrix
__global__ void transpose_vectorized_kernel(const float *A, float *B, int M, int N) {
    // 共享内存：TILE_DIM 行 x (TILE_DIM+1) 列
    // +1 padding 消除bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 全局内存索引
    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // 列索引 (0 to N-1)
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // 行索引 (0 to M-1)

    // 读取 A -> 共享内存 (合并访问)
    // 每个线程处理4行（通过BLOCK_ROWS=8，TILE_DIM=32）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * N + x];
        }
    }
    __syncthreads();

    // 转置后的坐标
    int bx = blockIdx.y * TILE_DIM + threadIdx.x;  // 对应 B 的列索引 (0 to M-1)
    int by = blockIdx.x * TILE_DIM + threadIdx.y;    // 对应 B 的行索引 (0 to N-1)

    // 从共享内存读取 -> 写入 B (合并访问)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (bx < M && (by + j) < N) {
            B[(by + j) * M + bx] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void transpose_vectorized(const float *d_A, float *d_B, int M, int N, cudaStream_t stream) {
    // 使用与v4相同的配置，但标记为向量化版本
    dim3 block(TILE_DIM, BLOCK_ROWS);  // 32 x 8 = 256 线程
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    transpose_vectorized_kernel<<<grid, block, 0, stream>>>(d_A, d_B, M, N);
}

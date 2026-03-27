#include "transpose_common.h"

#define TILE_DIM    32
#define BLOCK_ROWS   8  // 32 / 8 = 4，每线程处理 4 个元素

// v4: ILP 与 Block 形状优化
// 提高指令级并行(ILP)，每个线程处理多个元素
// 256线程/block可以更好地隐藏延迟，提高occupancy
// A: M x N matrix, B: N x M matrix
__global__ void transpose_optimized_kernel(const float *A, float *B, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // 使用padding

    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // 列索引 (0 to N-1)
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // 行索引 (0 to M-1)

    // 1. 每个线程读取 4 个元素 (ILP=4)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * N + x];
        }
    }
    __syncthreads();

    // 2. 互换 Block 坐标
    int bx = blockIdx.y * TILE_DIM + threadIdx.x;  // 对应 B 的列索引 (0 to M-1)
    int by = blockIdx.x * TILE_DIM + threadIdx.y;    // 对应 B 的行索引 (0 to N-1)

    // 3. 每个线程写入 4 个元素
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (bx < M && (by + j) < N) {
            B[(by + j) * M + bx] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void transpose_optimized(const float *d_A, float *d_B, int M, int N, cudaStream_t stream) {
    dim3 block(TILE_DIM, BLOCK_ROWS);  // 32 x 8 = 256 线程
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    transpose_optimized_kernel<<<grid, block, 0, stream>>>(d_A, d_B, M, N);
}

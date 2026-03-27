#include "transpose_common.h"

#define TILE_DIM 32

// v2: 共享内存优化
// 使用共享内存中转数据，但存在 bank conflict
// A: M x N matrix, B: N x M matrix
__global__ void transpose_shared_kernel(const float *A, float *B, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // 列索引 (0 to N-1)
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // 行索引 (0 to M-1)

    // 1. 合并读取 A -> 共享内存
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();

    // 2. 互换 Block 坐标
    int bx = blockIdx.y * TILE_DIM + threadIdx.x;  // 对应 B 的列索引 (0 to M-1)
    int by = blockIdx.x * TILE_DIM + threadIdx.y;  // 对应 B 的行索引 (0 to N-1)

    // 3. 从共享内存读取并合并写入 B
    // 注意：这里存在 bank conflict！
    // 线程0读取 tile[0][0] -> bank 0
    // 线程1读取 tile[1][0] -> bank 0 (冲突!)
    if (bx < M && by < N) {
        B[by * M + bx] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_shared_memory(const float *d_A, float *d_B, int M, int N, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    transpose_shared_kernel<<<grid, block, 0, stream>>>(d_A, d_B, M, N);
}

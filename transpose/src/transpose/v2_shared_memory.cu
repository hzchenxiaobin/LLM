#include "transpose_common.h"

#define TILE_DIM 32

// v2: 共享内存优化
// 使用共享内存中转数据，但存在 bank conflict
__global__ void transpose_shared_kernel(const float *A, float *B, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 合并读取 A -> 共享内存
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();

    // 2. 互换 Block 坐标
    int bx = blockIdx.y * TILE_DIM + threadIdx.x;
    int by = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. 从共享内存读取并合并写入 B
    // 注意：这里存在 bank conflict！
    // 线程0读取 tile[0][0] -> bank 0
    // 线程1读取 tile[1][0] -> bank 0 (冲突!)
    if (bx < N && by < N) {
        B[by * N + bx] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_shared_memory(const float *d_A, float *d_B, int N, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    transpose_shared_kernel<<<grid, block, 0, stream>>>(d_A, d_B, N);
}

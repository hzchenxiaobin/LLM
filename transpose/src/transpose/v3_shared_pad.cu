#include "transpose_common.h"

#define TILE_DIM 32

// v3: Padding 消除 Bank Conflict
// 通过将列宽加1，使同一行的数据分散到不同 bank
__global__ void transpose_shared_pad_kernel(const float *A, float *B, int N) {
    // 关键修改：TILE_DIM + 1，消除 bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

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
    // 现在访问 tile[threadIdx.x][threadIdx.y] 时
    // 由于列宽是33，相邻行的数据会在不同 bank
    if (bx < N && by < N) {
        B[by * N + bx] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_shared_pad(const float *d_A, float *d_B, int N, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    transpose_shared_pad_kernel<<<grid, block, 0, stream>>>(d_A, d_B, N);
}

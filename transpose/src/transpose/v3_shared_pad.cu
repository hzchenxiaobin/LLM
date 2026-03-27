#include "transpose_common.h"

#define TILE_DIM 32

// v3: Padding 消除 Bank Conflict
// 通过将列宽加1，使同一行的数据分散到不同 bank
// A: M x N matrix, B: N x M matrix
__global__ void transpose_shared_pad_kernel(const float *A, float *B, int M, int N) {
    // 关键修改：TILE_DIM + 1，消除 bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

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
    // 现在访问 tile[threadIdx.x][threadIdx.y] 时
    // 由于列宽是33，相邻行的数据会在不同 bank
    if (bx < M && by < N) {
        B[by * M + bx] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_shared_pad(const float *d_A, float *d_B, int M, int N, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    transpose_shared_pad_kernel<<<grid, block, 0, stream>>>(d_A, d_B, M, N);
}

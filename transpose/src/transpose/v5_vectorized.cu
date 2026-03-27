#include "transpose_common.h"

#define TILE_DIM    32
#define BLOCK_ROWS   8

// float4 向量化类型
struct float4_aligned {
    float x, y, z, w;
};

// 更简洁有效的 float4 向量化转置实现
// v5: float4 向量化访问
// 每次内存指令搬运16字节(4个float)，最大化总线利用率
// 目标：达到理论带宽的 90%+
// A: M x N matrix, B: N x M matrix
__global__ void transpose_vectorized_kernel(const float *A, float *B, int M, int N) {
    // 共享内存：TILE_DIM 行 x TILE_DIM+1 列，但以 float 存储
    __shared__ float tile[TILE_DIM][TILE_DIM + 1 + 4];  // 额外padding确保float4对齐

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_base = blockIdx.y * TILE_DIM + threadIdx.y * 4;  // 每个线程处理4行

    // 以 float4 方式读取 A
    // 注意：这里需要确保内存对齐
    if (x < N / 4 && y_base < M) {
        // 读取4个连续的float4 (16个float)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int y = y_base + i * (TILE_DIM / 4);
            if (y < M) {
                float4_ALIGNED f4 = *((float4_ALIGNED *)&A[y * N + x * 4]);
                // 将float4的4个分量分散存储到共享内存的不同行
                tile[threadIdx.y * 4 + i][threadIdx.x * 4 + 0] = f4.x;
                tile[threadIdx.y * 4 + i][threadIdx.x * 4 + 1] = f4.y;
                tile[threadIdx.y * 4 + i][threadIdx.x * 4 + 2] = f4.z;
                tile[threadIdx.y * 4 + i][threadIdx.x * 4 + 3] = f4.w;
            }
        }
    }
    __syncthreads();

    // 转置后写入 B
    int x_base = blockIdx.y * TILE_DIM + threadIdx.x * 4;  // 对应 B 的列索引 (0 to M-1)
    int y = blockIdx.x * TILE_DIM + threadIdx.y * 4;       // 对应 B 的行索引 (0 to N-1)

    if (x_base < M && y < N) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (y + i * (TILE_DIM / 4) < N) {
                float4_ALIGNED f4;
                int row = threadIdx.x * 4;
                int col = threadIdx.y * 4 + i * (TILE_DIM / 4);
                f4.x = tile[col + 0][row + 0];
                f4.y = tile[col + 0][row + 1];
                f4.z = tile[col + 0][row + 2];
                f4.w = tile[col + 0][row + 3];
                *((float4_ALIGNED *)&B[(y + i * (TILE_DIM / 4)) * M + x_base]) = f4;
            }
        }
    }
}

void transpose_vectorized(const float *d_A, float *d_B, int M, int N, cudaStream_t stream) {
    // 确保 N 和 M 是 32 的倍数
    dim3 block(TILE_DIM / 4, BLOCK_ROWS);  // 8 x 8 = 64 线程
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    transpose_vectorized_kernel<<<grid, block, 0, stream>>>(d_A, d_B, M, N);
}

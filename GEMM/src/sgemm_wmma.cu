#include "common.h"
#include "gemm_kernels.h"
#include <mma.h>

// ==========================================
// Tensor Core GEMM using WMMA API
// 使用 FP16 输入 + FP32 累加 (FP16 Tensor Core)
// ==========================================

using namespace nvcuda::wmma;

// WMMA 分块尺寸 (符合 Volta/Turing/Ampere 架构的 Tensor Core 约束)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 大的分块尺寸 (用于 Warp 级别的分块策略)
#define BLOCK_ROW_WARPS 2  // 每个 Block 中 Warp 的行数
#define BLOCK_COL_WARPS 4  // 每个 Block 中 Warp 的列数
#define WARP_ROWS 4        // 每个 Warp 处理的 WMMA_M 块数
#define WARP_COLS 2        // 每个 Warp 处理的 WMMA_N 块数

#define BLOCK_ROWS (BLOCK_ROW_WARPS * WARP_ROWS * WMMA_M)  // 128
#define BLOCK_COLS (BLOCK_COL_WARPS * WARP_COLS * WMMA_N) // 128
#define BLOCK_K (WMMA_K * 4) // 64

// 每个 Warp 的计算量
#define WARP_ROW_TILES (WARP_ROWS)
#define WARP_COL_TILES (WARP_COLS)

// 数据类型转换核函数：float -> half
__global__ void float_to_half_kernel(int n, const float *src, half *dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void sgemm_wmma_kernel(int M, int N, int K, float alpha, const half *A, const half *B, float beta, float *C) {
    // 当前 Warp 在 Block 中的位置
    int warp_id = threadIdx.x / warpSize;
    int warp_row = warp_id / BLOCK_COL_WARPS;
    int warp_col = warp_id % BLOCK_COL_WARPS;

    // 当前 Warp 负责的输出块起始位置
    int warp_row_start = warp_row * WARP_ROW_TILES * WMMA_M;
    int warp_col_start = warp_col * WARP_COL_TILES * WMMA_N;

    // Block 在全局的位置
    int block_row = blockIdx.y * BLOCK_ROWS;
    int block_col = blockIdx.x * BLOCK_COLS;

    // 声明 WMMA Fragment
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[WARP_ROW_TILES];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag[WARP_COL_TILES];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_ROW_TILES][WARP_COL_TILES];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    // 声明共享内存用于预取数据
    __shared__ half sA[BLOCK_ROWS][BLOCK_K];
    __shared__ half sB[BLOCK_K][BLOCK_COLS];

    // 计算当前线程负责的共享内存加载位置
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // 加载 A 到共享内存: BLOCK_ROWS x BLOCK_K 个 half 元素
    int a_elements = BLOCK_ROWS * BLOCK_K;
    int a_per_thread = (a_elements + total_threads - 1) / total_threads;

    // 加载 B 到共享内存: BLOCK_K x BLOCK_COLS 个 half 元素
    int b_elements = BLOCK_K * BLOCK_COLS;
    int b_per_thread = (b_elements + total_threads - 1) / total_threads;

    // 沿 K 维度滑动
    for (int k_step = 0; k_step < K; k_step += BLOCK_K) {
        // 1. 从全局内存加载 A 到共享内存
        #pragma unroll
        for (int i = 0; i < a_per_thread; ++i) {
            int idx = tid + i * total_threads;
            if (idx < a_elements) {
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                int global_row = block_row + row;
                int global_col = k_step + col;

                if (global_row < M && global_col < K) {
                    sA[row][col] = A[global_row * K + global_col];
                } else {
                    sA[row][col] = __float2half(0.0f);
                }
            }
        }

        // 2. 从全局内存加载 B 到共享内存
        #pragma unroll
        for (int i = 0; i < b_per_thread; ++i) {
            int idx = tid + i * total_threads;
            if (idx < b_elements) {
                int row = idx / BLOCK_COLS;
                int col = idx % BLOCK_COLS;
                int global_row = k_step + row;
                int global_col = block_col + col;

                if (global_row < K && global_col < N) {
                    sB[row][col] = B[global_row * N + global_col];
                } else {
                    sB[row][col] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // 3. 在共享内存分块内使用 WMMA 进行计算
        for (int k_frag = 0; k_frag < BLOCK_K; k_frag += WMMA_K) {
            // 加载 A fragment
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; ++i) {
                int a_row = warp_row_start + i * WMMA_M;
                int a_col = k_frag;
                load_matrix_sync(a_frag[i], &sA[a_row][a_col], BLOCK_K);
            }

            // 加载 B fragment
            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; ++j) {
                int b_row = k_frag;
                int b_col = warp_col_start + j * WMMA_N;
                load_matrix_sync(b_frag[j], &sB[b_row][b_col], BLOCK_COLS);
            }

            // 执行矩阵乘法累加
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; ++j) {
                    mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // 4. 将结果写回全局内存
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            int c_row = block_row + warp_row_start + i * WMMA_M;
            int c_col = block_col + warp_col_start + j * WMMA_N;

            // 检查边界
            if (c_row + WMMA_M <= M && c_col + WMMA_N <= N) {
                // 首先加载 C 的当前值 (用于 beta 缩放)
                load_matrix_sync(c_frag, &C[c_row * N + c_col], N, mem_row_major);

                // 应用 alpha * acc + beta * c
                for (int t = 0; t < c_frag.num_elements; ++t) {
                    c_frag.x[t] = alpha * acc_frag[i][j].x[t] + beta * c_frag.x[t];
                }

                // 存储回全局内存
                store_matrix_sync(&C[c_row * N + c_col], c_frag, N, mem_row_major);
            } else {
                // 边界处理：逐个元素写入
                for (int ii = 0; ii < WMMA_M; ++ii) {
                    for (int jj = 0; jj < WMMA_N; ++jj) {
                        int global_row = c_row + ii;
                        int global_col = c_col + jj;
                        if (global_row < M && global_col < N) {
                            float val = alpha * acc_frag[i][j].x[ii * WMMA_N + jj];
                            C[global_row * N + global_col] = val + beta * C[global_row * N + global_col];
                        }
                    }
                }
            }
        }
    }
}

// 包装函数：处理数据类型转换并启动核函数
void run_sgemm_wmma(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 分配 half 类型的设备内存用于输入
    half *d_A_half, *d_B_half;
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);

    CHECK_CUDA(cudaMalloc(&d_A_half, size_A));
    CHECK_CUDA(cudaMalloc(&d_B_half, size_B));

    // 将 float 输入转换为 half
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;

    // 启动转换核函数
    float_to_half_kernel<<<blocks_A, threads>>>(M * K, A, d_A_half);
    float_to_half_kernel<<<blocks_B, threads>>>(K * N, B, d_B_half);

    CHECK_CUDA(cudaDeviceSynchronize());

    // 启动 WMMA GEMM 核函数 (warpSize = 32)
    dim3 block(32 * BLOCK_ROW_WARPS * BLOCK_COL_WARPS); // 32 * 2 * 4 = 256
    dim3 grid((N + BLOCK_COLS - 1) / BLOCK_COLS, (M + BLOCK_ROWS - 1) / BLOCK_ROWS);

    sgemm_wmma_kernel<<<grid, block>>>(M, N, K, alpha, d_A_half, d_B_half, beta, C);

    CHECK_CUDA(cudaDeviceSynchronize());

    // 释放临时内存
    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_B_half));
}

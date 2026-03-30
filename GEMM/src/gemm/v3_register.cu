#include "../common.h"
#include "gemm_kernels.h"

// ==========================================
// 算子 3: 二维寄存器分块 GEMM (Register Tiling)
// ==========================================

// 定义分块大小
#define BM 128  // Block在M维度的负责大小
#define BN 128  // Block在N维度的负责大小
#define BK 8    // Block在K维度的步长
#define TM 8    // Thread在M维度的负责大小
#define TN 8    // Thread在N维度的负责大小

__global__ void sgemm_register_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // Block 索引
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread 索引 (16x16 = 256 个线程 / Block)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 线程的全局一维 ID (Block内)
    int tid = ty * blockDim.x + tx;

    // 1. 申请共享内存
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // 2. 申请线程私有的寄存器数组，用于保存 C 的中间结果
    // 每个线程负责计算 8x8 = 64 个元素
    float accum[TM][TN] = {0.0f};

    // 计算当前线程负责的 C 矩阵元素的全局起始坐标
    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;

    // 预计算从全局内存搬运数据到共享内存时的坐标 (共 256 个线程协作)
    // 搬运 A: 需要加载 128x8 = 1024 个元素，256个线程每个加载 4 个
    int load_a_row = tid / BK; 
    int load_a_col = tid % BK; 

    // 搬运 B: 需要加载 8x128 = 1024 个元素，256个线程每个加载 4 个
    int load_b_row = tid / BN; 
    int load_b_col = tid % BN; 

    // 3. 沿 K 维度分块滑动
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;

        // --- 步骤 A: 协作加载 A 块到 sA ---
        #pragma unroll
        for (int i = 0; i < BM * BK / 256; ++i) {
            int a_row_idx = load_a_row + i * 32; // 256/8 = 32
            int a_col_idx = load_a_col;
            int global_a_row = by * BM + a_row_idx;
            int global_a_col = k_offset + a_col_idx;
            if (global_a_row < M && global_a_col < K) {
                sA[a_row_idx][a_col_idx] = A[global_a_row * K + global_a_col];
            } else {
                sA[a_row_idx][a_col_idx] = 0.0f;
            }
        }

        // --- 步骤 B: 协作加载 B 块到 sB ---
        #pragma unroll
        for (int i = 0; i < BK * BN / 256; ++i) {
            int b_row_idx = load_b_row + i * 2; // 256/128 = 2
            int b_col_idx = load_b_col;
            int global_b_row = k_offset + b_row_idx;
            int global_b_col = bx * BN + b_col_idx;
            if (global_b_row < K && global_b_col < N) {
                sB[b_row_idx][b_col_idx] = B[global_b_row * N + global_b_col];
            } else {
                sB[b_row_idx][b_col_idx] = 0.0f;
            }
        }

        __syncthreads(); // 等待数据全部加载到共享内存

        // --- 步骤 C: 寄存器分块计算 ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 将 A 和 B 的数据从共享内存拉取到寄存器中
            float frag_a[TM];
            float frag_b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];

            // 在寄存器中完成 8x8 的外积 (FFMA)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }
        __syncthreads(); // 等待当前块计算完成，再进入下一个 K 步
    }

    // 4. 将寄存器中的累加结果写回全局内存
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_row = row_start + i;
            int global_col = col_start + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = alpha * accum[i][j] + beta * C[global_row * N + global_col];
            }
        }
    }
}

// 包装函数
void run_sgemm_register(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 使用 16x16 的线程块，每个线程计算 8x8，因此一个 Block 负责计算 128x128 的 C 矩阵块
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
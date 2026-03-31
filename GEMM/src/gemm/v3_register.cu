#include "../common.h"
#include "gemm_kernels.h"

// ==========================================
// 算子 3: 二维寄存器分块 GEMM (Register Tiling)
// 优化版本：减少局部变量，降低寄存器压力
// ==========================================

// 定义分块大小
#define BM 128  // Block在M维度的负责大小
#define BN 128  // Block在N维度的负责大小
#define BK 8    // Block在K维度的步长
#define TM 8    // Thread在M维度的负责大小
#define TN 8    // Thread在N维度的负责大小

__global__ void sgemm_register_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 1. 申请共享内存
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // 2. 申请线程私有的寄存器数组，用于保存 C 的中间结果
    // 每个线程负责计算 8x8 = 64 个元素
    float accum[TM][TN] = {0.0f};

    // 计算当前线程负责的 C 矩阵元素的全局起始坐标
    int row_start = blockIdx.y * BM + threadIdx.y * TM;
    int col_start = blockIdx.x * BN + threadIdx.x * TN;

    // 计算线程的一维 ID (Block内) 用于协作加载
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 3. 沿 K 维度分块滑动
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;

        // --- 步骤 A: 协作加载 A 块到 sA ---
        // 计算从全局内存搬运数据到共享内存时的坐标
        // 256个线程协作加载 128x8 = 1024 个元素，每个线程加载 4 个
        #pragma unroll
        for (int i = 0; i < BM * BK / 256; ++i) {
            int a_row_idx = (tid / BK) + i * 32;  // tid / 8
            int a_col_idx = tid % BK;             // tid % 8
            int global_a_row = blockIdx.y * BM + a_row_idx;
            int global_a_col = k_offset + a_col_idx;
            if (global_a_row < M && global_a_col < K) {
                sA[a_row_idx][a_col_idx] = A[global_a_row * K + global_a_col];
            } else {
                sA[a_row_idx][a_col_idx] = 0.0f;
            }
        }

        // --- 步骤 B: 协作加载 B 块到 sB ---
        // 256个线程协作加载 8x128 = 1024 个元素，每个线程加载 4 个
        #pragma unroll
        for (int i = 0; i < BK * BN / 256; ++i) {
            int b_row_idx = (tid / BN) + i * 2;   // tid / 128
            int b_col_idx = tid % BN;             // tid % 128
            int global_b_row = k_offset + b_row_idx;
            int global_b_col = blockIdx.x * BN + b_col_idx;
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
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[threadIdx.y * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][threadIdx.x * TN + j];

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

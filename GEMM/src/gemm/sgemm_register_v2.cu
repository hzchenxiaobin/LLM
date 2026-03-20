#include "../common.h"
#include "gemm_kernels.h"

// ==========================================
// 算子 3: 优化版二维寄存器分块 (Vectorized + Padding)
// ==========================================

// 定义分块大小
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// 辅助宏：使用 float4 进行 128-bit 向量化访存
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

__global__ void sgemm_register_kernel_v2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // 0~255

    // 1. 申请共享内存 
    // 【优化】给 sA 和 sB 增加 Padding (+4) 消除 Shared Memory Bank 冲突
    __shared__ float sA[BM][BK + 4]; 
    __shared__ float sB[BK][BN + 4];

    float accum[TM][TN] = {0.0f};

    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;

    // 【优化】重构预计算坐标，适应 float4 向量化加载 (每次加载4个元素)
    // 256个线程共需加载 128*8=1024 个元素，即 256 个 float4
    // 负责搬运 A: 128 行，每行 2 个 float4
    int load_a_row = tid / 2;       
    int load_a_col = (tid % 2) * 4; 

    // 负责搬运 B: 8 行，每行 32 个 float4
    int load_b_row = tid / 32;      
    int load_b_col = (tid % 32) * 4;

    // 3. 沿 K 维度分块滑动
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;

        // --- 步骤 A: 向量化加载 A 块到 sA ---
        int global_a_row = by * BM + load_a_row;
        int global_a_col = k_offset + load_a_col;
        // 假设 M, N, K 是 4 的倍数，直接使用 float4 读取以榨干带宽
        if (global_a_row < M && global_a_col < K) {
            float4 vec_a = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
            sA[load_a_row][load_a_col + 0] = vec_a.x;
            sA[load_a_row][load_a_col + 1] = vec_a.y;
            sA[load_a_row][load_a_col + 2] = vec_a.z;
            sA[load_a_row][load_a_col + 3] = vec_a.w;
        } else {
            sA[load_a_row][load_a_col + 0] = 0.0f;
            sA[load_a_row][load_a_col + 1] = 0.0f;
            sA[load_a_row][load_a_col + 2] = 0.0f;
            sA[load_a_row][load_a_col + 3] = 0.0f;
        }

        // --- 步骤 B: 向量化加载 B 块到 sB ---
        int global_b_row = k_offset + load_b_row;
        int global_b_col = bx * BN + load_b_col;
        if (global_b_row < K && global_b_col < N) {
            float4 vec_b = FETCH_FLOAT4(B[global_b_row * N + global_b_col]);
            sB[load_b_row][load_b_col + 0] = vec_b.x;
            sB[load_b_row][load_b_col + 1] = vec_b.y;
            sB[load_b_row][load_b_col + 2] = vec_b.z;
            sB[load_b_row][load_b_col + 3] = vec_b.w;
        } else {
            sB[load_b_row][load_b_col + 0] = 0.0f;
            sB[load_b_row][load_b_col + 1] = 0.0f;
            sB[load_b_row][load_b_col + 2] = 0.0f;
            sB[load_b_row][load_b_col + 3] = 0.0f;
        }

        __syncthreads();

        // --- 步骤 C: 寄存器分块计算 ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 【优化】由于前面加了 Padding，这里的读取将完全没有 Bank 冲突
            float frag_a[TM];
            float frag_b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];

            // 在寄存器中完成外积
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }
        __syncthreads(); 
    }

    // 4. 写回全局内存（此处依然可继续优化为 float4 写入，为简便保持原样）
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
void run_sgemm_register_v2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_kernel_v2<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
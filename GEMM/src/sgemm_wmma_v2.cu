#include "common.h"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// ==========================================
// 算子 4: Tensor Core WMMA GEMM
// ==========================================

// 定义 Block 负责计算的输出块大小
#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 16 // FP16 WMMA 的标准 K 维度大小

// 标准 WMMA Fragment 的大小
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void sgemm_wmma_kernel_v2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 申请共享内存，用于暂存并进行 float -> half 的转换
    __shared__ half sA[BLOCK_M][BLOCK_K];
    __shared__ half sB[BLOCK_K][BLOCK_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0..31 (Warp 内的线程 Lane ID)
    int ty = threadIdx.y; // 0..3  (Warp ID)
    
    // Block 内的全局线性线程 ID (共 128 个线程)
    int tid = ty * 32 + tx; 
    
    int warpId = ty;

    // 当前 Warp 负责计算的 16x16 Tile 在 Block (32x32) 中的偏移量
    // Warp 0: (0, 0), Warp 1: (0, 16), Warp 2: (16, 0), Warp 3: (16, 16)
    int warp_row = (warpId / 2) * 16;
    int warp_col = (warpId % 2) * 16;

    // 声明 WMMA Fragment (寄存器级别的矩阵碎片)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 维度分块滑动
    for (int k_step = 0; k_step < K; k_step += BLOCK_K) {
        
        // --- 步骤 A: 协同加载 A 矩阵到 sA (并转为 half) ---
        // 128 个线程需要加载 32x16 = 512 个元素，每个线程负责 4 个元素
        for(int i = tid; i < 512; i += 128) {
            int r = i / BLOCK_K;
            int c = i % BLOCK_K;
            int global_r = by * BLOCK_M + r;
            int global_c = k_step + c;
            
            sA[r][c] = (global_r < M && global_c < K) 
                       ? __float2half(A[global_r * K + global_c]) 
                       : __float2half(0.0f);
        }

        // --- 步骤 B: 协同加载 B 矩阵到 sB (并转为 half) ---
        // 128 个线程需要加载 16x32 = 512 个元素，每个线程负责 4 个元素
        for(int i = tid; i < 512; i += 128) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            int global_r = k_step + r;
            int global_c = bx * BLOCK_N + c;
            
            sB[r][c] = (global_r < K && global_c < N) 
                       ? __float2half(B[global_r * N + global_c]) 
                       : __float2half(0.0f);
        }

        // 必须同步，等待所有数据转化为 half 并落入共享内存
        __syncthreads();

        // --- 步骤 C: Tensor Core 矩阵乘加 ---
        // Warp 从共享内存加载对应的 half 矩阵块到 Fragment
        wmma::load_matrix_sync(a_frag, &sA[warp_row][0], BLOCK_K);
        wmma::load_matrix_sync(b_frag, &sB[0][warp_col], BLOCK_N);

        // 调用硬件 Tensor Core 进行极速相乘！
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // 同步等待当前 K-step 算完，防止下一轮循环覆盖 sA 和 sB
        __syncthreads();
    }

    // --- 步骤 D: 结果写回 ---
    // 缩放 alpha (针对当前框架 beta=0 的情况直接覆盖写入)
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = alpha * c_frag.x[i];
    }

    int global_r = by * BLOCK_M + warp_row;
    int global_c = bx * BLOCK_N + warp_col;

    // 写回 Global Memory 矩阵 C
    if (global_r < M && global_c < N) {
        wmma::store_matrix_sync(&C[global_r * N + global_c], c_frag, N, wmma::mem_row_major);
    }
}

// 包装函数
void run_sgemm_wmma_v2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 32(线程) x 4(Warp) = 128 个线程
    dim3 block(32, 4);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    sgemm_wmma_kernel_v2<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
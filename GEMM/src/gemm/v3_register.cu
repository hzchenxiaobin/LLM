#include "../common.h"
#include "gemm_kernels.h"

// ==========================================
// 算子 3: 二维寄存器分块 GEMM (Register Tiling)
// 用命名变量标出块/线程坐标与协作加载步长，便于对照论文或示意图阅读
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

    // 当前 Block 在 C 上负责的子块左上角（全局行列）
    const int block_c_row = blockIdx.y * BM;
    const int block_c_col = blockIdx.x * BN;
    // 当前线程在该子块内的局部行/列起点 → 对应 C 的全局坐标
    const int thread_local_row = threadIdx.y * TM;
    const int thread_local_col = threadIdx.x * TN;
    const int row_start = block_c_row + thread_local_row;
    const int col_start = block_c_col + thread_local_col;

    // Block 内一维线程号，用于协作把全局 A/B 条带搬进共享内存
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;
    // 协作加载：每个线程负责 (BM*BK)/num_threads 个 A 元素、(BK*BN)/num_threads 个 B 元素
    const int a_load_iters = (BM * BK) / num_threads;
    const int b_load_iters = (BK * BN) / num_threads;
    // 沿共享块行方向每次跨过一整「列组」的线程数，使 tid 扫完 sA 的所有行
    const int a_row_stride = num_threads / BK;
    const int b_row_stride = num_threads / BN;

    // 3. 沿 K 维度分块滑动
    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        // --- 步骤 A: 协作加载 A 块到 sA ---
        // 计算从全局内存搬运数据到共享内存时的坐标
        // 全体线程协作加载 BM×BK 个元素，均摊到每个线程 a_load_iters 次
        #pragma unroll
        for (int i = 0; i < a_load_iters; ++i) {
            const int a_row_idx = (tid / BK) + i * a_row_stride;
            const int a_col_idx = tid % BK;
            const int global_a_row = block_c_row + a_row_idx;
            const int global_a_col = k_offset + a_col_idx;
            if (global_a_row < M && global_a_col < K) {
                sA[a_row_idx][a_col_idx] = A[global_a_row * K + global_a_col];
            } else {
                sA[a_row_idx][a_col_idx] = 0.0f;
            }
        }

        // --- 步骤 B: 协作加载 B 块到 sB ---
        #pragma unroll
        for (int i = 0; i < b_load_iters; ++i) {
            const int b_row_idx = (tid / BN) + i * b_row_stride;
            const int b_col_idx = tid % BN;
            const int global_b_row = k_offset + b_row_idx;
            const int global_b_col = block_c_col + b_col_idx;
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
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[thread_local_row + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][thread_local_col + j];

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

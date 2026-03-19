#include "common.h"
#include "gemm_kernels.h"

// ==========================================
// 算子: 二维寄存器分块 GEMM - Bank Conflict 优化版本
// ==========================================
// 通过添加padding解决shared memory bank conflict问题

// 定义分块大小
#define BM 128  // Block在M维度的负责大小
#define BN 128  // Block在N维度的负责大小
#define BK 8    // Block在K维度的步长
#define TM 8    // Thread在M维度的负责大小
#define TN 8    // Thread在N维度的负责大小

// Padding定义: 在K维度添加1个元素的padding，避免bank conflict
// 32个bank，每个bank宽度4字节(float)
// sA的宽度BK=8，如果不加padding，row 0和row 8会访问相同的bank (0-7)
// 添加padding后，sA[BM][BK+1]，row 8会从bank 8-15读取，避免冲突
#define BK_PAD (BK + 1)   // sA的padding宽度
#define BN_PAD (BN + 1)   // sB的padding宽度 (虽然sB按行访问本来无冲突，但保持一致性)

__global__ void sgemm_register_bank_conflict_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // Block 索引
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread 索引 (16x16 = 256 个线程 / Block)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 线程的全局一维 ID (Block内)
    int tid = ty * blockDim.x + tx;

    // 1. 申请共享内存 - 添加padding解决bank conflict
    __shared__ float sA[BM][BK_PAD];  // 128 x 9，避免bank conflict
    __shared__ float sB[BK][BN_PAD];  // 8 x 129，保持一致性

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

        // --- 步骤 C: 寄存器分块计算 (Bank Conflict优化) ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 将 A 和 B 的数据从共享内存拉取到寄存器中
            float frag_a[TM];
            float frag_b[TN];

            // 加载A: 每个线程加载TM个元素
            // 由于sA添加了padding，线程(ty=0)访问row 0-7 -> banks 0-7
            // 线程(ty=1)访问row 8-15 -> banks 8-15 (因为padding，实际是9-16列)
            // 这样就不会发生bank conflict
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                frag_a[i] = sA[ty * TM + i][k];
            }

            // 加载B: 每个线程加载TN个元素
            // sB[k][tx * TN + j]，tx * TN + j在warp内是连续的
            // 本来就没有bank conflict，但添加padding保持一致性
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                frag_b[j] = sB[k][tx * TN + j];
            }

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
void run_sgemm_register_bank_conflict(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 使用 16x16 的线程块，每个线程计算 8x8，因此一个 Block 负责计算 128x128 的 C 矩阵块
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_bank_conflict_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

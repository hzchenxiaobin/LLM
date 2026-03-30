#include "../common.h"
#include "gemm_kernels.h"

// ==========================================
// 算子: 二维寄存器分块 GEMM - Bank Conflict 优化版本 (整合自 register.cu)
// ==========================================
// 使用 Shared Memory Padding + 转置存储布局 消除 Bank Conflict

// --- 分块参数配置 ---
// Block 级别的分块大小 (Shared Memory 缓存大小)
#define BM 128
#define BN 128
#define BK 8

// Thread 级别的分块大小 (寄存器缓存大小)
#define TM 8
#define TN 8

// ==========================================================
// 核心 Kernel：2D寄存器分块 + Shared Memory Padding 消除 Bank Conflict
// 整合自 register.cu - 使用更优化的线程映射和存储布局
// ==========================================================
__global__ void sgemm_register_bank_conflict_kernel(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta,
    float *C
) {
    // Block 的二维索引
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread 的一维索引 (每个 Block 有 256 个线程)
    int tid = threadIdx.x;

    // 线程在 Block 内负责的输出瓦片(Tile)的坐标映射
    // 巧妙的映射：保证 Warp 内的线程在 N 维度上连续，从而消除读取 Bs 的冲突，并且有利于 C 的合并写入
    int tx = tid % 16;
    int ty = tid / 16;

    // 申请 Shared Memory
    // As 加入 PAD = 4：128+4=132。这样写入转置矩阵时，
    // col=0映射到Bank 0~15，col=4映射到Bank 16~31，完美避开 Bank Conflict！
    __shared__ float As[BK][BM + 4];
    // Bs 不需要 PAD：Warp 内连续线程写入连续地址，完美铺满 32 个 Bank，无冲突。
    __shared__ float Bs[BK][BN];

    // 将指针移动到当前 Block 需要处理的起始位置
    const float *A_ptr = A + by * BM * K;
    const float *B_ptr = B + bx * BN;

    // 计算当前线程从 Global Memory 加载 A 时的索引 (128x8)
    // 256个线程，需要加载 128 * 8 = 1024 个元素，每个线程加载 4 个
    // 按行读取，保证部分合并访存 (Coalesced Memory Access)
    int load_a_row = tid / 8;               // 0 ~ 31
    int load_a_col = tid % 8;               // 0 ~ 7

    // 计算当前线程从 Global Memory 加载 B 时的索引 (8x128)
    // 256个线程，需要加载 8 * 128 = 1024 个元素，每个线程加载 4 个
    // 按列读取，保证完全的合并访存
    int load_b_row = tid / 128;             // 0 ~ 1
    int load_b_col = tid % 128;             // 0 ~ 127

    // 为每个线程分配本地寄存器，用于累加 C 的结果
    float accum[TM][TN] = {0.0f};

    // 为每个线程分配本地寄存器，用于缓存 A 和 B 的数据
    float frag_a[TM];
    float frag_b[TN];

    // 主循环：沿着 K 维度分块推进
    for (int k = 0; k < K; k += BK) {

        // 1. 加载 A 到 Shared Memory (标量加载)
        // 并直接在 Shared Memory 中进行转置存储，方便后续计算阶段连续读取
        #pragma unroll
        for (int step = 0; step < BM; step += 32) {
            As[load_a_col][load_a_row + step] = A_ptr[(load_a_row + step) * K + load_a_col];
        }

        // 2. 加载 B 到 Shared Memory (标量加载)
        #pragma unroll
        for (int step = 0; step < BK; step += 2) {
            Bs[load_b_row + step][load_b_col] = B_ptr[(load_b_row + step) * N + load_b_col];
        }

        // 同步，等待所有线程完成 Shared Memory 的加载
        __syncthreads();

        // 3. 计算当前 Block 的矩阵乘法
        #pragma unroll
        for (int i = 0; i < BK; ++i) {

            // 将 A 的数据从 Shared Memory 加载到寄存器 (利用 Broadcast 机制)
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                frag_a[m] = As[i][ty + m * 16];
            }

            // 将 B 的数据从 Shared Memory 加载到寄存器 (利用连续读取无冲突机制)
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                frag_b[n] = Bs[i][tx + n * 16];
            }

            // 执行 8x8 的外积 (Outer Product) 累加
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += frag_a[m] * frag_b[n];
                }
            }
        }

        // 同步，确保在加载下一个 K 块之前，当前的 Shared Memory 数据已被读取完毕
        __syncthreads();

        // 移动 A 和 B 的指针到下一个 K 分块
        A_ptr += BK;
        B_ptr += BK * N;
    }

    // 4. 将计算结果写回 Global Memory
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            // 计算在全局 C 矩阵中的行列坐标
            int c_row = by * BM + ty + m * 16;
            int c_col = bx * BN + tx + n * 16;

            // 边界检查（仅针对写回阶段）
            if (c_row < M && c_col < N) {
                int idx = c_row * N + c_col;
                C[idx] = alpha * accum[m][n] + beta * C[idx];
            }
        }
    }
}

// 包装函数
void run_sgemm_register_bank_conflict(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 使用 1D 线程块 256 线程，每个线程计算 8x8，因此一个 Block 负责计算 128x128 的 C 矩阵块
    // 线程映射方式：tx = tid % 16, ty = tid / 16
    dim3 block(256); // 1D Thread Block, 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_bank_conflict_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

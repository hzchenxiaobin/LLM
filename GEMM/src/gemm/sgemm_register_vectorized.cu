#include "../common.h"
#include "gemm_kernels.h"

// ==========================================
// 算子: 向量化访存优化的二维寄存器分块 GEMM (Vectorized Loads/Stores)
// ==========================================
// 基于 sgemm_register.cu，使用 float4 (128-bit) 向量化读取全局内存
// 优化点:
// 1. 使用 float4 一次性读取4个 float，减少内存指令数量
// 2. 提高显存带宽利用率，隐藏访存延迟
// 3. 保持寄存器分块计算结构，最大化计算吞吐量

// 定义分块大小
#define BM 128  // Block在M维度的负责大小
#define BN 128  // Block在N维度的负责大小
#define BK 8    // Block在K维度的步长
#define TM 8    // Thread在M维度的负责大小
#define TN 8    // Thread在N维度的负责大小

// 辅助宏：使用 float4 进行 128-bit 向量化访存
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

__global__ void sgemm_register_vectorized_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // Block 索引
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread 索引 (16x16 = 256 个线程 / Block)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 线程的全局一维 ID (Block内)
    int tid = ty * blockDim.x + tx; // 0~255

    // 1. 申请共享内存
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // 2. 申请线程私有的寄存器数组，用于保存 C 的中间结果
    // 每个线程负责计算 8x8 = 64 个元素
    float accum[TM][TN] = {0.0f};

    // 计算当前线程负责的 C 矩阵元素的全局起始坐标
    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;

    // ==========================================
    // 【核心优化】向量化加载的坐标重构
    // ==========================================
    // 原版本：256个线程各加载4个 float 元素（共1024个元素）
    // 向量化版本：256个线程各加载1个 float4（包含4个float），同样覆盖1024个元素
    //
    // 共享内存布局:
    // sA: 128行 x 8列 (BM x BK)
    // sB: 8行 x 128列 (BK x BN)
    //
    // 每个 float4 包含4个连续的 float，所以:
    // - 对于 A: 列方向每4个元素为一组，共 8/4 = 2 组
    // - 对于 B: 列方向每4个元素为一组，共 128/4 = 32 组

    // 负责搬运 A: 128行，每行分为 8/4=2 个 float4
    // 总共需要 128 * 2 = 256 个 float4，正好由 256 个线程各加载1个
    int load_a_row = tid / (BK / 4);           // 行索引: 0~127
    int load_a_col = (tid % (BK / 4)) * 4;     // 列索引: 0 或 4

    // 负责搬运 B: 8行，每行分为 128/4=32 个 float4
    // 总共需要 8 * 32 = 256 个 float4，正好由 256 个线程各加载1个
    int load_b_row = tid / (BN / 4);           // 行索引: 0~7
    int load_b_col = (tid % (BN / 4)) * 4;     // 列索引: 0, 4, 8, ..., 124

    // 3. 沿 K 维度分块滑动
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;

        // --- 步骤 A: 向量化加载 A 块到 sA ---
        int global_a_row = by * BM + load_a_row;
        int global_a_col = k_offset + load_a_col;

        // 使用 float4 一次性读取4个 float 到寄存器
        if (global_a_row < M && global_a_col + 3 < K) {
            // 完全在边界内，使用向量化加载
            float4 vec_a = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
            sA[load_a_row][load_a_col + 0] = vec_a.x;
            sA[load_a_row][load_a_col + 1] = vec_a.y;
            sA[load_a_row][load_a_col + 2] = vec_a.z;
            sA[load_a_row][load_a_col + 3] = vec_a.w;
        } else if (global_a_row < M && global_a_col < K) {
            // 边界处理：逐个元素检查并加载
            for (int i = 0; i < 4; ++i) {
                if (global_a_col + i < K) {
                    sA[load_a_row][load_a_col + i] = A[global_a_row * K + global_a_col + i];
                } else {
                    sA[load_a_row][load_a_col + i] = 0.0f;
                }
            }
        } else {
            // 完全越界，填充0
            sA[load_a_row][load_a_col + 0] = 0.0f;
            sA[load_a_row][load_a_col + 1] = 0.0f;
            sA[load_a_row][load_a_col + 2] = 0.0f;
            sA[load_a_row][load_a_col + 3] = 0.0f;
        }

        // --- 步骤 B: 向量化加载 B 块到 sB ---
        int global_b_row = k_offset + load_b_row;
        int global_b_col = bx * BN + load_b_col;

        // 使用 float4 一次性读取4个 float 到寄存器
        if (global_b_row < K && global_b_col + 3 < N) {
            // 完全在边界内，使用向量化加载
            float4 vec_b = FETCH_FLOAT4(B[global_b_row * N + global_b_col]);
            sB[load_b_row][load_b_col + 0] = vec_b.x;
            sB[load_b_row][load_b_col + 1] = vec_b.y;
            sB[load_b_row][load_b_col + 2] = vec_b.z;
            sB[load_b_row][load_b_col + 3] = vec_b.w;
        } else if (global_b_row < K && global_b_col < N) {
            // 边界处理：逐个元素检查并加载
            for (int j = 0; j < 4; ++j) {
                if (global_b_col + j < N) {
                    sB[load_b_row][load_b_col + j] = B[global_b_row * N + global_b_col + j];
                } else {
                    sB[load_b_row][load_b_col + j] = 0.0f;
                }
            }
        } else {
            // 完全越界，填充0
            sB[load_b_row][load_b_col + 0] = 0.0f;
            sB[load_b_row][load_b_col + 1] = 0.0f;
            sB[load_b_row][load_b_col + 2] = 0.0f;
            sB[load_b_row][load_b_col + 3] = 0.0f;
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
    // 【可选优化】此处也可使用 float4 向量化写入
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int global_row = row_start + i;
            int global_col = col_start + j;

            if (global_row < M && global_col + 3 < N) {
                // 使用 float4 向量化写入
                float4 vec_c;
                vec_c.x = alpha * accum[i][j + 0] + beta * C[global_row * N + global_col + 0];
                vec_c.y = alpha * accum[i][j + 1] + beta * C[global_row * N + global_col + 1];
                vec_c.z = alpha * accum[i][j + 2] + beta * C[global_row * N + global_col + 2];
                vec_c.w = alpha * accum[i][j + 3] + beta * C[global_row * N + global_col + 3];
                reinterpret_cast<float4*>(C)[(global_row * N + global_col) / 4] = vec_c;
            } else if (global_row < M && global_col < N) {
                // 边界处理：逐个元素写入
                for (int jj = 0; jj < 4 && global_col + jj < N; ++jj) {
                    C[global_row * N + global_col + jj] = alpha * accum[i][j + jj] + beta * C[global_row * N + global_col + jj];
                }
            }
        }
    }
}

// 包装函数
void run_sgemm_register_vectorized(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 使用 16x16 的线程块，每个线程计算 8x8，因此一个 Block 负责计算 128x128 的 C 矩阵块
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_vectorized_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

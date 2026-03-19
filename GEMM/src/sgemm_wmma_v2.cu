#include "common.h"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// ==========================================
// 算子 4: 极致优化版 WMMA Tensor Core GEMM
// 核心技术: 128x128大分块 + float4向量化 + Padding + 双缓冲流水线
// ==========================================

// 定义 Block 负责计算的输出块大小
#define BM 128
#define BN 128
#define BK 32 // 每次沿 K 维度前进 32

// 标准 WMMA Fragment 的大小 (16x16x16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 辅助宏：向量化访存
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

__global__ void sgemm_wmma_opt_kernel_v2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 1. 申请双缓冲共享内存，并加入 Padding(+8) 以消除 Bank Conflict
    __shared__ half sA[2][BM][BK + 8]; 
    __shared__ half sB[2][BK][BN + 8]; 

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0~255 (共256个线程, 8个Warp)

    int warpId = tid / 32; // Warp 编号 0~7
    
    // Warp 在 128x128 的输出块中负责 64x32 的子块
    // 将 8 个 Warp 排列为 2(M) x 4(N) 的网格
    int wy = warpId / 4; 
    int wx = warpId % 4; 
    int warp_row = wy * 64; 
    int warp_col = wx * 32; 

    // 2. 声明 WMMA 寄存器碎片 (Fragment)
    // 一个 Warp 负责 64x32，需要 4x2=8 个 16x16 的 Accumulator
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[4][2];
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // 3. 预计算从全局内存加载数据到共享内存的坐标 (256 线程协同)
    // A 块: 128x32，需 4096 元素，每个线程读 16 元素 (4 个 float4)
    int load_a_row = tid / 8;        // 256/8=32 行
    int load_a_col = (tid % 8) * 4;  // 每行 8个 float4 (32个元素)

    // B 块: 32x128，需 4096 元素，每个线程读 16 元素 (4 个 float4)
    int load_b_row = tid / 32;       // 256/32=8 行
    int load_b_col = (tid % 32) * 4; // 每行 32个 float4 (128个元素)

    int global_a_row = by * BM + load_a_row;
    int global_a_col = load_a_col;
    int global_b_row = load_b_row;
    int global_b_col = bx * BN + load_b_col;

    // --- 阶段 A (Prologue): 加载第 0 块数据并进行 float2half 转换 ---
    float4 vec_a[4], vec_b[4];
    
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int r = global_a_row + i * 32;
        int c = global_a_col;
        vec_a[i] = (r < M && c < K) ? FETCH_FLOAT4(A[r * K + c]) : make_float4(0,0,0,0);
    }
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int r = global_b_row + i * 8;
        int c = global_b_col;
        vec_b[i] = (r < K && c < N) ? FETCH_FLOAT4(B[r * N + c]) : make_float4(0,0,0,0);
    }

    // 存入 sA[0] 和 sB[0]
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        sA[0][load_a_row + i * 32][load_a_col + 0] = __float2half(vec_a[i].x);
        sA[0][load_a_row + i * 32][load_a_col + 1] = __float2half(vec_a[i].y);
        sA[0][load_a_row + i * 32][load_a_col + 2] = __float2half(vec_a[i].z);
        sA[0][load_a_row + i * 32][load_a_col + 3] = __float2half(vec_a[i].w);
    }
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        sB[0][load_b_row + i * 8][load_b_col + 0] = __float2half(vec_b[i].x);
        sB[0][load_b_row + i * 8][load_b_col + 1] = __float2half(vec_b[i].y);
        sB[0][load_b_row + i * 8][load_b_col + 2] = __float2half(vec_b[i].z);
        sB[0][load_b_row + i * 8][load_b_col + 3] = __float2half(vec_b[i].w);
    }
    
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // --- 阶段 B (Main Loop): 双缓冲流水线 ---
    for (int k_step = 1; k_step < (K + BK - 1) / BK; ++k_step) {
        
        // 1. (后台) 从全局内存读取【下一块】数据到寄存器 (掩盖延迟)
        global_a_col += BK;
        global_b_row += BK;
        
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            int r = global_a_row + i * 32;
            int c = global_a_col;
            vec_a[i] = (r < M && c < K) ? FETCH_FLOAT4(A[r * K + c]) : make_float4(0,0,0,0);
        }
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            int r = global_b_row + i * 8;
            int c = global_b_col;
            vec_b[i] = (r < K && c < N) ? FETCH_FLOAT4(B[r * N + c]) : make_float4(0,0,0,0);
        }

        // 2. (前台) 使用 Tensor Core 计算【当前块】的共享内存数据
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[4];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];

        // BK=32, 分 2 步 (k=0, 1) 送入 16x16x16 的 WMMA
        #pragma unroll
        for(int k = 0; k < 2; k++) {
            // 加载 A 的 4 个子块
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                wmma::load_matrix_sync(a_frag[i], &sA[read_idx][warp_row + i * 16][k * 16], BK + 8);
            }
            // 加载 B 的 2 个子块
            #pragma unroll
            for(int j = 0; j < 2; j++) {
                wmma::load_matrix_sync(b_frag[j], &sB[read_idx][k * 16][warp_col + j * 16], BN + 8);
            }
            // 矩阵乘加！
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        // 3. (后台) 将之前读取在寄存器的【下一块】数据转换并存入【另一个】共享内存缓冲
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            sA[write_idx][load_a_row + i * 32][load_a_col + 0] = __float2half(vec_a[i].x);
            sA[write_idx][load_a_row + i * 32][load_a_col + 1] = __float2half(vec_a[i].y);
            sA[write_idx][load_a_row + i * 32][load_a_col + 2] = __float2half(vec_a[i].z);
            sA[write_idx][load_a_row + i * 32][load_a_col + 3] = __float2half(vec_a[i].w);
        }
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            sB[write_idx][load_b_row + i * 8][load_b_col + 0] = __float2half(vec_b[i].x);
            sB[write_idx][load_b_row + i * 8][load_b_col + 1] = __float2half(vec_b[i].y);
            sB[write_idx][load_b_row + i * 8][load_b_col + 2] = __float2half(vec_b[i].z);
            sB[write_idx][load_b_row + i * 8][load_b_col + 3] = __float2half(vec_b[i].w);
        }

        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }

    // --- 阶段 C (Epilogue): 计算最后一块暂存在缓冲中的数据 ---
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[4];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];

    #pragma unroll
    for(int k = 0; k < 2; k++) {
        #pragma unroll
        for(int i = 0; i < 4; i++) wmma::load_matrix_sync(a_frag[i], &sA[read_idx][warp_row + i * 16][k * 16], BK + 8);
        #pragma unroll
        for(int j = 0; j < 2; j++) wmma::load_matrix_sync(b_frag[j], &sB[read_idx][k * 16][warp_col + j * 16], BN + 8);
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            #pragma unroll
            for(int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
    }

    // --- 阶段 D: 写回结果 ---
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int j = 0; j < 2; j++) {
            // 放缩 alpha 
            for(int t = 0; t < c_frag[i][j].num_elements; t++) c_frag[i][j].x[t] *= alpha;
            
            int g_row = by * BM + warp_row + i * 16;
            int g_col = bx * BN + warp_col + j * 16;
            
            // 写入 Global Memory
            if(g_row < M && g_col < N) {
                wmma::store_matrix_sync(&C[g_row * N + g_col], c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// 包装函数
void run_sgemm_wmma_v2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 使用 256 线程 (8 Warps)
    dim3 block(32, 8); 
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_wmma_opt_kernel_v2<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
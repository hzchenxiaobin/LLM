#include "../common.h"
#include "gemm_kernels.h"

// ============================================================
// Kernel 3: 2D Register Tiling GEMM
//
// Each thread computes a TMxTN sub-matrix of C using registers
// Block size: 16x16 threads
// Tile sizes: BM=128, BN=128, BK=8
// Each thread handles: TM=8, TN=8 → 64 output elements
// ============================================================

// Tile dimension constants
#define BM 128  // Block tile size in M dimension
#define BN 128  // Block tile size in N dimension
#define BK 8    // Inner dimension tile size (K)
#define TM 8    // Thread tile size in M dimension
#define TN 8    // Thread tile size in N dimension

__global__ void sgemm_register_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // -------------------------------------------------------------------
    // 1. Shared memory allocation for input tiles
    // -------------------------------------------------------------------
    __shared__ float shared_A[BM][BK];
    __shared__ float shared_B[BK][BN];

    // -------------------------------------------------------------------
    // 2. Register allocation for accumulation
    // Each thread accumulates TM × TN = 64 output values
    // -------------------------------------------------------------------
    float accum[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    // -------------------------------------------------------------------
    // 3. Calculate global and local coordinates
    // -------------------------------------------------------------------
    // Block's starting position in output matrix C
    const int block_row_c = blockIdx.y * BM;
    const int block_col_c = blockIdx.x * BN;

    // Thread's starting position within the block's tile
    const int thread_row_in_block = threadIdx.y * TM;
    const int thread_col_in_block = threadIdx.x * TN;

    // Thread's global starting position in output matrix C
    const int thread_row_global = block_row_c + thread_row_in_block;
    const int thread_col_global = block_col_c + thread_col_in_block;

    // Thread indexing for cooperative loading
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    // Calculate how many elements each thread loads
    const int a_elements_per_thread = (BM * BK) / num_threads;  // (128*8)/(16*16) = 4
    const int b_elements_per_thread = (BK * BN) / num_threads;  // (8*128)/(16*16) = 4

    // Stride for loading A and B tiles (for multi-pass loading)
    const int a_row_stride = num_threads / BK;  // 256/8 = 32
    const int b_row_stride = num_threads / BN;  // 256/128 = 2

    // -------------------------------------------------------------------
    // 4. Main loop: iterate over K dimension in BK-sized chunks
    // -------------------------------------------------------------------
    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += BK) {

        // ==================== Load A tile to shared memory ====================
        #pragma unroll
        for (int load_iter = 0; load_iter < a_elements_per_thread; ++load_iter) {
            const int local_row = (tid / BK) + load_iter * a_row_stride;
            const int local_col = tid % BK;

            const int global_row_a = block_row_c + local_row;
            const int global_col_a = k_tile_start + local_col;

            // Load with bounds checking
            if (global_row_a < M && global_col_a < K) {
                shared_A[local_row][local_col] = A[global_row_a * K + global_col_a];
            } else {
                shared_A[local_row][local_col] = 0.0f;
            }
        }

        // ==================== Load B tile to shared memory ====================
        #pragma unroll
        for (int load_iter = 0; load_iter < b_elements_per_thread; ++load_iter) {
            const int local_row = (tid / BN) + load_iter * b_row_stride;
            const int local_col = tid % BN;

            const int global_row_b = k_tile_start + local_row;
            const int global_col_b = block_col_c + local_col;

            // Load with bounds checking
            if (global_row_b < K && global_col_b < N) {
                shared_B[local_row][local_col] = B[global_row_b * N + global_col_b];
            } else {
                shared_B[local_row][local_col] = 0.0f;
            }
        }

        __syncthreads();

        // ==================== Register tiling computation ====================
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A and B fragments from shared memory to registers
            float frag_a[TM];
            float frag_b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                frag_a[i] = shared_A[thread_row_in_block + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                frag_b[j] = shared_B[k][thread_col_in_block + j];
            }

            // Outer product: accumulate TM × TN results
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

    // -------------------------------------------------------------------
    // 5. Write accumulated results to global memory (with alpha/beta scaling)
    // -------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int global_row = thread_row_global + i;
            const int global_col = thread_col_global + j;

            if (global_row < M && global_col < N) {
                const int idx = global_row * N + global_col;
                C[idx] = alpha * accum[i][j] + beta * C[idx];
            }
        }
    }
}

// ============================================================================
// Host wrapper function
// ============================================================================
void run_sgemm_register(
    int M, int N, int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    // Grid configuration:
    // - Block: 16x16 threads
    // - Each thread computes 8x8 = 64 elements
    // - Each block handles 128x128 = 16384 elements
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (N + BN - 1) / BN,  // Ceiling division for columns
        (M + BM - 1) / BM   // Ceiling division for rows
    );

    sgemm_register_kernel<<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}

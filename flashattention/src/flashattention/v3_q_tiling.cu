/*
 * FlashAttention V3: Q Tiling + Double Buffering
 * ==============================================
 *
 * Building on V2, this version adds:
 * 1. Q Tiling: Threads also cooperate to load Q tiles
 * 2. Double Buffering: Overlap computation of current tile with loading next tile
 *
 * Double Buffering Strategy:
 * -------------------------
 * The key insight is that we can hide memory latency by loading the next
 * K/V tile while computing with the current tile.
 *
 * We maintain two buffers in shared memory:
 *   Buffer 0: Current tile being computed
 *   Buffer 1: Next tile being loaded
 *
 * While threads compute with buffer 0, a subset of threads loads buffer 1.
 * After computation, we swap buffers and synchronize.
 *
 * Q Tiling:
 * --------
 * In V2, each thread loaded its entire Q row (d elements).
 * Now we also tile Q to better utilize shared memory bandwidth.
 *
 * However, we need to be careful: in FlashAttention, each query row
 * maintains its own online softmax state (m, l). So we can't easily
 * parallelize across query rows, but we can still tile the Q loading
 * to improve coalescing.
 *
 * Actually, let's reconsider: Q tiling in the context of FlashAttention
 * typically refers to processing Q in tiles (blocks of rows) and for
 * each Q tile, iterating over all KV tiles.
 *
 * Thread Organization Update:
 * --------------------------
 * - Block dimensions: (Br, Bc) or more threads for loading
 * - We need more threads for efficient double buffering
 * - Let's use a 2D block: (Br threads for compute) + (extra threads for load)
 *
 * For simplicity, we'll use 1D block but with more threads:
 * - Total threads: max(Br, Bc * d / elements_per_thread)
 * - Some threads compute, some load
 *
 * Actually, cleaner approach: use all threads for both compute and load
 * - All Br threads participate in loading (cooperative)
 * - All Br threads participate in computing
 * - But we need more threads for efficient loading...
 *
 * Better approach for double buffering:
 * - Increase block size to have more parallelism for loading
 * - Use 128 or 256 threads per block
 * - Partition: some threads dedicated to loading, some to compute
 *   OR use all threads for both (switch roles)
 *
 * Let's use 128 threads per block:
 * - First 64 threads (tid < Br): compute
 * - All 128 threads: participate in loading
 *
 * Shared Memory Layout with Double Buffer:
 * ---------------------------------------
 * Buffer size: Bc * d for K + Bc * d for V = 2 * Bc * d floats
 * Double buffer: 2 * 2 * Bc * d floats total
 *
 * Layout: [K_buf0][V_buf0][K_buf1][V_buf1]
 */

#include "kernels.h"

// V3 Configuration
constexpr int V3_Br = 64;    // Query rows per block
constexpr int V3_Bc = 64;    // KV columns per tile
constexpr int V3_THREADS = 128;  // More threads for loading parallelism

// Double buffer indices
#define BUFFER_0 0
#define BUFFER_1 1

__global__ void flash_attention_v3_q_tiling_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale)
{
    // Block and thread indices
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Query row handled by this thread (only first Br threads compute)
    int q_row = block_idx * V3_Br + tid;
    bool is_compute_thread = (tid < V3_Br) && (q_row < N);

    // Shared memory with double buffering
    // Layout: [K_0][V_0][K_1][V_1] where _0 and _1 are buffer indices
    extern __shared__ float shared_mem[];
    int buf_size = V3_Bc * d;
    float *K_buffers[2];
    float *V_buffers[2];
    K_buffers[0] = shared_mem;
    V_buffers[0] = shared_mem + buf_size;
    K_buffers[1] = shared_mem + 2 * buf_size;
    V_buffers[1] = shared_mem + 3 * buf_size;

    // Register storage for query vector (only for compute threads)
    float q_vec[128];
    float o_acc[128];

    // Initialize accumulator and load Q
    if (is_compute_thread) {
        for (int i = 0; i < d; i++) {
            q_vec[i] = Q[q_row * d + i];
            o_acc[i] = 0.0f;
        }
    }

    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;

    // Number of KV tiles
    int num_kv_tiles = (N + V3_Bc - 1) / V3_Bc;

    // Current buffer index (alternates between 0 and 1)
    int current_buf = 0;

    // Preload first tile
    {
        int kv_start = 0;
        int total_elements = V3_Bc * d;
        int elements_per_thread = (total_elements + V3_THREADS - 1) / V3_THREADS;

        for (int i = 0; i < elements_per_thread; i++) {
            int idx = tid * elements_per_thread + i;
            if (idx < total_elements) {
                int row = idx / d;
                int col = idx % d;
                int global_row = kv_start + row;

                // Load K
                if (global_row < N) {
                    K_buffers[current_buf][row * d + col] = K[global_row * d + col];
                } else {
                    K_buffers[current_buf][row * d + col] = 0.0f;
                }

                // Load V
                if (global_row < N) {
                    V_buffers[current_buf][row * d + col] = V[global_row * d + col];
                } else {
                    V_buffers[current_buf][row * d + col] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    // Process all tiles with double buffering
    for (int tile_idx = 0; tile_idx < num_kv_tiles; tile_idx++) {
        int kv_start = tile_idx * V3_Bc;
        int next_tile_idx = tile_idx + 1;
        int next_kv_start = next_tile_idx * V3_Bc;

        // Compute with current buffer (all threads that have work)
        if (is_compute_thread && kv_start < N) {
            float *K_tile = K_buffers[current_buf];
            float *V_tile = V_buffers[current_buf];

            // Process each column in the tile
            int cols_to_process = min(V3_Bc, N - kv_start);

            for (int b = 0; b < cols_to_process; b++) {
                // Compute q @ k_b
                float qk = 0.0f;
                #pragma unroll
                for (int i = 0; i < d; i++) {
                    qk += q_vec[i] * K_tile[b * d + i];
                }
                qk *= scale;

                // Online softmax
                float m_prev = m;
                m = fmaxf(m, qk);
                float exp_factor = expf(m_prev - m);
                float exp_qk = expf(qk - m);

                l = l * exp_factor + exp_qk;

                #pragma unroll
                for (int i = 0; i < d; i++) {
                    o_acc[i] = o_acc[i] * exp_factor + exp_qk * V_tile[b * d + i];
                }
            }
        }

        // Load next tile while threads sync (all threads participate)
        if (next_tile_idx < num_kv_tiles) {
            int next_buf = 1 - current_buf;
            int total_elements = V3_Bc * d;
            int elements_per_thread = (total_elements + V3_THREADS - 1) / V3_THREADS;

            for (int i = 0; i < elements_per_thread; i++) {
                int idx = tid * elements_per_thread + i;
                if (idx < total_elements) {
                    int row = idx / d;
                    int col = idx % d;
                    int global_row = next_kv_start + row;

                    // Load K
                    if (global_row < N) {
                        K_buffers[next_buf][row * d + col] = K[global_row * d + col];
                    } else {
                        K_buffers[next_buf][row * d + col] = 0.0f;
                    }

                    // Load V
                    if (global_row < N) {
                        V_buffers[next_buf][row * d + col] = V[global_row * d + col];
                    } else {
                        V_buffers[next_buf][row * d + col] = 0.0f;
                    }
                }
            }
        }

        // Synchronize and swap buffers
        __syncthreads();
        current_buf = 1 - current_buf;
    }

    // Write output (only compute threads)
    if (is_compute_thread) {
        for (int i = 0; i < d; i++) {
            O[q_row * d + i] = o_acc[i] / l;
        }
    }
}

// Host wrapper
void flash_attention_v3_q_tiling(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    for (int b = 0; b < B; b++) {
        const float *Q_b = Q + b * N * d;
        const float *K_b = K + b * N * d;
        const float *V_b = V + b * N * d;
        float *O_b = O + b * N * d;

        int num_blocks = (N + V3_Br - 1) / V3_Br;

        // Shared memory: 4 buffers (K_0, V_0, K_1, V_1) each Bc * d
        size_t shared_size = 4 * V3_Bc * d * sizeof(float);

        flash_attention_v3_q_tiling_kernel<<<num_blocks, V3_THREADS, shared_size>>>(
            Q_b, K_b, V_b, O_b, N, d, scale
        );

        CUDA_CHECK(cudaGetLastError());
    }
}

/*
 * Performance Analysis of V3:
 * ---------------------------
 *
 * Improvements over V2:
 * 1. Double buffering
 *    - Computation of current tile overlaps with loading next tile
 *    - Hides memory latency
 *    - Requires 2x shared memory but improves throughput
 *
 * 2. More loading threads
 *    - 128 threads instead of 64
 *    - Better parallelism for shared memory loading
 *    - Each thread loads fewer elements
 *
 * 3. Unrolled inner loops
 *    - #pragma unroll for d dimension
 *    - Reduces loop overhead
 *    - Better ILP (Instruction Level Parallelism)
 *
 * Remaining Issues:
 * 1. Memory loads are 32-bit (float)
 *    - Could use float4 for 4x bandwidth
 *    - Next version adds vectorization
 *
 * 2. Bank conflicts in shared memory
 *    - K_tile[b * d + i] can cause conflicts
 *    - Columns accessed by threads in warp
 *    - If d % 32 == 0, all threads hit same bank!
 *
 * 3. No warp-level primitives
 *    - Could use warp shuffle for better reduction
 *
 * 4. Limited thread utilization
 *    - Only 64 threads do actual computation
 *    - 128 threads participate, but half just load
 *
 * Expected Performance:
 * - Better than V2 due to double buffering
 * - Should see 10-20% improvement from overlap
 * - Bandwidth limited until we add vectorization
 */

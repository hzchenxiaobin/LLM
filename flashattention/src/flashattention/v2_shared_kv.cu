/*
 * FlashAttention V2: Shared Memory KV Tiling
 * ===========================================
 *
 * Building on V1, this version adds shared memory tiling for K and V.
 * This is the first major optimization that significantly improves performance.
 *
 * Key Insight:
 * -----------
 * In V1, every thread independently loaded K and V rows from global memory.
 * With 64 threads per block and N=1024, we loaded each K/V row 64 times!
 *
 * In V2, threads cooperate to load K and V tiles into shared memory,
 * then all threads in the block access the shared data.
 *
 * Tiling Strategy:
 * ---------------
 * - Block dimensions: (Br threads) x (Bc columns processed per thread)
 * - Each block handles Br rows of Q
 * - K and V are processed in tiles of size Bc x d
 * - Shared memory: 2 * Bc * d floats (double buffer for K and V tiles)
 *
 * Online Softmax with Tiling:
 * --------------------------
 * When processing a tile of KV (size Bc), we compute Bc qk values.
 * The online softmax needs to be updated for each value in the tile.
 *
 * For a tile with values [x_1, x_2, ..., x_Bc]:
 *   1. Find local max: m_tile = max(x_1, ..., x_Bc)
 *   2. Update global max: m_new = max(m_old, m_tile)
 *   3. Rescale previous accumulator: factor = exp(m_old - m_new)
 *   4. Compute new sum: l_new = l_old * factor + sum(exp(x_i - m_new))
 *   5. Update output: o_new = o_old * factor + sum(exp(x_i - m_new) * v_i)
 *
 * Memory Layout:
 * -------------
 * Shared memory layout for K tile (Bc x d):
 *   - Row-major: K_tile[b][i] where b in [0, Bc), i in [0, d)
 *   - Access pattern: threads in a warp access same column (broadcast)
 *
 * Thread Organization:
 * -------------------
 * - Block: (Br threads) - one thread per query row
 * - Each thread computes dot products with all columns in K tile
 * - Threads synchronize after loading each tile
 *
 * Performance Improvements:
 * -------------------------
 * - K/V rows loaded once per block, not once per thread
 * - Bc times reduction in global memory traffic for K/V
 * - For Bc=64, we reduce K/V traffic by 64x!
 */

#include "kernels.h"

// V2 Configuration
constexpr int V2_Br = 64;   // Query rows per block
constexpr int V2_Bc = 64;   // KV columns per tile
constexpr int V2_d = 64;    // Head dimension (compile-time constant for simplicity)

// Shared memory size: K tile + V tile
// Each tile is Bc x d floats
// We'll use a single buffer (no double buffering yet)
#define V2_SHARED_SIZE (V2_Bc * V2_d * 2)  // K and V tiles

__global__ void flash_attention_v2_shared_kv_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale)
{
    // Block and thread indices
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Each block handles Br query rows
    // Thread tid handles query row: block_idx * Br + tid
    int q_row = block_idx * V2_Br + tid;

    // Shared memory for K and V tiles
    // Layout: [K_tile (Bc x d)][V_tile (Bc x d)]
    extern __shared__ float shared_mem[];
    float *K_tile = shared_mem;
    float *V_tile = shared_mem + V2_Bc * d;

    // Register storage for this thread's query vector
    float q_vec[128];  // Max d assumed 128
    float o_acc[128];  // Output accumulator

    // Initialize output accumulator
    for (int i = 0; i < d; i++) {
        o_acc[i] = 0.0f;
    }

    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;

    // Load Q row into registers (coalesced)
    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            q_vec[i] = Q[q_row * d + i];
        }
    }

    // Iterate over KV tiles
    // Each tile covers Bc rows of K and V
    int num_kv_tiles = (N + V2_Bc - 1) / V2_Bc;

    for (int tile_idx = 0; tile_idx < num_kv_tiles; tile_idx++) {
        int kv_start = tile_idx * V2_Bc;

        // Step 1: Cooperative loading of K tile into shared memory
        // Threads cooperate: each thread loads d elements from K
        // Total elements: Bc * d
        // Threads: Br (we need more parallelism for loading)

        // For loading, we use all threads in the block
        // Each thread loads (Bc * d) / Br elements
        int load_per_thread = (V2_Bc * d + V2_Br - 1) / V2_Br;

        for (int i = 0; i < load_per_thread; i++) {
            int idx = tid * load_per_thread + i;
            if (idx < V2_Bc * d) {
                int k_row = idx / d;
                int k_col = idx % d;
                int global_k_row = kv_start + k_row;

                if (global_k_row < N) {
                    K_tile[k_row * d + k_col] = K[global_k_row * d + k_col];
                } else {
                    K_tile[k_row * d + k_col] = 0.0f;
                }
            }
        }

        // Step 2: Cooperative loading of V tile
        for (int i = 0; i < load_per_thread; i++) {
            int idx = tid * load_per_thread + i;
            if (idx < V2_Bc * d) {
                int v_row = idx / d;
                int v_col = idx % d;
                int global_v_row = kv_start + v_row;

                if (global_v_row < N) {
                    V_tile[v_row * d + v_col] = V[global_v_row * d + v_col];
                } else {
                    V_tile[v_row * d + v_col] = 0.0f;
                }
            }
        }

        // Synchronize to ensure all shared memory is loaded
        __syncthreads();

        // Step 3: Each thread processes all Bc columns in this tile
        // For each column in the tile, compute q @ k and update softmax
        if (q_row < N) {
            for (int b = 0; b < V2_Bc; b++) {
                // Compute q @ k_b (dot product with b-th row of K tile)
                float qk = 0.0f;
                for (int i = 0; i < d; i++) {
                    qk += q_vec[i] * K_tile[b * d + i];
                }
                qk *= scale;

                // Online softmax update
                float m_prev = m;
                m = fmaxf(m, qk);
                float exp_factor = expf(m_prev - m);
                float exp_qk = expf(qk - m);

                // Update l and o_acc
                l = l * exp_factor + exp_qk;

                // Update output accumulator
                for (int i = 0; i < d; i++) {
                    o_acc[i] = o_acc[i] * exp_factor + exp_qk * V_tile[b * d + i];
                }
            }
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write final output
    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            O[q_row * d + i] = o_acc[i] / l;
        }
    }
}

// Host wrapper
void flash_attention_v2_shared_kv(
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

        int num_blocks = (N + V2_Br - 1) / V2_Br;

        // Calculate shared memory size
        size_t shared_size = V2_Bc * d * 2 * sizeof(float);

        flash_attention_v2_shared_kv_kernel<<<num_blocks, V2_Br, shared_size>>>(
            Q_b, K_b, V_b, O_b, N, d, scale
        );

        CUDA_CHECK(cudaGetLastError());
    }
}

/*
 * Performance Analysis of V2:
 * ---------------------------
 *
 * Improvements over V1:
 * 1. Shared memory tiling for K and V
 *    - K/V rows loaded once per block, not per thread
 *    - Traffic reduction: ~64x for K/V (with Bc=64)
 *    - All threads in block share the same K/V tile
 *
 * 2. Better memory access patterns
 *    - Cooperative loading: threads load consecutive elements
 *    - Coalesced global memory access during tile loading
 *    - Shared memory broadcast for K/V access
 *
 * 3. Reuse within block
 *    - Bc x d tile reused by all Br threads
 *    - Exploits data reuse across query rows
 *
 * Remaining Issues:
 * 1. No Q tiling
 *    - Each thread still loads its entire Q row from global mem
 *    - Q is loaded once per thread (better than K/V in V1)
 *
 * 2. Sequential tile processing
 *    - No double buffering (compute stalls during load)
 *    - Next version adds double buffering
 *
 * 3. Bank conflicts
 *    - K_tile[b * d + i] has potential bank conflicts
 *    - Threads in warp access different banks for column i
 *    - Padding needed for conflict-free access
 *
 * 4. Limited parallelism
 *    - Only Br threads per block
 *    - Could use more threads for better occupancy
 *
 * Expected Performance:
 * - Significant improvement over V1
 * - Likely 5-10x faster than V1
 * - Still not optimal due to issues above
 */

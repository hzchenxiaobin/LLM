/*
 * FlashAttention V1: Naive Implementation with Online Softmax
 * =============================================================
 *
 * This is the most basic FlashAttention implementation. It demonstrates
 * the core online softmax algorithm without any memory optimization.
 *
 * Algorithm Overview:
 * -------------------
 * Standard Attention: O = softmax(QK^T / sqrt(d)) V
 *
 * FlashAttention uses tiling and online softmax to avoid materializing
 * the full NxN attention matrix in HBM.
 *
 * Online Softmax Technique:
 * -------------------------
 * Instead of computing softmax in three passes (max, exp, sum),
 * online softmax computes it in a single pass while maintaining
 * running statistics.
 *
 * For each tile j with values x_j:
 *   m_new = max(m_old, max(x_j))
 *   l_new = l_old * exp(m_old - m_new) + sum(exp(x_j - m_new))
 *
 * Where:
 *   m = running maximum
 *   l = running sum of exp(x - m)
 *
 * This allows us to compute softmax incrementally without storing
 * all intermediate values.
 *
 * Kernel Design:
 * --------------
 * - Each block handles Br rows of Q
 * - Each thread handles d elements of one query row
 * - Threads loop over entire KV sequence (no tiling yet)
 * - All accesses go to global memory (slow!)
 *
 * Performance Characteristics:
 * ---------------------------
 * - Memory bound: Every load from global memory
 * - No shared memory usage
 * - Simplest implementation, worst performance
 * - Useful for understanding the algorithm
 */

#include "kernels.h"

// Kernel configuration for V1
constexpr int V1_Br = 64;  // Rows per block (query block size)
constexpr int V1_Bc = 64;  // Cols for KV iteration

__global__ void flash_attention_v1_naive_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale)
{
    // Block index - each block handles a Br x d tile of O
    int block_idx = blockIdx.x;

    // Thread index within block
    int tid = threadIdx.x;

    // Each thread handles one row of Q (one query vector)
    // Block handles rows [block_idx * Br, block_idx * Br + Br)
    int q_row = block_idx * V1_Br + tid;

    // Shared memory for online softmax state
    // We'll use registers for per-thread state

    // Allocate local storage for this thread's query row
    // Each thread loads d elements from Q
    float q_vec[128];  // Max head dimension assumed 128
    float o_acc[128];  // Accumulator for output

    // Initialize output accumulator to zero
    for (int i = 0; i < d; i++) {
        o_acc[i] = 0.0f;
    }

    // Online softmax state
    float m = -INFINITY;  // Running max
    float l = 0.0f;       // Running sum

    // Load Q row into registers (coalesced access)
    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            q_vec[i] = Q[q_row * d + i];
        }
    }

    // Iterate over all KV positions (no tiling yet)
    // Each iteration processes a single KV position
    for (int k_idx = 0; k_idx < N; k_idx++) {
        // Step 1: Compute q @ k^T for this KV position
        // This is a dot product: sum(q[i] * k[i])
        float qk = 0.0f;

        // Load K row and compute dot product
        if (q_row < N) {
            for (int i = 0; i < d; i++) {
                float k_val = K[k_idx * d + i];
                qk += q_vec[i] * k_val;
            }
        }

        qk *= scale;

        // Step 2: Online softmax update
        float m_prev = m;
        float l_prev = l;

        // Update running max
        m = fmaxf(m, qk);

        // Compute exp factors
        float exp_prev = expf(m_prev - m);  // Rescale previous sum
        float exp_curr = expf(qk - m);       // Current value contribution

        // Update running sum
        l = l_prev * exp_prev + exp_curr;

        // Step 3: Update output accumulator
        // The contribution of this KV position to output
        // We need to incorporate the softmax weight
        if (q_row < N) {
            for (int i = 0; i < d; i++) {
                float v_val = V[k_idx * d + i];
                // Rescale previous accumulator and add new contribution
                o_acc[i] = o_acc[i] * exp_prev + exp_curr * v_val;
            }
        }
    }

    // Final normalization and write output
    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            O[q_row * d + i] = o_acc[i] / l;
        }
    }
}

// Host wrapper
void flash_attention_v1_naive(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    // For simplicity, process one batch at a time
    for (int b = 0; b < B; b++) {
        const float *Q_b = Q + b * N * d;
        const float *K_b = K + b * N * d;
        const float *V_b = V + b * N * d;
        float *O_b = O + b * N * d;

        // Grid: enough blocks to cover N rows with Br rows per block
        int num_blocks = (N + V1_Br - 1) / V1_Br;

        // Launch kernel with 1D block
        flash_attention_v1_naive_kernel<<<num_blocks, V1_Br>>>(
            Q_b, K_b, V_b, O_b, N, d, scale
        );

        CUDA_CHECK(cudaGetLastError());
    }
}

/*
 * Performance Analysis of V1:
 * ---------------------------
 *
 * Problems:
 * 1. Global memory bandwidth bottleneck
 *    - Each thread loads entire K and V rows from global memory
 *    - No reuse between threads in the same block
 *    - For N=1024, d=64: each thread loads 1024*64*2 = 131K floats from global mem
 *
 * 2. Memory access pattern
 *    - K and V access is sequential but not coalesced well across threads
 *    - Each thread does its own independent loads
 *
 * 3. No parallelism in KV dimension
 *    - Sequential loop over N KV positions
 *    - No thread cooperation
 *
 * 4. Occupancy
 *    - Only 64 threads per block
 *    - Limited by register usage (q_vec[128] + o_acc[128] = 256 floats per thread)
 *
 * Expected Performance (RTX 4090/5090):
 * - Very slow, likely < 10% of peak bandwidth
 * - Mainly for educational purposes
 * - Next version adds shared memory tiling
 *
 * Roofline Analysis:
 * - Arithmetic Intensity: O(1) (compute is O(N*d), memory is O(N*d))
 * - Memory bound region
 * - Not utilizing shared memory at all
 */

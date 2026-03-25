/*
 * FlashAttention V4: Vectorized Loads + Bank Conflict Free
 * ==========================================================
 *
 * Building on V3, this version adds:
 * 1. Vectorized memory loads (float4) for 4x bandwidth
 * 2. Bank conflict elimination through padding
 *
 * Vectorized Memory Access:
 * ------------------------
 * Modern GPUs have 128-bit load/store units. Using float4 (4 floats = 128 bits)
 * allows us to utilize the full bandwidth of these units.
 *
 * Without vectorization:
 *   - Each float load uses 1/4 of the memory bus width
 *   - 4 separate instructions needed
 *   - Wasted memory bandwidth
 *
 * With float4:
 *   - One instruction loads 4 floats
 *   - Full utilization of memory bus
 *   - 4x fewer memory instructions
 *
 * Shared Memory Bank Conflicts:
 * ---------------------------
 * Shared memory is organized into 32 banks (on modern GPUs).
 * When threads in a warp access different banks, they run in parallel.
 * When multiple threads access the same bank, they serialize (conflict).
 *
 * Layout: K_tile[row][col] = K_tile[row * d + col]
 *
 * If d = 64 (multiple of 32):
 *   - Thread 0 accesses column 0: bank 0
 *   - Thread 1 accesses column 1: bank 1
 *   - ...
 *   - Thread 31 accesses column 31: bank 31
 *   - Thread 32 accesses column 32: bank 0 (CONFLICT with thread 0!)
 *
 * Solution: Add padding to make row stride not a multiple of 32
 *   - Padded d' = d + 1 (or d + padding)
 *   - Thread 32 accesses column 32: bank 1 (no conflict!)
 *
 * Implementation Details:
 * ---------------------
 * 1. Global memory loads: Use float4 when possible
 * 2. Shared memory layout: Pad row stride to avoid conflicts
 * 3. Compute: Still use float, but load/store with float4
 *
 * Shared Memory Layout with Padding:
 * ---------------------------------
 * Original: K_tile[row * d + col]
 * Padded:   K_tile[row * (d + PAD) + col]
 *
 * Where PAD is chosen so (d + PAD) % 32 != 0
 * For d = 64, we can use PAD = 1, giving stride 65
 */

#include "kernels.h"

// V4 Configuration
constexpr int V4_Br = 64;
constexpr int V4_Bc = 64;
constexpr int V4_THREADS = 128;

// Padding for bank conflict elimination
// Make sure (d + SMEM_PAD) % 32 != 0
constexpr int SMEM_PAD = 1;

// Helper: Load float4 from global memory
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

// Helper: Store float4 to global memory
__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

__global__ void flash_attention_v4_vectorized_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale)
{
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int q_row = block_idx * V4_Br + tid;
    bool is_compute_thread = (tid < V4_Br) && (q_row < N);

    // Padded shared memory dimensions
    int d_padded = d + SMEM_PAD;
    int buf_size = V4_Bc * d_padded;

    extern __shared__ float shared_mem[];
    float *K_buffers[2];
    float *V_buffers[2];
    K_buffers[0] = shared_mem;
    V_buffers[0] = shared_mem + buf_size;
    K_buffers[1] = shared_mem + 2 * buf_size;
    V_buffers[1] = shared_mem + 3 * buf_size;

    // Register storage
    float q_vec[128];
    float o_acc[128];

    // Load Q with vectorization
    if (is_compute_thread) {
        // Load using float4 where possible
        int d_vec4 = d / 4;  // Number of float4 elements
        int d_remainder = d % 4;

        const float4* Q_vec4 = reinterpret_cast<const float4*>(Q + q_row * d);
        float* q_ptr = q_vec;

        // Load float4 chunks
        for (int i = 0; i < d_vec4; i++) {
            float4 val = load_float4(reinterpret_cast<const float*>(Q_vec4 + i));
            q_ptr[0] = val.x;
            q_ptr[1] = val.y;
            q_ptr[2] = val.z;
            q_ptr[3] = val.w;
            q_ptr += 4;
        }

        // Load remaining elements
        for (int i = d_vec4 * 4; i < d; i++) {
            *q_ptr++ = Q[q_row * d + i];
        }

        for (int i = 0; i < d; i++) {
            o_acc[i] = 0.0f;
        }
    }

    float m = -INFINITY;
    float l = 0.0f;

    int num_kv_tiles = (N + V4_Bc - 1) / V4_Bc;
    int current_buf = 0;

    // Helper lambda for loading tiles with vectorization and padding
    auto load_tile_vectorized = [&](int buf_idx, int kv_start) {
        // Total elements in tile (with padding)
        int total_elements = V4_Bc * d;

        // Use float4 loading: each thread loads 4 elements at a time
        int float4_per_thread = (total_elements / 4 + V4_THREADS - 1) / V4_THREADS;

        for (int i = 0; i < float4_per_thread; i++) {
            int idx4 = tid * float4_per_thread + i;
            int base_idx = idx4 * 4;

            if (base_idx < total_elements) {
                int row = base_idx / d;
                int col = base_idx % d;
                int global_row = kv_start + row;

                // Load 4 floats from K
                if (global_row < N && col + 3 < d) {
                    const float4 k_val = load_float4(&K[global_row * d + col]);
                    // Store to padded shared memory
                    K_buffers[buf_idx][row * d_padded + col] = k_val.x;
                    K_buffers[buf_idx][row * d_padded + col + 1] = k_val.y;
                    K_buffers[buf_idx][row * d_padded + col + 2] = k_val.z;
                    K_buffers[buf_idx][row * d_padded + col + 3] = k_val.w;
                } else if (global_row < N) {
                    // Handle boundary with scalar loads
                    for (int c = 0; c < 4 && col + c < d; c++) {
                        K_buffers[buf_idx][row * d_padded + col + c] = K[global_row * d + col + c];
                    }
                }

                // Load 4 floats from V
                if (global_row < N && col + 3 < d) {
                    const float4 v_val = load_float4(&V[global_row * d + col]);
                    V_buffers[buf_idx][row * d_padded + col] = v_val.x;
                    V_buffers[buf_idx][row * d_padded + col + 1] = v_val.y;
                    V_buffers[buf_idx][row * d_padded + col + 2] = v_val.z;
                    V_buffers[buf_idx][row * d_padded + col + 3] = v_val.w;
                } else if (global_row < N) {
                    for (int c = 0; c < 4 && col + c < d; c++) {
                        V_buffers[buf_idx][row * d_padded + col + c] = V[global_row * d + col + c];
                    }
                }
            }
        }

        // Handle remaining elements (if d not divisible by 4)
        int remainder_start = (total_elements / 4) * 4;
        int remainder_elements = total_elements - remainder_start;
        int remainder_per_thread = (remainder_elements + V4_THREADS - 1) / V4_THREADS;

        for (int i = 0; i < remainder_per_thread; i++) {
            int idx = remainder_start + tid * remainder_per_thread + i;
            if (idx < total_elements) {
                int row = idx / d;
                int col = idx % d;
                int global_row = kv_start + row;

                if (global_row < N) {
                    K_buffers[buf_idx][row * d_padded + col] = K[global_row * d + col];
                    V_buffers[buf_idx][row * d_padded + col] = V[global_row * d + col];
                }
            }
        }
    };

    // Preload first tile
    load_tile_vectorized(current_buf, 0);
    __syncthreads();

    // Process tiles
    for (int tile_idx = 0; tile_idx < num_kv_tiles; tile_idx++) {
        int kv_start = tile_idx * V4_Bc;
        int next_kv_start = (tile_idx + 1) * V4_Bc;

        // Compute
        if (is_compute_thread && kv_start < N) {
            float *K_tile = K_buffers[current_buf];
            float *V_tile = V_buffers[current_buf];
            int cols_to_process = min(V4_Bc, N - kv_start);

            for (int b = 0; b < cols_to_process; b++) {
                // Access with padded stride: b * d_padded + i
                float qk = 0.0f;
                #pragma unroll
                for (int i = 0; i < d; i++) {
                    qk += q_vec[i] * K_tile[b * d_padded + i];
                }
                qk *= scale;

                float m_prev = m;
                m = fmaxf(m, qk);
                float exp_factor = expf(m_prev - m);
                float exp_qk = expf(qk - m);

                l = l * exp_factor + exp_qk;

                #pragma unroll
                for (int i = 0; i < d; i++) {
                    o_acc[i] = o_acc[i] * exp_factor + exp_qk * V_tile[b * d_padded + i];
                }
            }
        }

        // Load next tile
        if (tile_idx + 1 < num_kv_tiles) {
            load_tile_vectorized(1 - current_buf, next_kv_start);
        }

        __syncthreads();
        current_buf = 1 - current_buf;
    }

    // Write output with vectorization
    if (is_compute_thread) {
        int d_vec4 = d / 4;
        float4* O_vec4 = reinterpret_cast<float4*>(O + q_row * d);
        float* o_ptr = o_acc;

        // Store float4 chunks
        for (int i = 0; i < d_vec4; i++) {
            float4 val;
            val.x = o_ptr[0] / l;
            val.y = o_ptr[1] / l;
            val.z = o_ptr[2] / l;
            val.w = o_ptr[3] / l;
            store_float4(reinterpret_cast<float*>(O_vec4 + i), val);
            o_ptr += 4;
        }

        // Store remaining elements
        for (int i = d_vec4 * 4; i < d; i++) {
            O[q_row * d + i] = o_acc[i] / l;
        }
    }
}

// Host wrapper
void flash_attention_v4_vectorized(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    // Check alignment for vectorization
    if (d % 4 != 0) {
        printf("Warning: d=%d is not divisible by 4, vectorization may be inefficient\n", d);
    }

    for (int b = 0; b < B; b++) {
        const float *Q_b = Q + b * N * d;
        const float *K_b = K + b * N * d;
        const float *V_b = V + b * N * d;
        float *O_b = O + b * N * d;

        int num_blocks = (N + V4_Br - 1) / V4_Br;

        // Shared memory with padding
        int d_padded = d + SMEM_PAD;
        size_t shared_size = 4 * V4_Bc * d_padded * sizeof(float);

        flash_attention_v4_vectorized_kernel<<<num_blocks, V4_THREADS, shared_size>>>(
            Q_b, K_b, V_b, O_b, N, d, scale
        );

        CUDA_CHECK(cudaGetLastError());
    }
}

/*
 * Performance Analysis of V4:
 * ---------------------------
 *
 * Improvements over V3:
 * 1. Vectorized global memory loads (float4)
 *    - 4x fewer load instructions
 *    - Better utilization of memory bus (128-bit)
 *    - Significant bandwidth improvement
 *
 * 2. Bank conflict elimination
 *    - Added padding to shared memory
 *    - (d + PAD) % 32 != 0 ensures conflict-free access
 *    - Threads in warp access different banks
 *
 * 3. Vectorized stores
 *    - Output writes also use float4
 *    - Reduces store instruction count
 *
 * Remaining Issues:
 * 1. Still FP32 computation
 *    - RTX 5090 has strong FP16/BF16 Tensor Core support
 *    - Next version adds Tensor Core usage
 *
 * 2. Thread utilization
 *    - Still only using 64 threads for compute
 *    - Could parallelize more aggressively
 *
 * 3. No warp-level reduction
 *    - Using per-thread state for softmax
 *    - Could use warp shuffle for some operations
 *
 * Expected Performance:
 * - Significant improvement from vectorization
 * - 2-3x bandwidth improvement expected
 * - Should approach memory bandwidth limit
 * - Good foundation for Tensor Core version
 */

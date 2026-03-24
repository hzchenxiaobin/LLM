// V5: Blackwell (RTX 5090) Architecture Optimizations - Kernel implementations only
// Based on TUTORIAL.md Chapter V5

#include "../include/topk_common.h"

#define WARP_SIZE 32

// Device function: Insert into top-k (optimized for Blackwell)
template <int K>
__device__ __forceinline__ void insert_topk_blackwell(TopKNode* local_top, float value, int index) {
    if (value <= local_top[K - 1].value) {
        return;
    }
    int j = K - 1;
    while (j > 0 && value > local_top[j - 1].value) {
        local_top[j] = local_top[j - 1];
        j--;
    }
    local_top[j].value = value;
    local_top[j].index = index;
}

// Device function: Warp shuffle merge
template <int K>
__device__ __forceinline__ void warp_shuffle_merge_blackwell(TopKNode* local_top) {
    for (int step = 16; step > 0; step /= 2) {
        TopKNode received[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            received[i].value = __shfl_down_sync(0xffffffff, local_top[i].value, step);
            received[i].index = __shfl_down_sync(0xffffffff, local_top[i].index, step);
        }

        // Merge two sorted arrays
        TopKNode merged[2 * K];
        int a = 0, b = 0;
        for (int i = 0; i < 2 * K; i++) {
            if (b >= K || (a < K && local_top[a].value > received[b].value)) {
                merged[i] = local_top[a++];
            } else {
                merged[i] = received[b++];
            }
        }

        #pragma unroll
        for (int i = 0; i < K; i++) {
            local_top[i] = merged[i];
        }
    }
}

// V5.1: Blackwell optimized top-k kernel
template <int K, int BLOCK_THREADS>
__global__ void topk_v5_blackwell_kernel(const float* __restrict__ input,
                                        int N,
                                        TopKNode* block_tops) {
    const int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;
    extern __shared__ TopKNode smem[];

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int global_tid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Local top-k in registers
    TopKNode local_top[K];
    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_top[i].value = -1e20f;
        local_top[i].index = -1;
    }

    // Grid-stride loop
    const float* aligned_input = input;
    for (int idx = global_tid; idx < N; idx += stride) {
        float val = aligned_input[idx];
        insert_topk_blackwell<K>(local_top, val, idx);
    }

    // Warp reduction
    warp_shuffle_merge_blackwell<K>(local_top);

    // Warp leader writes to shared memory
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            smem[warp_id * K + i] = local_top[i];
        }
    }
    __syncthreads();

    // Block reduction - only warp 0
    if (warp_id == 0) {
        TopKNode block_top[K];
        if (lane_id < WARPS_PER_BLOCK) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                block_top[i] = smem[lane_id * K + i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                block_top[i].value = -1e20f;
                block_top[i].index = -1;
            }
        }

        warp_shuffle_merge_blackwell<K>(block_top);

        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                block_tops[blockIdx.x * K + i] = block_top[i];
            }
        }
    }
}

// V5.2: Final reduction kernel
template <int K>
__global__ void topk_v5_reduce_kernel(const TopKNode* block_tops,
                                       int num_blocks,
                                       TopKNode* output) {
    const int WARP_SIZE_LOCAL = 32;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE_LOCAL;
    int warp_id = tid / WARP_SIZE_LOCAL;
    int num_warps = blockDim.x / WARP_SIZE_LOCAL;

    extern __shared__ TopKNode smem[];

    // Load block results
    TopKNode local_top[K];
    if (tid < num_blocks) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            local_top[i] = block_tops[tid * K + i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            local_top[i].value = -1e20f;
            local_top[i].index = -1;
        }
    }

    // Warp shuffle reduction
    warp_shuffle_merge_blackwell<K>(local_top);

    // Lane 0 writes to shared memory
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            smem[warp_id * K + i] = local_top[i];
        }
    }
    __syncthreads();

    // Warp 0 final reduction
    if (warp_id == 0) {
        TopKNode final_top[K];
        if (lane_id < num_warps) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                final_top[i] = smem[lane_id * K + i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                final_top[i].value = -1e20f;
                final_top[i].index = -1;
            }
        }

        warp_shuffle_merge_blackwell<K>(final_top);

        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                output[i] = final_top[i];
            }
        }
    }
}

// V5.3: Pipeline optimized kernel for large datasets
template <int K, int BLOCK_THREADS, int CHUNK_SIZE>
__global__ void topk_v5_pipeline_kernel(const float* __restrict__ input,
                                        int N,
                                        TopKNode* block_tops) {
    const int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;
    extern __shared__ TopKNode smem[];

    // Extra shared memory for data chunk cache
    float* s_data = (float*)&smem[WARPS_PER_BLOCK * K];

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    TopKNode local_top[K];
    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_top[i].value = -1e20f;
        local_top[i].index = -1;
    }

    // Calculate block's data range
    int elements_per_block = (N + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * elements_per_block;
    int block_end = min(block_start + elements_per_block, N);

    // Cooperative loading and processing
    for (int base_idx = block_start; base_idx < block_end; base_idx += CHUNK_SIZE) {
        int chunk_end = min(base_idx + CHUNK_SIZE, block_end);
        int chunk_size = chunk_end - base_idx;

        // Cooperative load to shared memory
        for (int i = tid; i < chunk_size; i += BLOCK_THREADS) {
            s_data[i] = input[base_idx + i];
        }
        __syncthreads();

        // Process from shared memory
        for (int i = tid; i < chunk_size; i += BLOCK_THREADS) {
            float val = s_data[i];
            int global_idx = base_idx + i;
            insert_topk_blackwell<K>(local_top, val, global_idx);
        }
        __syncthreads();
    }

    // Warp reduction
    warp_shuffle_merge_blackwell<K>(local_top);

    // Warp leader writes to shared memory
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            smem[warp_id * K + i] = local_top[i];
        }
    }
    __syncthreads();

    // Block reduction
    if (warp_id == 0) {
        TopKNode block_top[K];
        if (lane_id < WARPS_PER_BLOCK) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                block_top[i] = smem[lane_id * K + i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                block_top[i].value = -1e20f;
                block_top[i].index = -1;
            }
        }

        warp_shuffle_merge_blackwell<K>(block_top);

        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                block_tops[blockIdx.x * K + i] = block_top[i];
            }
        }
    }
}

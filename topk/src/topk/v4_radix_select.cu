// V4: Radix Select (基数选择算法) - Kernel implementations only
// 基于 TUTORIAL.md 章节 V4

#include "../include/topk_common.h"

#define RADIX_BITS 8
#define RADIX_BUCKETS 256
#define RADIX_MASK 0xFF

// Convert float to sortable unsigned int (for descending order)
__device__ __forceinline__ unsigned int float_to_sortable_uint(float value) {
    unsigned int bits;
    memcpy(&bits, &value, sizeof(float));
    // Flip sign bit and adjust for descending order
    if (bits & 0x80000000) {
        bits = ~bits;
    } else {
        bits |= 0x80000000;
    }
    return bits;
}

// Kernel: Compute histogram of radix values
template <int BLOCK_SIZE>
__global__ void radix_histogram_kernel(const float* input, int N,
                                        unsigned int* histograms,
                                        int bit_start,
                                        unsigned int current_threshold) {
    __shared__ unsigned int s_histogram[RADIX_BUCKETS];

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared histogram
    for (int i = tid; i < RADIX_BUCKETS; i += blockDim.x) {
        s_histogram[i] = 0;
    }
    __syncthreads();

    // Process input data
    for (int idx = gid; idx < N; idx += stride) {
        float val = input[idx];
        unsigned int bits = float_to_sortable_uint(val);

        // Check threshold condition
        bool include = true;
        if (bit_start < 24) {
            unsigned int high_bits = bits >> (bit_start + RADIX_BITS);
            unsigned int threshold_high = current_threshold >> RADIX_BITS;
            include = (high_bits == threshold_high);
        }

        if (include) {
            unsigned int key = (bits >> bit_start) & RADIX_MASK;
            atomicAdd(&s_histogram[key], 1);
        }
    }
    __syncthreads();

    // Write to global memory
    unsigned int* block_hist = histograms + blockIdx.x * RADIX_BUCKETS;
    for (int i = tid; i < RADIX_BUCKETS; i += blockDim.x) {
        block_hist[i] = s_histogram[i];
    }
}

// Kernel: Merge block histograms into global histogram
__global__ void merge_histograms_kernel(const unsigned int* block_histograms,
                                         unsigned int* global_histogram,
                                         int num_blocks) {
    int tid = threadIdx.x;

    for (int bucket = tid; bucket < RADIX_BUCKETS; bucket += blockDim.x) {
        unsigned int sum = 0;
        for (int b = 0; b < num_blocks; b++) {
            sum += block_histograms[b * RADIX_BUCKETS + bucket];
        }
        global_histogram[bucket] = sum;
    }
}

// Kernel: Collect candidates meeting threshold
__global__ void collect_candidates_kernel(const float* input, int N,
                                            TopKNode* candidates,
                                            int* candidate_count,
                                            unsigned int threshold_bits,
                                            int num_bits_checked) {
    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = gid; idx < N; idx += stride) {
        float val = input[idx];
        unsigned int bits = float_to_sortable_uint(val);

        unsigned int checked_bits = bits >> (32 - num_bits_checked);
        unsigned int threshold_checked = threshold_bits >> (32 - num_bits_checked);

        if (checked_bits == threshold_checked) {
            int pos = atomicAdd(candidate_count, 1);
            if (pos < N) {
                candidates[pos].value = val;
                candidates[pos].index = idx;
            }
        }
    }
}

// Device function: Insert into top-k
__device__ __forceinline__ void dev_insert_topk(TopKNode* local_top, int K, float value, int index) {
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

// Kernel: Final top-k selection from candidates
template <int K>
__global__ void final_select_topk_kernel(TopKNode* candidates,
                                        int candidate_count,
                                        TopKNode* output) {
    extern __shared__ TopKNode smem[];
    int tid = threadIdx.x;

    // Local top-k
    TopKNode local_top[K];
    for (int i = 0; i < K; i++) {
        local_top[i].value = -1e20f;
        local_top[i].index = -1;
    }

    // Process candidates
    int stride = blockDim.x;
    for (int idx = tid; idx < candidate_count; idx += stride) {
        float val = candidates[idx].value;
        int index = candidates[idx].index;
        dev_insert_topk(local_top, K, val, index);
    }

    // Store to shared memory
    for (int i = 0; i < K; i++) {
        smem[tid * K + i] = local_top[i];
    }
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            TopKNode temp[2 * K];
            for (int i = 0; i < K; i++) {
                temp[i] = smem[tid * K + i];
                temp[i + K] = smem[(tid + s) * K + i];
            }

            // Sort first K
            for (int i = 0; i < K; i++) {
                int max_idx = i;
                for (int j = i + 1; j < 2 * K; j++) {
                    if (temp[j].value > temp[max_idx].value) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    TopKNode swap = temp[i];
                    temp[i] = temp[max_idx];
                    temp[max_idx] = swap;
                }
            }

            for (int i = 0; i < K; i++) {
                smem[tid * K + i] = temp[i];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes output
    if (tid == 0) {
        for (int i = 0; i < K; i++) {
            output[i] = smem[i];
        }
    }
}

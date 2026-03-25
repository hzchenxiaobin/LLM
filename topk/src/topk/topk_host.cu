// Host wrapper functions for all Top-K implementations
// Separated from kernel implementations for cleaner code organization

#include "../include/topk_common.h"

// Thrust headers needed for V0
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

// ============================================================================
// V0: Thrust Baseline (host-only implementation)
// ============================================================================

struct greater_float_with_index {
    __host__ __device__
    bool operator()(const TopKNode& a, const TopKNode& b) const {
        return a.value > b.value;
    }
};

void topk_v0_thrust(const float* d_input, int N, int K, TopKNode* d_output) {
    thrust::device_vector<TopKNode> d_vec(N);
    TopKNode* d_vec_ptr = thrust::raw_pointer_cast(d_vec.data());

    // Initialize with values and indices
    thrust::counting_iterator<int> start(0);
    thrust::counting_iterator<int> end(N);
    thrust::for_each(thrust::device, start, end,
        [=] __device__ (int idx) {
            d_vec_ptr[idx].value = d_input[idx];
            d_vec_ptr[idx].index = idx;
        });

    // Sort descending
    thrust::sort(thrust::device, d_vec.begin(), d_vec.end(), greater_float_with_index());

    // Copy top K
    thrust::copy(d_vec.begin(), d_vec.begin() + K, d_output);
}

// ============================================================================
// External kernel declarations from v1_thread.cu
// ============================================================================

template <int K>
__global__ void topk_v1_kernel(const float* input, int N, TopKNode* thread_tops);

template <int K>
__global__ void topk_v1_reduce_kernel(const TopKNode* thread_tops, int num_threads, TopKNode* output);

// ============================================================================
// V1 Host Function: Thread-level
// ============================================================================

void topk_v1_thread(const float* d_input, int N, int K, TopKNode* d_output,
                    int num_blocks, int num_threads) {
    int total_threads = num_blocks * num_threads;

    TopKNode* d_thread_tops;
    CUDA_CHECK(cudaMalloc(&d_thread_tops, total_threads * K * sizeof(TopKNode)));

    switch (K) {
        case 8:
            topk_v1_kernel<8><<<num_blocks, num_threads>>>(d_input, N, d_thread_tops);
            break;
        case 16:
            topk_v1_kernel<16><<<num_blocks, num_threads>>>(d_input, N, d_thread_tops);
            break;
        case 32:
            topk_v1_kernel<32><<<num_blocks, num_threads>>>(d_input, N, d_thread_tops);
            break;
        case 64:
            topk_v1_kernel<64><<<num_blocks, num_threads>>>(d_input, N, d_thread_tops);
            break;
        case 128:
            topk_v1_kernel<128><<<num_blocks, num_threads>>>(d_input, N, d_thread_tops);
            break;
        case 256:
            topk_v1_kernel<256><<<num_blocks, num_threads>>>(d_input, N, d_thread_tops);
            break;
        default:
            topk_v1_kernel<256><<<num_blocks, num_threads>>>(d_input, N, d_thread_tops);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int reduce_threads = 256;
    while (reduce_threads > total_threads) {
        reduce_threads >>= 1;
    }
    reduce_threads = max(reduce_threads, 32);

    size_t smem_size = reduce_threads * K * sizeof(TopKNode);

    switch (K) {
        case 8:
            topk_v1_reduce_kernel<8><<<1, reduce_threads, smem_size>>>(d_thread_tops, total_threads, d_output);
            break;
        case 16:
            topk_v1_reduce_kernel<16><<<1, reduce_threads, smem_size>>>(d_thread_tops, total_threads, d_output);
            break;
        case 32:
            topk_v1_reduce_kernel<32><<<1, reduce_threads, smem_size>>>(d_thread_tops, total_threads, d_output);
            break;
        case 64:
            topk_v1_reduce_kernel<64><<<1, reduce_threads, smem_size>>>(d_thread_tops, total_threads, d_output);
            break;
        case 128:
            topk_v1_reduce_kernel<128><<<1, reduce_threads, smem_size>>>(d_thread_tops, total_threads, d_output);
            break;
        case 256:
            topk_v1_reduce_kernel<256><<<1, reduce_threads, smem_size>>>(d_thread_tops, total_threads, d_output);
            break;
        default:
            topk_v1_reduce_kernel<256><<<1, reduce_threads, smem_size>>>(d_thread_tops, total_threads, d_output);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_thread_tops));
}

// ============================================================================
// External kernel declarations from v2_shared_memory.cu
// ============================================================================

template <int K>
__global__ void topk_v2_block_kernel(const float* input, int N, TopKNode* block_tops);

template <int K>
__global__ void topk_v2_reduce_kernel(const TopKNode* block_tops, int num_blocks, TopKNode* output);

// ============================================================================
// V2 Host Function: Shared Memory
// ============================================================================

void topk_v2_shared_memory(const float* d_input, int N, int K, TopKNode* d_output,
                           int num_blocks, int num_threads) {
    TopKNode* d_block_tops;
    CUDA_CHECK(cudaMalloc(&d_block_tops, num_blocks * K * sizeof(TopKNode)));

    size_t smem_size = num_threads * K * sizeof(TopKNode);

    switch (K) {
        case 8:
            topk_v2_block_kernel<8><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 16:
            topk_v2_block_kernel<16><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 32:
            topk_v2_block_kernel<32><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 64:
            topk_v2_block_kernel<64><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 128:
            topk_v2_block_kernel<128><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 256:
            topk_v2_block_kernel<256><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        default:
            topk_v2_block_kernel<256><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int reduce_threads = 256;
    while (reduce_threads > num_blocks) {
        reduce_threads >>= 1;
    }
    reduce_threads = max(reduce_threads, 32);

    size_t reduce_smem_size = reduce_threads * K * sizeof(TopKNode);

    switch (K) {
        case 8:
            topk_v2_reduce_kernel<8><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 16:
            topk_v2_reduce_kernel<16><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 32:
            topk_v2_reduce_kernel<32><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 64:
            topk_v2_reduce_kernel<64><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 128:
            topk_v2_reduce_kernel<128><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 256:
            topk_v2_reduce_kernel<256><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        default:
            topk_v2_reduce_kernel<256><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_block_tops));
}

// ============================================================================
// External kernel declarations from v3_warp_shuffle.cu
// ============================================================================

template <int K>
__global__ void topk_v3_warp_kernel(const float* input, int N, TopKNode* block_tops);

template <int K>
__global__ void topk_v3_warp_shuffle_kernel(const float* input, int N, TopKNode* block_tops);

template <int K>
__global__ void topk_v3_reduce_kernel(const TopKNode* block_tops, int num_blocks, TopKNode* output);

// ============================================================================
// V3 Host Function: Warp Shuffle
// ============================================================================

void topk_v3_warp_shuffle(const float* d_input, int N, int K, TopKNode* d_output,
                          int num_blocks, int num_threads) {
    TopKNode* d_block_tops;
    CUDA_CHECK(cudaMalloc(&d_block_tops, num_blocks * K * sizeof(TopKNode)));

    int num_warps = num_threads / 32;
    size_t smem_size = num_warps * K * sizeof(TopKNode);

    switch (K) {
        case 8:
            topk_v3_warp_shuffle_kernel<8><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 16:
            topk_v3_warp_shuffle_kernel<16><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 32:
            topk_v3_warp_shuffle_kernel<32><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 64:
            topk_v3_warp_shuffle_kernel<64><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 128:
            topk_v3_warp_shuffle_kernel<128><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        case 256:
            topk_v3_warp_shuffle_kernel<256><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
        default:
            topk_v3_warp_shuffle_kernel<256><<<num_blocks, num_threads, smem_size>>>(d_input, N, d_block_tops);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int reduce_threads = 256;
    while (reduce_threads > num_blocks) {
        reduce_threads >>= 1;
    }
    reduce_threads = max(reduce_threads, 32);

    size_t reduce_smem_size = (reduce_threads / 32) * K * sizeof(TopKNode);

    switch (K) {
        case 8:
            topk_v3_reduce_kernel<8><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 16:
            topk_v3_reduce_kernel<16><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 32:
            topk_v3_reduce_kernel<32><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 64:
            topk_v3_reduce_kernel<64><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 128:
            topk_v3_reduce_kernel<128><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        case 256:
            topk_v3_reduce_kernel<256><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
        default:
            topk_v3_reduce_kernel<256><<<1, reduce_threads, reduce_smem_size>>>(d_block_tops, num_blocks, d_output);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_block_tops));
}

// ============================================================================
// External kernel declarations from v4_radix_select.cu
// ============================================================================

template <int BLOCK_SIZE>
__global__ void radix_histogram_kernel(const float* input, int N,
                                        unsigned int* histograms,
                                        int bit_start,
                                        unsigned int current_threshold);

__global__ void merge_histograms_kernel(const unsigned int* block_histograms,
                                         unsigned int* global_histogram,
                                         int num_blocks);

__global__ void collect_candidates_kernel(const float* input, int N,
                                            TopKNode* candidates,
                                            int* candidate_count,
                                            unsigned int threshold_bits,
                                            int num_bits_checked);

template <int K>
__global__ void final_select_topk_kernel(TopKNode* candidates,
                                        int candidate_count,
                                        TopKNode* output);

// ============================================================================
// External kernel declarations from v5_blackwell.cu
// ============================================================================

template <int K, int BLOCK_THREADS>
__global__ void topk_v5_blackwell_kernel(const float* __restrict__ input,
                                        int N,
                                        TopKNode* block_tops);

template <int K>
__global__ void topk_v5_reduce_kernel(const TopKNode* block_tops,
                                       int num_blocks,
                                       TopKNode* output);

template <int K, int BLOCK_THREADS, int CHUNK_SIZE>
__global__ void topk_v5_pipeline_kernel(const float* __restrict__ input,
                                        int N,
                                        TopKNode* block_tops);

// ============================================================================
// V4 Host Function: Radix Select
// ============================================================================

#define RADIX_BUCKETS 256

void topk_v4_radix_select(const float* d_input, int N, int K, TopKNode* d_output,
                          int num_blocks, int num_threads) {
    // Allocate histogram storage
    unsigned int* d_histograms;
    unsigned int* d_global_histogram;
    CUDA_CHECK(cudaMalloc(&d_histograms, num_blocks * RADIX_BUCKETS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_global_histogram, RADIX_BUCKETS * sizeof(unsigned int)));

    // Allocate candidate storage
    TopKNode* d_candidates;
    int* d_candidate_count;
    CUDA_CHECK(cudaMalloc(&d_candidates, N * sizeof(TopKNode)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(int)));

    unsigned int current_threshold = 0;
    int remaining_k = K;

    // Pass 1-4: Process 32 bits (8 bits per pass)
    for (int pass = 0; pass < 4; pass++) {
        int bit_start = (3 - pass) * 8;  // 24, 16, 8, 0

        // Clear histograms
        CUDA_CHECK(cudaMemset(d_histograms, 0, num_blocks * RADIX_BUCKETS * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_global_histogram, 0, RADIX_BUCKETS * sizeof(unsigned int)));

        // Compute histogram
        radix_histogram_kernel<256><<<num_blocks, num_threads>>>(
            d_input, N, d_histograms, bit_start, current_threshold);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Merge histograms
        merge_histograms_kernel<<<1, 256>>>(d_histograms, d_global_histogram, num_blocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read histogram to find bucket containing K-th element
        unsigned int h_histogram[RADIX_BUCKETS];
        CUDA_CHECK(cudaMemcpy(h_histogram, d_global_histogram,
                              RADIX_BUCKETS * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        unsigned int count = 0;
        int selected_bucket = -1;
        for (int bucket = RADIX_BUCKETS - 1; bucket >= 0; bucket--) {
            count += h_histogram[bucket];
            if ((int)count >= remaining_k && selected_bucket == -1) {
                selected_bucket = bucket;
            }
        }

        if (selected_bucket == -1) {
            selected_bucket = 0;
        }

        current_threshold = (current_threshold << 8) | (unsigned int)selected_bucket;
    }

    // Final pass: collect elements meeting threshold
    CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)));

    collect_candidates_kernel<<<num_blocks, num_threads>>>(
        d_input, N, d_candidates, d_candidate_count,
        current_threshold, 32);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read candidate count
    int h_candidate_count;
    CUDA_CHECK(cudaMemcpy(&h_candidate_count, d_candidate_count, sizeof(int),
                          cudaMemcpyDeviceToHost));
    h_candidate_count = min(h_candidate_count, N);

    // Final top-k selection
    size_t smem_size = num_threads * K * sizeof(TopKNode);

    switch (K) {
        case 8:
            final_select_topk_kernel<8><<<1, num_threads, smem_size>>>(
                d_candidates, h_candidate_count, d_output);
            break;
        case 16:
            final_select_topk_kernel<16><<<1, num_threads, smem_size>>>(
                d_candidates, h_candidate_count, d_output);
            break;
        case 32:
            final_select_topk_kernel<32><<<1, num_threads, smem_size>>>(
                d_candidates, h_candidate_count, d_output);
            break;
        case 64:
            final_select_topk_kernel<64><<<1, num_threads, smem_size>>>(
                d_candidates, h_candidate_count, d_output);
            break;
        case 128:
            final_select_topk_kernel<128><<<1, num_threads, smem_size>>>(
                d_candidates, h_candidate_count, d_output);
            break;
        case 256:
            final_select_topk_kernel<256><<<1, num_threads, smem_size>>>(
                d_candidates, h_candidate_count, d_output);
            break;
        default:
            final_select_topk_kernel<256><<<1, num_threads, smem_size>>>(
                d_candidates, h_candidate_count, d_output);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFree(d_histograms));
    CUDA_CHECK(cudaFree(d_global_histogram));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
}

// ============================================================================
// L2 Persistence helpers (CUDA 11.8+)
// ============================================================================

#if CUDA_VERSION >= 11080
void setup_l2_persistence(void* ptr, size_t size, float fraction) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t l2_size = prop.l2CacheSize;
    size_t persist_size = (size_t)(l2_size * fraction);

    cudaStreamAttrValue access_policy;
    access_policy.accessPolicyWindow.base_ptr = ptr;
    access_policy.accessPolicyWindow.num_bytes = min(size, persist_size);
    access_policy.accessPolicyWindow.hitRatio = 1.0f;
    access_policy.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    access_policy.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &access_policy);
}

void reset_l2_persistence() {
    cudaStreamAttrValue access_policy = {};
    access_policy.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &access_policy);
}
#else
void setup_l2_persistence(void* ptr, size_t size, float fraction) {
    (void)ptr; (void)size; (void)fraction;
}
void reset_l2_persistence() {}
#endif

// ============================================================================
// V5 Host Function: Blackwell Optimized
// ============================================================================

void topk_v5_blackwell(const float* d_input, int N, int K, TopKNode* d_output,
                       int num_blocks, int num_threads) {
    // Check device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Try L2 persistence for large data reuse
    size_t data_size = N * sizeof(float);
    bool use_l2_persist = (data_size < (size_t)prop.l2CacheSize * 0.8f);

    if (use_l2_persist) {
        setup_l2_persistence((void*)d_input, data_size, 0.6f);
    }

    // Allocate temporary storage
    TopKNode* d_block_tops;
    CUDA_CHECK(cudaMalloc(&d_block_tops, num_blocks * K * sizeof(TopKNode)));

    // Calculate shared memory size
    int num_warps = num_threads / 32;
    size_t smem_size = num_warps * K * sizeof(TopKNode);

    // Choose optimal kernel
    bool use_pipeline = (N > 1000000);

    if (use_pipeline) {
        const int CHUNK_SIZE = 2048;
        size_t pipeline_smem = smem_size + CHUNK_SIZE * sizeof(float);

        switch (K) {
            case 8:
                topk_v5_pipeline_kernel<8, 256, CHUNK_SIZE><<<num_blocks, 256, pipeline_smem>>>(
                    d_input, N, d_block_tops);
                break;
            case 16:
                topk_v5_pipeline_kernel<16, 256, CHUNK_SIZE><<<num_blocks, 256, pipeline_smem>>>(
                    d_input, N, d_block_tops);
                break;
            case 32:
                topk_v5_pipeline_kernel<32, 256, CHUNK_SIZE><<<num_blocks, 256, pipeline_smem>>>(
                    d_input, N, d_block_tops);
                break;
            case 64:
                topk_v5_pipeline_kernel<64, 256, CHUNK_SIZE><<<num_blocks, 256, pipeline_smem>>>(
                    d_input, N, d_block_tops);
                break;
            case 128:
                topk_v5_pipeline_kernel<128, 256, CHUNK_SIZE><<<num_blocks, 256, pipeline_smem>>>(
                    d_input, N, d_block_tops);
                break;
            case 256:
                topk_v5_pipeline_kernel<256, 256, CHUNK_SIZE><<<num_blocks, 256, pipeline_smem>>>(
                    d_input, N, d_block_tops);
                break;
            default:
                topk_v5_pipeline_kernel<256, 256, CHUNK_SIZE><<<num_blocks, 256, pipeline_smem>>>(
                    d_input, N, d_block_tops);
                break;
        }
    } else {
        switch (K) {
            case 8:
                topk_v5_blackwell_kernel<8, 256><<<num_blocks, 256, smem_size>>>(
                    d_input, N, d_block_tops);
                break;
            case 16:
                topk_v5_blackwell_kernel<16, 256><<<num_blocks, 256, smem_size>>>(
                    d_input, N, d_block_tops);
                break;
            case 32:
                topk_v5_blackwell_kernel<32, 256><<<num_blocks, 256, smem_size>>>(
                    d_input, N, d_block_tops);
                break;
            case 64:
                topk_v5_blackwell_kernel<64, 256><<<num_blocks, 256, smem_size>>>(
                    d_input, N, d_block_tops);
                break;
            case 128:
                topk_v5_blackwell_kernel<128, 256><<<num_blocks, 256, smem_size>>>(
                    d_input, N, d_block_tops);
                break;
            case 256:
                topk_v5_blackwell_kernel<256, 256><<<num_blocks, 256, smem_size>>>(
                    d_input, N, d_block_tops);
                break;
            default:
                topk_v5_blackwell_kernel<256, 256><<<num_blocks, 256, smem_size>>>(
                    d_input, N, d_block_tops);
                break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (use_l2_persist) {
        reset_l2_persistence();
    }

    // Final reduction
    int reduce_threads = 256;
    while (reduce_threads > num_blocks) {
        reduce_threads >>= 1;
    }
    reduce_threads = max(reduce_threads, 32);

    size_t reduce_smem_size = (reduce_threads / 32) * K * sizeof(TopKNode);

    switch (K) {
        case 8:
            topk_v5_reduce_kernel<8><<<1, reduce_threads, reduce_smem_size>>>(
                d_block_tops, num_blocks, d_output);
            break;
        case 16:
            topk_v5_reduce_kernel<16><<<1, reduce_threads, reduce_smem_size>>>(
                d_block_tops, num_blocks, d_output);
            break;
        case 32:
            topk_v5_reduce_kernel<32><<<1, reduce_threads, reduce_smem_size>>>(
                d_block_tops, num_blocks, d_output);
            break;
        case 64:
            topk_v5_reduce_kernel<64><<<1, reduce_threads, reduce_smem_size>>>(
                d_block_tops, num_blocks, d_output);
            break;
        case 128:
            topk_v5_reduce_kernel<128><<<1, reduce_threads, reduce_smem_size>>>(
                d_block_tops, num_blocks, d_output);
            break;
        case 256:
            topk_v5_reduce_kernel<256><<<1, reduce_threads, reduce_smem_size>>>(
                d_block_tops, num_blocks, d_output);
            break;
        default:
            topk_v5_reduce_kernel<256><<<1, reduce_threads, reduce_smem_size>>>(
                d_block_tops, num_blocks, d_output);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_block_tops));
}

// Include kernel implementations for template instantiation
// These are included at the end to ensure template definitions are visible
#include "v1_thread.cu"
#include "v2_shared_memory.cu"
#include "v3_warp_shuffle.cu"
#include "v4_radix_select.cu"
#include "v5_blackwell.cu"

#include <cuda_runtime.h>
#include <cfloat>

// ==========================================
// Top-K Selection Solution
// 从 src/topk_v5_multi_block.cu 抽取的核心实现
// ==========================================

#define MAX_K 128
#define WARP_SIZE 32

// Device function: Insert value into sorted array (descending)
__device__ __forceinline__ void insert_topk(float* vals, int K, float new_val) {
    if (new_val <= vals[K - 1]) return;
    int pos = K - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
}

// Phase 1: Find local top-k in each block
__global__ void local_topk_kernel(const float* input, float* block_out, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int lane = tid % WARP_SIZE;
    int warpid = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Data range for this block
    int chunk = (N + num_blocks - 1) / num_blocks;
    int start = bid * chunk;
    int end = start + chunk;
    if (end > N) end = N;
    
    // Thread-local top-k
    float local[MAX_K];
    #pragma unroll
    for (int i = 0; i < MAX_K; ++i) local[i] = -FLT_MAX;
    
    // Process elements
    for (int i = start + tid; i < end; i += blockDim.x) {
        insert_topk(local, K, input[i]);
    }
    
    // Shared memory for warp aggregation
    extern __shared__ float smem[];
    float* warp_buf = smem;
    
    // Warp-level reduction
    for (int k = 0; k < K; ++k) {
        float v = local[k];
        #pragma unroll
        for (int off = 16; off > 0; off /= 2) {
            float o = __shfl_down_sync(0xffffffff, v, off);
            if (o > v) v = o;
        }
        if (lane == 0) warp_buf[warpid * K + k] = v;
    }
    
    __syncthreads();
    
    // First warp: merge and output
    if (warpid == 0) {
        float best[MAX_K];
        #pragma unroll
        for (int i = 0; i < MAX_K; ++i) best[i] = -FLT_MAX;
        
        for (int w = lane; w < num_warps; w += WARP_SIZE) {
            for (int k = 0; k < K; ++k) {
                insert_topk(best, K, warp_buf[w * K + k]);
            }
        }
        
        for (int k = 0; k < K; ++k) {
            float v = best[k];
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) {
                float o = __shfl_down_sync(0xffffffff, v, off);
                if (o > v) v = o;
            }
            if (best[k] == v) best[k] = -FLT_MAX;
            if (lane == 0) block_out[bid * K + k] = v;
        }
    }
}

// Phase 2: Merge all block results
__global__ void merge_topk_kernel(const float* block_results, float* output,
                                   int num_blocks, int K) {
    int tid = threadIdx.x;
    int total = num_blocks * K;
    
    float my_best[MAX_K];
    for (int i = 0; i < K; ++i) my_best[i] = -FLT_MAX;
    
    for (int i = tid; i < total; i += blockDim.x) {
        insert_topk(my_best, K, block_results[i]);
    }
    
    __shared__ float s_data[256 * MAX_K / WARP_SIZE];
    
    int lane = tid % WARP_SIZE;
    int warpid = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    for (int k = 0; k < K; ++k) {
        s_data[warpid * K + k] = my_best[k];
    }
    __syncthreads();
    
    if (warpid == 0) {
        float final_best[MAX_K];
        for (int i = 0; i < K; ++i) final_best[i] = -FLT_MAX;
        
        for (int w = lane; w < num_warps; w += WARP_SIZE) {
            for (int k = 0; k < K; ++k) {
                insert_topk(final_best, K, s_data[w * K + k]);
            }
        }
        
        for (int k = 0; k < K; ++k) {
            float v = final_best[k];
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) {
                float o = __shfl_down_sync(0xffffffff, v, off);
                if (o > v) v = o;
            }
            if (final_best[k] == v) final_best[k] = -FLT_MAX;
            if (lane == 0) output[k] = v;
        }
    }
}

// Host interface
void solve(const float* input, float* output, int N, int k) {
    int K = (k > MAX_K) ? MAX_K : k;
    
    float *d_in = nullptr, *d_blocks = nullptr, *d_out = nullptr;
    
    // Determine number of blocks based on N
    int nblocks = 256;
    if (N < 1000000) nblocks = 128;
    if (N < 100000) nblocks = 32;
    if (N < 10000) nblocks = 4;
    if (N < 1024) nblocks = 1;
    
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_blocks, nblocks * K * sizeof(float));
    cudaMalloc(&d_out, K * sizeof(float));
    
    cudaMemcpy(d_in, input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Phase 1
    size_t smem = (256 / WARP_SIZE) * K * sizeof(float);
    local_topk_kernel<<<nblocks, 256, smem>>>(d_in, d_blocks, N, K);
    
    // Phase 2
    merge_topk_kernel<<<1, 256>>>(d_blocks, d_out, nblocks, K);
    
    cudaMemcpy(output, d_out, K * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_blocks);
    cudaFree(d_out);
}

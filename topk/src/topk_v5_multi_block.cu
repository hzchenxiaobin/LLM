#include "../include/topk_common.h"

// ==========================================
// V5: Multi-Block Top-K (支持超大N，K > 32)
// ==========================================
// 核心思想：两阶段算法，适用于超大规模数据
// Phase 1: 多个Block并行处理数据的不同部分，每个Block找出局部Top-K
// Phase 2: 单Block合并所有局部结果，找出全局Top-K
//
// 适用场景：
// - N 超大（如 50M, 100M）
// - K 较大（如 K > 32，最大支持 128 或更大）
// - 需要饱和GPU利用率
//
// 修改说明：
// - 1D输入（无Batch轴）
// - 只输出Top-K值，不输出索引
// ==========================================

#define V5_MAX_K 128
#define V5_WARP_SIZE 32

// ==========================================
// 辅助函数：插入排序维护Local Top-K
// ==========================================
__device__ __forceinline__ void v5_insert_topk(float* vals, int K, float new_val) {
    if (new_val <= vals[K - 1]) return;
    
    int pos = K - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
}

// ==========================================
// Phase 1 Kernel: 每个Block找局部Top-K
// ==========================================
__global__ void topk_v5_local_kernel(const float* input, float* block_out, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int lane = tid % V5_WARP_SIZE;
    int warpid = tid / V5_WARP_SIZE;
    int num_warps = blockDim.x / V5_WARP_SIZE;
    
    // 计算本Block的数据范围
    int chunk = (N + num_blocks - 1) / num_blocks;
    int start = bid * chunk;
    int end = start + chunk;
    if (end > N) end = N;
    
    // Thread-local Top-K（寄存器存储）
    float local[V5_MAX_K];
    #pragma unroll
    for (int i = 0; i < V5_MAX_K; ++i) local[i] = -1e20f;
    
    // 处理数据
    for (int i = start + tid; i < end; i += blockDim.x) {
        v5_insert_topk(local, K, input[i]);
    }
    
    // Shared Memory用于Warp级归约
    extern __shared__ char smem[];
    float* warp_buf = (float*)smem;
    
    // Warp级归约：找每个Warp的最佳K个值
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
    
    // 第一个Warp：合并所有Warp结果并输出Block的Top-K
    if (warpid == 0) {
        float best[V5_MAX_K];
        #pragma unroll
        for (int i = 0; i < V5_MAX_K; ++i) best[i] = -1e20f;
        
        // 合并所有Warp结果
        for (int w = lane; w < num_warps; w += V5_WARP_SIZE) {
            for (int k = 0; k < K; ++k) {
                v5_insert_topk(best, K, warp_buf[w * K + k]);
            }
        }
        
        // 输出本Block的K个最佳值
        for (int k = 0; k < K; ++k) {
            float v = best[k];
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) {
                float o = __shfl_down_sync(0xffffffff, v, off);
                if (o > v) v = o;
            }
            // 标记为已使用
            if (best[k] == v) best[k] = -1e20f;
            
            if (lane == 0) block_out[bid * K + k] = v;
        }
    }
}

// ==========================================
// Phase 2 Kernel: 合并所有Block结果
// ==========================================
__global__ void topk_v5_merge_kernel(const float* block_results, float* output,
                                     int num_blocks, int K) {
    int tid = threadIdx.x;
    int total = num_blocks * K;
    
    // 每个线程维护自己的Top-K
    float my_best[V5_MAX_K];
    for (int i = 0; i < K; ++i) my_best[i] = -1e20f;
    
    // 处理所有候选值（strided方式）
    for (int i = tid; i < total; i += blockDim.x) {
        v5_insert_topk(my_best, K, block_results[i]);
    }
    
    // Shared Memory收集结果
    __shared__ float s_data[256 * V5_MAX_K / V5_WARP_SIZE];  // 8 warps * K values
    
    int lane = tid % V5_WARP_SIZE;
    int warpid = tid / V5_WARP_SIZE;
    int num_warps = blockDim.x / V5_WARP_SIZE;
    
    // 存储结果到Shared Memory
    for (int k = 0; k < K; ++k) {
        s_data[warpid * K + k] = my_best[k];
    }
    __syncthreads();
    
    // 第一个Warp做最终合并
    if (warpid == 0) {
        float final_best[V5_MAX_K];
        for (int i = 0; i < K; ++i) final_best[i] = -1e20f;
        
        // 合并所有Warp结果
        for (int w = lane; w < num_warps; w += V5_WARP_SIZE) {
            for (int k = 0; k < K; ++k) {
                v5_insert_topk(final_best, K, s_data[w * K + k]);
            }
        }
        
        // 输出最终的K个值
        for (int k = 0; k < K; ++k) {
            float v = final_best[k];
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) {
                float o = __shfl_down_sync(0xffffffff, v, off);
                if (o > v) v = o;
            }
            if (final_best[k] == v) final_best[k] = -1e20f;
            if (lane == 0) output[k] = v;
        }
    }
}

// ==========================================
// V5 包装函数（供外部调用）
// ==========================================
void topk_v5(const float* d_input, float* d_output, int N, int K, 
             int nblocks = 256, int block_size = 256) {
    // 限制K在支持范围内
    if (K > V5_MAX_K) K = V5_MAX_K;
    
    // 分配中间结果内存
    float* d_block_results;
    cudaMalloc(&d_block_results, nblocks * K * sizeof(float));
    
    // Phase 1: 局部Top-K
    int num_warps = block_size / V5_WARP_SIZE;
    size_t smem_size = num_warps * K * sizeof(float);
    topk_v5_local_kernel<<<nblocks, block_size, smem_size>>>
        (d_input, d_block_results, N, K);
    
    // Phase 2: 全局合并
    topk_v5_merge_kernel<<<1, block_size>>>(d_block_results, d_output, nblocks, K);
    
    cudaFree(d_block_results);
}

// ==========================================
// 调用说明：
// ==========================================
// 对于大规模数据 (N = 50M, K = 100):
//   int nblocks = 256;
//   int block_size = 256;
//   float *d_input, *d_output;
//   cudaMalloc(&d_input, N * sizeof(float));
//   cudaMalloc(&d_output, K * sizeof(float));
//   topk_v5(d_input, d_output, N, K, nblocks, block_size);
//
// 注意：
// - K 最大支持 V5_MAX_K (默认128)
// - 输出按降序排列
// - 只输出值，不输出索引

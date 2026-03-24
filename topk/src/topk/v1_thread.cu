// V1: 线程级局部 Top-K
// Map-Reduce 方案: 每个线程维护自己的局部 Top-K, 写入全局内存
// 基于 TUTORIAL.md 章节 V1

#include "../include/topk_common.h"

// 内核: 每个线程使用 grid-stride 循环处理元素
// 在寄存器中维护局部 Top-K, 将结果写入全局内存
template <int K>
__global__ void topk_v1_kernel(const float* input, int N, TopKNode* thread_tops) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // 寄存器中的局部 Top-K (初始化为最小值)
    // 注意: 当 K > 16 时, 可能会溢出到局部内存
    TopKNode local_top[K];
    for (int i = 0; i < K; ++i) {
        local_top[i].value = -1e20f;  // 非常小的数
        local_top[i].index = -1;
    }

    // 在输入上执行 grid-stride 循环
    for (int idx = tid; idx < N; idx += stride) {
        float val = input[idx];

        // 如果比当前最小值大, 则插入到局部 top-K
        // local_top 按降序维护 (最大值在索引 0 处)
        if (val > local_top[K - 1].value) {
            // 移位并插入
            int j = K - 1;
            while (j > 0 && val > local_top[j - 1].value) {
                local_top[j] = local_top[j - 1];
                j--;
            }
            local_top[j].value = val;
            local_top[j].index = idx;
        }
    }

    // 将局部 top-K 写入全局内存
    // 输出布局: 每个线程写入 K 个连续元素
    int out_offset = tid * K;
    for (int i = 0; i < K; ++i) {
        thread_tops[out_offset + i] = local_top[i];
    }
}

// 最终归约内核: 合并所有线程结果得到最终的 top-K
// 使用单个块进行树形归约
template <int K>
__global__ void topk_v1_reduce_kernel(const TopKNode* thread_tops, int num_threads,
                                        TopKNode* output) {
    extern __shared__ TopKNode smem[];

    int tid = threadIdx.x;

    // 每个线程从全局内存加载自己的 top-K
    // 注意: thread_tops 布局是 [num_threads][K]
    TopKNode local_top[K];
    for (int i = 0; i < K; ++i) {
        if (tid < num_threads) {
            local_top[i] = thread_tops[tid * K + i];
        } else {
            local_top[i].value = -1e20f;
            local_top[i].index = -1;
        }
    }

    // 存储到共享内存以进行树形归约
    for (int i = 0; i < K; ++i) {
        smem[tid * K + i] = local_top[i];
    }
    __syncthreads();

    // 树形归约: 成对合并 top-K 数组
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 合并 smem[tid] 和 smem[tid + s]
            TopKNode temp[2 * K];

            // 加载两个数组
            for (int i = 0; i < K; ++i) {
                temp[i] = smem[tid * K + i];
                temp[i + K] = smem[(tid + s) * K + i];
            }

            // 对 2K 个元素排序 (小 K 情况下使用简单的冒泡排序)
            for (int i = 0; i < 2 * K - 1; ++i) {
                for (int j = 0; j < 2 * K - i - 1; ++j) {
                    if (temp[j].value < temp[j + 1].value) {
                        TopKNode swap = temp[j];
                        temp[j] = temp[j + 1];
                        temp[j + 1] = swap;
                    }
                }
            }

            // 写回前 K 个
            for (int i = 0; i < K; ++i) {
                smem[tid * K + i] = temp[i];
            }
        }
        __syncthreads();
    }

    // 线程 0 写入最终结果
    if (tid == 0) {
        for (int i = 0; i < K; ++i) {
            output[i] = smem[i];
        }
    }
}

// Explicit template instantiations for kernels
// This ensures kernels are compiled for all K values we use
template __global__ void topk_v1_kernel<8>(const float* input, int N, TopKNode* thread_tops);
template __global__ void topk_v1_kernel<16>(const float* input, int N, TopKNode* thread_tops);
template __global__ void topk_v1_kernel<32>(const float* input, int N, TopKNode* thread_tops);
template __global__ void topk_v1_kernel<64>(const float* input, int N, TopKNode* thread_tops);
template __global__ void topk_v1_kernel<128>(const float* input, int N, TopKNode* thread_tops);
template __global__ void topk_v1_kernel<256>(const float* input, int N, TopKNode* thread_tops);

template __global__ void topk_v1_reduce_kernel<8>(const TopKNode* thread_tops, int num_threads, TopKNode* output);
template __global__ void topk_v1_reduce_kernel<16>(const TopKNode* thread_tops, int num_threads, TopKNode* output);
template __global__ void topk_v1_reduce_kernel<32>(const TopKNode* thread_tops, int num_threads, TopKNode* output);
template __global__ void topk_v1_reduce_kernel<64>(const TopKNode* thread_tops, int num_threads, TopKNode* output);
template __global__ void topk_v1_reduce_kernel<128>(const TopKNode* thread_tops, int num_threads, TopKNode* output);
template __global__ void topk_v1_reduce_kernel<256>(const TopKNode* thread_tops, int num_threads, TopKNode* output);

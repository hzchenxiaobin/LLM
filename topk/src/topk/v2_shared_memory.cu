// V2: 块级共享内存合并
// 优化方案: 在写入全局内存前在共享内存中进行块级合并
// 基于 TUTORIAL.md 章节 V2

#include "../include/topk_common.h"

// 设备函数: 在共享内存中合并两个已排序的 top-K 数组
// 将结果写回第一个数组的位置
template <int K>
__device__ __forceinline__ void merge_topk_in_smem(TopKNode* smem, int dst_offset, int src_offset) {
    // 栈上的临时数组 (寄存器/局部内存)
    TopKNode temp[2 * K];

    // 加载两个数组
    for (int i = 0; i < K; ++i) {
        temp[i] = smem[dst_offset + i];
        temp[i + K] = smem[src_offset + i];
    }

    // 对 2K 个元素使用简单选择排序 (K <= 256, 所以 2K <= 512)
    // 降序排序
    for (int i = 0; i < K; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < 2 * K; ++j) {
            if (temp[j].value > temp[max_idx].value) {
                max_idx = j;
            }
        }
        // 交换
        if (max_idx != i) {
            TopKNode swap = temp[i];
            temp[i] = temp[max_idx];
            temp[max_idx] = swap;
        }
    }

    // 将前 K 个写回目标位置
    for (int i = 0; i < K; ++i) {
        smem[dst_offset + i] = temp[i];
    }
}

// 内核: 块级 Top-K, 使用共享内存优化
// 每个块精确输出 K 个元素 (输入数据块中的前 K 个)
template <int K>
__global__ void topk_v2_block_kernel(const float* input, int N, TopKNode* block_tops) {
    // 共享内存布局: 每个线程存储 K 个元素
    // 大小 = blockDim.x * K * sizeof(TopKNode)
    extern __shared__ TopKNode smem[];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int num_threads = blockDim.x;

    // 每个线程使用 grid-stride 循环处理输入的一部分
    // 但只在该块分配范围内处理
    int elements_per_block = (N + num_blocks - 1) / num_blocks;
    int block_start = block_id * elements_per_block;
    int block_end = min(block_start + elements_per_block, N);

    // 寄存器中的局部 Top-K (按降序维护)
    TopKNode local_top[K];
    for (int i = 0; i < K; ++i) {
        local_top[i].value = -1e20f;
        local_top[i].index = -1;
    }

    // 块分配范围内的 grid-stride 循环
    // 块中的每个线程处理每第 num_threads 个元素
    for (int idx = block_start + tid; idx < block_end; idx += num_threads) {
        float val = input[idx];

        // 如果比当前最小值大, 则插入到局部 top-K
        if (val > local_top[K - 1].value) {
            // 从末尾开始查找插入位置, 边查边移位
            int j = K - 1;
            while (j > 0 && val > local_top[j - 1].value) {
                local_top[j] = local_top[j - 1];
                j--;
            }
            local_top[j].value = val;
            local_top[j].index = idx;
        }
    }

    // 将局部 top-K 写入共享内存
    for (int i = 0; i < K; ++i) {
        smem[tid * K + i] = local_top[i];
    }
    __syncthreads();

    // 使用共享内存进行块级树形归约
    // 每次迭代将活动线程数减半
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 合并 smem[tid] 和 smem[tid + s]
            merge_topk_in_smem<K>(smem, tid * K, (tid + s) * K);
        }
        __syncthreads();
    }

    // 线程 0 将块的最终 top-K 写入全局内存
    if (tid == 0) {
        for (int i = 0; i < K; ++i) {
            block_tops[block_id * K + i] = smem[i];
        }
    }
}

// 最终归约内核: 合并所有块结果
template <int K>
__global__ void topk_v2_reduce_kernel(const TopKNode* block_tops, int num_blocks, TopKNode* output) {
    extern __shared__ TopKNode smem[];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 加载块结果到共享内存
    // 每个线程加载其分配块的结果
    TopKNode local_top[K];
    for (int i = 0; i < K; ++i) {
        if (tid < num_blocks) {
            local_top[i] = block_tops[tid * K + i];
        } else {
            local_top[i].value = -1e20f;
            local_top[i].index = -1;
        }
    }

    // 存储到共享内存
    for (int i = 0; i < K; ++i) {
        smem[tid * K + i] = local_top[i];
    }
    __syncthreads();

    // 树形归约
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < num_blocks) {
            // 合并两个 top-K 数组
            TopKNode temp[2 * K];
            for (int i = 0; i < K; ++i) {
                temp[i] = smem[tid * K + i];
                temp[i + K] = smem[(tid + s) * K + i];
            }

            // 对前 K 个排序
            for (int i = 0; i < K; ++i) {
                int max_idx = i;
                for (int j = i + 1; j < 2 * K; ++j) {
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

// Explicit template instantiations
template __global__ void topk_v2_block_kernel<8>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v2_block_kernel<16>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v2_block_kernel<32>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v2_block_kernel<64>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v2_block_kernel<128>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v2_block_kernel<256>(const float* input, int N, TopKNode* block_tops);

template __global__ void topk_v2_reduce_kernel<8>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v2_reduce_kernel<16>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v2_reduce_kernel<32>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v2_reduce_kernel<64>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v2_reduce_kernel<128>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v2_reduce_kernel<256>(const TopKNode* block_tops, int num_blocks, TopKNode* output);

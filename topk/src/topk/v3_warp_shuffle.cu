// V3: Warp Shuffle 原语
// 优化方案: 使用 warp 级 shuffle 进行寄存器到寄存器通信
// 基于 TUTORIAL.md 章节 V3

#include "../include/topk_common.h"

// 设备函数: 使用 warp shuffle 合并两个 top-K 数组
// 在 warp 内使用, 用于合并不同 lane 的结果
template <int K>
__device__ __forceinline__ void warp_merge_topk(TopKNode* local_top, int lane_id) {
    // 对于 top-K 中的每个位置, 需要在整个 warp 中找到最佳值
    // 使用 shuffle 成对比较值

    // 合并结果的临时存储
    TopKNode warp_best[K];
    for (int i = 0; i < K; ++i) {
        warp_best[i] = local_top[i];
    }

    // 使用 shuffle 与其他 lane 比较和交换
    // 使用蝶形模式合并
    for (int offset = 16; offset > 0; offset /= 2) {
        // 对于 top-K 中的每个位置, 从配对 lane 获取值
        for (int k = 0; k < K; ++k) {
            // Shuffle 整个 TopKNode (值 + 索引)
            // 需要分别 shuffle 值和索引
            float other_val = __shfl_down_sync(0xffffffff, warp_best[k].value, offset);
            int other_idx = __shfl_down_sync(0xffffffff, warp_best[k].index, offset);

            // 如果其他 lane 有更好的值, 则采用
            // 但需要进行适当的合并, 而不仅仅是替换
            // 为简化 warp 级操作, 只进行成对比较
            if (other_val > warp_best[k].value) {
                // 存储其他值
                TopKNode other_node;
                other_node.value = other_val;
                other_node.index = other_idx;

                // 简单冒泡插入
                // 移位当前和后续元素
                for (int j = K - 1; j > k; --j) {
                    warp_best[j] = warp_best[j - 1];
                }
                warp_best[k] = other_node;
            }
        }
    }

    // 复制结果回去
    for (int i = 0; i < K; ++i) {
        local_top[i] = warp_best[i];
    }
}

// 优化的 warp 级归约: 使用双调式合并找到 warp 内的 top-K
template <int K>
__device__ __forceinline__ void warp_reduce_topk(TopKNode* local_top) {
    // 对于 local_top 中的每个元素, 跨 warp 归约以找到最佳值
    // 更简单的方法: 对每个排名位置进行并行归约

    // 使用锦标赛式归约
    for (int offset = 16; offset > 0; offset /= 2) {
        // 获取配对 lane 的数据
        for (int i = 0; i < K; ++i) {
            float partner_val = __shfl_down_sync(0xffffffff, local_top[i].value, offset);
            int partner_idx = __shfl_down_sync(0xffffffff, local_top[i].index, offset);

            // 合并逻辑: 如果 partner 有更好的值, 需要合并数组
            // 对于 warp 级, 对边界进行简单的比较-交换
            if (partner_val > local_top[K - 1].value) {
                // Partner 的值属于我们的 top-K
                // 移位并在适当位置插入
                int pos = K - 1;
                while (pos > 0 && partner_val > local_top[pos - 1].value) {
                    local_top[pos] = local_top[pos - 1];
                    pos--;
                }
                local_top[pos].value = partner_val;
                local_top[pos].index = partner_idx;
            }
        }
    }
}

// 内核: Warp-shuffle 优化的 Top-K
// 1. 每个线程维护自己的 Top-K
// 2. 使用 shuffle 进行 warp 级归约 (只有 lane 0 有有效结果)
// 3. Lane 0 写入共享内存
// 4. 在共享内存中进行块级归约
// 5. 块 leader 写入全局内存
template <int K>
__global__ void topk_v3_warp_kernel(const float* input, int N, TopKNode* block_tops) {
    // 共享内存: 只存储 warp leader 的结果
    // 每块最多 32 个 warp (1024 线程), 所以需要 32 * K 个元素
    extern __shared__ TopKNode smem[];

    int tid = threadIdx.x;
    int lane_id = tid % 32;      // Warp 内的 lane
    int warp_id = tid / 32;      // 块内的 warp 索引
    int num_warps = blockDim.x / 32;
    int global_tid = tid + blockIdx.x * blockDim.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // 寄存器中的局部 Top-K (按降序维护)
    TopKNode local_top[K];
    for (int i = 0; i < K; ++i) {
        local_top[i].value = -1e20f;
        local_top[i].index = -1;
    }

    // Grid-stride 循环
    for (int idx = global_tid; idx < N; idx += num_threads_total) {
        float val = input[idx];

        // 插入到局部 top-K
        if (val > local_top[K - 1].value) {
            int j = K - 1;
            while (j > 0 && val > local_top[j - 1].value) {
                local_top[j] = local_top[j - 1];
                j--;
            }
            local_top[j].value = val;
            local_top[j].index = idx;
        }
    }

    // 步骤 1: 使用 shuffle 进行 warp 级归约
    // 此后, 每个 warp 的 lane 0 持有该 warp 合并后的 top-K
    warp_reduce_topk<K>(local_top);

    // 步骤 2: Warp leader (lane 0) 写入共享内存
    if (lane_id == 0) {
        for (int i = 0; i < K; ++i) {
            smem[warp_id * K + i] = local_top[i];
        }
    }
    __syncthreads();

    // 步骤 3: 使用第一个 warp 进行块级归约
    // 只有 warp 0 参与最终合并
    if (warp_id == 0) {
        // 从共享内存加载 (只需要前 num_warps 个条目)
        TopKNode warp_top[K];
        if (lane_id < num_warps) {
            for (int i = 0; i < K; ++i) {
                warp_top[i] = smem[lane_id * K + i];
            }
        } else {
            for (int i = 0; i < K; ++i) {
                warp_top[i].value = -1e20f;
                warp_top[i].index = -1;
            }
        }

        // Warp 0 内的 warp 级归约
        warp_reduce_topk<K>(warp_top);

        // Warp 0 的 lane 0 写入块的最终结果
        if (lane_id == 0) {
            for (int i = 0; i < K; ++i) {
                block_tops[blockIdx.x * K + i] = warp_top[i];
            }
        }
    }
}

// 替代方案: 使用显式通信的更简单的 warp shuffle 方法
// 使用 warp shuffle 广播和合并
template <int K>
__global__ void topk_v3_warp_shuffle_kernel(const float* input, int N, TopKNode* block_tops) {
    extern __shared__ TopKNode smem[];

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int global_tid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // 寄存器中的局部 top-K
    TopKNode local_top[K];
    for (int i = 0; i < K; ++i) {
        local_top[i].value = -1e20f;
        local_top[i].index = -1;
    }

    // Grid-stride 循环
    for (int idx = global_tid; idx < N; idx += stride) {
        float val = input[idx];

        if (val > local_top[K - 1].value) {
            int j = K - 1;
            while (j > 0 && val > local_top[j - 1].value) {
                local_top[j] = local_top[j - 1];
                j--;
            }
            local_top[j].value = val;
            local_top[j].index = idx;
        }
    }

    // Warp shuffle 归约: 使用双调式合并构建有序列表
    // 每步与配对 lane 交换
    for (int step = 16; step > 0; step /= 2) {
        // 为简化, 使用不同方法:
        // 逐个 shuffle 值并合并

        // 临时缓冲区保存 shuffle 来的值
        TopKNode received[K];

        // 从 partner 逐个元素 shuffle
        for (int i = 0; i < K; ++i) {
            received[i].value = __shfl_down_sync(0xffffffff, local_top[i].value, step);
            received[i].index = __shfl_down_sync(0xffffffff, local_top[i].index, step);
        }

        // 合并 local_top 和 received
        // 简单合并: 从 2K 元素中取最佳 K 个
        TopKNode merged[2 * K];
        for (int i = 0; i < K; ++i) {
            merged[i] = local_top[i];
            merged[i + K] = received[i];
        }

        // 选择排序前 K 个
        for (int i = 0; i < K; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < 2 * K; ++j) {
                if (merged[j].value > merged[max_idx].value) {
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                TopKNode temp = merged[i];
                merged[i] = merged[max_idx];
                merged[max_idx] = temp;
            }
        }

        // 复制回前 K 个
        for (int i = 0; i < K; ++i) {
            local_top[i] = merged[i];
        }
    }

    // Warp 归约后, 只有 lane 0 有有效结果
    // 写入共享内存
    if (lane_id == 0) {
        for (int i = 0; i < K; ++i) {
            smem[warp_id * K + i] = local_top[i];
        }
    }
    __syncthreads();

    // 最终块级归约使用共享内存 (只有第一个 warp)
    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        // 加载所有 warp 结果
        TopKNode block_top[K];
        if (lane_id < num_warps) {
            for (int i = 0; i < K; ++i) {
                block_top[i] = smem[lane_id * K + i];
            }
        } else {
            for (int i = 0; i < K; ++i) {
                block_top[i].value = -1e20f;
                block_top[i].index = -1;
            }
        }

        // Warp 0 内再进行一次 warp shuffle 归约
        for (int step = 16; step > 0; step /= 2) {
            TopKNode received[K];
            for (int i = 0; i < K; ++i) {
                received[i].value = __shfl_down_sync(0xffffffff, block_top[i].value, step);
                received[i].index = __shfl_down_sync(0xffffffff, block_top[i].index, step);
            }

            // 合并
            TopKNode merged[2 * K];
            for (int i = 0; i < K; ++i) {
                merged[i] = block_top[i];
                merged[i + K] = received[i];
            }

            for (int i = 0; i < K; ++i) {
                int max_idx = i;
                for (int j = i + 1; j < 2 * K; ++j) {
                    if (merged[j].value > merged[max_idx].value) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    TopKNode temp = merged[i];
                    merged[i] = merged[max_idx];
                    merged[max_idx] = temp;
                }
            }

            for (int i = 0; i < K; ++i) {
                block_top[i] = merged[i];
            }
        }

        // Lane 0 写入最终结果
        if (lane_id == 0) {
            for (int i = 0; i < K; ++i) {
                block_tops[blockIdx.x * K + i] = block_top[i];
            }
        }
    }
}

// 最终归约内核
template <int K>
__global__ void topk_v3_reduce_kernel(const TopKNode* block_tops, int num_blocks, TopKNode* output) {
    extern __shared__ TopKNode smem[];

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / 32;

    // 加载块结果
    TopKNode local_top[K];
    if (tid < num_blocks) {
        for (int i = 0; i < K; ++i) {
            local_top[i] = block_tops[tid * K + i];
        }
    } else {
        for (int i = 0; i < K; ++i) {
            local_top[i].value = -1e20f;
            local_top[i].index = -1;
        }
    }

    // Warp shuffle 归约
    for (int step = 16; step > 0; step /= 2) {
        TopKNode received[K];
        for (int i = 0; i < K; ++i) {
            received[i].value = __shfl_down_sync(0xffffffff, local_top[i].value, step);
            received[i].index = __shfl_down_sync(0xffffffff, local_top[i].index, step);
        }

        TopKNode merged[2 * K];
        for (int i = 0; i < K; ++i) {
            merged[i] = local_top[i];
            merged[i + K] = received[i];
        }

        for (int i = 0; i < K; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < 2 * K; ++j) {
                if (merged[j].value > merged[max_idx].value) {
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                TopKNode temp = merged[i];
                merged[i] = merged[max_idx];
                merged[max_idx] = temp;
            }
        }

        for (int i = 0; i < K; ++i) {
            local_top[i] = merged[i];
        }
    }

    // 写入共享内存 (只有 lane 0)
    if (lane_id == 0) {
        for (int i = 0; i < K; ++i) {
            smem[warp_id * K + i] = local_top[i];
        }
    }
    __syncthreads();

    // Warp 0 中的最终归约
    if (warp_id == 0) {
        TopKNode final_top[K];
        if (lane_id < num_warps) {
            for (int i = 0; i < K; ++i) {
                final_top[i] = smem[lane_id * K + i];
            }
        } else {
            for (int i = 0; i < K; ++i) {
                final_top[i].value = -1e20f;
                final_top[i].index = -1;
            }
        }

        for (int step = 16; step > 0; step /= 2) {
            TopKNode received[K];
            for (int i = 0; i < K; ++i) {
                received[i].value = __shfl_down_sync(0xffffffff, final_top[i].value, step);
                received[i].index = __shfl_down_sync(0xffffffff, final_top[i].index, step);
            }

            TopKNode merged[2 * K];
            for (int i = 0; i < K; ++i) {
                merged[i] = final_top[i];
                merged[i + K] = received[i];
            }

            for (int i = 0; i < K; ++i) {
                int max_idx = i;
                for (int j = i + 1; j < 2 * K; ++j) {
                    if (merged[j].value > merged[max_idx].value) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    TopKNode temp = merged[i];
                    merged[i] = merged[max_idx];
                    merged[max_idx] = temp;
                }
            }

            for (int i = 0; i < K; ++i) {
                final_top[i] = merged[i];
            }
        }

        if (lane_id == 0) {
            for (int i = 0; i < K; ++i) {
                output[i] = final_top[i];
            }
        }
    }
}

// Explicit template instantiations
template __global__ void topk_v3_warp_kernel<8>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_kernel<16>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_kernel<32>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_kernel<64>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_kernel<128>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_kernel<256>(const float* input, int N, TopKNode* block_tops);

template __global__ void topk_v3_warp_shuffle_kernel<8>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_shuffle_kernel<16>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_shuffle_kernel<32>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_shuffle_kernel<64>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_shuffle_kernel<128>(const float* input, int N, TopKNode* block_tops);
template __global__ void topk_v3_warp_shuffle_kernel<256>(const float* input, int N, TopKNode* block_tops);

template __global__ void topk_v3_reduce_kernel<8>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v3_reduce_kernel<16>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v3_reduce_kernel<32>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v3_reduce_kernel<64>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v3_reduce_kernel<128>(const TopKNode* block_tops, int num_blocks, TopKNode* output);
template __global__ void topk_v3_reduce_kernel<256>(const TopKNode* block_tops, int num_blocks, TopKNode* output);

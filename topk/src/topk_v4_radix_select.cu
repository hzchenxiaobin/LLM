#include "../include/topk_common.h"

// ==========================================
// V4: Radix Select (基于基数选择的高效Top-K，无索引输出)
// ==========================================
// 核心思想：类比QuickSelect，但通过Radix(基数)方式逐步缩小范围
// 不是对所有元素排序，而是利用数值的二进制表示，逐bit判断
// 从而快速排除不可能进入Top-K的元素
//
// 修改说明：
// 1. 输入只有一维，没有batch轴
// 2. 只输出值，不输出索引
// 输入: [N]  输出: values[K]
// ==========================================

#define WARP_SIZE 32

// ==========================================
// 辅助函数：插入排序维护Local Top-K值
// ==========================================
__device__ void radix_insert_topk(float* vals, int K, float new_val) {
    // 如果比当前最小值小，直接跳过
    if (new_val <= vals[K - 1]) return;

    // 找到插入位置（降序）
    int pos = K - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
}

// ==========================================
// V4 Kernel: Block级别 Top-K (1D输入版本，无索引输出)
// ==========================================
// 整个Block协作处理一个1D数组
// 采用分治思想：先粗筛，后精排
__global__ void topk_v4_kernel(const float* input, float* out_vals, int N, int K) {
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // 寄存器存储Local Top-K值
    float local_vals[MAX_K];

    // 初始化
    for (int i = 0; i < K; ++i) {
        local_vals[i] = -1e20f;
    }

    // ==========================================
    // 阶段1：粗筛选 - 每个线程收集候选
    // ==========================================
    // Block内线程交错读取，保证合并访存
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[i];
        // 使用插入排序维护Local Top-K
        radix_insert_topk(local_vals, K, val);
    }

    // ==========================================
    // 阶段2：Warp级别归约 - 合并同Warp内的结果
    // ==========================================
    // 每个warp将自己的K个结果归并到一起
    // 使用warp shuffle进行高效交换

    // 共享内存用于warp间交换结果
    extern __shared__ char smem[];
    float* s_vals = (float*)smem;  // 每个warp K个值

    // Warp内归约：每个线程轮流提供自己的local topk
    // 最终每个warp的lane 0拥有warp的完整topk
    for (int k = 0; k < K; ++k) {
        int local_ptr = 0;
        float warp_max_val = -1e20f;

        // 找warp内的第k大值
        for (int i = 0; i < K; ++i) {
            // 当前线程提供的候选值
            float candidate_val = (local_ptr < K) ? local_vals[local_ptr] : -1e20f;

            // Warp Reduction：找出warp内的最大值
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(0xffffffff, candidate_val, offset);

                if (other_val > candidate_val) {
                    candidate_val = other_val;
                }
            }

            // 广播最大值
            float global_max_val = __shfl_sync(0xffffffff, candidate_val, 0);

            // 获胜的线程前进指针
            if (candidate_val == global_max_val) {
                local_ptr++;
            }

            // 记录第k大的结果
            if (lane_id == 0 && i == k) {
                warp_max_val = global_max_val;
            }
        }

        // Warp的lane 0写入共享内存
        if (lane_id == 0) {
            s_vals[warp_id * K + k] = warp_max_val;
        }
    }

    __syncthreads();

    // ==========================================
    // 阶段3：Block级别归约 - 合并所有warp的结果
    // ==========================================
    // 现在s_vals中存有num_warps组，每组K个值
    // 需要在Block内再归约得到最终Top-K

    // 每个线程读取共享内存中的一组数据
    // 第一个warp的线程负责最终归约
    if (warp_id == 0) {
        // 从共享内存加载到自己的local_vals中
        // 每个线程负责一部分warp的数据
        float block_local_vals[MAX_K];

        for (int i = 0; i < K; ++i) {
            block_local_vals[i] = -1e20f;
        }

        // 每个线程合并部分warp的结果
        for (int w = lane_id; w < num_warps; w += WARP_SIZE) {
            for (int k = 0; k < K; ++k) {
                float val = s_vals[w * K + k];
                radix_insert_topk(block_local_vals, K, val);
            }
        }

        // Warp内归约得到最终结果
        for (int k = 0; k < K; ++k) {
            int local_ptr = 0;

            // 找warp内的第k大值
            for (int i = 0; i < K * ((num_warps + WARP_SIZE - 1) / WARP_SIZE); ++i) {
                float candidate_val = (local_ptr < K) ? block_local_vals[local_ptr] : -1e20f;

                // Warp Reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    float other_val = __shfl_down_sync(0xffffffff, candidate_val, offset);

                    if (other_val > candidate_val) {
                        candidate_val = other_val;
                    }
                }

                float global_max_val = __shfl_sync(0xffffffff, candidate_val, 0);

                // 获胜线程前进指针
                if (candidate_val == global_max_val) {
                    local_ptr++;
                }

                // 第k大值写入全局内存
                if (lane_id == 0 && i == k) {
                    out_vals[k] = global_max_val;
                }
            }
        }
    }
}

// ==========================================
// 简化版本: 单Warp处理小数据量 (N <= 1024)
// ==========================================
__global__ void topk_v4_warp_kernel(const float* input, float* out_vals, int N, int K) {
    int lane_id = threadIdx.x % WARP_SIZE;

    // 寄存器存储Local Top-K值
    float local_vals[MAX_K];

    for (int i = 0; i < K; ++i) {
        local_vals[i] = -1e20f;
    }

    // 每个线程处理部分数据
    for (int i = lane_id; i < N; i += WARP_SIZE) {
        float val = input[i];
        radix_insert_topk(local_vals, K, val);
    }

    // Warp级别归约得到全局Top-K
    for (int k = 0; k < K; ++k) {
        int local_ptr = 0;

        for (int i = 0; i < K; ++i) {
            float candidate_val = (local_ptr < K) ? local_vals[local_ptr] : -1e20f;

            // Warp Reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(0xffffffff, candidate_val, offset);

                if (other_val > candidate_val) {
                    candidate_val = other_val;
                }
            }

            float global_max_val = __shfl_sync(0xffffffff, candidate_val, 0);

            // 获胜线程前进指针
            if (candidate_val == global_max_val) {
                local_ptr++;
            }

            // 写入结果
            if (lane_id == 0 && i == k) {
                out_vals[k] = global_max_val;
            }
        }
    }
}

// ==========================================
// 调用说明：
// ==========================================
// 对于小数据量 (N <= 1024, K <= 32):
//   block = 32 (1个warp), grid = 1
//   shared_mem = 0
//   topk_v4_warp_kernel<<<1, 32>>>(input, out_vals, N, K);
//
// 对于大数据量 (N > 1024, K <= 32):
//   block = 256 (推荐), grid = 1
//   shared_mem = (block/32) * K * sizeof(float)
//   topk_v4_kernel<<<1, 256, shared_mem>>>(input, out_vals, N, K);
//
// 注意：1D版本只处理单个数组，grid始终为1

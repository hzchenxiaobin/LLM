#include "../include/topk_common.h"

// ==========================================
// V2: Block级别 Shared Memory Top-K (1D输入，无Batch轴，无索引输出)
// ==========================================
// 修改说明：输入只有一维，没有batch轴，只输出值不输出索引
// 输入: [N]  输出: values[K]
// 使用单个Block协作处理整个数组
// ==========================================

__global__ void topk_v2_kernel(const float* input, float* out_vals, int N, int K) {
    // 只使用block 0处理（grid = 1）
    if (blockIdx.x != 0) return;

    int tid = threadIdx.x;

    // 1. 每个线程维护自己的 Local Top-K值
    float local_vals[MAX_K];
    for (int i = 0; i < K; ++i) {
        local_vals[i] = -1e20f;
    }

    // 合并访存读取，每个线程处理一部分数据
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[i];
        if (val > local_vals[K - 1]) {
            int p = K - 1;
            while (p > 0 && val > local_vals[p - 1]) {
                local_vals[p] = local_vals[p - 1];
                p--;
            }
            local_vals[p] = val;
        }
    }

    // 2. Block 级别规约 (利用 Shared Memory)
    __shared__ float s_vals[256];
    __shared__ int s_owners[256]; // 记录最大值属于哪个线程

    int local_ptr = 0;
    float current_val = local_vals[0];

    // 循环 K 次找最大值
    for (int k = 0; k < K; ++k) {
        s_vals[tid] = current_val;
        s_owners[tid] = tid;
        __syncthreads();

        // 树状规约找最大值
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_vals[tid + stride] > s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + stride];
                    s_owners[tid] = s_owners[tid + stride];
                }
            }
            __syncthreads();
        }

        // Thread 0 写入全局内存
        if (tid == 0) {
            out_vals[k] = s_vals[0];
        }

        // 胜出的线程，指针下移，取出其下一个候选值参与下一次规约
        int winner = s_owners[0];
        if (tid == winner) {
            local_ptr++;
            if (local_ptr < K) {
                current_val = local_vals[local_ptr];
            } else {
                current_val = -1e20f;
            }
        }
        __syncthreads();
    }
}

// ==========================================
// 调用说明：
// ==========================================
// grid = 1, block = 256 (或根据硬件调整)
// 适用于中等规模数据
//
// topk_v2_kernel<<<1, 256>>>(input, out_vals, N, K);

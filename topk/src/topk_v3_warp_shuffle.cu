#include "../include/topk_common.h"

// ==========================================
// V3: Warp-level Shuffle Top-K (1D输入，无Batch轴，无索引输出)
// ==========================================
// 修改说明：输入只有一维，没有batch轴，只输出值不输出索引
// 输入: [N]  输出: values[K]
// 使用单个Warp协作处理整个数组
// ==========================================

#define WARP_SIZE 32

__global__ void topk_v3_kernel(const float* input, float* out_vals, int N, int K) {
    // 只使用 warp 0 (threads 0-31) 处理（block 可以为任意大小，但只有warp 0工作）
    if (threadIdx.x >= WARP_SIZE || blockIdx.x != 0) return;

    int lane_id = threadIdx.x; // 0~31

    // 1. 线程局部 Top-K值
    float local_vals[MAX_K];
    for (int i = 0; i < K; ++i) {
        local_vals[i] = -1e20f;
    }

    // Warp 内 32 个线程交错读取数据 (完美的合并访存)
    for (int i = lane_id; i < N; i += WARP_SIZE) {
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

    // 2. Warp 级归并
    int local_ptr = 0;
    float current_val = local_vals[0];

    for (int k = 0; k < K; ++k) {
        float max_val = current_val;
        int max_lane = lane_id;

        // Warp Reduction: 使用 __shfl_down_sync 直接在寄存器间交互
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_lane = __shfl_down_sync(0xffffffff, max_lane, offset);

            if (other_val > max_val) {
                max_val = other_val;
                max_lane = other_lane;
            }
        }

        // 只有 lane 0 拥有真正的最大值，它将胜利者的 lane_id 广播给所有线程
        max_lane = __shfl_sync(0xffffffff, max_lane, 0);

        // lane 0 写入输出
        if (lane_id == 0) {
            out_vals[k] = max_val;
        }

        // 胜出的线程更新它的候选值
        if (lane_id == max_lane) {
            local_ptr++;
            if (local_ptr < K) {
                current_val = local_vals[local_ptr];
            } else {
                current_val = -1e20f;
            }
        }
    }
}

// ==========================================
// 调用说明：
// ==========================================
// grid = 1, block = 32 (1个warp)
// 使用Warp Shuffle指令进行高效归约，无需Shared Memory
//
// topk_v3_kernel<<<1, 32>>>(input, out_vals, N, K);

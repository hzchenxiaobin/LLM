#include "../include/topk_common.h"

// ==========================================
// V3: Warp-per-Row + Warp Primitives (工业级)
// ==========================================
__global__ void topk_v3_kernel(const float* input, float* out_vals, int* out_inds, int Batch, int N, int K) {
    // 假设 blockDim.x = 32 (1 Warp), blockDim.y = 4 (4 Warps per Block)
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= Batch) return;

    int lane_id = threadIdx.x; // 0~31
    const float* row_input = input + row * N;

    // 1. 线程局部 Top-K
    float local_vals[MAX_K];
    int local_inds[MAX_K];
    for (int i = 0; i < K; ++i) { local_vals[i] = -1e20f; local_inds[i] = -1; }

    // Warp 内 32 个线程交错读取数据 (完美的合并访存)
    for (int i = lane_id; i < N; i += 32) {
        float val = row_input[i];
        if (val > local_vals[K - 1]) {
            int p = K - 1;
            while (p > 0 && val > local_vals[p - 1]) {
                local_vals[p] = local_vals[p - 1];
                local_inds[p] = local_inds[p - 1];
                p--;
            }
            local_vals[p] = val;
            local_inds[p] = i;
        }
    }

    // 2. Warp 级归并
    int local_ptr = 0;
    float current_val = local_vals[0];
    int current_idx = local_inds[0];

    float* row_out_vals = out_vals + row * K;
    int* row_out_inds = out_inds + row * K;

    for (int k = 0; k < K; ++k) {
        float max_val = current_val;
        int max_idx = current_idx;
        int max_lane = lane_id;

        // Warp Reduction: 使用 __shfl_down_sync 直接在寄存器间交互
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            int other_lane = __shfl_down_sync(0xffffffff, max_lane, offset);

            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
                max_lane = other_lane;
            }
        }

        // 只有 lane 0 拥有真正的最大值，它将胜利者的 lane_id 广播给所有线程
        max_lane = __shfl_sync(0xffffffff, max_lane, 0);

        if (lane_id == 0) {
            row_out_vals[k] = max_val;
            row_out_inds[k] = max_idx;
        }

        // 胜出的线程更新它的候选值
        if (lane_id == max_lane) {
            local_ptr++;
            if (local_ptr < K) {
                current_val = local_vals[local_ptr];
                current_idx = local_inds[local_ptr];
            } else {
                current_val = -1e20f;
            }
        }
    }
}

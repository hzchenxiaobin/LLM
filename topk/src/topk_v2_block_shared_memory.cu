#include "../include/topk_common.h"

// ==========================================
// V2: Block-per-Row + Shared Memory
// ==========================================
__global__ void topk_v2_kernel(const float* input, float* out_vals, int* out_inds, int Batch, int N, int K) {
    int row = blockIdx.x; // 1 个 Block 处理 1 行
    if (row >= Batch) return;

    int tid = threadIdx.x;
    const float* row_input = input + row * N;

    // 1. 每个线程维护自己的 Local Top-K
    float local_vals[MAX_K];
    int local_inds[MAX_K];
    for (int i = 0; i < K; ++i) { local_vals[i] = -1e20f; local_inds[i] = -1; }

    for (int i = tid; i < N; i += blockDim.x) { // 合并访存读取
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

    // 2. Block 级别规约 (利用 Shared Memory)
    __shared__ float s_vals[256];
    __shared__ int s_inds[256];
    __shared__ int s_owners[256]; // 记录最大值属于哪个线程

    int local_ptr = 0;
    float current_val = local_vals[0];
    int current_idx = local_inds[0];

    float* row_out_vals = out_vals + row * K;
    int* row_out_inds = out_inds + row * K;

    // 循环 K 次找最大值
    for (int k = 0; k < K; ++k) {
        s_vals[tid] = current_val;
        s_inds[tid] = current_idx;
        s_owners[tid] = tid;
        __syncthreads();

        // 树状规约找最大值
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_vals[tid + stride] > s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + stride];
                    s_inds[tid] = s_inds[tid + stride];
                    s_owners[tid] = s_owners[tid + stride];
                }
            }
            __syncthreads();
        }

        // Thread 0 写入全局内存
        if (tid == 0) {
            row_out_vals[k] = s_vals[0];
            row_out_inds[k] = s_inds[0];
        }

        // 胜出的线程，指针下移，取出其下一个候选值参与下一次规约
        int winner = s_owners[0];
        if (tid == winner) {
            local_ptr++;
            if (local_ptr < K) {
                current_val = local_vals[local_ptr];
                current_idx = local_inds[local_ptr];
            } else {
                current_val = -1e20f;
            }
        }
        __syncthreads();
    }
}

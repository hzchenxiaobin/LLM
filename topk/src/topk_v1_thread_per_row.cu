#include "../include/topk_common.h"

// ==========================================
// V1: 朴素版本 (Thread-per-Row)
// ==========================================
__global__ void topk_v1_kernel(const float* input, float* out_vals, int* out_inds, int Batch, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Batch) return;

    const float* row_input = input + row * N;
    float* row_out_vals = out_vals + row * K;
    int* row_out_inds = out_inds + row * K;

    float top_vals[MAX_K];
    int top_inds[MAX_K];
    for (int i = 0; i < K; ++i) {
        top_vals[i] = -1e20f;
        top_inds[i] = -1;
    }

    // 串行遍历 N 个元素，插入排序
    for (int i = 0; i < N; ++i) {
        float val = row_input[i];
        if (val > top_vals[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && val > top_vals[pos - 1]) {
                top_vals[pos] = top_vals[pos - 1];
                top_inds[pos] = top_inds[pos - 1];
                pos--;
            }
            top_vals[pos] = val;
            top_inds[pos] = i;
        }
    }

    for (int i = 0; i < K; ++i) {
        row_out_vals[i] = top_vals[i];
        row_out_inds[i] = top_inds[i];
    }
}

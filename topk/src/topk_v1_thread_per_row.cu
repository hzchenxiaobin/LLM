#include "../include/topk_common.h"

// ==========================================
// V1: 朴素版本 (1D输入，无Batch轴，无索引输出)
// ==========================================
// 修改说明：输入只有一维，没有batch轴，只输出值不输出索引
// 输入: [N]  输出: values[K]
// 使用单个线程处理整个数组（适合小数据量调试）
// ==========================================

__global__ void topk_v1_kernel(const float* input, float* out_vals, int N, int K) {
    // 只使用一个线程处理（线程0）
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // 寄存器存储Top-K值
    float top_vals[MAX_K];
    for (int i = 0; i < K; ++i) {
        top_vals[i] = -1e20f;
    }

    // 串行遍历 N 个元素，插入排序维护Top-K
    for (int i = 0; i < N; ++i) {
        float val = input[i];
        if (val > top_vals[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && val > top_vals[pos - 1]) {
                top_vals[pos] = top_vals[pos - 1];
                pos--;
            }
            top_vals[pos] = val;
        }
    }

    // 写入输出
    for (int i = 0; i < K; ++i) {
        out_vals[i] = top_vals[i];
    }
}

// ==========================================
// 调用说明：
// ==========================================
// grid = 1, block = 1 (单线程串行处理)
// 适用于小数据量调试，大数据量请使用V4版本
//
// topk_v1_kernel<<<1, 1>>>(input, out_vals, N, K);

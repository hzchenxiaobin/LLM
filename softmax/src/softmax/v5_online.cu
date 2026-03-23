// V5: Online Softmax
// Single-pass algorithm that updates max and sum simultaneously
// Based on FlashAttention's online softmax

#include "../include/softmax_common.h"

__global__ void softmax_v5_online_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Single pass: update max and sum together
    for (int i = lane_id; i < N; i += 32) {
        float val = x[i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }

    // Warp reduce with online correction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);

        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max) + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    float row_max = __shfl_sync(0xffffffff, local_max, 0);
    float row_sum = __shfl_sync(0xffffffff, local_sum, 0);

    // Second pass: write results
    for (int i = lane_id; i < N; i += 32) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}

// Host function for V5 - Optimized for RTX 5090 (Blackwell)
// Online Softmax with 256 threads for best performance
void softmax_v5(const float* d_input, float* d_output, int M, int N) {
    // RTX 5090: 256 threads (8 warps) for optimal L2 cache usage and occupancy
    int threads = 256;  // 8 warps per block
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v5_online_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// V5+Vec: Ultimate (Online Softmax + Vectorized)
// Combines V4 (float4) and V5 (online) for maximum performance

#include "../include/softmax_common.h"

__global__ void softmax_v5_vec_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float4* x_vec = reinterpret_cast<const float4*>(input + row * N);
    float4* y_vec = reinterpret_cast<float4*>(output + row * N);
    int N_vec = N / 4;

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Single pass with float4
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];

        // Process x component
        float new_max = fmaxf(local_max, val.x);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.x - new_max);
        local_max = new_max;

        // Process y component
        new_max = fmaxf(local_max, val.y);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.y - new_max);
        local_max = new_max;

        // Process z component
        new_max = fmaxf(local_max, val.z);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.z - new_max);
        local_max = new_max;

        // Process w component
        new_max = fmaxf(local_max, val.w);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.w - new_max);
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
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        float4 out_val;
        out_val.x = expf(val.x - row_max) / row_sum;
        out_val.y = expf(val.y - row_max) / row_sum;
        out_val.z = expf(val.z - row_max) / row_sum;
        out_val.w = expf(val.w - row_max) / row_sum;
        y_vec[i] = out_val;
    }
}

// Host function for V5+Vec - Ultimate optimization for RTX 5090 (Blackwell)
// Combines Online Softmax with float4 vectorization and 256 threads
void softmax_v5_vec(const float* d_input, float* d_output, int M, int N) {
    // RTX 5090: 256 threads (8 warps) for ultimate bandwidth utilization with GDDR7
    int threads = 256;  // 8 warps per block
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v5_vec_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

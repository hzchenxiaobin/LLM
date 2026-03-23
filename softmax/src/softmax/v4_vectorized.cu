// V4: Vectorized Memory Access (float4)
// Uses 128-bit vectorized loads/stores to maximize bandwidth

#include "../include/softmax_common.h"

__global__ void softmax_v4_vectorized_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float4* x_vec = reinterpret_cast<const float4*>(input + row * N);
    float4* y_vec = reinterpret_cast<float4*>(output + row * N);

    int N_vec = N / 4;

    // 1. Find max
    float local_max = -INFINITY;
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        local_max = fmaxf(local_max, val.x);
        local_max = fmaxf(local_max, val.y);
        local_max = fmaxf(local_max, val.z);
        local_max = fmaxf(local_max, val.w);
    }
    float row_max = warpReduceMax(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // 2. Compute sum
    float local_sum = 0.0f;
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        local_sum += expf(val.x - row_max);
        local_sum += expf(val.y - row_max);
        local_sum += expf(val.z - row_max);
        local_sum += expf(val.w - row_max);
    }
    float row_sum = warpReduceSum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // 3. Write back
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

// Host function for V4 - Optimized for RTX 5090 (Blackwell) with GDDR7
// Use 256 threads (8 warps) to maximize GDDR7 bandwidth utilization
void softmax_v4(const float* d_input, float* d_output, int M, int N) {
    // RTX 5090: 256 threads (8 warps) for better memory coalescing on GDDR7
    int threads = 256;  // 8 warps per block, optimal for RTX 5090 GDDR7
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v4_vectorized_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

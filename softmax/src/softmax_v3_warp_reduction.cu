// V3: Warp-level Reduction
// Uses warp shuffle instructions instead of shared memory

#include "softmax_common.h"

__global__ void softmax_v3_warp_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    // 1. Find max
    float local_max = -INFINITY;
    for (int i = lane_id; i < N; i += 32) {
        local_max = fmaxf(local_max, x[i]);
    }
    float row_max = warpReduceMax(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // 2. Compute sum
    float local_sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        local_sum += expf(x[i] - row_max);
    }
    float row_sum = warpReduceSum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // 3. Write back
    for (int i = lane_id; i < N; i += 32) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}

// Host function for V3 - Optimized for RTX 5090 (Blackwell)
// Use 256 threads (8 warps) per block for better SM utilization on RTX 5090
void softmax_v3(const float* d_input, float* d_output, int M, int N) {
    // RTX 5090 has more SMs, use larger blocks with more warps
    int threads = 256;  // 8 warps per block, increased for RTX 5090
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v3_warp_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void run_benchmark_v3(const float* d_input, float* d_output,
                      float* h_output, const float* h_ref,
                      int M, int N, int warmup_iters, int benchmark_iters,
                      size_t data_size_bytes, BenchmarkResult* result) {

    printf("Benchmarking V3: Warp-level Reduction...\n");

    for (int i = 0; i < warmup_iters; i++) {
        softmax_v3(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        softmax_v3(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V3: Warp-level Reduction";
    result->time_ms = (end - start) / benchmark_iters;

    double total_bytes = 2.0 * M * N * sizeof(float);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size_bytes, cudaMemcpyDeviceToHost));
    result->max_error = max_error(h_output, h_ref, M * N);
    result->passed = result->max_error < 1e-4;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASSED" : "FAILED");
}

// V1: Naive Softmax (3 separate kernels)
// Most basic implementation - 3 kernels, 6 global memory accesses per element

#include "softmax_common.h"

// Kernel 1: Find max value per row
__global__ void kernel_max_v1(const float* input, float* max_vals, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float local_max = -INFINITY;
    for (int i = 0; i < N; i++) {
        local_max = fmaxf(local_max, input[row * N + i]);
    }
    max_vals[row] = local_max;
}

// Kernel 2: Compute sum of exp(x - max) per row
__global__ void kernel_sum_v1(const float* input, const float* max_vals, float* sum_vals, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float row_max = max_vals[row];
    float local_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        local_sum += expf(input[row * N + i] - row_max);
    }
    sum_vals[row] = local_sum;
}

// Kernel 3: Normalize and write output
__global__ void kernel_div_v1(const float* input, const float* max_vals, const float* sum_vals,
                               float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float row_max = max_vals[row];
    float row_sum = sum_vals[row];
    for (int i = 0; i < N; i++) {
        output[row * N + i] = expf(input[row * N + i] - row_max) / row_sum;
    }
}

// Host function for V1 - Optimized for RTX 5090 (Blackwell)
// Use larger thread blocks for better occupancy on RTX 5090's many SMs
void softmax_v1(const float* d_input, float* d_output, float* d_max, float* d_sum, int M, int N) {
    // RTX 5090 has many SMs, use larger blocks for better parallelism
    int threads = 512;  // Increased from 256 for RTX 5090
    int blocks = (M + threads - 1) / threads;

    kernel_max_v1<<<blocks, threads>>>(d_input, d_max, M, N);
    kernel_sum_v1<<<blocks, threads>>>(d_input, d_max, d_sum, M, N);
    kernel_div_v1<<<blocks, threads>>>(d_input, d_max, d_sum, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Standalone benchmark for V1
void run_benchmark_v1(const float* d_input, float* d_output, float* d_max, float* d_sum,
                      float* h_output, const float* h_ref,
                      int M, int N, int warmup_iters, int benchmark_iters,
                      size_t data_size_bytes, BenchmarkResult* result) {

    printf("Benchmarking V1: Naive (3 kernels)...\n");

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        softmax_v1(d_input, d_output, d_max, d_sum, M, N);
    }

    // Benchmark
    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        softmax_v1(d_input, d_output, d_max, d_sum, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    // Calculate metrics
    result->name = "V1: Naive (3 kernels)";
    result->time_ms = (end - start) / benchmark_iters;

    // V1 reads input 3 times, writes output 1 time
    double total_bytes = 4.0 * M * N * sizeof(float);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size_bytes, cudaMemcpyDeviceToHost));
    result->max_error = max_error(h_output, h_ref, M * N);
    result->passed = result->max_error < 1e-4;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASSED" : "FAILED");
}

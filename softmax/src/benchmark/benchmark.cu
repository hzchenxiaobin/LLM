// Benchmark implementations for all softmax versions
// Centralized benchmark functions that call the respective softmax implementations

#include "../include/softmax_common.h"

// V1: Naive (3 kernels) - requires d_max and d_sum buffers
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

// V2: Block Shared Memory
void run_benchmark_v2(const float* d_input, float* d_output,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result) {

    printf("Benchmarking V2: Block Shared Memory...\n");

    for (int i = 0; i < warmup_iters; i++) {
        softmax_v2(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        softmax_v2(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V2: Block Shared Memory";
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

// V3: Warp-level Reduction
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

// V4: Vectorized (float4)
void run_benchmark_v4(const float* d_input, float* d_output,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result) {

    printf("Benchmarking V4: Vectorized (float4)...\n");

    for (int i = 0; i < warmup_iters; i++) {
        softmax_v4(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        softmax_v4(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V4: Vectorized (float4)";
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

// V5: Online Softmax
void run_benchmark_v5(const float* d_input, float* d_output,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result) {

    printf("Benchmarking V5: Online Softmax...\n");

    for (int i = 0; i < warmup_iters; i++) {
        softmax_v5(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        softmax_v5(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V5: Online Softmax";
    result->time_ms = (end - start) / benchmark_iters;

    // Online softmax: 1 full pass + 1 read pass (L2 cache hit)
    double total_bytes = 2.0 * M * N * sizeof(float);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size_bytes, cudaMemcpyDeviceToHost));
    result->max_error = max_error(h_output, h_ref, M * N);
    result->passed = result->max_error < 1e-4;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASSED" : "FAILED");
}

// V5+Vec: Ultimate (Online + float4)
void run_benchmark_v5_vec(const float* d_input, float* d_output,
                            float* h_output, const float* h_ref,
                            int M, int N, int warmup_iters, int benchmark_iters,
                            size_t data_size_bytes, BenchmarkResult* result) {

    printf("Benchmarking V5+Vec: Ultimate (Online + float4)...\n");

    for (int i = 0; i < warmup_iters; i++) {
        softmax_v5_vec(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        softmax_v5_vec(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V5+Vec: Ultimate (Online + float4)";
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

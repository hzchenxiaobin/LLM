// Benchmark implementations for all Top-K versions

#include "../include/topk_common.h"

void run_benchmark_v0(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result) {
    printf("Testing V0: Thrust Baseline...\n");

    TopKNode* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(TopKNode)));

    for (int i = 0; i < warmup_iters; i++) {
        topk_v0_thrust(d_input, N, K, d_output);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        topk_v0_thrust(d_input, N, K, d_output);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V0: Thrust Baseline";
    result->time_ms = (end - start) / benchmark_iters;
    double total_bytes = N * sizeof(float) + N * sizeof(TopKNode) + K * sizeof(TopKNode);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(TopKNode), cudaMemcpyDeviceToHost));
    result->max_error = max_error_topk(h_output, h_ref, K);
    result->passed = result->max_error < 1e-4;
    result->speedup_vs_baseline = 1.0;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_output));
}

void run_benchmark_v1(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result) {
    printf("Testing V1: Thread-level...\n");

    TopKNode* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(TopKNode)));

    int num_threads = 256;
    int num_blocks = 256;

    for (int i = 0; i < warmup_iters; i++) {
        topk_v1_thread(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        topk_v1_thread(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V1: Thread-level";
    result->time_ms = (end - start) / benchmark_iters;
    double total_bytes = N * sizeof(float) + num_blocks * num_threads * K * sizeof(TopKNode) + K * sizeof(TopKNode);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(TopKNode), cudaMemcpyDeviceToHost));
    result->max_error = max_error_topk(h_output, h_ref, K);
    result->passed = result->max_error < 1e-4;
    result->speedup_vs_baseline = 1.0;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_output));
}

void run_benchmark_v2(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result) {
    printf("Testing V2: Shared Memory...\n");

    TopKNode* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(TopKNode)));

    int num_threads = 256;
    int num_blocks = 256;

    for (int i = 0; i < warmup_iters; i++) {
        topk_v2_shared_memory(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        topk_v2_shared_memory(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V2: Shared Memory";
    result->time_ms = (end - start) / benchmark_iters;
    double total_bytes = N * sizeof(float) + num_blocks * K * sizeof(TopKNode) + K * sizeof(TopKNode);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(TopKNode), cudaMemcpyDeviceToHost));
    result->max_error = max_error_topk(h_output, h_ref, K);
    result->passed = result->max_error < 1e-4;
    result->speedup_vs_baseline = 1.0;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_output));
}

void run_benchmark_v3(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result) {
    printf("Testing V3: Warp Shuffle...\n");

    TopKNode* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(TopKNode)));

    int num_threads = 256;
    int num_blocks = 256;

    for (int i = 0; i < warmup_iters; i++) {
        topk_v3_warp_shuffle(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        topk_v3_warp_shuffle(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V3: Warp Shuffle";
    result->time_ms = (end - start) / benchmark_iters;
    double total_bytes = N * sizeof(float) + num_blocks * K * sizeof(TopKNode) + K * sizeof(TopKNode);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(TopKNode), cudaMemcpyDeviceToHost));
    result->max_error = max_error_topk(h_output, h_ref, K);
    result->passed = result->max_error < 1e-4;
    result->speedup_vs_baseline = 1.0;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_output));
}

void run_benchmark_v4(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result) {
    printf("Testing V4: Radix Select...\n");

    TopKNode* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(TopKNode)));

    int num_threads = 256;
    int num_blocks = 256;

    for (int i = 0; i < warmup_iters; i++) {
        topk_v4_radix_select(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        topk_v4_radix_select(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V4: Radix Select";
    result->time_ms = (end - start) / benchmark_iters;
    double total_bytes = 4 * N * sizeof(float) + 2 * N * sizeof(TopKNode);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(TopKNode), cudaMemcpyDeviceToHost));
    result->max_error = max_error_topk(h_output, h_ref, K);
    result->passed = result->max_error < 1e-4;
    result->speedup_vs_baseline = 1.0;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_output));
}

void run_benchmark_v5(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result) {
    printf("Testing V5: Blackwell Optimized...\n");

    TopKNode* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(TopKNode)));

    int num_threads = 256;
    int num_blocks = 256;

    for (int i = 0; i < warmup_iters; i++) {
        topk_v5_blackwell(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        topk_v5_blackwell(d_input, N, K, d_output, num_blocks, num_threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V5: Blackwell Opt";
    result->time_ms = (end - start) / benchmark_iters;
    double total_bytes = N * sizeof(float) + num_blocks * K * sizeof(TopKNode) + K * sizeof(TopKNode);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(TopKNode), cudaMemcpyDeviceToHost));
    result->max_error = max_error_topk(h_output, h_ref, K);
    result->passed = result->max_error < 1e-4;
    result->speedup_vs_baseline = 1.0;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_output));
}

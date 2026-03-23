// Main benchmark driver - tests all softmax versions
// Includes: V1, V2, V3, V4, V5, V5+Vec

#include "softmax_common.h"

// Declare benchmark functions from each version
extern void run_benchmark_v1(const float* d_input, float* d_output, float* d_max, float* d_sum,
                             float* h_output, const float* h_ref,
                             int M, int N, int warmup_iters, int benchmark_iters,
                             size_t data_size_bytes, BenchmarkResult* result);

extern void run_benchmark_v2(const float* d_input, float* d_output,
                             float* h_output, const float* h_ref,
                             int M, int N, int warmup_iters, int benchmark_iters,
                             size_t data_size_bytes, BenchmarkResult* result);

extern void run_benchmark_v3(const float* d_input, float* d_output,
                             float* h_output, const float* h_ref,
                             int M, int N, int warmup_iters, int benchmark_iters,
                             size_t data_size_bytes, BenchmarkResult* result);

extern void run_benchmark_v4(const float* d_input, float* d_output,
                             float* h_output, const float* h_ref,
                             int M, int N, int warmup_iters, int benchmark_iters,
                             size_t data_size_bytes, BenchmarkResult* result);

extern void run_benchmark_v5(const float* d_input, float* d_output,
                             float* h_output, const float* h_ref,
                             int M, int N, int warmup_iters, int benchmark_iters,
                             size_t data_size_bytes, BenchmarkResult* result);

extern void run_benchmark_v5_vec(const float* d_input, float* d_output,
                                 float* h_output, const float* h_ref,
                                 int M, int N, int warmup_iters, int benchmark_iters,
                                 size_t data_size_bytes, BenchmarkResult* result);

// RTX 5090 Blackwell Architecture Specific Optimizations:
// - Architecture: sm_100 (Blackwell)
// - Memory: GDDR7 with 1792 GB/s theoretical bandwidth
// - Optimized thread counts: 512 for block-level, 256 for warp-level
// - Enhanced L2 cache utilization for Online Softmax variants

int main(int argc, char** argv) {
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("===============================================\n");
    printf("  Softmax Performance Benchmark\n");
    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("  Memory Clock: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  Theoretical Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * prop.memoryBusWidth / (8.0 * 1e6));
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("===============================================\n");
    printf("  Optimized for RTX 5090 (Blackwell sm_100)\n");
    printf("  Thread config: 512 (V1/V2), 256 (V3-V5) for max occupancy\n");
    printf("===============================================\n\n");

    // Test configurations (M, N) - typical LLM attention shapes
    int test_configs[][2] = {
        {1024, 128},      // Small: batch=32, heads=32, seq=128
        {1024, 512},      // Medium: batch=32, heads=32, seq=512
        {1024, 1024},     // Large: batch=32, heads=32, seq=1024
        {1024, 2048},     // XL: batch=32, heads=32, seq=2048
        {4096, 1024},     // Batch large
        {1024, 4096},     // Very long sequence
    };
    int num_configs = sizeof(test_configs) / sizeof(test_configs[0]);

    const int warmup_iters = 10;
    const int benchmark_iters = 100;

    for (int cfg = 0; cfg < num_configs; cfg++) {
        int M = test_configs[cfg][0];
        int N = test_configs[cfg][1];

        // Skip vectorized versions if N is not divisible by 4
        bool can_vectorize = (N % 4 == 0);

        printf("\n========================================\n");
        printf("Configuration: M=%d, N=%d (rows x cols)\n", M, N);
        printf("Total elements: %d (%.2f MB)\n", M * N, M * N * sizeof(float) / (1024.0 * 1024));
        printf("========================================\n\n");

        size_t data_size = M * N * sizeof(float);
        size_t row_size = M * sizeof(float);

        // Allocate host memory
        float *h_input = (float*)malloc(data_size);
        float *h_output = (float*)malloc(data_size);
        float *h_ref = (float*)malloc(data_size);

        // Initialize data
        srand(42);
        init_random(h_input, M * N);

        // Compute CPU reference
        softmax_cpu(h_input, h_ref, M, N);

        // Allocate device memory
        float *d_input, *d_output, *d_max, *d_sum;
        CUDA_CHECK(cudaMalloc(&d_input, data_size));
        CUDA_CHECK(cudaMalloc(&d_output, data_size));
        CUDA_CHECK(cudaMalloc(&d_max, row_size));
        CUDA_CHECK(cudaMalloc(&d_sum, row_size));

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));

        // Run benchmarks
        BenchmarkResult results[7];
        int result_idx = 0;

        // V1: Naive (3 kernels)
        run_benchmark_v1(d_input, d_output, d_max, d_sum,
                         h_output, h_ref, M, N,
                         warmup_iters, benchmark_iters, data_size,
                         &results[result_idx++]);

        // V2: Block-level Shared Memory
        run_benchmark_v2(d_input, d_output,
                         h_output, h_ref, M, N,
                         warmup_iters, benchmark_iters, data_size,
                         &results[result_idx++]);

        // V3: Warp-level Reduction
        run_benchmark_v3(d_input, d_output,
                         h_output, h_ref, M, N,
                         warmup_iters, benchmark_iters, data_size,
                         &results[result_idx++]);

        // V4: Vectorized (if possible)
        if (can_vectorize) {
            run_benchmark_v4(d_input, d_output,
                             h_output, h_ref, M, N,
                             warmup_iters, benchmark_iters, data_size,
                             &results[result_idx++]);
        }

        // V5: Online Softmax
        run_benchmark_v5(d_input, d_output,
                         h_output, h_ref, M, N,
                         warmup_iters, benchmark_iters, data_size,
                         &results[result_idx++]);

        // V5+Vec: Ultimate (Online + Vectorized)
        if (can_vectorize) {
            run_benchmark_v5_vec(d_input, d_output,
                                 h_output, h_ref, M, N,
                                 warmup_iters, benchmark_iters, data_size,
                                 &results[result_idx++]);
        }

        // Print summary table
        printf("\n--- Summary for M=%d, N=%d ---\n", M, N);
        printf("%-40s %10s %12s %12s\n", "Version", "Time(ms)", "BW(GB/s)", "Error");
        printf("--------------------------------------------------------------------------------\n");
        for (int i = 0; i < result_idx; i++) {
            printf("%-40s %10.4f %12.2f %12.2e %s\n",
                   results[i].name, results[i].time_ms,
                   results[i].bandwidth_gbps, results[i].max_error,
                   results[i].passed ? "✓" : "✗");
        }

        // Find best performing
        double best_time = results[0].time_ms;
        const char* best_name = results[0].name;
        for (int i = 1; i < result_idx; i++) {
            if (results[i].time_ms < best_time) {
                best_time = results[i].time_ms;
                best_name = results[i].name;
            }
        }
        printf("\nBest performer: %s (%.4f ms)\n", best_name, best_time);

        // Cleanup
        free(h_input);
        free(h_output);
        free(h_ref);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_max));
        CUDA_CHECK(cudaFree(d_sum));
    }

    printf("\n\nBenchmark complete!\n");
    return 0;
}

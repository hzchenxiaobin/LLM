// CUDA Top-K Benchmark Driver

#include "include/topk_common.h"

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("===============================================================================\n");
    printf("  CUDA Top-K Performance Benchmark\n");
    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("  L2 Cache: %d KB\n", prop.l2CacheSize / 1024);
    printf("===============================================================================\n\n");

    struct TestConfig { int N; int K; };
    TestConfig configs[] = {
        {1000000, 8},
        {1000000, 32},
        {1000000, 64},
        {5000000, 8},
        {5000000, 32},
        {5000000, 64},
        {10000000, 8},
        {10000000, 32},
        {10000000, 64},
        {10000000, 128},
        {10000000, 256},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    const int warmup_iters = 5;
    const int benchmark_iters = 20;

    for (int cfg = 0; cfg < num_configs; cfg++) {
        int N = configs[cfg].N;
        int K = configs[cfg].K;

        printf("\n");
        printf("Configuration: N=%d, K=%d\n", N, K);
        printf("---------------------------------------------------------------\n\n");

        size_t data_size = N * sizeof(float);
        size_t output_size = K * sizeof(TopKNode);

        float *h_input = (float*)malloc(data_size);
        TopKNode *h_output = (TopKNode*)malloc(output_size);
        TopKNode *h_ref = (TopKNode*)malloc(output_size);

        init_random_positive(h_input, N);

        printf("Computing CPU reference...\n");
        double cpu_start = get_time_ms();
        topk_cpu_reference(h_input, h_ref, N, K);
        double cpu_end = get_time_ms();
        printf("  CPU time: %.4f ms\n\n", cpu_end - cpu_start);

        float *d_input;
        CUDA_CHECK(cudaMalloc(&d_input, data_size));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));

        BenchmarkResult results[6];
        int result_idx = 0;

        run_benchmark_v0(d_input, N, K, h_output, h_ref, warmup_iters, benchmark_iters, data_size, &results[result_idx++]);
        run_benchmark_v1(d_input, N, K, h_output, h_ref, warmup_iters, benchmark_iters, data_size, &results[result_idx++]);
        run_benchmark_v2(d_input, N, K, h_output, h_ref, warmup_iters, benchmark_iters, data_size, &results[result_idx++]);
        run_benchmark_v3(d_input, N, K, h_output, h_ref, warmup_iters, benchmark_iters, data_size, &results[result_idx++]);
        run_benchmark_v4(d_input, N, K, h_output, h_ref, warmup_iters, benchmark_iters, data_size, &results[result_idx++]);
        run_benchmark_v5(d_input, N, K, h_output, h_ref, warmup_iters, benchmark_iters, data_size, &results[result_idx++]);

        double baseline_time = results[0].time_ms;
        for (int i = 0; i < result_idx; i++) {
            results[i].speedup_vs_baseline = baseline_time / results[i].time_ms;
        }

        printf("\n");
        printf("Summary (N=%d, K=%d):\n", N, K);
        printf("%-20s %10s %12s %10s %10s\n", "Version", "Time(ms)", "Bandwidth", "Speedup", "Status");
        printf("--------------------------------------------------------------------------------\n");
        for (int i = 0; i < result_idx; i++) {
            printf("%-20s %10.4f %10.2f GB/s %8.2fx %10s\n",
                   results[i].name, results[i].time_ms,
                   results[i].bandwidth_gbps, results[i].speedup_vs_baseline,
                   results[i].passed ? "PASS" : "FAIL");
        }

        double best_time = results[0].time_ms;
        const char* best_name = results[0].name;
        for (int i = 1; i < result_idx; i++) {
            if (results[i].time_ms < best_time) {
                best_time = results[i].time_ms;
                best_name = results[i].name;
            }
        }
        printf("\nBest: %s (%.4f ms, %.2fx speedup)\n", best_name, best_time, baseline_time / best_time);

        free(h_input);
        free(h_output);
        free(h_ref);
        CUDA_CHECK(cudaFree(d_input));
    }

    printf("\n\nBenchmark complete!\n");
    return 0;
}

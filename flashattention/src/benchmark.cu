/*
 * FlashAttention Benchmark Suite
 * ==============================
 *
 * This benchmark compares different FlashAttention implementations:
 * - V1: Naive (global memory only)
 * - V2: Shared memory KV tiling
 * - V3: Double buffering
 * - V4: Vectorized loads + bank conflict free
 * - V5: FlashAttention-2 style (warp partitioned KV)
 * - V6: Tensor Core (demonstration)
 * - CPU reference
 * - cuBLAS reference (partial)
 *
 * For each version, we measure:
 * - Execution time (ms)
 * - TFLOPs/s
 * - Memory bandwidth (GB/s)
 * - Correctness (max error vs reference)
 */

#include "common.h"
#include "flashattention/kernels.h"
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <functional>
#include <iomanip>
#include <iostream>

// Test configuration
struct TestConfig {
    int B;  // Batch size
    int N;  // Sequence length
    int d;  // Head dimension
    int iterations;  // Number of runs for averaging
};

// Result structure
struct BenchmarkResult {
    std::string name;
    float time_ms;
    double tflops;
    double bandwidth_gb_s;
    float max_error;
    bool passed;
};

// =============================================================================
// Utility Functions
// =============================================================================

void print_header(const TestConfig& config) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "FlashAttention Benchmark\n";
    std::cout << "Configuration: B=" << config.B
              << ", N=" << config.N
              << ", d=" << config.d
              << ", iterations=" << config.iterations << "\n";

    double flops = flash_attention_flops(config.B, config.N, config.d);
    std::cout << "Total FLOPs per run: " << std::scientific << flops << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left << std::setw(20) << "Version"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(15) << "TFLOPs/s"
              << std::setw(18) << "Bandwidth (GB/s)"
              << std::setw(12) << "Max Error"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(15) << r.time_ms
                  << std::setw(15) << r.tflops
                  << std::setw(18) << r.bandwidth_gb_s
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.max_error
                  << std::setw(10) << (r.passed ? "PASS" : "FAIL")
                  << "\n";
    }
    std::cout << std::string(100, '-') << "\n";
}

// =============================================================================
// Benchmark Function
// =============================================================================

BenchmarkResult run_benchmark(
    const std::string& name,
    std::function<void()> kernel_func,
    const float* reference_output,
    float* test_output,
    int total_elements,
    const TestConfig& config,
    double flops)
{
    BenchmarkResult result;
    result.name = name;

    // Warmup
    for (int i = 0; i < 3; i++) {
        kernel_func();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    GpuTimer timer;
    float total_time = 0.0f;

    for (int i = 0; i < config.iterations; i++) {
        // Clear output buffer
        CUDA_CHECK(cudaMemset(test_output, 0, total_elements * sizeof(float)));

        timer.start();
        kernel_func();
        timer.stop();
        total_time += timer.elapsed_msecs();
    }

    result.time_ms = total_time / config.iterations;
    result.tflops = calculate_tflops(flops, result.time_ms);
    result.bandwidth_gb_s = calculate_memory_bandwidth(config.B, config.N, config.d, result.time_ms);

    // Check correctness
    std::vector<float> h_output(total_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), test_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    result.max_error = compare_matrices(reference_output, h_output.data(), total_elements);
    result.passed = (result.max_error < 1e-3);  // Tolerance for FP32

    return result;
}

// =============================================================================
// Main Benchmark
// =============================================================================

int main(int argc, char** argv) {
    // Parse command line arguments
    TestConfig config = {1, 1024, 64, 10};  // Default

    if (argc > 1) config.B = atoi(argv[1]);
    if (argc > 2) config.N = atoi(argv[2]);
    if (argc > 3) config.d = atoi(argv[3]);
    if (argc > 4) config.iterations = atoi(argv[4]);

    print_header(config);

    // Calculate sizes
    int qkv_size = config.B * config.N * config.d;
    int output_size = qkv_size;
    double flops = flash_attention_flops(config.B, config.N, config.d);

    // Host memory
    std::vector<float> h_Q(qkv_size);
    std::vector<float> h_K(qkv_size);
    std::vector<float> h_V(qkv_size);
    std::vector<float> h_O_cpu(output_size);
    std::vector<float> h_O_ref(output_size);

    // Initialize with random data
    srand(42);
    init_matrix(h_Q.data(), config.B * config.N, config.d);
    init_matrix(h_K.data(), config.B * config.N, config.d);
    init_matrix(h_V.data(), config.B * config.N, config.d);

    // Device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, output_size * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));

    // Compute CPU reference
    std::cout << "Computing CPU reference...\n";
    standard_attention_cpu(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(),
                            config.B, config.N, config.d);

    // Run benchmarks
    std::vector<BenchmarkResult> results;

    // V1: Naive
    std::cout << "\nBenchmarking V1 (Naive)...\n";
    results.push_back(run_benchmark(
        "V1 Naive",
        [&]() { flash_attention_v1_naive(d_Q, d_K, d_V, d_O, config.B, config.N, config.d); },
        h_O_cpu.data(), d_O, output_size, config, flops
    ));

    // V2: Shared KV
    std::cout << "Benchmarking V2 (Shared KV)...\n";
    results.push_back(run_benchmark(
        "V2 Shared KV",
        [&]() { flash_attention_v2_shared_kv(d_Q, d_K, d_V, d_O, config.B, config.N, config.d); },
        h_O_cpu.data(), d_O, output_size, config, flops
    ));

    // V3: Q Tiling + Double Buffer
    std::cout << "Benchmarking V3 (Double Buffer)...\n";
    results.push_back(run_benchmark(
        "V3 Double Buffer",
        [&]() { flash_attention_v3_q_tiling(d_Q, d_K, d_V, d_O, config.B, config.N, config.d); },
        h_O_cpu.data(), d_O, output_size, config, flops
    ));

    // V4: Vectorized
    if (config.d % 4 == 0) {
        std::cout << "Benchmarking V4 (Vectorized)...\n";
        results.push_back(run_benchmark(
            "V4 Vectorized",
            [&]() { flash_attention_v4_vectorized(d_Q, d_K, d_V, d_O, config.B, config.N, config.d); },
            h_O_cpu.data(), d_O, output_size, config, flops
        ));
    } else {
        std::cout << "Skipping V4 (d not divisible by 4)\n";
    }

    // V5: FlashAttention-2
    std::cout << "Benchmarking V5 (FA-2 Style)...\n";
    results.push_back(run_benchmark(
        "V5 FA-2 Style",
        [&]() { flash_attention_v5_fa2(d_Q, d_K, d_V, d_O, config.B, config.N, config.d); },
        h_O_cpu.data(), d_O, output_size, config, flops
    ));

    // Print results
    print_results(results);

    // Speedup summary
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "Speedup Summary (relative to V1):\n";
    std::cout << std::string(50, '=') << "\n";

    float v1_time = results[0].time_ms;
    for (const auto& r : results) {
        float speedup = v1_time / r.time_ms;
        std::cout << std::left << std::setw(20) << r.name
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << speedup << "x\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    std::cout << "\nBenchmark complete!\n";
    return 0;
}

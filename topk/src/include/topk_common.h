// Top-K Common Header
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <vector>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Top-K element structure
struct TopKNode {
    float value;
    int index;
};

// Benchmark result structure
struct BenchmarkResult {
    const char* name;
    double time_ms;
    double bandwidth_gbps;
    float max_error;
    bool passed;
    double speedup_vs_baseline;
};

// Initialize array with random values
inline void init_random(float* data, int size) {
    srand(42);
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// Initialize array with positive random values
inline void init_random_positive(float* data, int size) {
    srand(42);
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

// Get current time in milliseconds
inline double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Compare function for descending order
inline bool compare_topk_desc(const TopKNode& a, const TopKNode& b) {
    return a.value > b.value;
}

// CPU reference implementation
inline void topk_cpu_reference(const float* input, TopKNode* output, int N, int K) {
    std::vector<TopKNode> nodes(N);
    for (int i = 0; i < N; i++) {
        nodes[i].value = input[i];
        nodes[i].index = i;
    }
    std::partial_sort(nodes.begin(), nodes.begin() + K, nodes.end(), compare_topk_desc);
    for (int i = 0; i < K; i++) {
        output[i] = nodes[i];
    }
}

// Verify results
inline bool verify_topk_results(const TopKNode* gpu_results, const TopKNode* cpu_results, int K, float tolerance = 1e-4f) {
    for (int i = 0; i < K; i++) {
        if (fabsf(gpu_results[i].value - cpu_results[i].value) > tolerance) {
            return false;
        }
    }
    return true;
}

// Max error calculation
inline float max_error_topk(const TopKNode* a, const TopKNode* b, int K) {
    float max_err = 0.0f;
    for (int i = 0; i < K; i++) {
        max_err = fmaxf(max_err, fabsf(a[i].value - b[i].value));
    }
    return max_err;
}

// Host function declarations
void topk_v0_thrust(const float* d_input, int N, int K, TopKNode* d_output);
void topk_v1_thread(const float* d_input, int N, int K, TopKNode* d_output, int num_blocks, int num_threads);
void topk_v2_shared_memory(const float* d_input, int N, int K, TopKNode* d_output, int num_blocks, int num_threads);
void topk_v3_warp_shuffle(const float* d_input, int N, int K, TopKNode* d_output, int num_blocks, int num_threads);
void topk_v4_radix_select(const float* d_input, int N, int K, TopKNode* d_output, int num_blocks, int num_threads);
void topk_v5_blackwell(const float* d_input, int N, int K, TopKNode* d_output, int num_blocks, int num_threads);

// Benchmark function declarations
void run_benchmark_v0(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result);
void run_benchmark_v1(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result);
void run_benchmark_v2(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result);
void run_benchmark_v3(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result);
void run_benchmark_v4(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result);
void run_benchmark_v5(const float* d_input, int N, int K, TopKNode* h_output, const TopKNode* h_ref,
                      int warmup_iters, int benchmark_iters, size_t data_size_bytes, BenchmarkResult* result);

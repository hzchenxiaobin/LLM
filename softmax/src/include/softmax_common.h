// Common utilities for Softmax benchmarks
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Warp-level reduction helpers (used by V3, V4, V5)
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CPU reference implementation
inline void softmax_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        float row_max = -INFINITY;
        for (int i = 0; i < N; i++) {
            row_max = fmaxf(row_max, input[row * N + i]);
        }

        float row_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            row_sum += expf(input[row * N + i] - row_max);
        }

        for (int i = 0; i < N; i++) {
            output[row * N + i] = expf(input[row * N + i] - row_max) / row_sum;
        }
    }
}

// Utility functions
inline void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

inline float max_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
    }
    return max_err;
}

inline double get_time_ms() {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

// Benchmark result structure
struct BenchmarkResult {
    const char* name;
    double time_ms;
    double bandwidth_gbps;
    float max_error;
    bool passed;
};

// Softmax function declarations (implemented in respective .cu files)
void softmax_v1(const float* d_input, float* d_output, float* d_max, float* d_sum, int M, int N);
void softmax_v2(const float* d_input, float* d_output, int M, int N);
void softmax_v3(const float* d_input, float* d_output, int M, int N);
void softmax_v4(const float* d_input, float* d_output, int M, int N);
void softmax_v5(const float* d_input, float* d_output, int M, int N);
void softmax_v5_vec(const float* d_input, float* d_output, int M, int N);

// Benchmark function declarations (implemented in softmax_benchmark.cu)
void run_benchmark_v1(const float* d_input, float* d_output, float* d_max, float* d_sum,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result);

void run_benchmark_v2(const float* d_input, float* d_output,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result);

void run_benchmark_v3(const float* d_input, float* d_output,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result);

void run_benchmark_v4(const float* d_input, float* d_output,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result);

void run_benchmark_v5(const float* d_input, float* d_output,
                        float* h_output, const float* h_ref,
                        int M, int N, int warmup_iters, int benchmark_iters,
                        size_t data_size_bytes, BenchmarkResult* result);

void run_benchmark_v5_vec(const float* d_input, float* d_output,
                            float* h_output, const float* h_ref,
                            int M, int N, int warmup_iters, int benchmark_iters,
                            size_t data_size_bytes, BenchmarkResult* result);

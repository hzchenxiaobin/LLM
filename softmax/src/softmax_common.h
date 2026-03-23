// Common utilities for Softmax benchmarks
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

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
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Benchmark result structure
struct BenchmarkResult {
    const char* name;
    double time_ms;
    double bandwidth_gbps;
    float max_error;
    bool passed;
};

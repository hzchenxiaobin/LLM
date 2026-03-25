#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// =============================================================================
// CUDA Error Checking Macros
// =============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    status);                                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// =============================================================================
// Constants for FlashAttention
// =============================================================================

// FlashAttention typically uses 64 or 128 for Br (rows per block)
// and 64 for Bc (cols per block, i.e., sequence length per tile)
#ifndef Br
#define Br 64   // Block size for query (rows)
#endif

#ifndef Bc
#define Bc 64   // Block size for key/value (cols)
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64    // Head dimension
#endif

// Warp size
#define WARP_SIZE 32

// Maximum sequence length supported
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 32768
#endif

// =============================================================================
// Utility Functions
// =============================================================================

inline float rand_float() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
}

// Initialize matrix with random values
inline void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = rand_float() * 0.01f;  // Small values for numerical stability
    }
}

// Initialize to identity-like pattern for debugging
inline void init_matrix_identity(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// Compare two matrices, return max error
inline float compare_matrices(const float *a, const float *b, int n) {
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = fabsf(a[i] - b[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

// Print matrix for debugging (only first few elements)
inline void print_matrix(const char *name, const float *mat, int rows, int cols, int limit = 8) {
    printf("\n%s (%dx%d, showing first %dx%d):\n", name, rows, cols,
           (rows < limit ? rows : limit), (cols < limit ? cols : limit));
    for (int i = 0; i < (rows < limit ? rows : limit); i++) {
        for (int j = 0; j < (cols < limit ? cols : limit); j++) {
            printf("%8.4f ", mat[i * cols + j]);
        }
        printf("...\n");
    }
    printf("\n");
}

// =============================================================================
// Timer Utilities
// =============================================================================

class GpuTimer {
public:
    cudaEvent_t start_event, stop_event;

    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, 0);
    }

    void stop() {
        cudaEventRecord(stop_event, 0);
    }

    float elapsed_msecs() {
        float elapsed;
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed, start_event, stop_event);
        return elapsed;
    }

    float elapsed_secs() {
        return elapsed_msecs() / 1000.0f;
    }
};

// =============================================================================
// FLOPs Calculation
// =============================================================================

// Standard Attention FLOPs
inline double standard_attention_flops(int B, int N, int d) {
    // Q @ K^T: B * N * N * d
    // Softmax: approximately B * N * N (neglect for large N)
    // Attn @ V: B * N * d * N
    return 2.0 * B * N * N * d;
}

// FlashAttention has same theoretical FLOPs but better memory efficiency
inline double flash_attention_flops(int B, int N, int d) {
    return standard_attention_flops(B, N, d);
}

// Calculate TFLOPs/s
inline double calculate_tflops(double flops, float time_ms) {
    return flops / (time_ms * 1e-3) / 1e12;
}

// Memory bandwidth calculation (GB/s)
// FlashAttention: O(N) memory complexity vs O(N^2) for standard
inline double calculate_memory_bandwidth(int B, int N, int d, float time_ms) {
    // Bytes transferred: Q + K + V + O (ignoring tiling overhead)
    // Q: B * N * d, K: B * N * d, V: B * N * d, O: B * N * d
    double bytes = 4.0 * B * N * d * sizeof(float);
    return bytes / (time_ms * 1e-3) / 1e9;  // GB/s
}

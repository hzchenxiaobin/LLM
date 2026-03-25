/*
 * Reference Implementations for FlashAttention Benchmarking
 * =========================================================
 *
 * This file contains:
 * 1. CPU reference implementation (standard attention)
 * 2. cuBLAS reference implementation (for comparison)
 *
 * These are used to verify correctness and measure speedup.
 */

#include "kernels.h"
#include <cmath>
#include <cstring>
#include <vector>

// =============================================================================
// CPU Reference Implementation (Standard Attention)
// =============================================================================

void standard_attention_cpu(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    // Allocate temporary buffers
    float *S = new float[N * N];      // Attention scores
    float *softmax_S = new float[N * N];  // After softmax

    for (int b = 0; b < B; b++) {
        const float *Q_b = Q + b * N * d;
        const float *K_b = K + b * N * d;
        const float *V_b = V + b * N * d;
        float *O_b = O + b * N * d;

        // Step 1: Compute S = Q @ K^T
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < d; k++) {
                    sum += Q_b[i * d + k] * K_b[j * d + k];
                }
                S[i * N + j] = sum * scale;
            }
        }

        // Step 2: Softmax on each row
        for (int i = 0; i < N; i++) {
            // Find max for numerical stability
            float max_val = S[i * N];
            for (int j = 1; j < N; j++) {
                max_val = fmaxf(max_val, S[i * N + j]);
            }

            // Compute exp and sum
            float exp_sum = 0.0f;
            for (int j = 0; j < N; j++) {
                float exp_val = expf(S[i * N + j] - max_val);
                softmax_S[i * N + j] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            for (int j = 0; j < N; j++) {
                softmax_S[i * N + j] /= exp_sum;
            }
        }

        // Step 3: O = Softmax(S) @ V
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < d; k++) {
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    sum += softmax_S[i * N + j] * V_b[j * d + k];
                }
                O_b[i * d + k] = sum;
            }
        }
    }

    delete[] S;
    delete[] softmax_S;
}

// =============================================================================
// cuBLAS Reference Implementation
// =============================================================================

void standard_attention_cublas(
    cublasHandle_t handle,
    const float *Q, const float *K, const float *V,
    float *O,
    float *workspace,
    int B, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    float alpha = scale;
    float beta = 0.0f;

    // Workspace layout: [S (N*N)][softmax_S (N*N)]
    float *S = workspace;
    float *softmax_S = workspace + N * N;

    for (int b = 0; b < B; b++) {
        const float *Q_b = Q + b * N * d;
        const float *K_b = K + b * N * d;
        const float *V_b = V + b * N * d;
        float *O_b = O + b * N * d;

        // Step 1: S = Q @ K^T
        // Q: N x d, K: N x d, S: N x N
        // S = 1.0 * Q @ K^T + 0.0 * S
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T,    // K^T
            CUBLAS_OP_N,    // Q
            N, N, d,
            &alpha,
            K_b, d,
            Q_b, d,
            &beta,
            S, N
        ));

        // Step 2: Softmax (manual, cuBLAS doesn't have softmax)
        // For simplicity, we do this on CPU in the benchmark
        // In practice, you'd use a custom softmax kernel

        // Copy to host for softmax (not efficient, but for reference)
        std::vector<float> h_S(N * N);
        std::vector<float> h_softmax(N * N);
        CUDA_CHECK(cudaMemcpy(h_S.data(), S, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; i++) {
            float max_val = h_S[i * N];
            for (int j = 1; j < N; j++) {
                max_val = fmaxf(max_val, h_S[i * N + j]);
            }

            float exp_sum = 0.0f;
            for (int j = 0; j < N; j++) {
                float exp_val = expf(h_S[i * N + j] - max_val);
                h_softmax[i * N + j] = exp_val;
                exp_sum += exp_val;
            }

            for (int j = 0; j < N; j++) {
                h_softmax[i * N + j] /= exp_sum;
            }
        }

        CUDA_CHECK(cudaMemcpy(softmax_S, h_softmax.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

        // Step 3: O = Softmax(S) @ V
        // Softmax_S: N x N, V: N x d, O: N x d
        float one = 1.0f;
        float zero = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,    // V
            CUBLAS_OP_N,    // Softmax_S
            d, N, N,
            &one,
            V_b, d,
            softmax_S, N,
            &zero,
            O_b, d
        ));
    }
}

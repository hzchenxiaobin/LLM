#pragma once

#include "../common.h"

// =============================================================================
// FlashAttention Kernel Versions
// =============================================================================

// -----------------------------------------------------------------------------
// V1: Naive FlashAttention (Online Softmax, No Tiling)
// -----------------------------------------------------------------------------
// Basic implementation demonstrating the core online softmax algorithm.
// Each thread block handles one query block, iterates over entire KV sequence.
// - Uses global memory for all accesses (very slow)
// - Demonstrates online softmax technique
// - Good for understanding the algorithm, bad for performance
// -----------------------------------------------------------------------------
__global__ void flash_attention_v1_naive_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale
);

// -----------------------------------------------------------------------------
// V2: Shared Memory Tiling
// -----------------------------------------------------------------------------
// Optimizes memory access by loading K and V tiles into shared memory.
// - K and V tiles cached in shared memory
// - Still uses global memory for Q (loaded once per thread)
// - Basic online softmax with tiling
// -----------------------------------------------------------------------------
__global__ void flash_attention_v2_shared_kv_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale
);

// -----------------------------------------------------------------------------
// V3: Q Tiling + Double Buffering
// -----------------------------------------------------------------------------
// Adds Q tiling and double buffering for better memory throughput.
// - Both Q and KV are tiled
// - Double buffering for KV tiles (compute while loading next tile)
// - Still 32-bit float loads
// -----------------------------------------------------------------------------
__global__ void flash_attention_v3_q_tiling_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale
);

// -----------------------------------------------------------------------------
// V4: Vectorized Loads (float4)
// -----------------------------------------------------------------------------
// Uses vectorized memory instructions for better bandwidth.
// - float4 loads for Q, K, V when dimensions permit
// - Bank conflict aware shared memory layout
// - Warp-level primitives for reduction
// -----------------------------------------------------------------------------
__global__ void flash_attention_v4_vectorized_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale
);

// -----------------------------------------------------------------------------
// V5: FlashAttention-2 Optimized (Split KV, Warp Specialization)
// -----------------------------------------------------------------------------
// Implements key FlashAttention-2 optimizations:
// - Split KV across warps (different from FA-1 which split Q)
// - Better parallelism for long sequences
// - Warp-level online softmax
// - Optimized for RTX 5090 (Blackwell)
// -----------------------------------------------------------------------------
__global__ void flash_attention_v5_fa2_kernel(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float scale
);

// -----------------------------------------------------------------------------
// V6: Tensor Core Version (WMMA/MMA)
// -----------------------------------------------------------------------------
// Uses Tensor Cores for matrix multiplication.
// - WMMA for Q@K^T and Attn@V
// - FP16/BF16 computation
// - Requires compatible dimensions
// -----------------------------------------------------------------------------
__global__ void flash_attention_v6_tensor_core_kernel(
    const __half *Q, const __half *K, const __half *V,
    __half *O,
    int N, int d,
    float scale
);

// =============================================================================
// Host Wrapper Functions
// =============================================================================

void flash_attention_v1_naive(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d
);

void flash_attention_v2_shared_kv(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d
);

void flash_attention_v3_q_tiling(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d
);

void flash_attention_v4_vectorized(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d
);

void flash_attention_v5_fa2(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d
);

// FP16 version for Tensor Cores
void flash_attention_v6_tensor_core(
    const __half *Q, const __half *K, const __half *V,
    __half *O,
    int B, int N, int d
);

// =============================================================================
// Reference Implementation (CPU)
// =============================================================================

void standard_attention_cpu(
    const float *Q, const float *K, const float *V,
    float *O,
    int B, int N, int d
);

// cuBLAS reference for comparison
void standard_attention_cublas(
    cublasHandle_t handle,
    const float *Q, const float *K, const float *V,
    float *O,
    float *workspace,
    int B, int N, int d
);

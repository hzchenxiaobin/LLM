/**
 * CUDA Reduction 算子 - Kernel 声明头文件
 */

#ifndef REDUCTION_KERNELS_H
#define REDUCTION_KERNELS_H

#include "common.h"
#include <cub/cub.cuh>

// ============================================
// 版本 1: 朴素版本 (Interleaved Addressing)
// 问题: Warp Divergence
// ============================================
__global__ void reduce_v1(float *g_idata, float *g_odata, unsigned int n);

// ============================================
// 版本 2: 解决分支发散 (Strided Index)
// 问题: Bank Conflict
// ============================================
__global__ void reduce_v2(float *g_idata, float *g_odata, unsigned int n);

// ============================================
// 版本 3: 解决 Bank Conflict (Sequential Addressing)
// ============================================
__global__ void reduce_v3(float *g_idata, float *g_odata, unsigned int n);

// ============================================
// 版本 4: 提高指令吞吐与隐藏延迟 (First Add During Load)
// ============================================
__global__ void reduce_v4(float *g_idata, float *g_odata, unsigned int n);

// ============================================
// 版本 5: Warp Shuffle (终结 Shared Memory)
// ============================================
__inline__ __device__ float warpReduceSum(float val);
__global__ void reduce_v5(float *g_idata, float *g_odata, unsigned int n);

// ============================================
// 版本 6: 向量化访存 (Vectorized Memory Access)
// ============================================
__global__ void reduce_v6(float *g_idata, float *g_odata, unsigned int n);

// ============================================
// CUB 库基准版本
// ============================================
void reduce_cub(float *d_in, float *d_out, unsigned int n,
                void *&d_temp_storage, size_t &temp_storage_bytes,
                cudaStream_t stream);

// ============================================
// Kernel 运行函数
// ============================================
extern "C" {
float run_kernel(int version, float *d_in, float *d_out, unsigned int n,
                 int num_blocks, int num_threads, int shared_mem_bytes,
                 cudaStream_t stream, int warmup_iters, int test_iters);

float run_cub(float *d_in, float *d_out, unsigned int n,
              void *&d_temp_storage, size_t &temp_storage_bytes,
              cudaStream_t stream, int warmup_iters, int test_iters);
}

#endif // REDUCTION_KERNELS_H

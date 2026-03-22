/**
 * CUDA Reduction 算子 - 版本 1
 * 朴素版本 (Interleaved Addressing)
 * 问题: Warp Divergence
 */

#include "reduction_kernels.h"

// ============================================
// 版本 1: 朴素版本 (Interleaved Addressing)
// 问题: Warp Divergence
// ============================================
__global__ void reduce_v1(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

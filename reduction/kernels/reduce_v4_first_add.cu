/**
 * CUDA Reduction 算子 - 版本 4
 * 提高指令吞吐与隐藏延迟 (First Add During Load)
 */

#include "reduction_kernels.h"

// ============================================
// 版本 4: 提高指令吞吐与隐藏延迟 (First Add During Load)
// ============================================
__global__ void reduce_v4(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

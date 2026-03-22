/**
 * CUDA Reduction 算子 - 版本 3
 * 解决 Bank Conflict (Sequential Addressing)
 */

#include "reduction_kernels.h"

// ============================================
// 版本 3: 解决 Bank Conflict (Sequential Addressing)
// ============================================
__global__ void reduce_v3(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

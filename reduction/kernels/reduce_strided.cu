/**
 * CUDA Reduction 算子 - 版本 2
 * 解决分支发散 (Strided Index)
 * 问题: Bank Conflict
 */

#include "reduction_kernels.h"

// ============================================
// 版本 2: 解决分支发散 (Strided Index)
// 问题: Bank Conflict
// ============================================
__global__ void reduce_v2(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

/**
 * CUDA Reduction 算子 - 版本 5
 * Warp Shuffle (终结 Shared Memory)
 */

#include "reduction_kernels.h"

// ============================================
// 版本 5: Warp Shuffle (终结 Shared Memory)
// ============================================
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v5(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockDim.x >= 64) sum = sdata[tid] + sdata[tid + 32];
        else sum = sdata[tid];

        sum = warpReduceSum(sum);

        if (tid == 0) atomicAdd(g_odata, sum);
    }
}

/**
 * CUDA Reduction 算子 - 版本 6
 * 向量化访存 (Vectorized Memory Access)
 */

#include "reduction_kernels.h"

// ============================================
// 版本 6: 向量化访存 (Vectorized Memory Access)
// ============================================
__global__ void reduce_v6(float *g_idata, float *g_odata, unsigned int n) {
    float4 *g_idata_f4 = reinterpret_cast<float4*>(g_idata);

    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop
    while (i < n / 4) {
        float4 vec = g_idata_f4[i];
        sum += vec.x + vec.y + vec.z + vec.w;
        i += stride;
    }

    // 处理剩余元素
    i = i * 4;
    while (i < n) {
        sum += g_idata[i];
        i++;
    }

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

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (tid == 0) atomicAdd(g_odata, sum);
    }
}

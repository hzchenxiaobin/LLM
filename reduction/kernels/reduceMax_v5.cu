/**
 * CUDA ReduceMax 算子 - 基于 V5 Warp Shuffle 架构
 * 仅返回最大值（无索引）
 */

#include "../include/reduction_kernels.h"
#include <float.h>

// ============================================
// Warp级别找最大值
// ============================================
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// ============================================
// 版本 5: ReduceMax - 混合归约架构（仅最大值）
// 输入: g_idata - 输入数组
// 输出: g_maxval - 全局最大值
//       n - 数组长度
// ============================================
__global__ void reduceMax_v5(
    const float *g_idata, 
    float *g_maxval, 
    unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // Grid-stride: 每个线程处理2个元素（First Add During Load风格）
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 加载并比较，保留局部最大值
    float max_val = -FLT_MAX;
    
    if (i < n) {
        max_val = g_idata[i];
    }
    if (i + blockDim.x < n) {
        max_val = fmaxf(max_val, g_idata[i + blockDim.x]);
    }
    
    // 存入Shared Memory
    sdata[tid] = max_val;
    __syncthreads();
    
    // 阶段1: Shared Memory树形归约（减少到32个元素）
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // 阶段2: Warp Shuffle最终归约（最后32个元素）
    if (tid < 32) {
        // 从Shared Memory加载当前最大值
        if (blockDim.x >= 64) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
        }
        
        max_val = sdata[tid];
        
        // Warp Shuffle找全局最大值
        max_val = warpReduceMax(max_val);
        
        // 线程0写入全局内存
        if (tid == 0) {
            // 注意：此版本假设单block运行
            // 多block需要使用原子操作或其他归并策略
            *g_maxval = max_val;
        }
    }
}

// ============================================
// 多Block版本：支持原子操作合并结果
// ============================================
__global__ void reduceMax_v5_multiblock(
    const float *g_idata, 
    float *g_maxval, 
    unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // Grid-stride遍历（V6风格）
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    
    float max_val = -FLT_MAX;
    
    // 每个线程处理多个元素
    while (i < n) {
        max_val = fmaxf(max_val, g_idata[i]);
        if (i + blockDim.x < n) {
            max_val = fmaxf(max_val, g_idata[i + blockDim.x]);
        }
        i += gridSize;
    }
    
    // 存入Shared Memory
    sdata[tid] = max_val;
    __syncthreads();
    
    // Shared Memory归约到32元素
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Warp Shuffle最终归约
    if (tid < 32) {
        if (blockDim.x >= 64) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
        }
        
        max_val = sdata[tid];
        
        // Warp内归约
        max_val = warpReduceMax(max_val);
        
        // 线程0使用原子操作更新全局结果
        if (tid == 0) {
            // 原子比较并更新最大值
            // 注意：CUDA没有原生atomicMax for float，使用atomicCAS模拟
            int* maxval_as_int = (int*)g_maxval;
            int old = *maxval_as_int;
            float old_val = __int_as_float(old);
            
            while (old_val < max_val) {
                int assumed = old;
                old = atomicCAS(maxval_as_int, assumed, __float_as_int(max_val));
                old_val = __int_as_float(old);
            }
        }
    }
}

// ============================================
// 快速调用接口（单Block版本，适合小数据）
// ============================================
__global__ void reduceMax_v5_quick(
    const float *g_idata, 
    float *g_maxval, 
    unsigned int n
) {
    // 假设只启动1个block，256线程
    // 适合 n <= 512 的情况
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // 每个线程加载2个元素
    float max_val = -FLT_MAX;
    
    if (tid < n) {
        max_val = g_idata[tid];
    }
    if (tid + 256 < n) {
        max_val = fmaxf(max_val, g_idata[tid + 256]);
    }
    
    sdata[tid] = max_val;
    __syncthreads();
    
    // 完全展开的树形归约（适合256线程）
    #pragma unroll
    for (int s = 128; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Warp Shuffle最终归约
    if (tid < 32) {
        if (sdata[tid + 32] > sdata[tid]) {
            sdata[tid] = sdata[tid + 32];
        }
        
        max_val = sdata[tid];
        max_val = warpReduceMax(max_val);
        
        if (tid == 0) {
            *g_maxval = max_val;
        }
    }
}

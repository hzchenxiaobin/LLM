/**
 * CUDA ReduceMax 算子 - 基于 V5 Warp Shuffle 架构
 * 返回最大值及其索引
 */

#include "../include/reduction_kernels.h"
#include <float.h>

// ============================================
// Warp级别找最大值（带索引）
// ============================================
__inline__ __device__ float warpReduceMax(float val, int idx, int* out_idx) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    
    // 广播warp内的最大值和索引
    *out_idx = __shfl_sync(0xffffffff, idx, 0);
    return __shfl_sync(0xffffffff, val, 0);
}

// ============================================
// 简化版：只返回最大值（不带索引）
// ============================================
__inline__ __device__ float warpReduceMaxOnly(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// ============================================
// 版本 5: ReduceMax - 混合归约架构
// 输入: g_idata - 输入数组
// 输出: g_maxval - 全局最大值
//       g_maxidx - 全局最大值索引（可为NULL）
//       n - 数组长度
// ============================================
__global__ void reduceMax_v5(
    const float *g_idata, 
    float *g_maxval, 
    int *g_maxidx,  // 可为NULL，表示不需要索引
    unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // 动态共享内存布局：前半存值，后半存索引
    float* s_vals = sdata;
    int* s_idxs = (int*)(sdata + blockDim.x);
    
    // Grid-stride: 每个线程处理2个元素（First Add During Load风格）
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 加载并比较，保留局部最大值
    float max_val = -FLT_MAX;
    int max_idx = -1;
    
    if (i < n) {
        max_val = g_idata[i];
        max_idx = i;
    }
    if (i + blockDim.x < n) {
        float val2 = g_idata[i + blockDim.x];
        if (val2 > max_val) {
            max_val = val2;
            max_idx = i + blockDim.x;
        }
    }
    
    // 存入Shared Memory
    s_vals[tid] = max_val;
    if (g_maxidx != NULL) {
        s_idxs[tid] = max_idx;
    }
    __syncthreads();
    
    // 阶段1: Shared Memory树形归约（减少到32个元素）
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                if (g_maxidx != NULL) {
                    s_idxs[tid] = s_idxs[tid + s];
                }
            }
        }
        __syncthreads();
    }
    
    // 阶段2: Warp Shuffle最终归约（最后32个元素）
    if (tid < 32) {
        // 从Shared Memory加载当前最大值
        if (blockDim.x >= 64) {
            if (s_vals[tid + 32] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + 32];
                if (g_maxidx != NULL) {
                    s_idxs[tid] = s_idxs[tid + 32];
                }
            }
        }
        
        max_val = s_vals[tid];
        max_idx = (g_maxidx != NULL) ? s_idxs[tid] : tid;
        
        // Warp Shuffle找全局最大值
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        
        // 线程0写入全局内存
        if (tid == 0) {
            // 注意：此版本假设单block运行
            // 多block需要使用原子操作或其他归并策略
            *g_maxval = max_val;
            if (g_maxidx != NULL) {
                *g_maxidx = max_idx;
            }
        }
    }
}

// ============================================
// 多Block版本：支持原子操作合并结果
// ============================================
__global__ void reduceMax_v5_multiblock(
    const float *g_idata, 
    float *g_maxval, 
    int *g_maxidx,
    unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    float* s_vals = sdata;
    int* s_idxs = (int*)(sdata + blockDim.x);
    
    // Grid-stride遍历（V6风格）
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    
    float max_val = -FLT_MAX;
    int max_idx = -1;
    
    // 每个线程处理多个元素
    while (i < n) {
        if (g_idata[i] > max_val) {
            max_val = g_idata[i];
            max_idx = i;
        }
        if (i + blockDim.x < n && g_idata[i + blockDim.x] > max_val) {
            max_val = g_idata[i + blockDim.x];
            max_idx = i + blockDim.x;
        }
        i += gridSize;
    }
    
    // 存入Shared Memory
    s_vals[tid] = max_val;
    s_idxs[tid] = max_idx;
    __syncthreads();
    
    // Shared Memory归约到32元素
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s && s_vals[tid + s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + s];
            s_idxs[tid] = s_idxs[tid + s];
        }
        __syncthreads();
    }
    
    // Warp Shuffle最终归约
    if (tid < 32) {
        if (blockDim.x >= 64 && s_vals[tid + 32] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + 32];
            s_idxs[tid] = s_idxs[tid + 32];
        }
        
        max_val = s_vals[tid];
        max_idx = s_idxs[tid];
        
        // Warp内归约
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        
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
            
            // 如果值相等，取较小索引（可自定义策略）
            if (g_maxidx != NULL && __int_as_float(*maxval_as_int) == max_val) {
                atomicMin(g_maxidx, max_idx);
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
    int *g_maxidx,
    unsigned int n
) {
    // 假设只启动1个block，256线程
    // 适合 n <= 512 的情况
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    float* s_vals = sdata;
    int* s_idxs = (int*)(sdata + 256);
    
    // 每个线程加载2个元素
    float max_val = -FLT_MAX;
    int max_idx = -1;
    
    if (tid < n) {
        max_val = g_idata[tid];
        max_idx = tid;
    }
    if (tid + 256 < n) {
        float val2 = g_idata[tid + 256];
        if (val2 > max_val) {
            max_val = val2;
            max_idx = tid + 256;
        }
    }
    
    s_vals[tid] = max_val;
    s_idxs[tid] = max_idx;
    __syncthreads();
    
    // 完全展开的树形归约（适合256线程）
    #pragma unroll
    for (int s = 128; s > 32; s >>= 1) {
        if (tid < s && s_vals[tid + s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + s];
            s_idxs[tid] = s_idxs[tid + s];
        }
        __syncthreads();
    }
    
    // Warp Shuffle最终归约
    if (tid < 32) {
        if (s_vals[tid + 32] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + 32];
            s_idxs[tid] = s_idxs[tid + 32];
        }
        
        max_val = s_vals[tid];
        max_idx = s_idxs[tid];
        
        max_val = warpReduceMax(max_val, max_idx, &max_idx);
        
        if (tid == 0) {
            *g_maxval = max_val;
            if (g_maxidx != NULL) *g_maxidx = max_idx;
        }
    }
}

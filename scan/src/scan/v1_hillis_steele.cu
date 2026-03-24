#include "scan/scan.h"
#include <cuda_runtime.h>

namespace scan {

// V1: Hillis-Steele 朴素并行扫描 (Naive Parallel Scan)
// 使用双缓冲技术，在每个步骤中每个线程将当前元素与距离它 2^(d-1) 的元素相加
// 工作量: O(N log N)，不是最优的，但实现简单

template <typename T>
__global__ void hillis_steele_kernel(T* g_odata, const T* g_idata, int n) {
    // 分配两倍的 shared memory 用于双缓冲
    extern __shared__ T temp[];
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    // 将数据从 Global Memory 读入 Shared Memory
    // 注意：只加载有效数据，超出范围设为 0
    temp[pout * n + thid] = (thid < n) ? g_idata[thid] : 0;
    __syncthreads();

    // 迭代进行 scan，offset 每次翻倍 (1, 2, 4, 8...)
    for (int offset = 1; offset < n; offset *= 2) {
        // 交换双缓冲索引
        pout = 1 - pout;
        pin = 1 - pout;

        // 如果当前线程索引 >= offset，则进行累加
        if (thid >= offset) {
            temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
        } else {
            // 否则保持原值
            temp[pout * n + thid] = temp[pin * n + thid];
        }
        __syncthreads();
    }

    // 将结果写回 Global Memory
    if (thid < n) {
        g_odata[thid] = temp[pout * n + thid];
    }
}

// 包装函数：处理任意长度数组（限制在单个 block 内）
template <typename T>
__global__ void hillis_steele_exclusive_kernel(T* g_odata, const T* g_idata, int n) {
    extern __shared__ T temp[];
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    // 加载数据到共享内存（包含型扫描）
    temp[pout * n + thid] = (thid < n) ? g_idata[thid] : 0;
    __syncthreads();

    // Hillis-Steele 包含型扫描
    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (thid >= offset) {
            temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
        } else {
            temp[pout * n + thid] = temp[pin * n + thid];
        }
        __syncthreads();
    }

    // 转换为排他型扫描：右移，首元素置 0
    if (thid < n) {
        T inclusive_result = temp[pout * n + thid];
        T exclusive_result = (thid == 0) ? 0 : temp[pout * n + thid - 1];
        g_odata[thid] = exclusive_result;
    }
}

void hillis_steele_scan(const float* d_input, float* d_output, int n) {
    // 限制 block 大小不超过 1024 (CUDA 最大值)
    // 且 n 必须是 2 的幂次
    int threads = min(n, 1024);

    // 分配共享内存：需要 2 * n * sizeof(float) 用于双缓冲
    int smem_size = 2 * n * sizeof(float);

    hillis_steele_exclusive_kernel<<<1, threads, smem_size>>>(d_output, d_input, n);

    cudaDeviceSynchronize();
}

} // namespace scan

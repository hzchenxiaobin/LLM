#include "scan/scan.h"
#include <cuda_runtime.h>

namespace scan {

// V3: 消除 Bank Conflicts 的扫描
// CUDA 的 Shared Memory 被分为 32 个 Bank
// 当访问步长为 2^n 时（如 2, 4, 8...），会导致 Bank 冲突
// 解决方案：每 32 个元素插入一个 padding

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

// 计算无冲突的索引偏移量
// 每遇到 32 个元素，就跳过一个位置
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

template <typename T>
__global__ void bank_free_scan_kernel(T* g_odata, const T* g_idata, int n) {
    extern __shared__ T temp[];
    int thid = threadIdx.x;
    int offset = 1;

    // 计算原始索引（每个线程处理 2 个元素）
    int ai = thid;
    int bi = thid + (n / 2);

    // 计算带 padding 的无冲突索引
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // 加载数据到带 padding 的共享内存位置
    temp[ai + bankOffsetA] = (ai < n) ? g_idata[ai] : 0;
    temp[bi + bankOffsetB] = (bi < n) ? g_idata[bi] : 0;

    // ========== 1. Up-Sweep 阶段 (归约) ==========
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai_local = offset * (2 * thid + 1) - 1;
            int bi_local = offset * (2 * thid + 2) - 1;

            // 应用无冲突偏移
            ai_local += CONFLICT_FREE_OFFSET(ai_local);
            bi_local += CONFLICT_FREE_OFFSET(bi_local);

            temp[bi_local] += temp[ai_local];
        }
        offset *= 2;
    }

    // 根节点设为 0
    if (thid == 0) {
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    // ========== 2. Down-Sweep 阶段 (分发) ==========
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai_local = offset * (2 * thid + 1) - 1;
            int bi_local = offset * (2 * thid + 2) - 1;

            // 应用无冲突偏移
            ai_local += CONFLICT_FREE_OFFSET(ai_local);
            bi_local += CONFLICT_FREE_OFFSET(bi_local);

            // 交换并累加
            T t = temp[ai_local];
            temp[ai_local] = temp[bi_local];
            temp[bi_local] += t;
        }
    }
    __syncthreads();

    // 写回结果（注意移除 padding）
    if (ai < n) g_odata[ai] = temp[ai + bankOffsetA];
    if (bi < n) g_odata[bi] = temp[bi + bankOffsetB];
}

void bank_free_scan(const float* d_input, float* d_output, int n) {
    int threads = n / 2;
    if (threads > 1024) {
        threads = 1024;
    }

    // 计算带 padding 的共享内存大小
    // 需要额外空间存储 padding 元素
    int padded_n = n + CONFLICT_FREE_OFFSET(n - 1) + 1;
    int smem_size = padded_n * sizeof(float);

    bank_free_scan_kernel<<<1, threads, smem_size>>>(d_output, d_input, n);

    cudaDeviceSynchronize();
}

#undef NUM_BANKS
#undef LOG_NUM_BANKS
#undef CONFLICT_FREE_OFFSET

} // namespace scan

#include "scan/scan.h"
#include <cuda_runtime.h>

namespace scan {

// V2: Blelloch 工作高效扫描 (Work-Efficient Scan)
// 分为两个阶段：
// 1. Up-Sweep (归约阶段)：自底向上构建部分和树
// 2. Down-Sweep (分发阶段)：自顶向下计算前缀和
// 工作量: O(N)，比 Hillis-Steele 更高效

template <typename T>
__global__ void blelloch_scan_kernel(T* g_odata, const T* g_idata, int n) {
    extern __shared__ T temp[];
    int thid = threadIdx.x;
    int offset = 1;

    // 每个线程加载两个元素到 Shared Memory
    // 注意：这里假设 n 是偶数，且 blockDim.x = n/2
    int ai = 2 * thid;
    int bi = 2 * thid + 1;

    temp[ai] = (ai < n) ? g_idata[ai] : 0;
    temp[bi] = (bi < n) ? g_idata[bi] : 0;

    // ========== 1. Up-Sweep 阶段 (归约) ==========
    // 从叶子节点向上归约，构建部分和树
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai_local = offset * (2 * thid + 1) - 1;
            int bi_local = offset * (2 * thid + 2) - 1;
            temp[bi_local] += temp[ai_local];
        }
        offset *= 2;
    }

    // 将根节点设为 0 (这是排他型扫描的关键)
    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // ========== 2. Down-Sweep 阶段 (分发) ==========
    // 自顶向下计算前缀和
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai_local = offset * (2 * thid + 1) - 1;
            int bi_local = offset * (2 * thid + 2) - 1;

            // 交换并累加：
            // - ai 位置获得原来的 bi 值（当前前缀和）
            // - bi 位置累加上 ai 的值（传播前缀和）
            T t = temp[ai_local];
            temp[ai_local] = temp[bi_local];
            temp[bi_local] += t;
        }
    }
    __syncthreads();

    // 将结果写回 Global Memory
    if (ai < n) g_odata[ai] = temp[ai];
    if (bi < n) g_odata[bi] = temp[bi];
}

void blelloch_scan(const float* d_input, float* d_output, int n) {
    // Blelloch 算法需要 n 是 2 的幂次
    // 每个线程处理 2 个元素，所以线程数是 n/2
    int threads = n / 2;
    if (threads > 1024) {
        // 如果超过最大线程数，需要分块处理（这里简化处理）
        threads = 1024;
    }

    // 共享内存大小：n 个元素
    int smem_size = n * sizeof(float);

    blelloch_scan_kernel<<<1, threads, smem_size>>>(d_output, d_input, n);

    cudaDeviceSynchronize();
}

} // namespace scan

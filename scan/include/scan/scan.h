#pragma once

#include <cuda_runtime.h>

namespace scan {

// 各版本 scan 函数声明

// V1: Hillis-Steele 朴素并行扫描 (Naive Parallel Scan)
// 双缓冲方式，O(N log N) 工作量
void hillis_steele_scan(const float* d_input, float* d_output, int n);

// V2: Blelloch 工作高效扫描 (Work-Efficient Scan)
// Up-Sweep + Down-Sweep 两阶段，O(N) 工作量
void blelloch_scan(const float* d_input, float* d_output, int n);

// V3: 消除 Bank Conflicts 的扫描
// 使用 Padding 技巧避免 Shared Memory Bank Conflicts
void bank_free_scan(const float* d_input, float* d_output, int n);

// V4: Warp-Level Primitives 扫描
// 利用 __shfl_up_sync 在寄存器级别完成扫描
void warp_scan(const float* d_input, float* d_output, int n);

// 辅助函数：计算下一个 2 的幂次
inline int nextPowerOfTwo(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

} // namespace scan

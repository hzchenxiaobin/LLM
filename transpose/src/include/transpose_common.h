#ifndef TRANSPOSE_COMMON_H
#define TRANSPOSE_COMMON_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

// 统一使用 float4 类型定义
#ifndef float4_ALIGNED
#define float4_ALIGNED float4
#endif

// ============================================================================
// 各版本转置函数声明
// ============================================================================

// v1: 朴素实现 - 非合并写入
void transpose_naive(const float *d_A, float *d_B, int N, cudaStream_t stream = 0);

// v2: 共享内存优化 - 存在 bank conflict
void transpose_shared_memory(const float *d_A, float *d_B, int N, cudaStream_t stream = 0);

// v3: Padding 消除 Bank Conflict
void transpose_shared_pad(const float *d_A, float *d_B, int N, cudaStream_t stream = 0);

// v4: ILP 与 Block 形状优化
void transpose_optimized(const float *d_A, float *d_B, int N, cudaStream_t stream = 0);

// v5: float4 向量化访问
void transpose_vectorized(const float *d_A, float *d_B, int N, cudaStream_t stream = 0);

// ============================================================================
// 辅助函数
// ============================================================================

// 初始化矩阵数据
void init_matrix(float *h_A, int N, uint32_t seed = 42);

// 验证转置结果
bool verify_transpose(const float *h_A, const float *h_B, int N);

// 计算有效带宽 (GB/s)
// 转置操作：读取 N*N 个 float，写入 N*N 个 float
// 总数据量 = 2 * N * N * sizeof(float) 字节
inline float compute_effective_bandwidth_ms(float ms, int N) {
    float bytes = 2.0f * N * N * sizeof(float);
    return (bytes / (ms / 1000.0f)) / 1e9;  // GB/s
}

inline float compute_effective_bandwidth_us(float us, int N) {
    float bytes = 2.0f * N * N * sizeof(float);
    return (bytes / (us / 1000000.0f)) / 1e9;  // GB/s
}

// 获取 GPU 理论带宽
float get_theoretical_bandwidth();

// 打印 GPU 信息
void print_gpu_info();

#endif  // TRANSPOSE_COMMON_H

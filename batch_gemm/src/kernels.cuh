#pragma once

// 定义统一的 Kernel 启动函数指针类型
// 所有的 Kernel 包装函数都必须符合这个签名
typedef void (*BmmKernelFunc)(const float* A, const float* B, float* C, int B_size, int M, int N, int K);

// ==============================================================================
// 所有的自定义 Kernel 版本声明
// ==============================================================================

// V0: 朴素实现
void run_bmm_naive(const float* A, const float* B, float* C, int B_size, int M, int N, int K);

// V1: 共享内存分块优化 (Shared Memory Tiling)
void run_bmm_v1_shared_memory(const float* A, const float* B, float* C, int B_size, int M, int N, int K);
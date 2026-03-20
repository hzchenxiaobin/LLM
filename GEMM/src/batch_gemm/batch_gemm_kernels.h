#pragma once
#include "../common.h"

// ==========================================
// Batch GEMM 算子包装函数声明
// ==========================================

// 1. 朴素 Batch GEMM (通用 stride 版本)
void run_sgemm_batch_naive(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    int batch_count, long long int strideA, long long int strideB, long long int strideC
);

// 2. 朴素 Batch GEMM (连续存储版本)
void run_sgemm_batch_contiguous(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    int batch_count
);

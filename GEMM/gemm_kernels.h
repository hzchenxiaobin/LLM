#pragma once
#include <cublas_v2.h>
#include "common.h"

// 声明全局的 cuBLAS 句柄，供主函数和算子实现使用
extern cublasHandle_t cublas_handle;

// ==========================================
// GEMM 算子包装函数声明
// ==========================================

// 1. 朴素 GEMM
void run_sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

// 2. 共享内存 GEMM
void run_sgemm_shared(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

// 3. cuBLAS GEMM (基准)
void run_cublas(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
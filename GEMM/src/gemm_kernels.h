#pragma once

// 统一的头文件入口，包含所有 GEMM 和 Batch GEMM 算子声明

// 单矩阵 GEMM 算子
#include "gemm/gemm_kernels.h"

// Batch GEMM 算子
#include "batch_gemm/batch_gemm_kernels.h"

/**
 * CUDA Reduction 算子 - 公共头文件
 */

#ifndef REDUCTION_COMMON_H
#define REDUCTION_COMMON_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// 错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// CPU 参考实现
float reduce_cpu(const float *data, unsigned int n);

// 初始化函数
extern "C" {
void init_data(float *data, unsigned int n, unsigned int seed);
float get_bandwidth_gb_s(unsigned int n, float time_ms);
}

// 获取 GPU 信息
void get_gpu_info(char *name, int *major, int *minor, float *peak_bandwidth_gb_s);

// 辅助函数：计算配置参数
void get_kernel_config(int version, unsigned int n, int *num_blocks, int *num_threads, int *shared_mem_bytes);

#endif // REDUCTION_COMMON_H

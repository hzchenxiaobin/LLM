/**
 * CUDA Reduction 算子 - 公共函数实现
 */

#include "reduction_kernels.h"

// ============================================
// CPU 参考实现
// ============================================
float reduce_cpu(const float *data, unsigned int n) {
    double sum = 0.0;
    for (unsigned int i = 0; i < n; i++) {
        sum += data[i];
    }
    return (float)sum;
}

// ============================================
// 初始化函数
// ============================================
extern "C" {

void init_data(float *data, unsigned int n, unsigned int seed) {
    srand(seed);
    for (unsigned int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

float get_bandwidth_gb_s(unsigned int n, float time_ms) {
    double bytes = n * sizeof(float);
    double seconds = time_ms / 1000.0;
    return (float)(bytes / seconds / 1e9);
}

} // extern "C"

// ============================================
// 获取 GPU 信息
// ============================================
void get_gpu_info(char *name, int *major, int *minor, float *peak_bandwidth_gb_s) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    strcpy(name, prop.name);
    *major = prop.major;
    *minor = prop.minor;

    // 计算峰值带宽 (理论值)
    // 带宽 = 显存频率 * 位宽 / 8
    float mem_clock_ghz = prop.memoryClockRate / 1e6f;
    int bus_width_bits = prop.memoryBusWidth;
    *peak_bandwidth_gb_s = 2.0f * mem_clock_ghz * (bus_width_bits / 8.0f);
}

// ============================================
// 辅助函数：计算配置参数
// ============================================
void get_kernel_config(int version, unsigned int n, int *num_blocks, int *num_threads, int *shared_mem_bytes) {
    switch (version) {
        case 1:
        case 2:
        case 3:
            *num_threads = 256;
            *num_blocks = (n + *num_threads - 1) / *num_threads;
            *shared_mem_bytes = *num_threads * sizeof(float);
            break;
        case 4:
        case 5:
            *num_threads = 256;
            *num_blocks = (n + (*num_threads * 2) - 1) / (*num_threads * 2);
            *shared_mem_bytes = *num_threads * sizeof(float);
            break;
        case 6:
            *num_threads = 256;
            // 对于向量化版本，使用 grid-stride loop，blocks 数量可以更少
            *num_blocks = std::min(128, (int)((n / 4 + *num_threads - 1) / *num_threads));
            *shared_mem_bytes = *num_threads * sizeof(float);
            break;
    }
}

#include "transpose_common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 简单的线性同余随机数生成器 (避免与 nvcc 头文件冲突)
static uint32_t lcg_rand(uint32_t *seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffff;
    return *seed;
}

// 初始化矩阵数据
void init_matrix(float *h_A, int M, int N, uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < M * N; i++) {
        // 生成 [0, 1) 范围的随机数
        h_A[i] = (float)lcg_rand(&s) / (float)0x80000000;
    }
}

// 验证转置结果
// h_A: M x N matrix (input)
// h_B: N x M matrix (transposed output)
bool verify_transpose(const float *h_A, const float *h_B, int M, int N) {
    const float epsilon = 1e-5f;

    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            float expected = h_A[y * N + x];
            float actual = h_B[x * M + y];
            if (fabsf(expected - actual) > epsilon) {
                printf("Verification FAILED at (y=%d, x=%d): expected %.6f, got %.6f\n",
                       y, x, expected, actual);
                return false;
            }
        }
    }
    return true;
}

// 获取 GPU 理论带宽
float get_theoretical_bandwidth() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 计算理论带宽 (GB/s)
    // 带宽 = 显存时钟频率(MHz) * 显存位宽(bit) / 8 / 1000
    float mem_clock_ghz = prop.memoryClockRate / 1e6f;  // 转换为 GHz
    int mem_bus_width_bits = prop.memoryBusWidth;
    float bandwidth_gbps = mem_clock_ghz * 2.0f * (mem_bus_width_bits / 8.0f);

    return bandwidth_gbps;
}

// 打印 GPU 信息
void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    GPU Device Information                      ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Device Name:        %-40s ║\n", prop.name);
    printf("║ Compute Capability: %d.%d                                      ║\n",
           prop.major, prop.minor);
    printf("║ SMs:               %d                                          ║\n",
           prop.multiProcessorCount);
    printf("║ Max Threads/SM:    %d                                       ║\n",
           prop.maxThreadsPerMultiProcessor);
    printf("║ Memory Clock:      %.2f GHz                                    ║\n",
           prop.memoryClockRate / 1e6f);
    printf("║ Memory Bus Width:   %d bits                                    ║\n",
           prop.memoryBusWidth);
    printf("║ Theoretical BW:     %.2f GB/s                                  ║\n",
           get_theoretical_bandwidth());
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

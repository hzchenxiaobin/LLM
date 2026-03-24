#include "scan/scan.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

namespace scan {

// 性能计时辅助类
class GpuTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, 0);
    }

    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
    }

    float elapsed_ms() {
        float elapsed;
        cudaEventElapsedTime(&elapsed, start_event, stop_event);
        return elapsed;
    }
};

// CPU 参考实现：串行排他型前缀和
void cpu_exclusive_scan(const float* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = sum;
        sum += input[i];
    }
}

// 验证结果正确性
bool verify_result(const float* result, const float* expected, int n, float tolerance = 1e-4f) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(result[i] - expected[i]) > tolerance) {
            printf("  Mismatch at index %d: got %.6f, expected %.6f\n", i, result[i], expected[i]);
            return false;
        }
    }
    return true;
}

// 打印结果摘要
void print_summary(const char* name, int n, float time_ms, bool correct) {
    float bandwidth_gb_s = (n * sizeof(float) * 2.0f) / (time_ms * 1e6f);  // 读 + 写
    printf("  %-35s  N=%6d  Time=%8.4f ms  BW=%6.2f GB/s  %s\n",
           name, n, time_ms, bandwidth_gb_s, correct ? "[PASS]" : "[FAIL]");
}

// 基准测试配置
struct BenchmarkConfig {
    int n;
    int warmup_iterations;
    int test_iterations;
};

// 运行单个基准测试
typedef void (*ScanFunction)(const float*, float*, int);

float run_benchmark(ScanFunction func, const float* d_input, float* d_output,
                    int n, int warmup_iters, int test_iters,
                    const float* cpu_result, bool& correct) {
    GpuTimer timer;

    // 预热
    for (int i = 0; i < warmup_iters; i++) {
        func(d_input, d_output, n);
    }

    // 正式测试
    float total_time = 0.0f;
    correct = true;

    for (int i = 0; i < test_iters; i++) {
        timer.start();
        func(d_input, d_output, n);
        timer.stop();
        total_time += timer.elapsed_ms();

        // 验证正确性（只在第一次迭代验证）
        if (i == 0) {
            std::vector<float> h_output(n);
            cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
            correct = verify_result(h_output.data(), cpu_result, n);
        }
    }

    return total_time / test_iters;  // 返回平均时间
}

} // namespace scan

// 主函数
int main(int argc, char** argv) {
    using namespace scan;

    printf("================================================================================\n");
    printf("                   CUDA Scan (Prefix Sum) Performance Benchmark                  \n");
    printf("================================================================================\n\n");

    // 打印设备信息
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("Warning: No CUDA device available. Running in emulation mode.\n");
        printf("Error: %s\n\n", cudaGetErrorString(err));
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        printf("Device: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("\n");
    }

    // 测试配置
    std::vector<BenchmarkConfig> configs = {
        {256,   10, 100},   // 小规模
        {512,   10, 100},   // 中等规模
        {1024,  10, 100},   // 最大单 block
        {2048,  10, 100},   // 需要多 block（但我们的实现是单 block 简化版）
        {4096,  10, 100},   // 更大规模
    };

    // 分配 GPU 内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, 4096 * sizeof(float));
    cudaMalloc(&d_output, 4096 * sizeof(float));

    printf("Running benchmarks...\n");
    printf("--------------------------------------------------------------------------------\n");

    for (const auto& config : configs) {
        int n = config.n;
        int warmup = config.warmup_iterations;
        int test = config.test_iterations;

        // 生成随机输入数据
        std::vector<float> h_input(n);
        for (int i = 0; i < n; i++) {
            h_input[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        }

        // 计算 CPU 参考结果
        std::vector<float> cpu_result(n);
        cpu_exclusive_scan(h_input.data(), cpu_result.data(), n);

        // 拷贝到 GPU
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        printf("\nTest Size: N = %d elements\n", n);
        printf("--------------------------------------------------------------------------------\n");

        bool correct;
        float time;

        // V1: Hillis-Steele (只测试 n <= 1024 的情况)
        if (n <= 1024) {
            correct = true;
            time = run_benchmark(hillis_steele_scan, d_input, d_output, n,
                                warmup, test, cpu_result.data(), correct);
            print_summary("V1: Hillis-Steele (Naive)", n, time, correct);
        }

        // V2: Blelloch
        if (n <= 1024) {
            correct = true;
            time = run_benchmark(blelloch_scan, d_input, d_output, n,
                                warmup, test, cpu_result.data(), correct);
            print_summary("V2: Blelloch (Work-Efficient)", n, time, correct);
        }

        // V3: Bank-Free
        if (n <= 1024) {
            correct = true;
            time = run_benchmark(bank_free_scan, d_input, d_output, n,
                                warmup, test, cpu_result.data(), correct);
            print_summary("V3: Bank-Free (No Conflicts)", n, time, correct);
        }

        // V4: Warp Primitives
        correct = true;
        time = run_benchmark(warp_scan, d_input, d_output, n,
                           warmup, test, cpu_result.data(), correct);
        print_summary("V4: Warp Primitives", n, time, correct);

        printf("\n");
    }

    // 释放 GPU 内存
    cudaFree(d_input);
    cudaFree(d_output);

    printf("================================================================================\n");
    printf("                              Benchmark Complete                               \n");
    printf("================================================================================\n");

    return 0;
}

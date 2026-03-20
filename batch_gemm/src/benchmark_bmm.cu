#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 宏定义：用于检查 CUDA 运行时错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 宏定义：用于检查 cuBLAS 错误
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << stat << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#include "kernels.cuh"

// ==============================================================================
// 辅助函数：随机初始化矩阵
// ==============================================================================
void init_random_matrix(std::vector<float>& mat) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dis(gen);
    }
}

// ==============================================================================
// 辅助函数：验证结果正确性
// ==============================================================================
bool verify_result(const std::vector<float>& ref, const std::vector<float>& custom, float tol = 1e-4f) {
    if (ref.size() != custom.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(ref[i] - custom[i]);
        if (diff > tol) {
            std::cerr << "Mismatch at index " << i << ": expected " << ref[i] << ", got " << custom[i] 
                      << " (diff = " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// ==============================================================================
// 主函数
// ==============================================================================
int main() {
    // 1. 设置矩阵维度 (你可以随意修改这些参数进行测试)
    const int BATCH_SIZE = 16;
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t size_A = BATCH_SIZE * M * K;
    const size_t size_B = BATCH_SIZE * K * N;
    const size_t size_C = BATCH_SIZE * M * N;

    // 计算总 FLOPs (Multiply-Add 算两次浮点运算)
    const double total_flops = 2.0 * BATCH_SIZE * M * N * K;

    std::cout << "Batched GEMM Benchmark (FP32)" << std::endl;
    std::cout << "Dimensions: Batch=" << BATCH_SIZE << ", M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // 2. 分配主机内存并初始化
    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_ref(size_C, 0.0f);
    std::vector<float> h_C_custom(size_C, 0.0f);

    init_random_matrix(h_A);
    init_random_matrix(h_B);

    // 3. 分配设备内存
    float *d_A, *d_B, *d_C_ref, *d_C_custom;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_custom, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));

    // 4. 创建 CUDA Events 用于精确计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;
    const int WARMUP_ITERS = 5;
    const int TEST_ITERS = 20;

    // ==========================================================================
    // Benchmark 1: cuBLAS (基准)
    // ==========================================================================
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    // 注意：cuBLAS 默认使用列优先(Column-Major)。
    // 我们的矩阵是行优先(Row-Major)。
    // 根据矩阵转置公式：C_row = A_row * B_row  ==>  C_col^T = B_col^T * A_col^T
    // 因此传给 cuBLAS 的参数顺序需要交换 A 和 B，并且维度也要对应调整。
    auto run_cublas = [&]() {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,                     // 注意：m=N, n=M
            &alpha,
            d_B, N, K * N,               // A的位置传B, lda=N, stride=K*N
            d_A, K, M * K,               // B的位置传A, ldb=K, stride=M*K
            &beta,
            d_C_ref, N, M * N,           // ldc=N, stride=M*N
            BATCH_SIZE
        ));
    };

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) run_cublas();
    CUDA_CHECK(cudaDeviceSynchronize());

    // 测速
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TEST_ITERS; ++i) {
        run_cublas();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    double avg_ms_cublas = milliseconds / TEST_ITERS;
    double tflops_cublas = (total_flops / (avg_ms_cublas * 1e-3)) / 1e12;
    std::cout << "[cuBLAS]     Avg Time: " << avg_ms_cublas << " ms \t TFLOPS: " << tflops_cublas << std::endl;

    // ==========================================================================
    // Benchmark 2: Custom Kernels (动态测试多个版本)
    // ==========================================================================
    struct KernelConfig {
        std::string name;
        BmmKernelFunc func;
    };

    // ==========================================================================
    // 🚀 在这里注册你新写的所有 Kernel 版本！
    // ==========================================================================
    std::vector<KernelConfig> custom_kernels = {
        {"V0_Naive", run_bmm_naive},
        {"V1_SharedMem", run_bmm_v1_shared_memory},
        // {"V2_RegTiling", run_bmm_v2_register_tiling},
    };

    std::cout << "--------------------------------------------------------" << std::endl;

    for (const auto& kernel : custom_kernels) {
        // 重要：每次测新 kernel 前将输出内存置零，防止被上一次成功的结果掩盖逻辑错误
        CUDA_CHECK(cudaMemset(d_C_custom, 0, size_C * sizeof(float)));

        auto run_custom = [&]() {
            // 统一调用对应的 Host 端包装函数
            kernel.func(d_A, d_B, d_C_custom, BATCH_SIZE, M, N, K);
        };

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) run_custom();
        CUDA_CHECK(cudaDeviceSynchronize());

        // 测速
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < TEST_ITERS; ++i) {
            run_custom();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        double avg_ms_custom = milliseconds / TEST_ITERS;
        double tflops_custom = (total_flops / (avg_ms_custom * 1e-3)) / 1e12;
        std::cout << "[" << kernel.name << "] Avg Time: " << avg_ms_custom 
                  << " ms \t TFLOPS: " << tflops_custom 
                  << " \t (" << (tflops_custom / tflops_cublas) * 100.0 << "% of cuBLAS)" << std::endl;

        // 验证正确性
        CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (verify_result(h_C_ref, h_C_custom)) {
            std::cout << "  -> ✅ Validation PASSED!" << std::endl;
        } else {
            std::cout << "  -> ❌ Validation FAILED!" << std::endl;
        }
    }

    std::cout << "--------------------------------------------------------" << std::endl;

    // 5. 释放资源
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_ref));
    CUDA_CHECK(cudaFree(d_C_custom));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
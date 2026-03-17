#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include "common.h"
#include "gemm_kernels.h"

// ==========================================
// 框架核心功能 (验证与性能测试)
// ==========================================

// 初始化随机矩阵
void randomize_matrix(std::vector<float>& mat) {
    std::mt19937 gen(42); // 固定随机种子便于复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(gen);
    }
}

// 验证计算结果正确性 (基于 Max Error)
bool verify_result(const std::vector<float>& ref, const std::vector<float>& res, float tolerance = 1e-2f) {
    if (ref.size() != res.size()) return false;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float err = std::abs(ref[i] - res[i]);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            std::cerr << "Mismatch at index " << i << ": ref=" << ref[i] << ", res=" << res[i] << ", err=" << err << std::endl;
            return false;
        }
    }
    std::cout << "  [Correctness] Pass! (Max error: " << max_err << ")" << std::endl;
    return true;
}

// 统一 Benchmark 跑分函数
void benchmark_gemm(const char* name, GemmFunc gemm_func, 
                    int M, int N, int K, float alpha, 
                    const float* d_A, const float* d_B, float beta, float* d_C, 
                    std::vector<float>& h_C_res, const std::vector<float>& h_C_ref, 
                    bool check_correctness = true) {
    
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Running: " << name << std::endl;

    // 清空输出显存
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // 1. 预热 (Warm-up)
    for (int i = 0; i < 3; ++i) {
        gemm_func(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2. 如果需要，进行正确性验证
    if (check_correctness) {
        CHECK_CUDA(cudaMemcpy(h_C_res.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        bool passed = verify_result(h_C_ref, h_C_res);
        if (!passed) {
            std::cerr << "  [Correctness] FAILED! Skipping performance test." << std::endl;
            return;
        }
    } else {
        std::cout << "  [Correctness] Skipped (Used as Reference)." << std::endl;
    }

    // 3. 性能测试 (Timing)
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int num_repeats = 20; // 循环次数，求平均
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_repeats; ++i) {
        gemm_func(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 4. 计算指标
    float avg_ms = milliseconds / num_repeats;
    // TFLOPs = (2 * M * N * K) / (time_in_sec * 1e12)
    double tflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e12;

    std::cout << "  [Performance] Avg Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
    std::cout << "  [Performance] Throughput: " << std::fixed << std::setprecision(3) << tflops << " TFLOPs" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}


// ==========================================
// 主函数 Main
// ==========================================
int main() {
    // 矩阵维度 (M, N, K)
    int M = 4096;
    int N = 4096;
    int K = 4096;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::cout << "Starting GEMM Benchmark..." << std::endl;
    std::cout << "Matrix Size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 内存分配
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N, 0.0f); 
    std::vector<float> h_C_res(M * N, 0.0f); 

    // 初始化 Host 数据
    randomize_matrix(h_A);
    randomize_matrix(h_B);

    // Device 内存分配
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // 数据从 Host 拷贝到 Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 初始化 cuBLAS 句柄
    cublasCreate(&cublas_handle);

    // ==========================================
    // 运行测试
    // ==========================================

    // 0. cuBLAS 版本 (Reference)
    benchmark_gemm("cuBLAS SGEMM (Reference / Upper Bound)", run_cublas, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, false);
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 1. Naive 版本
    benchmark_gemm("SGEMM_Naive", run_sgemm_naive, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true);
    
    // 2. Shared Memory 版本
    benchmark_gemm("SGEMM_SharedMemory", run_sgemm_shared, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true);

    std::cout << "--------------------------------------------------------\n";
    std::cout << "Benchmark Completed." << std::endl;

    // 清理资源
    cublasDestroy(cublas_handle);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
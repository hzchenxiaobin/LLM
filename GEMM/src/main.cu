#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include "common.h"
#include "gemm_kernels.h"

// ==========================================
// 辅助功能：获取硬件信息并计算理论峰值
// ==========================================

// 根据 GPU 架构获取每个 SM 的 FP32 核心数
int get_cores_per_sm(int major, int minor) {
    switch (major) {
        case 7: return 64;  // Volta / Turing 架构
        case 8: return 128; // Ampere / Ada 架构 (包括 RTX 3060)
        case 9: return 128; // Hopper 架构
        case 10: return 128; // Blackwell 架构 (预估)
        default: return 128; // 默认 fallback
    }
}

// ==========================================
// 框架核心功能 (验证与性能测试)
// ==========================================

// 初始化随机矩阵
void randomize_matrix(std::vector<float>& mat) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(gen);
    }
}

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
    std::cout << "  [正确性检查] 通过! (最大误差: " << max_err << ")" << std::endl;
    return true;
}

// 统一 Benchmark 跑分函数 (新增了 peak_tflops 参数)
void benchmark_gemm(const char* name, GemmFunc gemm_func, 
                    int M, int N, int K, float alpha, 
                    const float* d_A, const float* d_B, float beta, float* d_C, 
                    std::vector<float>& h_C_res, const std::vector<float>& h_C_ref, 
                    bool check_correctness, double peak_tflops) {
    
    std::cout << "--------------------------------------------------------\n";
    std::cout << "正在运行: " << name << std::endl;

    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // 1. 预热
    for (int i = 0; i < 3; ++i) {
        gemm_func(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2. 验证
    if (check_correctness) {
        CHECK_CUDA(cudaMemcpy(h_C_res.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        bool passed = verify_result(h_C_ref, h_C_res);
        if (!passed) {
            std::cerr << "  [正确性检查] 失败! 跳过性能测试。" << std::endl;
            return;
        }
    } else {
        std::cout << "  [正确性检查] 跳过 (作为标准参考答案)." << std::endl;
    }

    // 3. 性能测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int num_repeats = 20; 
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_repeats; ++i) {
        gemm_func(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 4. 计算算力指标
    float avg_ms = milliseconds / num_repeats;
    // 实际跑出的 TFLOPs
    double tflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e12;

    std::cout << "  [性能统计] 平均耗时: " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
    
    // 输出实际算力和利用率百分比
    std::cout << "  [算力吞吐] 实际算力: " << std::fixed << std::setprecision(3) << tflops << " TFLOPs";
    if (peak_tflops > 0) {
        double utilization = (tflops / peak_tflops) * 100.0;
        std::cout << " (达理论峰值的 " << std::fixed << std::setprecision(2) << utilization << "%)";
    }
    std::cout << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}


// ==========================================
// 主函数 Main
// ==========================================
int main() {
    // ---- 动态获取 GPU 硬件信息和理论峰值算力 ----
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    int cores_per_sm = get_cores_per_sm(prop.major, prop.minor);
    // 理论峰值(FP32) = 2 (乘加指令) * SM数量 * 每SM核心数 * 核心频率(Hz)
    // prop.clockRate 单位是 kHz，因此乘以 1e3 转为 Hz
    double peak_tflops = 2.0 * prop.multiProcessorCount * cores_per_sm * (prop.clockRate * 1e3) / 1e12;

    std::cout << "========================================================\n";
    std::cout << "检测到显卡设备: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "SM 数量: " << prop.multiProcessorCount << ", 核心频率: " << prop.clockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "理论 FP32 峰值算力: " << std::fixed << std::setprecision(2) << peak_tflops << " TFLOPs" << std::endl;
    std::cout << "========================================================\n";

    // 矩阵维度 (M, N, K)
    int M = 4096;
    int N = 4096;
    int K = 4096;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::cout << "\n开始 GEMM 性能基准测试..." << std::endl;
    std::cout << "矩阵尺寸: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N, 0.0f); 
    std::vector<float> h_C_res(M * N, 0.0f); 

    randomize_matrix(h_A);
    randomize_matrix(h_B);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    cublasCreate(&cublas_handle);

    // ==========================================
    // 运行测试 (传入 peak_tflops)
    // ==========================================

    // 0. cuBLAS 版本 (Reference)
    benchmark_gemm("cuBLAS SGEMM (Reference / Upper Bound)", run_cublas, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, false, peak_tflops);
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 1. Naive 版本
    benchmark_gemm("SGEMM_Naive", run_sgemm_naive, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true, peak_tflops);
    
    // 2. Shared Memory 版本
    benchmark_gemm("SGEMM_SharedMemory", run_sgemm_shared, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true, peak_tflops);

    // 3. Register Tiling 版本
    benchmark_gemm("SGEMM_RegisterTiling", run_sgemm_register, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true, peak_tflops);

    // 3.5 Register Tiling V2 版本 (Vectorized + Padding)
    benchmark_gemm("SGEMM_RegisterTiling_V2", run_sgemm_register_v2, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true, peak_tflops);

    // 3.6 Register Tiling V3 版本 (Double Buffering)
    benchmark_gemm("SGEMM_RegisterTiling_V3", run_sgemm_register_v3, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true, peak_tflops);

    // 3.7 Register Tiling Bank Conflict 优化版本 (Shared Memory Padding)
    benchmark_gemm("SGEMM_RegisterTiling_BankConflict", run_sgemm_register_bank_conflict, M, N, K, alpha, d_A, d_B, beta, d_C, h_C_res, h_C_ref, true, peak_tflops);

    std::cout << "--------------------------------------------------------\n";
    std::cout << "测试完成." << std::endl;

    cublasDestroy(cublas_handle);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
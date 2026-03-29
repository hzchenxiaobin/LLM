#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include "../include/topk_common.h"

// 包含各个版本的 kernel
#include "../src/topk_v1_thread_per_row.cu"
#include "../src/topk_v2_block_shared_memory.cu"
#include "../src/topk_v3_warp_shuffle.cu"
#include "../src/topk_v4_radix_select.cu"

// ==========================================
// CPU Baseline: 供正确性验证 (1D版本，无索引输出)
// ==========================================
void topk_cpu_1d(const float* input, float* out_vals, int N, int K) {
    std::vector<float> arr(input, input + N);
    // 降序排序取前K个
    std::partial_sort(arr.begin(), arr.begin() + K, arr.end(),
        [](float a, float b) { return a > b; });

    for (int k = 0; k < K; ++k) {
        out_vals[k] = arr[k];
    }
}

// ==========================================
// 评测与辅助函数
// ==========================================
bool verify_1d(const float* cpu_vals, const float* gpu_vals, int K) {
    for (int i = 0; i < K; ++i) {
        if (std::abs(cpu_vals[i] - gpu_vals[i]) > 1e-4) {
            std::cout << "Value mismatch at " << i << ": CPU=" << cpu_vals[i] << ", GPU=" << gpu_vals[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 测试用例结构体 (1D版本)
struct TestCase1D {
    int n;
    int k;
    const char* name;
    const char* desc;
};

// 预定义的测试用例集 (1D版本 - LLM词表场景)
std::vector<TestCase1D> get_test_cases_1d() {
    return {
        // LLaMA 词表大小 (32K)
        {32000, 1, "LLaMA_Top1", "LLaMA vocab, greedy decoding"},
        {32000, 5, "LLaMA_Top5", "LLaMA vocab, top-5 sampling"},
        {32000, 16, "LLaMA_Top16", "LLaMA vocab, typical sampling"},
        {32000, 32, "LLaMA_Top32", "LLaMA vocab, large K sampling"},

        // Qwen 词表大小 (约 150K)
        {150000, 16, "Qwen_Top16", "Qwen vocab, typical sampling"},

        // 不同规模测试
        {1000, 4, "Small_1K", "Small vocab for debugging"},
        {5000, 8, "Medium_5K", "Medium vocab for debugging"},
        {10000, 16, "Large_10K", "Large vocab"},
        {50000, 16, "Vocab50K", "Non-standard vocab size"},
        {100000, 16, "Vocab100K", "Large non-standard vocab"},

        // 极端 K 值测试
        {32000, 2, "K=2", "Small K value"},
        {32000, 4, "K=4", "Small K value"},
        {32000, 8, "K=8", "Medium-small K"},
        {32000, 24, "K=24", "Medium-large K"},
    };
}

// 运行单个测试用例 - 测试所有版本
void run_test_case(const TestCase1D& tc) {
    int N = tc.n;
    int K = tc.k;

    if (K > MAX_K) {
        std::cout << "\n[跳过] " << tc.name << " - K=" << K << " 超过 MAX_K=" << MAX_K << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "测试: " << tc.name << " | " << tc.desc << std::endl;
    std::cout << "配置: N=" << N << ", K=" << K << " (1D输入，无Batch，无索引输出)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::vector<float> h_in(N);
    std::vector<float> h_out_vals_cpu(K, 0);

    // 随机数填充
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) h_in[i] = dis(gen);

    // 显存分配 (只分配值，不分配索引)
    float *d_in, *d_out_vals;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_vals, K * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 运行 CPU Baseline
    std::cout << "CPU Baseline... ";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    topk_cpu_1d(h_in.data(), h_out_vals_cpu.data(), N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::cout << cpu_us << " us" << std::endl;

    std::vector<float> h_out_vals_gpu(K);

    // --- Benchmark V1: 单线程串行 ---
    {
        std::cout << "V1_Thread: ";
        dim3 block(1);
        dim3 grid(1);

        // Warmup
        topk_v1_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 校验结果
        CHECK_CUDA(cudaMemcpy(h_out_vals_gpu.data(), d_out_vals, K * sizeof(float), cudaMemcpyDeviceToHost));
        bool passed = verify_1d(h_out_vals_cpu.data(), h_out_vals_gpu.data(), K);
        std::cout << "校验: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << " | ";

        if (passed) {
            int iters = (N <= 5000) ? 100 : 10;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) {
                topk_v1_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            float avg_us = (ms / iters) * 1000.0f;

            size_t total_bytes = N * sizeof(float) + K * sizeof(float);
            float bw_GBs = (total_bytes / 1e6) / (avg_us / 1000.0f);

            std::cout << "延迟: " << avg_us << " us | 带宽: " << bw_GBs << " GB/s" << std::endl;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    // --- Benchmark V2: Block级别 ---
    {
        std::cout << "V2_Block:  ";
        int block_size = 256;
        dim3 block(block_size);
        dim3 grid(1);

        // Warmup
        topk_v2_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out_vals_gpu.data(), d_out_vals, K * sizeof(float), cudaMemcpyDeviceToHost));
        bool passed = verify_1d(h_out_vals_cpu.data(), h_out_vals_gpu.data(), K);
        std::cout << "校验: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << " | ";

        if (passed) {
            int iters = 100;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) {
                topk_v2_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            float avg_us = (ms / iters) * 1000.0f;

            size_t total_bytes = N * sizeof(float) + K * sizeof(float);
            float bw_GBs = (total_bytes / 1e6) / (avg_us / 1000.0f);

            std::cout << "延迟: " << avg_us << " us | 带宽: " << bw_GBs << " GB/s" << std::endl;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    // --- Benchmark V3: Warp级别 ---
    {
        std::cout << "V3_Warp:   ";
        dim3 block(32);
        dim3 grid(1);

        // Warmup
        topk_v3_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out_vals_gpu.data(), d_out_vals, K * sizeof(float), cudaMemcpyDeviceToHost));
        bool passed = verify_1d(h_out_vals_cpu.data(), h_out_vals_gpu.data(), K);
        std::cout << "校验: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << " | ";

        if (passed) {
            int iters = 100;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) {
                topk_v3_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            float avg_us = (ms / iters) * 1000.0f;

            size_t total_bytes = N * sizeof(float) + K * sizeof(float);
            float bw_GBs = (total_bytes / 1e6) / (avg_us / 1000.0f);

            std::cout << "延迟: " << avg_us << " us | 带宽: " << bw_GBs << " GB/s" << std::endl;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    // --- Benchmark V4: Radix Select ---
    {
        std::cout << "V4_Radix:  ";

        if (N <= 1024) {
            // 小数据量：单warp
            dim3 block(32);
            dim3 grid(1);

            topk_v4_warp_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_out_vals_gpu.data(), d_out_vals, K * sizeof(float), cudaMemcpyDeviceToHost));
            bool passed = verify_1d(h_out_vals_cpu.data(), h_out_vals_gpu.data(), K);
            std::cout << "[Warp] 校验: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << " | ";

            if (passed) {
                int iters = 1000;
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start);
                for (int i = 0; i < iters; ++i) {
                    topk_v4_warp_kernel<<<grid, block>>>(d_in, d_out_vals, N, K);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                float avg_us = (ms / iters) * 1000.0f;

                size_t total_bytes = N * sizeof(float) + K * sizeof(float);
                float bw_GBs = (total_bytes / 1e6) / (avg_us / 1000.0f);

                std::cout << "延迟: " << avg_us << " us | 带宽: " << bw_GBs << " GB/s" << std::endl;

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        } else {
            // 大数据量：block级别
            int block_size = 256;
            int num_warps = block_size / 32;
            size_t shared_mem_size = num_warps * K * sizeof(float);

            dim3 block(block_size);
            dim3 grid(1);

            topk_v4_kernel<<<grid, block, shared_mem_size>>>(d_in, d_out_vals, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_out_vals_gpu.data(), d_out_vals, K * sizeof(float), cudaMemcpyDeviceToHost));
            bool passed = verify_1d(h_out_vals_cpu.data(), h_out_vals_gpu.data(), K);
            std::cout << "[Block] 校验: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << " | ";

            if (passed) {
                int iters = 100;
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start);
                for (int i = 0; i < iters; ++i) {
                    topk_v4_kernel<<<grid, block, shared_mem_size>>>(d_in, d_out_vals, N, K);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                float avg_us = (ms / iters) * 1000.0f;

                size_t total_bytes = N * sizeof(float) + K * sizeof(float);
                float bw_GBs = (total_bytes / 1e6) / (avg_us / 1000.0f);

                std::cout << "延迟: " << avg_us << " us | 带宽: " << bw_GBs << " GB/s" << std::endl;

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        }
    }

    // 打印Top-5结果
    std::cout << "Top-5 结果: ";
    CHECK_CUDA(cudaMemcpy(h_out_vals_gpu.data(), d_out_vals, std::min(5, K) * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < std::min(5, K); ++i) {
        std::cout << h_out_vals_gpu[i] << " ";
    }
    std::cout << std::endl;

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out_vals));
}

// 运行对比测试套件
void run_comparison_suite() {
    std::cout << "\n\n############################################" << std::endl;
    std::cout << "#        1D版本 对比测试套件              #" << std::endl;
    std::cout << "############################################" << std::endl;

    // 固定 K=16, 变化 N
    std::cout << "\n>>> 场景1: 固定 K=16, 变化 Vocab Size <<<" << std::endl;
    int vocab_sizes[] = {1000, 5000, 10000, 32000, 50000, 100000, 150000};
    for (int n : vocab_sizes) {
        TestCase1D tc = {n, 16, "VocabVar", "Varying vocab size"};
        run_test_case(tc);
    }

    // 固定 N=32000, 变化 K
    std::cout << "\n>>> 场景2: 固定 N=32000, 变化 K <<<" << std::endl;
    int k_values[] = {1, 2, 4, 8, 16, 24, 32};
    for (int k : k_values) {
        TestCase1D tc = {32000, k, "KVar", "Varying K"};
        run_test_case(tc);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "     Top-K CUDA Benchmark (1D)        " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nUsage: " << argv[0] << " [mode]" << std::endl;
    std::cout << "  mode: all      - all predefined tests (default)" << std::endl;
    std::cout << "        quick    - quick tests only" << std::endl;
    std::cout << "        llm      - LLM-shaped tests only" << std::endl;
    std::cout << "        compare  - comparison suite" << std::endl;
    std::cout << "        custom N K - custom N and K" << std::endl;

    std::string mode = "all";
    if (argc > 1) mode = argv[1];

    if (mode == "custom" && argc >= 4) {
        int N = std::atoi(argv[2]);
        int K = std::atoi(argv[3]);
        TestCase1D tc = {N, K, "Custom", "User defined test"};
        run_test_case(tc);
        return 0;
    }

    if (mode == "compare") {
        run_comparison_suite();
        return 0;
    }

    auto all_tests = get_test_cases_1d();

    for (const auto& tc : all_tests) {
        // 根据模式过滤测试
        if (mode == "quick") {
            if (tc.n > 10000) continue;
        } else if (mode == "llm") {
            bool is_typical_llm = (tc.n == 32000 || tc.n == 150000) &&
                                   (tc.k >= 1 && tc.k <= 32);
            if (!is_typical_llm) continue;
        }

        run_test_case(tc);
    }

    if (mode == "all" || mode == "compare") {
        run_comparison_suite();
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "         All tests done.               " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

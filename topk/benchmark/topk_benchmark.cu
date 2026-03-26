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

// ==========================================
// CPU Baseline: 供正确性验证
// ==========================================
void topk_cpu(const float* input, float* out_vals, int* out_inds, int Batch, int N, int K) {
    for (int b = 0; b < Batch; ++b) {
        std::vector<std::pair<float, int>> row(N);
        for (int i = 0; i < N; ++i) {
            row[i] = {input[b * N + i], i};
        }
        // 降序排序
        std::partial_sort(row.begin(), row.begin() + K, row.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        for (int k = 0; k < K; ++k) {
            out_vals[b * K + k] = row[k].first;
            out_inds[b * K + k] = row[k].second;
        }
    }
}

// ==========================================
// 评测与辅助函数
// ==========================================
bool verify(const float* cpu_vals, const float* gpu_vals, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(cpu_vals[i] - gpu_vals[i]) > 1e-4) {
            std::cout << "Mismatch at " << i << ": CPU=" << cpu_vals[i] << ", GPU=" << gpu_vals[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 测试用例结构体
struct TestCase {
    int batch;
    int n;
    int k;
    const char* name;
    const char* desc;
};

// 预定义的测试用例集
std::vector<TestCase> get_test_cases() {
    return {
        // ========== 基础 LLM 场景 ==========
        // LLaMA 词表大小 (32K)
        {128, 32000, 1, "LLaMA_Top1", "LLaMA vocab, greedy decoding"},
        {128, 32000, 5, "LLaMA_Top5", "LLaMA vocab, top-5 sampling"},
        {128, 32000, 16, "LLaMA_Top16", "LLaMA vocab, typical sampling"},
        {128, 32000, 32, "LLaMA_Top32", "LLaMA vocab, large K sampling"},

        // Qwen 词表大小 (约 150K)
        {128, 150000, 16, "Qwen_Top16", "Qwen vocab, typical sampling"},

        // ========== 不同 Batch 大小 ==========
        // 小 Batch (推理场景)
        {1, 32000, 16, "Batch1", "Single request inference"},
        {8, 32000, 16, "Batch8", "Small batch inference"},
        {32, 32000, 16, "Batch32", "Medium batch inference"},
        {256, 32000, 16, "Batch256", "Large batch inference"},
        {512, 32000, 16, "Batch512", "Very large batch"},

        // ========== 极端 K 值测试 ==========
        {128, 32000, 2, "K=2", "Small K value"},
        {128, 32000, 4, "K=4", "Small K value"},
        {128, 32000, 8, "K=8", "Medium-small K"},
        {128, 32000, 24, "K=24", "Medium-large K"},
        {128, 32000, 32, "K=32", "Maximum supported K"},

        // ========== Beam Search 场景 ==========
        {40, 32000, 4, "BeamSearch", "Beam search with 4 beams, 10 sequences"},

        // ========== 小规模测试（便于调试） ==========
        {4, 1000, 4, "Small_Debug", "Small scale for debugging"},
        {16, 5000, 8, "Medium_Debug", "Medium scale for debugging"},

        // ========== 非标准词表大小 ==========
        {128, 50000, 16, "Vocab50K", "Non-standard vocab size"},
        {128, 100000, 16, "Vocab100K", "Large non-standard vocab"},

        // ========== 超大 Batch 场景 (如推荐系统、嵌入查找) ==========
        {8192, 1024, 16, "Batch8K_N1K", "Large batch, small vocab - recommendation/embedding"},
    };
}

void benchmark(void (*kernel)(const float*, float*, int*, int, int, int),
               const char* name, dim3 grid, dim3 block,
               const float* d_in, float* d_out_vals, int* d_out_inds,
               const float* cpu_vals, int Batch, int N, int K) {

    // 初始化输出为 0
    CHECK_CUDA(cudaMemset(d_out_vals, 0, Batch * K * sizeof(float)));

    // Warmup
    kernel<<<grid, block>>>(d_in, d_out_vals, d_out_inds, Batch, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 校验结果
    float* h_out_vals = new float[Batch * K];
    CHECK_CUDA(cudaMemcpy(h_out_vals, d_out_vals, Batch * K * sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = verify(cpu_vals, h_out_vals, Batch * K);
    std::cout << "[" << name << "] 校验结果: " << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << " | ";
    delete[] h_out_vals;

    if (!passed) return;

    // 性能测算
    int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>(d_in, d_out_vals, d_out_inds, Batch, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float avg_ms = ms / iters;
    // 访存量：读 Batch * N 个 float, 写 Batch * K 个 float (和 int)
    size_t total_bytes = Batch * N * sizeof(float) + Batch * K * (sizeof(float) + sizeof(int));
    float bw_GBs = (total_bytes / 1e6) / avg_ms;

    std::cout << "延迟: " << avg_ms << " ms | 有效带宽: " << bw_GBs << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 运行单个测试用例
void run_test_case(const TestCase& tc) {
    int Batch = tc.batch;
    int N = tc.n;
    int K = tc.k;

    if (K > MAX_K) {
        std::cout << "\n[跳过] " << tc.name << " - K=" << K << " 超过 MAX_K=" << MAX_K << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "测试: " << tc.name << " | " << tc.desc << std::endl;
    std::cout << "配置: Batch=" << Batch << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    size_t input_size = static_cast<size_t>(Batch) * N;
    size_t output_size = static_cast<size_t>(Batch) * K;

    std::vector<float> h_in(input_size);
    std::vector<float> h_out_vals_cpu(output_size, 0);
    std::vector<int> h_out_inds_cpu(output_size, 0);

    // 随机数填充
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (size_t i = 0; i < input_size; ++i) h_in[i] = dis(gen);

    // 显存分配
    float *d_in, *d_out_vals;
    int *d_out_inds;
    CHECK_CUDA(cudaMalloc(&d_in, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_vals, output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_inds, output_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

    // 运行 CPU Baseline
    std::cout << "CPU Baseline... ";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    topk_cpu(h_in.data(), h_out_vals_cpu.data(), h_out_inds_cpu.data(), Batch, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    std::cout << cpu_ms << " ms" << std::endl;

    // --- Benchmark V1: Thread-per-Row ---
    int threads1 = 256;
    int blocks1 = (Batch + threads1 - 1) / threads1;
    benchmark(topk_v1_kernel, "V1_Thread", dim3(blocks1), dim3(threads1),
              d_in, d_out_vals, d_out_inds, h_out_vals_cpu.data(), Batch, N, K);

    // --- Benchmark V2: Block-per-Row ---
    int threads2 = 256;
    int blocks2 = Batch;
    benchmark(topk_v2_kernel, "V2_Block", dim3(blocks2), dim3(threads2),
              d_in, d_out_vals, d_out_inds, h_out_vals_cpu.data(), Batch, N, K);

    // --- Benchmark V3: Warp-per-Row ---
    dim3 block3(32, 4);
    dim3 grid3((Batch + block3.y - 1) / block3.y);
    benchmark(topk_v3_kernel, "V3_Warp", grid3, block3,
              d_in, d_out_vals, d_out_inds, h_out_vals_cpu.data(), Batch, N, K);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out_vals));
    CHECK_CUDA(cudaFree(d_out_inds));
}

// 运行特定场景的对比测试
void run_comparison_suite() {
    std::cout << "\n\n############################################" << std::endl;
    std::cout << "#           对比测试套件                    #" << std::endl;
    std::cout << "############################################" << std::endl;

    // 固定 N=32000, K=16, 变化 Batch
    std::cout << "\n>>> 场景1: 固定 N=32000, K=16, 变化 Batch Size <<<" << std::endl;
    int batches[] = {1, 8, 32, 128, 256, 512};
    for (int b : batches) {
        TestCase tc = {b, 32000, 16, "BatchVar", "Varying batch size"};
        run_test_case(tc);
    }

    // 固定 Batch=128, K=16, 变化 N
    std::cout << "\n>>> 场景2: 固定 Batch=128, K=16, 变化 Vocab Size <<<" << std::endl;
    int vocab_sizes[] = {1000, 5000, 10000, 32000, 50000, 100000, 150000};
    for (int n : vocab_sizes) {
        TestCase tc = {128, n, 16, "VocabVar", "Varying vocab size"};
        run_test_case(tc);
    }

    // 固定 Batch=128, N=32000, 变化 K
    std::cout << "\n>>> 场景3: 固定 Batch=128, N=32000, 变化 K <<<" << std::endl;
    int k_values[] = {1, 2, 4, 8, 16, 24, 32};
    for (int k : k_values) {
        TestCase tc = {128, 32000, k, "KVar", "Varying K"};
        run_test_case(tc);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "     Top-K CUDA Benchmark Suite       " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n用法: " << argv[0] << " [mode]" << std::endl;
    std::cout << "  mode: all      - 运行所有预定义测试 (默认)" << std::endl;
    std::cout << "        quick    - 仅运行快速测试 (小规模)" << std::endl;
    std::cout << "        llm      - 仅运行 LLM 相关测试" << std::endl;
    std::cout << "        compare  - 运行对比测试套件" << std::endl;
    std::cout << "        custom B N K - 自定义测试参数" << std::endl;

    std::string mode = "all";
    if (argc > 1) mode = argv[1];

    if (mode == "custom" && argc >= 5) {
        // 自定义测试模式: ./benchmark custom 128 32000 16
        int B = std::atoi(argv[2]);
        int N = std::atoi(argv[3]);
        int K = std::atoi(argv[4]);
        TestCase tc = {B, N, K, "Custom", "User defined test"};
        run_test_case(tc);
        return 0;
    }

    if (mode == "compare") {
        run_comparison_suite();
        return 0;
    }

    auto all_tests = get_test_cases();

    for (const auto& tc : all_tests) {
        // 根据模式过滤测试
        if (mode == "quick") {
            // 快速模式：只运行小规模测试
            if (tc.batch > 64 || tc.n > 10000) continue;
        } else if (mode == "llm") {
            // LLM模式：只运行典型LLM配置
            bool is_typical_llm = (tc.n == 32000 || tc.n == 150000) &&
                                   (tc.k >= 1 && tc.k <= 32) &&
                                   (tc.batch == 128 || tc.batch == 256);
            if (!is_typical_llm) continue;
        }

        run_test_case(tc);
    }

    if (mode == "all" || mode == "compare") {
        run_comparison_suite();
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "         所有测试完成!                 " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

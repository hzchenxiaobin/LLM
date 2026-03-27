#include "transpose_common.h"
#include <cstdio>
#include <vector>
#include <string>

// 基准测试结果结构
struct BenchmarkResult {
    std::string name;
    float min_time_ms;
    float max_time_ms;
    float avg_time_ms;
    float effective_bw_gbps;
    float theoretical_bw_gbps;
    float efficiency;
    bool verified;
};

// CUDA 事件计时器类
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }

    float elapsed_ms() const {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// 测试单个转置实现
BenchmarkResult benchmark_transpose(
    const std::string& name,
    void (*transpose_func)(const float*, float*, int, int, cudaStream_t),
    const float *d_A, float *d_B, float *h_A, float *h_B,
    int M, int N, int warmup_runs, int benchmark_runs
) {
    BenchmarkResult result;
    result.name = name;
    result.theoretical_bw_gbps = get_theoretical_bandwidth();

    CudaTimer timer;

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        transpose_func(d_A, d_B, M, N, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    float min_ms = 1e10f, max_ms = 0.0f, sum_ms = 0.0f;

    for (int i = 0; i < benchmark_runs; i++) {
        timer.start();
        transpose_func(d_A, d_B, M, N, 0);
        timer.stop();

        float ms = timer.elapsed_ms();
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
        sum_ms += ms;
    }

    result.min_time_ms = min_ms;
    result.max_time_ms = max_ms;
    result.avg_time_ms = sum_ms / benchmark_runs;
    result.effective_bw_gbps = compute_effective_bandwidth_ms(result.avg_time_ms, M, N);
    result.efficiency = result.effective_bw_gbps / result.theoretical_bw_gbps * 100.0f;

    // 验证结果
    size_t size_B = N * M * sizeof(float);
    cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost);
    result.verified = verify_transpose(h_A, h_B, M, N);

    return result;
}

// 打印结果表格
void print_results_table(const std::vector<BenchmarkResult>& results) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                              CUDA Transpose Benchmark Results                              ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-20s │ %-10s │ %-10s │ %-10s │ %-10s │ %-6s ║\n",
           "Version", "Min(ms)", "Max(ms)", "Avg(ms)", "BW(GB/s)", "Eff%");
    printf("╠══════════════════════╪════════════╪════════════╪════════════╪════════════╪════════════╣\n");

    for (const auto& r : results) {
        const char* status = r.verified ? "✓" : "✗";
        printf("║ %-20s │ %10.4f │ %10.4f │ %10.4f │ %10.2f │ %6.1f ║ %s\n",
               r.name.c_str(), r.min_time_ms, r.max_time_ms, r.avg_time_ms,
               r.effective_bw_gbps, r.efficiency, status);
    }

    printf("╚══════════════════════╧════════════╧════════════╧════════════╧════════════╧════════════╝\n");
    printf("\n");
    printf("Note: BW = Effective Bandwidth, Eff%% = Efficiency vs Theoretical BW\n");
    printf("      ✓ = Verified, ✗ = Failed verification\n");
    printf("\n");
}

// 打印性能对比
void print_performance_comparison(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                          Performance Comparison                            ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════════╣\n");

    // 以第一个版本为基准
    float baseline_bw = results[0].effective_bw_gbps;

    for (size_t i = 0; i < results.size(); i++) {
        float speedup = results[i].effective_bw_gbps / baseline_bw;
        printf("║ %-20s: %.2fx faster than baseline (v1_naive)              ║\n",
               results[i].name.c_str(), speedup);
    }

    printf("╚════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

// 运行完整的基准测试套件
void run_full_benchmark(int M, int N, int warmup_runs, int benchmark_runs) {
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Running Benchmark: M = %d, N = %d, Warmup = %d, Runs = %d\n", M, N, warmup_runs, benchmark_runs);
    printf("  Input Matrix: %d x %d = %.2f MB\n", M, N, (M * N * sizeof(float)) / (1024.0 * 1024.0));
    printf("  Output Matrix: %d x %d = %.2f MB\n", N, M, (N * M * sizeof(float)) / (1024.0 * 1024.0));
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 分配主机内存
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * M * sizeof(float);
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);

    if (!h_A || !h_B) {
        printf("Error: Failed to allocate host memory\n");
        return;
    }

    // 初始化输入数据
    init_matrix(h_A, M, N);

    // 分配设备内存
    float *d_A, *d_B;
    cudaError_t err_A = cudaMalloc(&d_A, size_A);
    cudaError_t err_B = cudaMalloc(&d_B, size_B);
    if (err_A != cudaSuccess || err_B != cudaSuccess) {
        printf("Error: Failed to allocate device memory: %s\n", cudaGetErrorString(err_A != cudaSuccess ? err_A : err_B));
        free(h_A);
        free(h_B);
        return;
    }
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // 运行各个版本的基准测试
    std::vector<BenchmarkResult> results;

    // v1: 朴素实现
    cudaMemset(d_B, 0, size_B);
    results.push_back(benchmark_transpose(
        "v1_naive", transpose_naive,
        d_A, d_B, h_A, h_B, M, N, warmup_runs, benchmark_runs));

    // v2: 共享内存 (有 bank conflict)
    cudaMemset(d_B, 0, size_B);
    results.push_back(benchmark_transpose(
        "v2_shared", transpose_shared_memory,
        d_A, d_B, h_A, h_B, M, N, warmup_runs, benchmark_runs));

    // v3: Padding 优化
    cudaMemset(d_B, 0, size_B);
    results.push_back(benchmark_transpose(
        "v3_shared_pad", transpose_shared_pad,
        d_A, d_B, h_A, h_B, M, N, warmup_runs, benchmark_runs));

    // v4: ILP 优化
    cudaMemset(d_B, 0, size_B);
    results.push_back(benchmark_transpose(
        "v4_optimized", transpose_optimized,
        d_A, d_B, h_A, h_B, M, N, warmup_runs, benchmark_runs));

    // v5: 向量化访问 (仅当 M 和 N 是 32 的倍数时)
    if (M % 32 == 0 && N % 32 == 0) {
        cudaMemset(d_B, 0, size_B);
        results.push_back(benchmark_transpose(
            "v5_vectorized", transpose_vectorized,
            d_A, d_B, h_A, h_B, M, N, warmup_runs, benchmark_runs));
    }

    // 打印结果
    print_results_table(results);
    print_performance_comparison(results);

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
}

// 运行所有10个测试用例（不同尺寸的矩阵）
void run_all_test_cases() {
    // 10个不同的测试用例，包含各种典型尺寸
    int test_cases[10][2] = {
        {1024, 1024},      // 1K x 1K - 小矩阵
        {2048, 2048},      // 2K x 2K
        {4096, 4096},      // 4K x 4K
        {8192, 8192},      // 8K x 8K - 默认大矩阵
        {16384, 4096},     // 16K x 4K - 宽矩阵
        {4096, 16384},     // 4K x 16K - 高矩阵
        {1024, 8192},      // 1K x 8K - 非方阵
        {8192, 1024},      // 8K x 1K - 非方阵
        {512, 32768},      // 极窄高矩阵
        {32768, 512}       // 极宽矮矩阵
    };

    const char* test_names[10] = {
        "Small Square (1K x 1K)",
        "Medium Square (2K x 2K)",
        "Large Square (4K x 4K)",
        "XLarge Square (8K x 8K)",
        "Wide Matrix (16K x 4K)",
        "Tall Matrix (4K x 16K)",
        "Non-square (1K x 8K)",
        "Non-square (8K x 1K)",
        "Very Tall (512 x 32K)",
        "Very Wide (32K x 512)"
    };

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                         Running All 10 Test Cases                          ║\n");
    printf("╚════════════════════════════════════════════════════════════════════════════╝\n");

    for (int i = 0; i < 10; i++) {
        int M = test_cases[i][0];
        int N = test_cases[i][1];

        printf("\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  Test Case %d/10: %s\n", i + 1, test_names[i]);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        run_full_benchmark(M, N, 10, 100);
    }

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                        All 10 Test Cases Completed!                      ║\n");
    printf("╚════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

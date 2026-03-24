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
    void (*transpose_func)(const float*, float*, int, cudaStream_t),
    const float *d_A, float *d_B, float *h_A, float *h_B,
    int N, int warmup_runs, int benchmark_runs
) {
    BenchmarkResult result;
    result.name = name;
    result.theoretical_bw_gbps = get_theoretical_bandwidth();

    CudaTimer timer;

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        transpose_func(d_A, d_B, N, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    float min_ms = 1e10f, max_ms = 0.0f, sum_ms = 0.0f;

    for (int i = 0; i < benchmark_runs; i++) {
        timer.start();
        transpose_func(d_A, d_B, N, 0);
        timer.stop();

        float ms = timer.elapsed_ms();
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
        sum_ms += ms;
    }

    result.min_time_ms = min_ms;
    result.max_time_ms = max_ms;
    result.avg_time_ms = sum_ms / benchmark_runs;
    result.effective_bw_gbps = compute_effective_bandwidth_ms(result.avg_time_ms, N);
    result.efficiency = result.effective_bw_gbps / result.theoretical_bw_gbps * 100.0f;

    // 验证结果
    cudaMemcpy(h_B, d_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    result.verified = verify_transpose(h_A, h_B, N);

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
void run_full_benchmark(int N, int warmup_runs, int benchmark_runs) {
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Running Benchmark: N = %d, Warmup = %d, Runs = %d\n", N, warmup_runs, benchmark_runs);
    printf("  Matrix Size: %d x %d = %.2f MB\n", N, N, (N * N * sizeof(float)) / (1024.0 * 1024.0));
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 分配主机内存
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);

    if (!h_A || !h_B) {
        printf("Error: Failed to allocate host memory\n");
        return;
    }

    // 初始化输入数据
    init_matrix(h_A, N);

    // 分配设备内存
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // 运行各个版本的基准测试
    std::vector<BenchmarkResult> results;

    // v1: 朴素实现
    cudaMemset(d_B, 0, size);
    results.push_back(benchmark_transpose(
        "v1_naive", transpose_naive,
        d_A, d_B, h_A, h_B, N, warmup_runs, benchmark_runs));

    // v2: 共享内存 (有 bank conflict)
    cudaMemset(d_B, 0, size);
    results.push_back(benchmark_transpose(
        "v2_shared", transpose_shared_memory,
        d_A, d_B, h_A, h_B, N, warmup_runs, benchmark_runs));

    // v3: Padding 优化
    cudaMemset(d_B, 0, size);
    results.push_back(benchmark_transpose(
        "v3_shared_pad", transpose_shared_pad,
        d_A, d_B, h_A, h_B, N, warmup_runs, benchmark_runs));

    // v4: ILP 优化
    cudaMemset(d_B, 0, size);
    results.push_back(benchmark_transpose(
        "v4_optimized", transpose_optimized,
        d_A, d_B, h_A, h_B, N, warmup_runs, benchmark_runs));

    // v5: 向量化访问 (仅当 N 是 32 的倍数时)
    if (N % 32 == 0) {
        cudaMemset(d_B, 0, size);
        results.push_back(benchmark_transpose(
            "v5_vectorized", transpose_vectorized,
            d_A, d_B, h_A, h_B, N, warmup_runs, benchmark_runs));
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

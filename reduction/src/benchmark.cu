/**
 * CUDA Reduction 性能测试 - 独立可执行文件
 *
 * 编译: make
 * 运行: ./benchmark --sizes 1M 10M 100M
 */

#include "reduction_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ============================================================
// 测试结构
// ============================================================
struct TestResult {
    std::string name;
    std::string desc;
    unsigned int n;
    float time_ms;
    float bandwidth_gb_s;
    float efficiency;
    bool correct;
    float error;
    float speedup_vs_cub;
};

struct VersionInfo {
    int id;
    std::string name;
    std::string desc;
};

const VersionInfo VERSIONS[] = {
    {1, "v1_interleaved", "naive interleaved"},
    {2, "v2_strided", "strided"},
    {3, "v3_sequential", "sequential"},
    {4, "v4_first_add", "first-add"},
    {5, "v5_warp_shuffle", "warp shuffle"},
    {6, "v6_vectorized", "vectorized"},
    {7, "cub", "NVIDIA CUB"},
};

// ============================================================
// 获取 Kernel 配置
// ============================================================
void get_kernel_config(int version, unsigned int n,
                       int &num_blocks, int &num_threads, int &shared_mem) {
    num_threads = 256;
    switch (version) {
        case 1:
        case 2:
        case 3:
            num_blocks = (n + num_threads - 1) / num_threads;
            break;
        case 4:
        case 5:
            num_blocks = (n + (num_threads * 2) - 1) / (num_threads * 2);
            break;
        case 6:
            num_blocks = std::min(128, (int)((n / 4 + num_threads - 1) / num_threads));
            break;
        default:
            num_blocks = 1;
    }
    shared_mem = num_threads * sizeof(float);
}

// ============================================================
// 运行单个测试
// ============================================================
TestResult run_test(int version, float *d_in, float *d_out,
                    float *h_in, float *h_out, unsigned int n,
                    int warmup_iters, int test_iters,
                    void *&d_cub_temp, size_t &cub_temp_bytes) {
    int num_blocks, num_threads, shared_mem;
    get_kernel_config(version, n, num_blocks, num_threads, shared_mem);

    float cpu_sum = reduce_cpu(h_in, n);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 清零输出
    float zero = 0.0f;
    CHECK_CUDA(cudaMemcpyAsync(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 预热
    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));
        switch (version) {
            case 1: reduce_v1<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 2: reduce_v2<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 3: reduce_v3<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 4: reduce_v4<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 5: reduce_v5<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 6: reduce_v6<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
        }
        if (version != 7) CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // CUB 预热
    if (version == 7) {
        cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_in, d_out, n, stream);
        if (d_cub_temp == nullptr) {
            CHECK_CUDA(cudaMalloc(&d_cub_temp, cub_temp_bytes));
        }
        for (int i = 0; i < warmup_iters; i++) {
            cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_in, d_out, n, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < test_iters; i++) {
        if (version != 7) {
            CHECK_CUDA(cudaMemcpyAsync(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        switch (version) {
            case 1: reduce_v1<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 2: reduce_v2<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 3: reduce_v3<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 4: reduce_v4<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 5: reduce_v5<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 6: reduce_v6<<<num_blocks, num_threads, shared_mem, stream>>>(d_in, d_out, n); break;
            case 7: cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_in, d_out, n, stream); break;
        }
        if (version != 7) CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // 验证结果
    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    float gpu_sum = h_out[0];
    float error = fabsf(gpu_sum - cpu_sum) / fabsf(cpu_sum);
    bool correct = error < 0.01f;

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    float time_ms = elapsed_ms / test_iters;
    double bytes = n * sizeof(float);
    double bandwidth_gb_s = (bytes / (time_ms / 1000.0)) / 1e9;

    return {
        VERSIONS[version-1].name,
        VERSIONS[version-1].desc,
        n, time_ms, (float)bandwidth_gb_s, 0.0f, correct, error, 0.0f
    };
}

// ============================================================
// 解析大小字符串
// ============================================================
unsigned int parse_size(const char *str) {
    char *end;
    double val = strtod(str, &end);
    if (*end == 'K' || *end == 'k') val *= 1024;
    else if (*end == 'M' || *end == 'm') val *= 1024 * 1024;
    else if (*end == 'G' || *end == 'g') val *= 1024 * 1024 * 1024;
    return (unsigned int)val;
}

const char* format_size(unsigned int n) {
    static char buf[32];
    if (n >= 1024*1024*1024) {
        snprintf(buf, sizeof(buf), "%.2f GB", n / (1024.0*1024.0*1024.0));
    } else if (n >= 1024*1024) {
        snprintf(buf, sizeof(buf), "%.2f MB", n / (1024.0*1024.0));
    } else if (n >= 1024) {
        snprintf(buf, sizeof(buf), "%.2f KB", n / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%u B", n);
    }
    return buf;
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char **argv) {
    // 默认参数
    std::vector<unsigned int> sizes;
    std::vector<int> versions_to_test;
    int warmup = 10;
    int iterations = 100;
    bool show_help = false;

    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else if (strcmp(argv[i], "--sizes") == 0 || strcmp(argv[i], "-s") == 0) {
            i++;
            while (i < argc && argv[i][0] != '-') {
                sizes.push_back(parse_size(argv[i]));
                i++;
            }
            i--;
        } else if (strcmp(argv[i], "--versions") == 0 || strcmp(argv[i], "-v") == 0) {
            i++;
            while (i < argc && argv[i][0] != '-') {
                if (strcmp(argv[i], "all") == 0) {
                    for (int v = 1; v <= 7; v++) versions_to_test.push_back(v);
                } else {
                    versions_to_test.push_back(atoi(argv[i]));
                }
                i++;
            }
            i--;
        } else if (strcmp(argv[i], "--warmup") == 0 || strcmp(argv[i], "-w") == 0) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 || strcmp(argv[i], "-i") == 0) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--quick") == 0 || strcmp(argv[i], "-q") == 0) {
            warmup = 3;
            iterations = 10;
        }
    }

    // 帮助信息
    if (show_help || argc == 1) {
        printf("CUDA Reduction benchmark\n");
        printf("Usage: %s [options]\n", argv[0]);
        printf("\nOptions:\n");
        printf("  -s, --sizes SIZES       problem sizes (e.g. 1M 10M 100M 1G)\n");
        printf("  -v, --versions VERS     versions 1-7 or all\n");
        printf("  -w, --warmup N          warmup iters (default 10)\n");
        printf("  -i, --iterations N      timed iters (default 100)\n");
        printf("  -q, --quick             quick (warmup=3, iters=10)\n");
        printf("  -h, --help              this help\n");
        printf("\nVersions:\n");
        for (const auto& v : VERSIONS) {
            printf("  %d: %s - %s\n", v.id, v.name.c_str(), v.desc.c_str());
        }
        printf("\nExamples:\n");
        printf("  %s -s 1M 10M 100M -v all\n", argv[0]);
        printf("  %s -s 100M -v 1 3 5 6 -q\n", argv[0]);
        return 0;
    }

    // 默认测试所有
    if (sizes.empty()) {
        sizes = {1024*1024, 10*1024*1024, 100*1024*1024};
    }
    if (versions_to_test.empty()) {
        for (int v = 1; v <= 7; v++) versions_to_test.push_back(v);
    }

    // 获取 GPU 信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    float peak_bw = 2.0f * (prop.memoryClockRate / 1e6f) * (prop.memoryBusWidth / 8.0f);

    printf("================================================================================\n");
    printf("CUDA Reduction benchmark\n");
    printf("================================================================================\n");
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Peak Memory Bandwidth: %.2f GB/s\n", peak_bw);
    printf("Warmup: %d, Test Iterations: %d\n", warmup, iterations);
    printf("================================================================================\n");
    printf("\n");

    // 分配设备内存 (最大尺寸)
    unsigned int max_n = *std::max_element(sizes.begin(), sizes.end());
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, max_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));

    // 分配主机内存
    float *h_in = (float*)malloc(max_n * sizeof(float));
    float *h_out = (float*)malloc(sizeof(float));

    // CUB 临时存储
    void *d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    // 存储结果
    std::vector<TestResult> all_results;

    // 运行测试
    for (unsigned int n : sizes) {
        n = (n / 4) * 4;  // 对齐
        if (n == 0) n = 4;

        printf("Size: %u elements (%s)\n", n, format_size(n * sizeof(float)));
        printf("--------------------------------------------------------------------------------\n");

        // 初始化数据
        init_data(h_in, n, 42);
        CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

        // 首先运行 CUB 作为基准
        float cub_time_ms = 0.0f;
        bool cub_tested = false;
        for (int version : versions_to_test) {
            if (version == 7) {
                try {
                    TestResult cub_result = run_test(7, d_in, d_out, h_in, h_out, n,
                                                     warmup, iterations, d_cub_temp, cub_temp_bytes);
                    cub_time_ms = cub_result.time_ms;
                    cub_result.efficiency = (cub_result.bandwidth_gb_s / peak_bw) * 100.0f;
                    cub_result.speedup_vs_cub = 1.0f;
                    all_results.push_back(cub_result);
                    cub_tested = true;

                    printf("  %s %-20s: %8.4f ms | %7.2f GB/s (%5.1f%%) [ref]\n",
                           "OK", cub_result.name.c_str(), cub_result.time_ms,
                           cub_result.bandwidth_gb_s, cub_result.efficiency);
                } catch (...) {
                    printf("  NO %-20s: error\n", VERSIONS[6].name.c_str());
                }
                break;
            }
        }

        // 运行 v1-v6，按版本号顺序（不是输入顺序）
        for (int version = 1; version <= 6; version++) {
            // 检查该版本是否在测试列表中
            bool should_test = false;
            for (int v : versions_to_test) {
                if (v == version) {
                    should_test = true;
                    break;
                }
            }
            if (!should_test) continue;

            try {
                TestResult result = run_test(version, d_in, d_out, h_in, h_out, n,
                                             warmup, iterations, d_cub_temp, cub_temp_bytes);
                result.efficiency = (result.bandwidth_gb_s / peak_bw) * 100.0f;
                if (cub_tested && cub_time_ms > 0) {
                    result.speedup_vs_cub = result.time_ms / cub_time_ms;
                } else {
                    result.speedup_vs_cub = 0.0f;
                }
                all_results.push_back(result);

                const char* status = result.correct ? "OK" : "NO";
                if (cub_tested) {
                    printf("  %s %-20s: %8.4f ms | %7.2f GB/s (%5.1f%%) | vs CUB: %.2fx\n",
                           status, result.name.c_str(), result.time_ms,
                           result.bandwidth_gb_s, result.efficiency,
                           result.speedup_vs_cub);
                } else {
                    printf("  %s %-20s: %8.4f ms | %7.2f GB/s (%5.1f%%)\n",
                           status, result.name.c_str(), result.time_ms,
                           result.bandwidth_gb_s, result.efficiency);
                }
            } catch (...) {
                printf("  NO %-20s: error\n", VERSIONS[version-1].name.c_str());
            }
        }
        printf("\n");
    }

    // 打印摘要 (以 CUB 为基准对比)
    printf("================================================================================\n");
    printf("Summary (vs CUB)\n");
    printf("================================================================================\n");

    for (unsigned int n : sizes) {
        n = (n / 4) * 4;
        std::vector<TestResult> results_for_size;
        for (const auto& r : all_results) {
            if (r.n == n) results_for_size.push_back(r);
        }
        std::sort(results_for_size.begin(), results_for_size.end(),
                  [](const TestResult& a, const TestResult& b) {
                      if (a.name == "cub") return true;
                      if (b.name == "cub") return false;
                      return a.speedup_vs_cub > b.speedup_vs_cub;
                  });

        float cub_bw = 0.0f;
        for (const auto& r : results_for_size) {
            if (r.name == "cub") {
                cub_bw = r.bandwidth_gb_s;
                break;
            }
        }

        printf("\n%s (%s):\n", format_size(n * sizeof(float)), format_size(n));
        if (cub_bw > 0) {
            printf("  CUB ref: %.2f GB/s\n", cub_bw);
        }
        printf("  %-20s %10s %12s %10s %12s %6s\n",
               "ver", "time(ms)", "BW(GB/s)", "eff%%", "vs CUB", "stat");
        printf("  --------------------------------------------------------------------------------\n");
        for (const auto& r : results_for_size) {
            const char* status = r.correct ? "OK" : "NO";
            char vs_cub_str[32];
            if (r.name == "cub") {
                snprintf(vs_cub_str, sizeof(vs_cub_str), "ref");
            } else if (r.speedup_vs_cub > 0) {
                if (r.speedup_vs_cub >= 1.0f) {
                    snprintf(vs_cub_str, sizeof(vs_cub_str), "%.2fx slow", r.speedup_vs_cub);
                } else {
                    snprintf(vs_cub_str, sizeof(vs_cub_str), "%.2fx fast", 1.0f / r.speedup_vs_cub);
                }
            } else {
                snprintf(vs_cub_str, sizeof(vs_cub_str), "N/A");
            }
            printf("  %-20s %10.4f %12.2f %9.1f%% %12s %6s\n",
                   r.name.c_str(), r.time_ms, r.bandwidth_gb_s, r.efficiency, vs_cub_str, status);
        }
    }

    // 整体最佳
    auto best = *std::max_element(all_results.begin(), all_results.end(),
                                  [](const TestResult& a, const TestResult& b) {
                                      return a.bandwidth_gb_s < b.bandwidth_gb_s;
                                  });
    printf("\nBest overall: %s @ %s: %.2f GB/s (%.1f%%)\n",
           best.name.c_str(), format_size(best.n * sizeof(float)),
           best.bandwidth_gb_s, best.efficiency);

    // 最佳手写版本 vs CUB
    auto best_handwritten = *std::max_element(all_results.begin(), all_results.end(),
                                              [](const TestResult& a, const TestResult& b) {
                                                  if (a.name == "cub") return false;
                                                  if (b.name == "cub") return true;
                                                  return a.bandwidth_gb_s < b.bandwidth_gb_s;
                                              });
    if (best_handwritten.name != "cub" && best_handwritten.speedup_vs_cub > 0) {
        if (best_handwritten.speedup_vs_cub >= 1.0f) {
            printf("Best handwritten (%s): %.2fx slower than CUB\n",
                   best_handwritten.name.c_str(), best_handwritten.speedup_vs_cub);
        } else {
            printf("Best handwritten (%s): %.1f%% of CUB BW\n",
                   best_handwritten.name.c_str(), (1.0f / best_handwritten.speedup_vs_cub) * 100.0f);
        }
    }
    printf("================================================================================\n");

    // 清理
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    if (d_cub_temp) CHECK_CUDA(cudaFree(d_cub_temp));

    return 0;
}

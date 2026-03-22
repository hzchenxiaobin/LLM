/**
 * CUDA Reduction 性能测试 - 独立可执行文件
 *
 * 编译: nvcc -O3 -o benchmark benchmark.cu
 * 运行: ./benchmark --sizes 1M 10M 100M
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// 版本 1: 朴素版本 (Interleaved Addressing)
// ============================================================
__global__ void reduce_v1(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ============================================================
// 版本 2: Strided Index
// ============================================================
__global__ void reduce_v2(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ============================================================
// 版本 3: Sequential Addressing (无 Bank Conflict)
// ============================================================
__global__ void reduce_v3(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ============================================================
// 版本 4: First Add During Load
// ============================================================
__global__ void reduce_v4(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ============================================================
// 版本 5: Warp Shuffle
// ============================================================
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v5(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        if (blockDim.x >= 64) sum = sdata[tid] + sdata[tid + 32];
        else sum = sdata[tid];
        sum = warpReduceSum(sum);
        if (tid == 0) atomicAdd(g_odata, sum);
    }
}

// ============================================================
// 版本 6: Vectorized Memory Access
// ============================================================
__global__ void reduce_v6(float *g_idata, float *g_odata, unsigned int n) {
    float4 *g_idata_f4 = reinterpret_cast<float4*>(g_idata);
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (i < n / 4) {
        float4 vec = g_idata_f4[i];
        sum += vec.x + vec.y + vec.z + vec.w;
        i += stride;
    }
    i = i * 4;
    while (i < n) {
        sum += g_idata[i];
        i++;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        if (blockDim.x >= 64) sum = sdata[tid] + sdata[tid + 32];
        else sum = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) atomicAdd(g_odata, sum);
    }
}

// ============================================================
// CPU 参考实现
// ============================================================
float reduce_cpu(const float *data, unsigned int n) {
    double sum = 0.0;
    for (unsigned int i = 0; i < n; i++) sum += data[i];
    return (float)sum;
}

void init_data(float *data, unsigned int n, unsigned int seed) {
    srand(seed);
    for (unsigned int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

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
};

struct VersionInfo {
    int id;
    std::string name;
    std::string desc;
};

const VersionInfo VERSIONS[] = {
    {1, "v1_interleaved", "朴素版本 - Warp Divergence"},
    {2, "v2_strided", "解决分支发散 - Bank Conflict"},
    {3, "v3_sequential", "解决 Bank Conflict"},
    {4, "v4_first_add", "加载时相加"},
    {5, "v5_warp_shuffle", "Warp Shuffle"},
    {6, "v6_vectorized", "向量化访存"},
    {7, "cub", "NVIDIA CUB 库"},
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
        n, time_ms, (float)bandwidth_gb_s, 0.0f, correct, error
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
        printf("CUDA Reduction 性能测试\n");
        printf("用法: %s [选项]\n", argv[0]);
        printf("\n选项:\n");
        printf("  -s, --sizes SIZES       测试数据规模 (如: 1M 10M 100M 1G)\n");
        printf("  -v, --versions VERS     测试版本 (1-7 或 all)\n");
        printf("  -w, --warmup N          预热迭代次数 (默认: 10)\n");
        printf("  -i, --iterations N      测试迭代次数 (默认: 100)\n");
        printf("  -q, --quick             快速模式 (warmup=3, iterations=10)\n");
        printf("  -h, --help              显示帮助\n");
        printf("\n版本说明:\n");
        for (const auto& v : VERSIONS) {
            printf("  %d: %s - %s\n", v.id, v.name.c_str(), v.desc.c_str());
        }
        printf("\n示例:\n");
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
    printf("CUDA Reduction 性能测试\n");
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

        printf("数据规模: %u 元素 (%s)\n", n, format_size(n * sizeof(float)));
        printf("--------------------------------------------------------------------------------\n");

        // 初始化数据
        init_data(h_in, n, 42);
        CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

        for (int version : versions_to_test) {
            if (version < 1 || version > 7) continue;

            try {
                TestResult result = run_test(version, d_in, d_out, h_in, h_out, n,
                                             warmup, iterations, d_cub_temp, cub_temp_bytes);
                result.efficiency = (result.bandwidth_gb_s / peak_bw) * 100.0f;
                all_results.push_back(result);

                const char* status = result.correct ? "✓" : "✗";
                printf("  %s %-20s: %8.4f ms | %7.2f GB/s (%5.1f%%)\n",
                       status, result.name.c_str(), result.time_ms,
                       result.bandwidth_gb_s, result.efficiency);
            } catch (...) {
                printf("  ✗ %-20s: 错误\n", VERSIONS[version-1].name.c_str());
            }
        }
        printf("\n");
    }

    // 打印摘要
    printf("================================================================================\n");
    printf("性能测试摘要\n");
    printf("================================================================================\n");

    for (unsigned int n : sizes) {
        n = (n / 4) * 4;
        std::vector<TestResult> results_for_size;
        for (const auto& r : all_results) {
            if (r.n == n) results_for_size.push_back(r);
        }
        std::sort(results_for_size.begin(), results_for_size.end(),
                  [](const TestResult& a, const TestResult& b) {
                      return a.bandwidth_gb_s > b.bandwidth_gb_s;
                  });

        printf("\n%s (%s):\n", format_size(n * sizeof(float)), format_size(n));
        printf("  %-20s %10s %12s %10s %6s\n", "版本", "时间(ms)", "带宽(GB/s)", "效率", "状态");
        printf("  --------------------------------------------------------------------------------\n");
        for (const auto& r : results_for_size) {
            const char* status = r.correct ? "✓" : "✗";
            printf("  %-20s %10.4f %12.2f %9.1f%% %6s\n",
                   r.name.c_str(), r.time_ms, r.bandwidth_gb_s, r.efficiency, status);
        }
    }

    // 整体最佳
    auto best = *std::max_element(all_results.begin(), all_results.end(),
                                  [](const TestResult& a, const TestResult& b) {
                                      return a.bandwidth_gb_s < b.bandwidth_gb_s;
                                  });
    printf("\n整体最佳: %s @ %s: %.2f GB/s (%.1f%%)\n",
           best.name.c_str(), format_size(best.n * sizeof(float)),
           best.bandwidth_gb_s, best.efficiency);
    printf("================================================================================\n");

    // 清理
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    if (d_cub_temp) CHECK_CUDA(cudaFree(d_cub_temp));

    return 0;
}

/**
 * CUDA Reduction 算子性能测试
 * 包含 6 个版本的实现，从朴素版本到向量化访存
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cfloat>

// 错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================
// 版本 1: 朴素版本 (Interleaved Addressing)
// 问题: Warp Divergence
// ============================================
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

    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

// ============================================
// 版本 2: 解决分支发散 (Strided Index)
// 问题: Bank Conflict
// ============================================
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

// ============================================
// 版本 3: 解决 Bank Conflict (Sequential Addressing)
// ============================================
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

// ============================================
// 版本 4: 提高指令吞吐与隐藏延迟 (First Add During Load)
// ============================================
__global__ void reduce_v4(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
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

// ============================================
// 版本 5: Warp Shuffle (终结 Shared Memory)
// ============================================
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

// ============================================
// 版本 6: 向量化访存 (Vectorized Memory Access)
// ============================================
__global__ void reduce_v6(float *g_idata, float *g_odata, unsigned int n) {
    float4 *g_idata_f4 = reinterpret_cast<float4*>(g_idata);

    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop
    while (i < n / 4) {
        float4 vec = g_idata_f4[i];
        sum += vec.x + vec.y + vec.z + vec.w;
        i += stride;
    }

    // 处理剩余元素
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

// ============================================
// CUB 库基准版本 (如果可用)
// ============================================
#include <cub/cub.cuh>

void reduce_cub(float *d_in, float *d_out, unsigned int n, void *&d_temp_storage, size_t &temp_storage_bytes, cudaStream_t stream) {
    // 首次调用确定临时存储大小
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
}

// ============================================
// CPU 参考实现
// ============================================
float reduce_cpu(const float *data, unsigned int n) {
    double sum = 0.0;
    for (unsigned int i = 0; i < n; i++) {
        sum += data[i];
    }
    return (float)sum;
}

// ============================================
// 初始化函数
// ============================================
extern "C" {

void init_data(float *data, unsigned int n, unsigned int seed) {
    srand(seed);
    for (unsigned int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

float get_bandwidth_gb_s(unsigned int n, float time_ms) {
    double bytes = n * sizeof(float);
    double seconds = time_ms / 1000.0;
    return (float)(bytes / seconds / 1e9);
}

// 运行指定版本的 kernel
float run_kernel(int version, float *d_in, float *d_out, unsigned int n,
                 int num_blocks, int num_threads, int shared_mem_bytes,
                 cudaStream_t stream, int warmup_iters, int test_iters) {

    // 清零输出
    float zero = 0.0f;
    CHECK_CUDA(cudaMemcpyAsync(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 预热
    for (int i = 0; i < warmup_iters; i++) {
        switch (version) {
            case 1: reduce_v1<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 2: reduce_v2<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 3: reduce_v3<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 4: reduce_v4<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 5: reduce_v5<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 6: reduce_v6<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 计时测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < test_iters; i++) {
        // 清零输出
        CHECK_CUDA(cudaMemcpyAsync(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));

        switch (version) {
            case 1: reduce_v1<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 2: reduce_v2<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 3: reduce_v3<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 4: reduce_v4<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 5: reduce_v5<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
            case 6: reduce_v6<<<num_blocks, num_threads, shared_mem_bytes, stream>>>(d_in, d_out, n); break;
        }
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms / test_iters;  // 平均每次迭代时间
}

// CUB 版本运行函数
float run_cub(float *d_in, float *d_out, unsigned int n,
              void *&d_temp_storage, size_t &temp_storage_bytes,
              cudaStream_t stream, int warmup_iters, int test_iters) {

    // 首次调用确定临时存储大小
    if (d_temp_storage == nullptr) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
        CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    // 预热
    for (int i = 0; i < warmup_iters; i++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 计时测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < test_iters; i++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms / test_iters;
}

// 获取 GPU 信息
void get_gpu_info(char *name, int *major, int *minor, float *peak_bandwidth_gb_s) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    strcpy(name, prop.name);
    *major = prop.major;
    *minor = prop.minor;

    // 计算峰值带宽 (理论值)
    // 带宽 = 显存频率 * 位宽 / 8
    float mem_clock_ghz = prop.memoryClockRate / 1e6f;
    int bus_width_bits = prop.memoryBusWidth;
    *peak_bandwidth_gb_s = 2.0f * mem_clock_ghz * (bus_width_bits / 8.0f);
}

// 辅助函数：计算配置参数
void get_kernel_config(int version, unsigned int n, int *num_blocks, int *num_threads, int *shared_mem_bytes) {
    switch (version) {
        case 1:
        case 2:
        case 3:
            *num_threads = 256;
            *num_blocks = (n + *num_threads - 1) / *num_threads;
            *shared_mem_bytes = *num_threads * sizeof(float);
            break;
        case 4:
        case 5:
            *num_threads = 256;
            *num_blocks = (n + (*num_threads * 2) - 1) / (*num_threads * 2);
            *shared_mem_bytes = *num_threads * sizeof(float);
            break;
        case 6:
            *num_threads = 256;
            // 对于向量化版本，使用 grid-stride loop，blocks 数量可以更少
            *num_blocks = min(128, (int)((n / 4 + *num_threads - 1) / *num_threads));
            *shared_mem_bytes = *num_threads * sizeof(float);
            break;
    }
}

} // extern "C"

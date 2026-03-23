// Softmax Performance Benchmark for RTX 5090
// Compares 5 versions of softmax implementation

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ==================== V1: Naive Softmax (3 kernels) ====================

__global__ void kernel_max_v1(const float* input, float* max_vals, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float local_max = -INFINITY;
    for (int i = 0; i < N; i++) {
        local_max = fmaxf(local_max, input[row * N + i]);
    }
    max_vals[row] = local_max;
}

__global__ void kernel_sum_v1(const float* input, const float* max_vals, float* sum_vals, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float row_max = max_vals[row];
    float local_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        local_sum += expf(input[row * N + i] - row_max);
    }
    sum_vals[row] = local_sum;
}

__global__ void kernel_div_v1(const float* input, const float* max_vals, const float* sum_vals, 
                               float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float row_max = max_vals[row];
    float row_sum = sum_vals[row];
    for (int i = 0; i < N; i++) {
        output[row * N + i] = expf(input[row * N + i] - row_max) / row_sum;
    }
}

void softmax_v1(const float* d_input, float* d_output, float* d_max, float* d_sum, int M, int N) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    
    kernel_max_v1<<<blocks, threads>>>(d_input, d_max, M, N);
    kernel_sum_v1<<<blocks, threads>>>(d_input, d_max, d_sum, M, N);
    kernel_div_v1<<<blocks, threads>>>(d_input, d_max, d_sum, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== V2: Block-level with Shared Memory ====================

__global__ void softmax_v2_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    const float* x = input + row * N;
    float* y = output + row * N;

    extern __shared__ float sdata[];

    // 1. Find max
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // 2. Compute sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // 3. Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}

void softmax_v2(const float* d_input, float* d_output, int M, int N) {
    int threads = 256;
    int blocks = M;
    size_t shared_mem = threads * sizeof(float);
    softmax_v2_kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== V3: Warp-level Reduction ====================

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_v3_warp_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    // 1. Find max
    float local_max = -INFINITY;
    for (int i = lane_id; i < N; i += 32) {
        local_max = fmaxf(local_max, x[i]);
    }
    float row_max = warpReduceMax(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // 2. Compute sum
    float local_sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        local_sum += expf(x[i] - row_max);
    }
    float row_sum = warpReduceSum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // 3. Write back
    for (int i = lane_id; i < N; i += 32) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}

void softmax_v3(const float* d_input, float* d_output, int M, int N) {
    int threads = 128;  // 4 warps per block
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v3_warp_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== V4: Vectorized Memory Access (float4) ====================

__global__ void softmax_v4_vectorized_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float4* x_vec = reinterpret_cast<const float4*>(input + row * N);
    float4* y_vec = reinterpret_cast<float4*>(output + row * N);

    int N_vec = N / 4;

    // 1. Find max
    float local_max = -INFINITY;
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        local_max = fmaxf(local_max, val.x);
        local_max = fmaxf(local_max, val.y);
        local_max = fmaxf(local_max, val.z);
        local_max = fmaxf(local_max, val.w);
    }
    float row_max = warpReduceMax(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // 2. Compute sum
    float local_sum = 0.0f;
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        local_sum += expf(val.x - row_max);
        local_sum += expf(val.y - row_max);
        local_sum += expf(val.z - row_max);
        local_sum += expf(val.w - row_max);
    }
    float row_sum = warpReduceSum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // 3. Write back
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        float4 out_val;
        out_val.x = expf(val.x - row_max) / row_sum;
        out_val.y = expf(val.y - row_max) / row_sum;
        out_val.z = expf(val.z - row_max) / row_sum;
        out_val.w = expf(val.w - row_max) / row_sum;
        y_vec[i] = out_val;
    }
}

void softmax_v4(const float* d_input, float* d_output, int M, int N) {
    int threads = 128;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v4_vectorized_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== V5: Online Softmax ====================

__global__ void softmax_v5_online_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Single pass: update max and sum together
    for (int i = lane_id; i < N; i += 32) {
        float val = x[i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }

    // Warp reduce with online correction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);

        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max) + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    float row_max = __shfl_sync(0xffffffff, local_max, 0);
    float row_sum = __shfl_sync(0xffffffff, local_sum, 0);

    // Second pass: write results
    for (int i = lane_id; i < N; i += 32) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}

void softmax_v5(const float* d_input, float* d_output, int M, int N) {
    int threads = 128;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v5_online_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== V5+Vectorized: Ultimate Version ====================

__global__ void softmax_v5_vec_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float4* x_vec = reinterpret_cast<const float4*>(input + row * N);
    float4* y_vec = reinterpret_cast<float4*>(output + row * N);
    int N_vec = N / 4;

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Single pass with float4
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        
        // Process x component
        float new_max = fmaxf(local_max, val.x);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.x - new_max);
        local_max = new_max;
        
        // Process y component
        new_max = fmaxf(local_max, val.y);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.y - new_max);
        local_max = new_max;
        
        // Process z component
        new_max = fmaxf(local_max, val.z);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.z - new_max);
        local_max = new_max;
        
        // Process w component
        new_max = fmaxf(local_max, val.w);
        local_sum = local_sum * expf(local_max - new_max) + expf(val.w - new_max);
        local_max = new_max;
    }

    // Warp reduce with online correction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);

        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max) + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    float row_max = __shfl_sync(0xffffffff, local_max, 0);
    float row_sum = __shfl_sync(0xffffffff, local_sum, 0);

    // Second pass: write results
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        float4 out_val;
        out_val.x = expf(val.x - row_max) / row_sum;
        out_val.y = expf(val.y - row_max) / row_sum;
        out_val.z = expf(val.z - row_max) / row_sum;
        out_val.w = expf(val.w - row_max) / row_sum;
        y_vec[i] = out_val;
    }
}

void softmax_v5_vec(const float* d_input, float* d_output, int M, int N) {
    int threads = 128;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    softmax_v5_vec_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== CPU Reference Implementation ====================

void softmax_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        // Find max
        float row_max = -INFINITY;
        for (int i = 0; i < N; i++) {
            row_max = fmaxf(row_max, input[row * N + i]);
        }
        
        // Compute sum of exp
        float row_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            row_sum += expf(input[row * N + i] - row_max);
        }
        
        // Normalize
        for (int i = 0; i < N; i++) {
            output[row * N + i] = expf(input[row * N + i] - row_max) / row_sum;
        }
    }
}

// ==================== Utility Functions ====================

void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

float max_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
    }
    return max_err;
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ==================== Benchmark Functions ====================

typedef void (*softmax_func_t)(const float*, float*, int, int);
typedef void (*softmax_v1_t)(const float*, float*, float*, float*, int, int);

struct BenchmarkResult {
    const char* name;
    double time_ms;
    double bandwidth_gbps;
    float max_error;
    bool passed;
};

void run_benchmark(const char* name, 
                   softmax_v1_t func_v1,
                   softmax_func_t func_v2_plus,
                   const float* d_input, float* d_output, 
                   float* d_max, float* d_sum,
                   float* h_output, const float* h_ref,
                   int M, int N, int warmup_iters, int benchmark_iters,
                   size_t data_size_bytes,
                   BenchmarkResult* result) {
    
    printf("Benchmarking %s...\n", name);
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        if (func_v1 != nullptr) {
            func_v1(d_input, d_output, d_max, d_sum, M, N);
        } else {
            func_v2_plus(d_input, d_output, M, N);
        }
    }
    
    // Benchmark
    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();
    
    for (int i = 0; i < benchmark_iters; i++) {
        if (func_v1 != nullptr) {
            func_v1(d_input, d_output, d_max, d_sum, M, N);
        } else {
            func_v2_plus(d_input, d_output, M, N);
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();
    
    // Calculate metrics
    result->name = name;
    result->time_ms = (end - start) / benchmark_iters;
    
    // Bandwidth: read input once, write output once = 2 * M * N * sizeof(float)
    // V1 reads 3 times, but we'll approximate
    double total_bytes = 2.0 * M * N * sizeof(float);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;
    
    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size_bytes, cudaMemcpyDeviceToHost));
    result->max_error = max_error(h_output, h_ref, M * N);
    result->passed = result->max_error < 1e-4;
    
    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASSED" : "FAILED");
}

// ==================== Main ====================

int main(int argc, char** argv) {
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("===============================================\n");
    printf("  Softmax Performance Benchmark\n");
    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("  Memory Clock: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  Theoretical Bandwidth: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * prop.memoryBusWidth / (8.0 * 1e6));
    printf("===============================================\n\n");
    
    // Test configurations (M, N) - typical LLM attention shapes
    int test_configs[][2] = {
        {1024, 128},      // Small: batch=32, heads=32, seq=128
        {1024, 512},      // Medium: batch=32, heads=32, seq=512
        {1024, 1024},     // Large: batch=32, heads=32, seq=1024
        {1024, 2048},     // XL: batch=32, heads=32, seq=2048
        {4096, 1024},     // Batch large
        {1024, 4096},     // Very long sequence
    };
    int num_configs = sizeof(test_configs) / sizeof(test_configs[0]);
    
    const int warmup_iters = 10;
    const int benchmark_iters = 100;
    
    for (int cfg = 0; cfg < num_configs; cfg++) {
        int M = test_configs[cfg][0];
        int N = test_configs[cfg][1];
        
        // Skip vectorized versions if N is not divisible by 4
        bool can_vectorize = (N % 4 == 0);
        
        printf("\n========================================\n");
        printf("Configuration: M=%d, N=%d (rows x cols)\n", M, N);
        printf("Total elements: %d (%.2f MB)\n", M * N, M * N * sizeof(float) / (1024.0 * 1024));
        printf("========================================\n\n");
        
        size_t data_size = M * N * sizeof(float);
        size_t row_size = M * sizeof(float);
        
        // Allocate host memory
        float *h_input = (float*)malloc(data_size);
        float *h_output = (float*)malloc(data_size);
        float *h_ref = (float*)malloc(data_size);
        
        // Initialize data
        srand(42);
        init_random(h_input, M * N);
        
        // Compute CPU reference
        softmax_cpu(h_input, h_ref, M, N);
        
        // Allocate device memory
        float *d_input, *d_output, *d_max, *d_sum;
        CUDA_CHECK(cudaMalloc(&d_input, data_size));
        CUDA_CHECK(cudaMalloc(&d_output, data_size));
        CUDA_CHECK(cudaMalloc(&d_max, row_size));
        CUDA_CHECK(cudaMalloc(&d_sum, row_size));
        
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));
        
        // Run benchmarks
        BenchmarkResult results[7];
        int result_idx = 0;
        
        // V1: Naive (3 kernels)
        run_benchmark("V1: Naive (3 kernels)", 
                      softmax_v1, nullptr,
                      d_input, d_output, d_max, d_sum,
                      h_output, h_ref, M, N, 
                      warmup_iters, benchmark_iters, data_size,
                      &results[result_idx++]);
        
        // V2: Block-level Shared Memory
        run_benchmark("V2: Block Shared Memory", 
                      nullptr, softmax_v2,
                      d_input, d_output, d_max, d_sum,
                      h_output, h_ref, M, N,
                      warmup_iters, benchmark_iters, data_size,
                      &results[result_idx++]);
        
        // V3: Warp-level Reduction
        run_benchmark("V3: Warp-level Reduction", 
                      nullptr, softmax_v3,
                      d_input, d_output, d_max, d_sum,
                      h_output, h_ref, M, N,
                      warmup_iters, benchmark_iters, data_size,
                      &results[result_idx++]);
        
        // V4: Vectorized (if possible)
        if (can_vectorize) {
            run_benchmark("V4: Vectorized (float4)", 
                          nullptr, softmax_v4,
                          d_input, d_output, d_max, d_sum,
                          h_output, h_ref, M, N,
                          warmup_iters, benchmark_iters, data_size,
                          &results[result_idx++]);
        }
        
        // V5: Online Softmax
        run_benchmark("V5: Online Softmax", 
                      nullptr, softmax_v5,
                      d_input, d_output, d_max, d_sum,
                      h_output, h_ref, M, N,
                      warmup_iters, benchmark_iters, data_size,
                      &results[result_idx++]);
        
        // V5+Vec: Ultimate (Online + Vectorized)
        if (can_vectorize) {
            run_benchmark("V5+Vec: Ultimate (Online + float4)", 
                          nullptr, softmax_v5_vec,
                          d_input, d_output, d_max, d_sum,
                          h_output, h_ref, M, N,
                          warmup_iters, benchmark_iters, data_size,
                          &results[result_idx++]);
        }
        
        // Print summary table
        printf("\n--- Summary for M=%d, N=%d ---\n", M, N);
        printf("%-40s %10s %12s %12s\n", "Version", "Time(ms)", "BW(GB/s)", "Error");
        printf("--------------------------------------------------------------------------------\n");
        for (int i = 0; i < result_idx; i++) {
            printf("%-40s %10.4f %12.2f %12.2e %s\n",
                   results[i].name, results[i].time_ms, 
                   results[i].bandwidth_gbps, results[i].max_error,
                   results[i].passed ? "✓" : "✗");
        }
        
        // Find best performing
        double best_time = results[0].time_ms;
        const char* best_name = results[0].name;
        for (int i = 1; i < result_idx; i++) {
            if (results[i].time_ms < best_time) {
                best_time = results[i].time_ms;
                best_name = results[i].name;
            }
        }
        printf("\nBest performer: %s (%.4f ms)\n", best_name, best_time);
        
        // Cleanup
        free(h_input);
        free(h_output);
        free(h_ref);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_max));
        CUDA_CHECK(cudaFree(d_sum));
    }
    
    printf("\n\nBenchmark complete!\n");
    return 0;
}

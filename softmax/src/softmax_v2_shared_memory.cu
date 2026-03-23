// V2: Block-level Shared Memory
// Kernel fusion with shared memory reduction

#include "softmax_common.h"

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

// Host function for V2 - Optimized for RTX 5090 (Blackwell)
// Use 512 threads per block for better utilization of RTX 5090's SMs
void softmax_v2(const float* d_input, float* d_output, int M, int N) {
    // RTX 5090: larger block size for higher occupancy and better memory coalescing
    int threads = 512;  // Increased from 256 for RTX 5090
    int blocks = M;
    size_t shared_mem = threads * sizeof(float);
    softmax_v2_kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void run_benchmark_v2(const float* d_input, float* d_output,
                    float* h_output, const float* h_ref,
                    int M, int N, int warmup_iters, int benchmark_iters,
                    size_t data_size_bytes, BenchmarkResult* result) {

    printf("Benchmarking V2: Block Shared Memory...\n");

    for (int i = 0; i < warmup_iters; i++) {
        softmax_v2(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double start = get_time_ms();

    for (int i = 0; i < benchmark_iters; i++) {
        softmax_v2(d_input, d_output, M, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    result->name = "V2: Block Shared Memory";
    result->time_ms = (end - start) / benchmark_iters;

    double total_bytes = 2.0 * M * N * sizeof(float);
    result->bandwidth_gbps = (total_bytes / (result->time_ms / 1000.0)) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size_bytes, cudaMemcpyDeviceToHost));
    result->max_error = max_error(h_output, h_ref, M * N);
    result->passed = result->max_error < 1e-4;

    printf("  Time: %.4f ms | Bandwidth: %.2f GB/s | Max Error: %.2e | %s\n\n",
           result->time_ms, result->bandwidth_gbps, result->max_error,
           result->passed ? "PASSED" : "FAILED");
}

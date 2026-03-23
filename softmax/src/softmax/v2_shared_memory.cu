// V2: Block-level Shared Memory
// Kernel fusion with shared memory reduction

#include "../include/softmax_common.h"

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

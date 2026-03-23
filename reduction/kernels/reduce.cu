#include <cuda_runtime.h>

__global__ void reduction(const float* input, float* output, int N) {
    extern __shared__ float data[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + tid;
    
    float preSum = index < N ? input[index] : 0.0;
    if(index + blockDim.x < N) {
        preSum += input[index + blockDim.x];
    }
    data[tid] = preSum;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            data[tid] += data[tid + s];
        }


        __syncthreads();
    }

    if(tid == 0) {
        atomicAdd(output, data[0]);
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 block(256);
    dim3 grid((N + block.x * 2 - 1) / block.x * 2);
    reduction<<<grid, block, 2 * block.x * sizeof(float)>>>(input, output, N);

    cudaDeviceSynchronize();
}
/**
 * CUDA Reduction 算子 - Kernel 运行调度
 */

#include "reduction_kernels.h"

// ============================================
// 运行指定版本的 kernel
// ============================================
extern "C" {

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

// ============================================
// CUB 版本运行函数
// ============================================
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

} // extern "C"

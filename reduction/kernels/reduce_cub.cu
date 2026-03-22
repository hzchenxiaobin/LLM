/**
 * CUDA Reduction 算子 - CUB 库基准版本
 */

#include "reduction_kernels.h"

// ============================================
// CUB 库基准版本
// ============================================
void reduce_cub(float *d_in, float *d_out, unsigned int n,
                void *&d_temp_storage, size_t &temp_storage_bytes,
                cudaStream_t stream) {
    // 首次调用确定临时存储大小
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
}

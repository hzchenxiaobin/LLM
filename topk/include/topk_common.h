#ifndef TOPK_COMMON_H
#define TOPK_COMMON_H

// ==========================================
// 错误检查宏
// ==========================================
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// 最大支持的 K 值，用于静态分配寄存器数组大小
#define MAX_K 32

#endif // TOPK_COMMON_H

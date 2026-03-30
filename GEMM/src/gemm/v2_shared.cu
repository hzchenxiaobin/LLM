#include "../common.h"
#include "gemm_kernels.h"


// ==========================================
// 算子 2: 共享内存分块 GEMM (Shared Memory Tiling)
// ==========================================

template <int BLOCK_SIZE>
__global__ void sgemm_shared_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 1. 申请共享内存 (Shared Memory)
    // 每个 Block 维护自己的一块 sA 和 sB，大小为 BLOCK_SIZE * BLOCK_SIZE
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // 2. 获取线程索引和计算全局坐标
    int tx = threadIdx.x; // Block 内部列索引
    int ty = threadIdx.y; // Block 内部行索引

    // 当前线程负责计算矩阵 C 的具体行号和列号
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // 寄存器变量，用于累加当前线程负责的 C 元素的点积结果
    float tmp = 0.0f;

    // 3. 沿 K 维度分块滑动（步长为 BLOCK_SIZE）
    // (K + BLOCK_SIZE - 1) / BLOCK_SIZE 实现了向上取整，确保覆盖所有 K
    for (int k_step = 0; k_step < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k_step) {
        
        // 当前滑动窗口在 K 维度上的起始偏移量
        int k_offset = k_step * BLOCK_SIZE;

        // ---------------------------------------------------
        // 步骤 A：协同加载数据到共享内存
        // ---------------------------------------------------
        // 将矩阵 A 的数据加载到 sA 中，并进行边界检查
        if (row < M && k_offset + tx < K) {
            sA[ty][tx] = A[row * K + k_offset + tx];
        } else {
            // 越界部分补零，防止影响后续相乘
            sA[ty][tx] = 0.0f;
        }

        // 将矩阵 B 的数据加载到 sB 中，并进行边界检查
        if (k_offset + ty < K && col < N) {
            sB[ty][tx] = B[(k_offset + ty) * N + col];
        } else {
            // 越界部分补零
            sB[ty][tx] = 0.0f;
        }

        // 【关键】同步线程：必须等待 Block 内所有线程都把数据加载到共享内存完毕
        __syncthreads();

        // ---------------------------------------------------
        // 步骤 B：在极速的共享内存中完成小块矩阵的乘加
        // ---------------------------------------------------
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += sA[ty][k] * sB[k][tx];
        }

        // 【关键】同步线程：必须等待 Block 内所有线程都算完这一轮，才能进入下一个 k_step 
        // 否则跑得快的线程会直接在下一次迭代中覆盖掉 sA 和 sB，导致数据错误
        __syncthreads();
    }

    // 4. 将最终结果写回全局内存矩阵 C
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// 包装函数：处理 Grid/Block 划分并调用 Kernel，符合测试框架中的 GemmFunc 签名
void run_sgemm_shared(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 经典的 32x32 线程块大小
    const int BLOCK_SIZE = 32;
    
    // 定义 Block 内部的线程排列 (32, 32)
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    // 定义 Grid 中 Block 的排列，使用向上取整确保覆盖完整的矩阵 C
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    // 启动 Kernel
    sgemm_shared_kernel<BLOCK_SIZE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
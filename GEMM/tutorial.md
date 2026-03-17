CUDA GEMM 算子性能优化教程 (面向 RTX 5090)

矩阵乘法 $C = A \times B$ 是 CUDA 优化中最经典的案例。假设我们要计算两个大小为 $M \times K$ 和 $K \times N$ 的矩阵乘法，得到 $M \times N$ 的矩阵 C。为了简化，我们以单精度浮点数 (FP32) 为例（即 SGEMM），并假设矩阵是按行主序（Row-Major）存储的。

在 RTX 5090 这样的顶级显卡上，优化的核心思想是：克服访存瓶颈，提高计算访存比（Arithmetic Intensity），最终利用专用硬件（Tensor Cores）。

优化第一步：朴素的 GEMM (Naive GEMM)

最基础的实现方式是让网格（Grid）中的每一个线程（Thread）负责计算矩阵 C 中的一个元素。

CUDA Kernel 代码：

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 计算当前线程负责的矩阵 C 的行号和列号
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 列 (N)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 行 (M)

    if (x < N && y < M) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            // A 的读取是连续的（合并访存），但 B 的读取是跳跃的（非合并访存）
            tmp += A[y * K + i] * B[i * N + x];
        }
        C[y * N + x] = alpha * tmp + beta * C[y * N + x];
    }
}


性能瓶颈分析：

极低的计算访存比：为了计算 1 个乘加（FMA），需要从全局内存（Global Memory）读取 2 个浮点数。RTX 5090 的算力远超其显存带宽（即使是 GDDR7 也是有极限的），这会导致计算单元都在“饿着肚子”等数据，性能极差（通常只有理论峰值的几十分之一）。

内存访问不合并（Uncoalesced Memory Access）：在读取矩阵 B 时，相邻线程读取的内存地址不连续，极大地浪费了显存带宽。

优化第二步：共享内存分块 (Shared Memory Tiling)

为了减少对全局内存的访问，我们可以利用每个 SM 内速度极快的共享内存（Shared Memory）。
我们将矩阵 C 分解成大小为 BLOCK_SIZE x BLOCK_SIZE 的小块。对于每个 C 块，我们分阶段将 A 和 B 对应的块加载到共享内存中，然后在共享内存中完成乘法，最后累加回 C。

CUDA Kernel 代码：

template <int BLOCK_SIZE>
__global__ void sgemm_shared_mem(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // 申请共享内存
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 当前线程负责的全局 C 的坐标
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float tmp = 0.0f;

    // 沿着 K 维度分块滑动
    for (int i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        // 协同加载数据到共享内存
        if (row < M && i * BLOCK_SIZE + tx < K)
            sA[ty][tx] = A[row * K + i * BLOCK_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;

        if (i * BLOCK_SIZE + ty < K && col < N)
            sB[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        // 必须同步，确保整个 Block 都加载完毕
        __syncthreads();

        // 在共享内存中进行矩阵相乘
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += sA[ty][k] * sB[k][tx];
        }

        // 再次同步，确保当前块计算完毕，才可以进行下一次迭代覆盖 sA 和 sB
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}


优化效果：
如果 BLOCK_SIZE=32，我们从全局内存读取数据的次数减少了约 32 倍！在早期的 GPU 上，这能达到 60% 左右的峰值性能，但在 RTX 5090 上这还远远不够。

优化第三步：一维 / 二维寄存器分块 (Thread/Register Tiling)

虽然共享内存很快，但它仍然比**寄存器（Register）**慢。在第二步中，每个线程在共享内存中循环计算 1 个 C 的元素，这会导致大量的共享内存读取指令。
终极理念是：让每个线程计算多个 C 的元素。

我们可以分配一个更大的 Block（例如 128x128），将其放入共享内存。然后在这个 Block 中，让每个线程负责计算 8x8（即 64 个）元素。由于这 64 个中间结果都存放在当前线程的寄存器中，我们可以成倍提升运算速度。

核心思想伪代码：

// 假设 BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128
// 假设每个线程处理 TM=8, TN=8 个元素
float frag_a[TM]; // 存放从 sA 读取的寄存器切片
float frag_b[TN]; // 存放从 sB 读取的寄存器切片
float accum[TM][TN] = {0.0f}; // 线程私有的寄存器累加器

// 外层循环：遍历 K 维度块
for (int k_idx = 0; k_idx < K; k_idx += BLOCK_SIZE_K) {
    // 1. 将 A, B 的块加载到 Shared Memory (代码略)
    __syncthreads();
    
    // 2. 寄存器分块计算
    for (int k = 0; k < BLOCK_SIZE_K; ++k) {
        // 从共享内存加载到寄存器
        for (int i=0; i<TM; ++i) frag_a[i] = sA[thread_y * TM + i][k];
        for (int j=0; j<TN; ++j) frag_b[j] = sB[k][thread_x * TN + j];
        
        // 寄存器级别的 FFMA (Fused Multiply-Add)
        for (int i=0; i<TM; ++i) {
            for (int j=0; j<TN; ++j) {
                accum[i][j] += frag_a[i] * frag_b[j];
            }
        }
    }
    __syncthreads();
}
// 3. 将寄存器 accum 中的结果写回 Global Memory


关键点： 寄存器是 GPU 上最快的存储，通过扩大每个线程的工作量（Instruction-Level Parallelism），隐藏访存延迟。

优化第四步：向量化访存与 Bank 冲突消除 (Vectorized & Bank Conflict)

为了让寄存器分块发挥到极致（也是目前纯 CUDA Core 优化的天花板，通常可以达到理论峰值的 80%-90%）：

向量化访存（Vectorized Loads/Stores）：
使用 float4（128-bit 访存）来代替 float（32-bit 访存）读取全局内存。这可以极大减少内存指令的数量，提高访存带宽利用率。
float4 val = reinterpret_cast<const float4*>(&A[idx])[0];

消除 Shared Memory Bank 冲突：
共享内存分为 32 个 Bank。如果多个线程同时访问同一个 Bank 的不同地址，就会发生序列化（Bank Conflict），导致性能下降。
解决方案：在申请共享内存时，增加 padding（例如 __shared__ float sA[128][128 + 1];）或者使用地址 Swizzling 技术。

双缓冲 / 软件流水线 (Double Buffering / Prefetching)：
分配两组 Shared Memory (sA[2][...], sB[2][...]) 和两组寄存器。当 GPU 在计算第 $i$ 块数据时，后台利用异步拷贝（Asynchronous Copy, cp.async）提前把第 $i+1$ 块的数据从全局内存拉到 Shared Memory 中，实现计算和访存的完美重叠。

优化第五步：释放 RTX 5090 真正的力量 —— Tensor Cores (WMMA)

你在用的是 RTX 5090！这张卡的绝大部分算力（FLOPs）都集中在 Tensor Cores 上。传统的 CUDA Core 计算 SGEMM 撑死只有几十 TFLOPs，而启用 Tensor Cores，FP16/BF16/FP8 的算力可以飙升到数百甚至上千 TFLOPs。

NVIDIA 提供了 nvcuda::wmma (Warp Matrix Multiply Accumulate) API。使用 Tensor Core 时，计算的主体不再是单个线程，而是整个 Warp (32个线程)。

WMMA 基础用法演示（FP16 精度）：

#include <mma.h>
using namespace nvcuda;

__global__ void wmma_gemm(half *A, half *B, float *C, int M, int N, int K) {
    // 定义 WMMA 矩阵片段 (Fragment)
    // 假设采用 16x16x16 的 Tensor Core 形状
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // 当前 Warp 对应的行
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);      // 当前 Warp 对应的列

    // K 维度循环
    for (int i = 0; i < K; i += 16) {
        // Warp 协同从内存加载矩阵块到 Fragment
        wmma::load_matrix_sync(a_frag, A + warpM * 16 * K + i, K);
        wmma::load_matrix_sync(b_frag, B + i * N + warpN * 16, N);

        // 激动人心的一步：调用 Tensor Core 进行 16x16x16 的矩阵乘加运算！
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 将结果存回全局内存
    wmma::store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N, wmma::mem_row_major);
}


进阶提示对于 5090： WMMA API 虽然好用，但为了达到 CUTLASS 或 cuBLAS 的水平，现代架构 (Hopper/Blackwell) 倾向于使用更底层的 PTX 指令 mma.sync，或者直接利用 TMA (Tensor Memory Accelerator) 技术来实现异步无缝的数据搬运。

总结与下一步学习建议

动手实践：先在你的 RTX 5090 上把 Naive 和 Shared Memory 的版本写出来，用 nvcc 编译，并写一个 host 端的计时函数对比它们的时间差异。

使用 Nsight Compute (ncu)：这是最重要的 profiling 工具。运行你的算子，查看 Compute Throughput 和 Memory Throughput，看看瓶颈是在计算还是访存。

学习 CUTLASS：当掌握了手写 Register Tiling 和 WMMA 后，建议直接阅读并使用 NVIDIA 开源的 CUTLASS 模板库，现代工业级的高性能算子（包括 LLM 推理的 FlashAttention 等）基本都是建立在 CUTLASS 的理念之上的。
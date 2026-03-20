// CuTe SGEMM: C = alpha * A * B + beta * C
// 使用 CuTe (CUTLASS 3.x) DSL 实现，展示张量编程的简洁与强大
//
// 矩阵按行主序 (Row-Major) 存储：
//   A: M×K, lda = K
//   B: K×N, ldb = N
//   C: M×N, ldc = N
//
// 依赖：NVIDIA CUTLASS 3.x 头文件（包含 cute/tensor.hpp）

#include "common.h"
#include "gemm_kernels.h"

#include <iostream>

#include <cute/tensor.hpp>  // 注意：CuTe 是 CUTLASS 3.x 的核心 DSL

using namespace cute;

// 共享内存存储结构
template <class TA, class TB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
    cute::ArrayEngine<TA, cute::cosize_v<ASmemLayout>> A;
    cute::ArrayEngine<TB, cute::cosize_v<BSmemLayout>> B;
};

// CuTe GEMM Kernel
// 使用 UniversalFMA (CUDA Core) 进行 FP32 计算
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void gemm_cute_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
                      TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
                      TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
                      TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
                      Alpha alpha, Beta beta)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    static_assert(is_static<AThreadLayout>::value);
    static_assert(is_static<BThreadLayout>::value);
    static_assert(is_static<CThreadLayout>::value);
    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

    // Full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);            // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory buffers
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K)

    // Partition the copying of A and B tiles across the threads
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

    Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

    // Define A/B partitioning and C accumulators
    // Partition sA by the rows of tC, sB by the cols of tC
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});    // (THR_M,BLK_K)
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});    // (THR_N,BLK_K)
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});    // (THR_M,THR_N)

    // Allocate the accumulators
    Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)
    clear(tCrC);

    // Main loop
    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // Copy gmem to smem
        copy(tAgA(_,_,k_tile), tAsA);
        copy(tBgB(_,_,k_tile), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Compute gemm on tC thread-partitioned smem
        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }

    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

// Host wrapper for Row-Major SGEMM (NT: No-Transpose)
void run_sgemm_cute(int M, int N, int K, float alpha, const float *A, const float *B,
                    float beta, float *C) {
    using namespace cute;

    // Define shapes (dynamic)
    auto m = int(M);
    auto n = int(N);
    auto k = int(K);
    auto prob_shape = make_shape(m, n, k);                 // (M, N, K)

    // Define NT strides for Row-Major layout
    // A is MxK, stride between rows is K (column-major stride in BLAS terms)
    auto dA = make_stride(Int<1>{}, K);                    // (dM, dK) -> step 1 within row, K between rows
    auto dB = make_stride(Int<1>{}, N);                    // (dN, dK) -> step 1 within row, N between rows
    auto dC = make_stride(Int<1>{}, N);                    // (dM, dN) -> step 1 within row, N between rows

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);               // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    // Row-major smem: m-major for A and B
    auto sA = make_layout(make_shape(bM, bK));             // (m,k) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK));             // (n,k) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN));             // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (m,k) -> thr_idx
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (n,k) -> thr_idx
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); // (m,n) -> thr_idx

    // Compute smem size
    int smem_size = int(sizeof(SharedStorage<float, float, decltype(sA), decltype(sB)>));

    // Launch kernel
    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(m, bM)),
                 size(ceil_div(n, bN)));

    gemm_cute_kernel<<<dimGrid, dimBlock, smem_size>>>
        (prob_shape, cta_tiler,
         A, dA, sA, tA,
         B, dB, sB, tB,
         C, dC, sC, tC,
         alpha, beta);

    CHECK_CUDA(cudaGetLastError());
}

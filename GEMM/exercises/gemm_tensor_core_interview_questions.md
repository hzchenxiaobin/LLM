# Tensor Core GEMM 面试题

本文档涵盖 NVIDIA Tensor Core 相关的深度面试题，包括 WMMA API、混合精度计算、性能调优等。

---

## Tensor Core 基础

### 问题 1：什么是 Tensor Core？它与传统 CUDA Core 有什么区别？

**参考答案**：

**Tensor Core 定义**：
NVIDIA GPU 中的专用矩阵计算单元，可以在一个时钟周期内完成小矩阵的乘加运算 (MMA)。

**硬件演进**：

| 架构 | Tensor Core 特性 | FP16 峰值 |
|-----|-----------------|----------|
| Volta (V100) | 第一代，仅 FP16 | 125 TFLOPS |
| Turing (RTX 20) | 增加 INT8/INT4 | ~130 TFLOPS |
| Ampere (A100/RTX 30) | 增加 BF16/TF32/FP64 | 312-624 TFLOPS |
| Ada (RTX 40) | 更多核心，更高频率 | 330-660 TFLOPS |
| Hopper (H100) | Transformer Engine | 989-1979 TFLOPS |
| Blackwell (RTX 50) | FP4/FP6 支持 | >2000 TFLOPS |

**与 CUDA Core 对比**：

| 特性 | CUDA Core | Tensor Core |
|-----|----------|-------------|
| 操作类型 | 标量 FMA | 矩阵 MMA |
| 每周期计算 | 2 FLOPS (乘加) | 512+ FLOPS (16x16x16 MMA) |
| 数据类型 | FP32/FP16 | FP16/BF16/TF32/INT8/FP64 |
| 累加精度 | 与输入相同 | 高于输入（如 FP16 输入 FP32 累加） |
| 编程接口 | 自动使用 | WMMA/PTX/mma.sync |
| 适用场景 | 通用计算 | 矩阵乘法、深度学习 |

**MMA 操作示例** (16×16×16)：
```
C(16×16) = A(16×16) × B(16×16) + C(16×16)
- 输入 A/B: FP16
- 累加 C: FP32
- 单周期完成 16×16×16 = 4096 次乘加 = 8192 FLOPS
```

---

### 问题 2：Tensor Core 对输入矩阵的尺寸和布局有什么要求？

**参考答案**：

**尺寸约束**：

| MMA 操作 | M | N | K | 典型用途 |
|---------|---|---|---|---------|
| m16n16k16 | 16 | 16 | 16 | 通用矩阵乘法 |
| m16n16k8 | 16 | 16 | 8 | FP16/BF16 |
| m32n8k16 | 32 | 8 | 16 | 特定优化 |
| m8n32k16 | 8 | 32 | 16 | 特定优化 |

**布局约束**：

```cpp
// WMMA Fragment 布局要求
wmma::fragment<matrix_a, M, N, K, T, layout> frag_a;
wmma::fragment<matrix_b, M, N, K, T, layout> frag_b;
wmma::fragment<accumulator, M, N, K, float> frag_c;

// A 矩阵可以是：
// - wmma::row_major (行主序)
// - wmma::col_major (列主序)

// B 矩阵可以是：
// - wmma::row_major
// - wmma::col_major

// C 矩阵（累加器）：
// - 自动匹配 A×B 的结果布局
```

**地址对齐要求**：
```
- A/B 矩阵首地址需要 16-byte 对齐
- 建议使用 padding 确保对齐
- 从 Shared Memory 加载时尤其要注意
```

**边界处理**：
```cpp
// 矩阵维度不是 16 的倍数时需要边界检查
if (row + WMMA_M <= M && col + WMMA_N <= N) {
    // 使用 WMMA 快速路径
    wmma::store_matrix_sync(&C[row*N+col], frag_c, N, wmma::mem_row_major);
} else {
    // 标量边界处理
    for (int i = 0; i < WMMA_M && row+i < M; i++) {
        for (int j = 0; j < WMMA_N && col+j < N; j++) {
            C[(row+i)*N + (col+j)] = frag_c.x[i*WMMA_N + j];
        }
    }
}
```

---

## WMMA API 编程

### 问题 3：使用 WMMA API 实现 GEMM 的基本步骤是什么？

**参考答案**：

**完整流程**：

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void wmma_gemm(int M, int N, int K, 
                          const half *A, const half *B, 
                          float *C) {
    // 步骤 1: 确定当前 Warp 负责的输出块
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_M;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) * WMMA_N;
    
    // 步骤 2: 声明 WMMA Fragment
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_b;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    // 步骤 3: 初始化累加器
    fill_fragment(frag_c, 0.0f);
    
    // 步骤 4: 沿 K 维度迭代
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载 A Fragment
        load_matrix_sync(frag_a, &A[warp_row * K + k], K);
        
        // 加载 B Fragment
        load_matrix_sync(frag_b, &B[k * N + warp_col], N);
        
        // 执行 MMA 操作
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    // 步骤 5: 存储结果
    store_matrix_sync(&C[warp_row * N + warp_col], frag_c, N, mem_row_major);
}
```

**关键 API 说明**：

| API | 作用 | 执行位置 |
|-----|------|---------|
| `load_matrix_sync` | 从内存加载矩阵到 Fragment | Warp 协作 |
| `mma_sync` | 执行矩阵乘加 | Warp 协作 |
| `store_matrix_sync` | 存储 Fragment 到内存 | Warp 协作 |
| `fill_fragment` | 初始化 Fragment | 每个线程独立 |

---

### 问题 4：为什么 Tensor Core GEMM 通常使用 FP16 输入和 FP32 累加？

**参考答案**：

**混合精度设计**：

```
输入: A, B (FP16)     累加: C (FP32)
         ↓                    ↑
         └────→ MMA ────┘
                ↓
              中间结果 (FP32)
```

**优势分析**：

1. **精度保证**：
```
FP16 范围: ~6e-5 到 6e4 (动态范围较小)
FP16 精度: ~3e-4 (约 3-4 位有效数字)

矩阵乘法中，K 个元素累加可能导致：
- FP16 累加：舍入误差累积，精度损失
- FP32 累加：精度高，可表示更大范围
```

2. **性能与精度平衡**：
```
场景: M=N=K=1024 的 GEMM

纯 FP16 (FP16 累加):
- 速度: 100%
- 精度: 可能有较大误差

混合精度 (FP16 输入 + FP32 累加):
- 速度: ~95-100% (几乎无损)
- 精度: 与 FP32 GEMM 相当

纯 FP32 (FP32 输入):
- 速度: ~50% (TF32 模式下)
- 精度: 最高
```

3. **数值稳定性**：
```cpp
// 大数加小数问题
float sum_fp16 = 0;  // 假设用 FP16 累加
for (int i = 0; i < 10000; i++) {
    sum_fp16 += 0.0001f;  // 可能最终 sum 仍是 0！
}

float sum_fp32 = 0;  // FP32 累加
for (int i = 0; i < 10000; i++) {
    sum_fp32 += 0.0001f;  // sum = 1.0
}
```

**实际应用场景**：
- **深度学习训练**: FP16/BF16 输入 + FP32 累加 (混合精度训练)
- **推理**: 纯 FP16 或 INT8（速度优先）
- **科学计算**: TF32 或 FP64（精度优先）

---

## 性能优化

### 问题 5：如何优化 Tensor Core GEMM 的性能？

**参考答案**：

**层次优化策略**：

1. **Warp 级分块**：
```cpp
// 每个 Warp 处理多个 WMMA 块
#define WARP_M_TILES 4  // 4 个 16x16 块 = 64 行
#define WARP_N_TILES 2  // 2 个 16x16 块 = 32 列

fragment<...> acc_frag[WARP_M_TILES][WARP_N_TILES];

// 每个 Warp 计算 64×32 输出块
for (int k = 0; k < K; k += WMMA_K) {
    for (int i = 0; i < WARP_M_TILES; i++) {
        load_matrix_sync(a_frag[i], ...);
    }
    for (int j = 0; j < WARP_N_TILES; j++) {
        load_matrix_sync(b_frag[j], ...);
    }
    for (int i = 0; i < WARP_M_TILES; i++) {
        for (int j = 0; j < WARP_N_TILES; j++) {
            mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
        }
    }
}
```

2. **Shared Memory 缓存**：
```cpp
// 预先将数据加载到 Shared Memory，减少 Global Memory 访问
__shared__ half sA[BLOCK_M][BLOCK_K];
__shared__ half sB[BLOCK_K][BLOCK_N];

// 协作加载
for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
    sA[i / BLOCK_K][i % BLOCK_K] = A[...];
}
__syncthreads();

// 从 Shared Memory 加载到 Fragment
load_matrix_sync(frag_a, &sA[warp_row][0], BLOCK_K);
```

3. **流水线优化**：
```cpp
// 双缓冲 + WMMA 流水线
__shared__ half sA[2][BLOCK_M][BLOCK_K];
int buf = 0;

// 预加载
load_to_shared(sA[buf]);
__syncthreads();

for (int k = BLOCK_K; k < K; k += BLOCK_K) {
    // 从当前缓冲计算
    load_matrix_sync(a_frag, &sA[buf][warp_row][0], BLOCK_K);
    mma_sync(...);
    
    // 同时加载到另一缓冲
    load_to_shared(sA[1-buf]);
    __syncthreads();
    
    buf = 1 - buf;
}
```

4. **数据布局优化**：
```cpp
// Swizzling 技术：重排数据以避免 Bank Conflict
// 适用于 Ampere+ 架构的 Shared Memory 布局
```

**性能调参指南**：

| 参数 | 推荐值 | 影响 |
|-----|-------|------|
| BLOCK_M/N | 128, 256 | 数据重用率 |
| BLOCK_K | 32, 64 | 管道深度 |
| WARP_M/N_TILES | 2-8 | 计算强度 |
| Shared Memory | 16-64 KB | Occupancy |

---

### 问题 6：cuBLAS 和 CUTLASS 在使用 Tensor Core 上有什么差异？

**参考答案**：

**对比分析**：

| 特性 | cuBLAS | CUTLASS |
|-----|--------|---------|
| 易用性 | 高（直接调用） | 中（模板编程） |
| 灵活性 | 低（固定实现） | 高（可定制） |
| 性能 | 优秀 | 接近 cuBLAS 或更好 |
| 学习价值 | 低 | 高（开源实现） |
| 适用场景 | 生产环境 | 研究/定制优化 |

**cuBLAS 使用**：
```cpp
// 自动选择最优算法（可能使用 Tensor Core）
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             d_A, CUDA_R_16F, lda,
             d_B, CUDA_R_16F, ldb,
             &beta,
             d_C, CUDA_R_32F, ldc,
             CUBLAS_COMPUTE_32F,  // FP32 累加
             CUBLAS_GEMM_DEFAULT); // 自动选择算法
```

**CUTLASS 特点**：
```cpp
// 显式控制所有优化参数
tusing Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,  // A
    cutlass::half_t, cutlass::layout::RowMajor,  // B
    float, cutlass::layout::RowMajor,             // C
    float,                                        // 累加器
    cutlass::arch::OpClassTensorCore,            // 使用 Tensor Core
    cutlass::arch::Sm80,                         // Ampere 架构
    cutlass::gemm::GemmShape<128, 256, 32>,      // 线程块形状
    cutlass::gemm::GemmShape<64, 64, 32>,        // Warp 形状
    cutlass::gemm::GemmShape<16, 16, 16>         // MMA 形状
>;
```

**选择建议**：
- **快速部署**: 使用 cuBLAS
- **学习研究**: 阅读 CUTLASS 源码
- **极致优化**: 基于 CUTLASS 定制或使用 PTX

---

## 架构演进

### 问题 7：Hopper/Blackwell 架构的 Tensor Core 有哪些新特性？

**参考答案**：

**Hopper (H100) 新特性**：

1. **FP8 支持**：
```
FP8 E4M3: 4 位指数，3 位尾数
FP8 E5M2: 5 位指数，2 位尾数

适用场景：
- Transformer 推理（精度要求较低）
- 吞吐量是 FP16 的 2 倍
```

2. **Transformer Engine**：
```
- 自动在前向传播中选择 FP8/FP16
- 动态缩放因子管理
- 与框架集成（PyTorch/TensorFlow）
```

3. **Thread Block Cluster**：
```
- 多个 Thread Block 协作
- 更大的分布式 Shared Memory
- 适用于大矩阵分块
```

**Blackwell (RTX 50) 新特性**：

1. **FP4/FP6 支持**：
```
FP6: 2-3 位指数，2-3 位尾数
FP4: 2 位指数，1 位尾数

极致压缩，适用于：
- 模型量化推理
- 边缘设备部署
```

2. **第二代 Transformer Engine**：
- 改进的动态范围管理
- 更好的训练稳定性

3. **更高的 FP16/FP8 峰值**：
```
RTX 5090 FP16 Tensor Core 峰值: ~660 TFLOPS
RTX 5090 FP8 峰值: ~1320 TFLOPS
```

---

## 总结

| 技术要点 | 核心价值 |
|---------|---------|
| WMMA API | 简化 Tensor Core 编程 |
| 混合精度 | 速度与精度的平衡 |
| 多级分块 | 最大化数据重用 |
| Shared Memory 缓存 | 减少 Global Memory 访问 |
| 架构演进 | 更快速度、更低精度选项 |

---

*文档生成时间：2026年3月19日*

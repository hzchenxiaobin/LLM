# GEMM 编程实践题

本文档提供 GEMM 相关的编程练习题，从简单到复杂，帮助巩固理论知识并提升实际编码能力。

---

## 初级练习

### 练习 1：Naive GEMM 实现

**任务**：
实现一个最基础的 CUDA GEMM，不使用任何优化。

**要求**：
- 实现 `sgemm_naive` kernel
- 每个线程计算 C 的一个元素
- 使用三重循环结构（外部两重并行，内部一重串行）

**代码框架**：
```cpp
__global__ void sgemm_naive(int M, int N, int K, 
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {
    // 计算当前线程负责的元素位置
    int row = ???;
    int col = ???;
    
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// 包装函数
void run_sgemm_naive(int M, int N, int K, 
                     float alpha, const float *A, const float *B,
                     float beta, float *C) {
    dim3 block(???);  // 选择合适的 block 大小
    dim3 grid(???);   // 计算 grid 大小
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
```

**提示**：
- 使用 2D Grid 布局：`blockIdx.y` 对应行，`blockIdx.x` 对应列
- 推荐 Block 大小：16×16 或 32×32

**验证**：
- 与 cuBLAS 结果对比，误差 tolerance = 1e-3
- 预期性能：< 1 TFLOPS (RTX 4090)

---

### 练习 2：Coalesced Memory Access 优化

**任务**：
优化 Naive GEMM 的内存访问模式，确保合并访问。

**分析原问题**：
```cpp
// Naive 版本的问题
for (int k = 0; k < K; k++) {
    float a = A[row * K + k];      // ✅ 行内连续，合并访问
    float b = B[k * N + col];      // ❌ 列访问，步长 N，非合并
    sum += a * b;
}
```

**优化方案**：
1. **方案 A**: 转置 B 矩阵，使访问连续
2. **方案 B**: 改变循环顺序（虽然会改变结果，但理解访问模式）

**任务要求**：
- 实现 B 矩阵转置后的 GEMM
- 对比转置前后的性能差异

**代码框架**：
```cpp
// 转置 B 矩阵的 kernel
__global__ void transpose_B(int K, int N, const float *B, float *B_T) {
    // 实现 B 的转置
}

// 使用转置后 B 的 GEMM
__global__ void sgemm_coalesced(int M, int N, int K, 
                                float alpha, const float *A, const float *B_T,
                                float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            // B_T 是列主序存储的原 B 矩阵
            sum += A[row * K + k] * B_T[col * K + k];  // ✅ 现在都是合并访问！
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
```

**预期收益**：
- 内存带宽利用率提升 2-5 倍
- 整体性能提升 1.5-3 倍

---

## 中级练习

### 练习 3：Shared Memory Tiling 实现

**任务**：
实现基础的 Shared Memory 分块 GEMM。

**参数设置**：
- Block 大小: 32×32 = 1024 线程
- 分块大小: BM=32, BN=32, BK=16
- 共享内存: sA[32][16], sB[16][32]

**实现步骤**：
```cpp
#define BM 32
#define BN 32
#define BK 16

__global__ void sgemm_shared(int M, int N, int K,
                             float alpha, const float *A, const float *B,
                             float beta, float *C) {
    // 1. 申请共享内存
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    
    // 2. 计算当前 Block 负责的输出块位置
    int block_row = ???;  // blockIdx.y * BM
    int block_col = ???;  // blockIdx.x * BN
    
    // 3. 线程在 Block 内的位置
    int thread_row = ???;  // threadIdx.y
    int thread_col = ???;  // threadIdx.x
    
    // 4. 沿 K 维度迭代
    float accum = 0;
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // 4.1 协作加载 A 到共享内存
        // 每个线程加载多少个元素？
        for (int i = ???; i < BM * BK; i += ???) {
            int r = i / BK;
            int c = i % BK;
            sA[r][c] = A[(block_row + r) * K + (k_tile + c)];
        }
        
        // 4.2 协作加载 B 到共享内存
        for (int i = ???; i < BK * BN; i += ???) {
            int r = i / BN;
            int c = i % BN;
            sB[r][c] = B[(k_tile + r) * N + (block_col + c)];
        }
        
        __syncthreads();
        
        // 4.3 计算当前分块的贡献
        for (int k = 0; k < BK; k++) {
            accum += sA[thread_row][k] * sB[k][thread_col];
        }
        
        __syncthreads();
    }
    
    // 5. 写回结果
    C[(block_row + thread_row) * N + (block_col + thread_col)] = 
        alpha * accum + beta * C[...];
}
```

**思考问题**：
1. 为什么要用两个 `__syncthreads()`？
2. 如果去掉第一个同步会发生什么？
3. 如何计算合适的每个线程加载数据量？

**预期性能**：
- 相比 Naive 版本提升 3-5 倍
- 约 3-8 TFLOPS (取决于 GPU)

---

### 练习 4：Bank Conflict 检测与消除

**任务**：
检测练习 3 的 Bank Conflict 并实现 Padding 优化。

**检测方法**：
使用 Nsight Compute 查看 Shared Memory 指标：
```
- Shared Memory Load/Store 效率
- Bank Conflict 比例
```

**问题分析**：
```cpp
// 原代码可能的 Conflict 情况
__shared__ float sA[BM][BK];  // BK=16
// 线程 0, 16 访问 sA[row][0] → 都访问 Bank 0 → Conflict！
```

**优化实现**：
```cpp
// 方案 1: Padding
#define PADDING 1
__shared__ float sA[BM][BK + PADDING];  // 17 列
__shared__ float sB[BK][BN + PADDING];  // 33 列

// 访问时需要调整索引
float val = sA[row][col];  // col 范围 0..BK-1

// 方案 2: 改变存储布局
__shared__ float sA[BK][BM];  // 转置存储，适配访问模式
```

**验证方法**：
- 使用 Nsight Compute 对比优化前后
- 预期 Bank Conflict 减少 80%+
- 性能提升 10-30%

---

### 练习 5：Register Tiling 实现

**任务**：
在 Shared Memory Tiling 基础上增加寄存器累加。

**参数设置**：
- Block: 16×16 = 256 线程
- 每线程计算: TM=8, TN=8 (8×8=64 个输出元素)
- 实际输出块: 128×128
- Shared Memory: sA[128][8], sB[8][128]

**实现框架**：
```cpp
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_register(int M, int N, int K,
                               float alpha, const float *A, const float *B,
                               float beta, float *C) {
    // 1. 共享内存声明
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    
    // 2. 寄存器累加器
    float accum[TM][TN] = {0};
    
    // 3. 线程位置计算
    int thread_row = threadIdx.y * TM;  // 0, 8, 16, ..., 120
    int thread_col = threadIdx.x * TN;  // 0, 8, 16, ..., 120
    
    // 4. 沿 K 维度迭代
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // 4.1 加载 A 到共享内存
        // 256 线程加载 128×8 = 1024 个元素，每个线程 4 个
        
        // 4.2 加载 B 到共享内存
        
        __syncthreads();
        
        // 4.3 计算：先加载到寄存器，再累加
        for (int k = 0; k < BK; k++) {
            // 加载 A 的一列到寄存器
            float frag_a[TM];
            for (int i = 0; i < TM; i++) {
                frag_a[i] = sA[thread_row + i][k];
            }
            
            // 加载 B 的一行到寄存器
            float frag_b[TN];
            for (int j = 0; j < TN; j++) {
                frag_b[j] = sB[k][thread_col + j];
            }
            
            // 寄存器外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    accum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 5. 写回全局内存
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = blockIdx.y * BM + thread_row + i;
            int col = blockIdx.x * BN + thread_col + j;
            if (row < M && col < N) {
                C[row * N + col] = alpha * accum[i][j] + beta * C[row * N + col];
            }
        }
    }
}
```

**思考问题**：
1. 计算这个 kernel 的寄存器使用量（估算）
2. 计算理论 Occupancy
3. 分析计算强度相比 Shared Memory Only 版本提升了多少

**预期性能**：
- 相比 Shared Memory 版本提升 2-3 倍
- 可达 20-40 TFLOPS

---

## 高级练习

### 练习 6：向量化加载实现

**任务**：
使用 `float4` 向量化加载优化练习 5。

**关键技术**：
```cpp
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// 向量化加载示例
float4 vec = FETCH_FLOAT4(A[global_idx]);
shared_mem[local_idx + 0] = vec.x;
shared_mem[local_idx + 1] = vec.y;
shared_mem[local_idx + 2] = vec.z;
shared_mem[local_idx + 3] = vec.w;
```

**实现要求**：
1. 确保全局内存地址 16-byte 对齐
2. 对 K 维度进行 padding 到 4 的倍数
3. 边界处理：剩余不足 4 个元素时使用标量加载

**框架代码**：
```cpp
__global__ void sgemm_vectorized(int M, int N, int K, ...) {
    // 共享内存声明（带 padding 确保对齐）
    __shared__ float sA[BM][BK + 4];  // +4 是为了避免 conflict 和对齐
    __shared__ float sB[BK][BN + 4];
    
    // 计算向量化加载参数
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // A: 128×8 = 1024 元素，256 线程，每个线程 4 元素 → 1 个 float4
    int load_a_row = tid / 2;       // 0-127
    int load_a_col = (tid % 2) * 4;  // 0 或 4
    
    // B: 8×128 = 1024 元素，类似计算
    int load_b_row = tid / 32;      // 0-7
    int load_b_col = (tid % 32) * 4; // 0, 4, 8, ..., 124
    
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // 向量化加载 A
        if (global_row < M && global_col + 3 < K) {
            float4 vec = FETCH_FLOAT4(A[global_row * K + global_col]);
            sA[load_a_row][load_a_col + 0] = vec.x;
            sA[load_a_row][load_a_col + 1] = vec.y;
            sA[load_a_row][load_a_col + 2] = vec.z;
            sA[load_a_row][load_a_col + 3] = vec.w;
        } else {
            // 边界处理：标量加载
            for (int i = 0; i < 4 && global_col + i < K; i++) {
                sA[load_a_row][load_a_col + i] = A[...];
            }
        }
        
        // 类似加载 B...
        
        __syncthreads();
        // 计算...
    }
}
```

**性能预期**：
- 带宽利用率提升 20-40%
- 整体性能提升 10-20%

---

### 练习 7：双缓冲 (Double Buffering) 实现

**任务**：
在练习 6 基础上实现双缓冲，重叠计算和数据加载。

**核心思想**：
```cpp
// 使用两个共享内存缓冲区
__shared__ float sA[2][BM][BK + 4];
__shared__ float sB[2][BK][BN + 4];

int write_buf = 0;  // 当前写入缓冲
int read_buf = 1;   // 当前读取缓冲

// 预加载第一块
load_to_buffer(sA[write_buf], sB[write_buf], 0);
__syncthreads();
swap(write_buf, read_buf);

// 主循环
for (int k = BK; k < K; k += BK) {
    // 1. 从读取缓冲计算（与加载并行）
    compute_with_buffer(sA[read_buf], sB[read_buf]);
    
    // 2. 预加载下一块到写入缓冲
    load_to_buffer(sA[write_buf], sB[write_buf], k);
    
    __syncthreads();
    swap(write_buf, read_buf);
}

// 计算最后一块
compute_with_buffer(sA[read_buf], sB[read_buf]);
```

**思考问题**：
1. 双缓冲会增加多少共享内存使用？对 Occupancy 有什么影响？
2. 为什么可以减少 `__syncthreads()` 的调用次数？
3. 在什么情况下双缓冲收益最大？

**性能预期**：
- 隐藏全局内存延迟
- 提升 10-25% 性能
- 在计算受限场景效果更明显

---

### 练习 8：WMMA Tensor Core 实现

**任务**：
使用 WMMA API 实现基于 Tensor Core 的 GEMM。

**要求**：
- 输入：FP16 (half)
- 累加：FP32 (float)
- 分块：Block 32×32，每 Block 4 个 Warp
- 每个 Warp 计算 16×16 输出块

**实现框架**：
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 16

__global__ void sgemm_wmma(int M, int N, int K,
                           half alpha, const half *A, const half *B,
                           float beta, float *C) {
    // 1. Warp 位置计算
    int warp_id = threadIdx.x / warpSize;  // 0-3
    int warp_row = (warp_id / 2) * WMMA_M;  // 0 或 16
    int warp_col = (warp_id % 2) * WMMA_N;  // 0 或 16
    
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;
    
    // 2. 声明 WMMA Fragment
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 3. 初始化累加器
    fill_fragment(c_frag, 0.0f);
    
    // 4. 申请共享内存
    __shared__ half sA[BLOCK_M][BLOCK_K];
    __shared__ half sB[BLOCK_K][BLOCK_N];
    
    // 5. 沿 K 维度迭代
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // 5.1 协作加载 A/B 到共享内存
        // 128 线程加载 32×16=512 个 half
        int tid = threadIdx.x;
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int r = i / BLOCK_K;
            int c = i % BLOCK_K;
            sA[r][c] = A[(block_row + r) * K + (k_tile + c)];
        }
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            sB[r][c] = B[(k_tile + r) * N + (block_col + c)];
        }
        __syncthreads();
        
        // 5.2 从共享内存加载到 Fragment
        load_matrix_sync(a_frag, &sA[warp_row][0], BLOCK_K);
        load_matrix_sync(b_frag, &sB[0][warp_col], BLOCK_N);
        
        // 5.3 执行 Tensor Core MMA
        mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    // 6. 存储结果
    // 应用 alpha 缩放
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] *= alpha;
    }
    
    int global_row = block_row + warp_row;
    int global_col = block_col + warp_col;
    store_matrix_sync(&C[global_row * N + global_col], c_frag, N, mem_row_major);
}
```

**拓展挑战**：
1. 实现每个 Warp 处理多个 16×16 块（类似 64×32）
2. 添加 FP32 输入自动转换为 FP16 的包装函数
3. 实现边界处理（非 16 对齐的矩阵）

**预期性能**：
- FP16 峰值可达 300-600 TFLOPS (RTX 4090/5090)
- 相比 FP32 版本提升 5-10 倍

---

## 综合挑战

### 练习 9：完整的 SGEMM 自动调优框架

**任务**：
设计一个可以自动搜索最优分块参数的框架。

**要求**：
1. 定义参数搜索空间：
   - Block 大小: {64, 128, 256} × {64, 128, 256}
   - BK: {8, 16, 32}
   - TM/TN: {4, 8, 16} × {4, 8, 16}

2. 实现参数验证：
   - 检查寄存器使用量是否超限
   - 检查共享内存是否超限
   - 估算 Occupancy

3. 实现基准测试：
   - 对每个参数组合运行性能测试
   - 记录最优参数

**框架结构**：
```cpp
struct GemmConfig {
    int BM, BN, BK;
    int TM, TN;
    int estimated_registers;
    int shared_memory_usage;
    float estimated_occupancy;
};

class GemmAutoTuner {
public:
    // 搜索最优配置
    GemmConfig tune(int M, int N, int K);
    
    // 验证配置是否可行
    bool validateConfig(const GemmConfig& config);
    
    // 估算寄存器使用量
    int estimateRegisters(const GemmConfig& config);
    
    // 运行基准测试
    float benchmark(const GemmConfig& config);
};
```

---

### 练习 10：与 cuBLAS 性能对比分析

**任务**：
将自己实现的各版本 GEMM 与 cuBLAS 进行全面对比。

**对比维度**：
1. **性能对比**：
   - 不同矩阵尺寸 (128, 512, 1024, 4096, 8192)
   - 不同实现版本 (Naive → Shared → Register → Vectorized → Double Buffer → WMMA)
   - 与 cuBLAS 的差距

2. **硬件利用率**：
   - 使用 Nsight Compute 分析
   - 内存带宽利用率
   - 计算单元利用率
   - Occupancy

3. **可扩展性**：
   - 批处理矩阵乘法 (Batched GEMM)
   - 非方阵处理
   - 不同数据类型 (FP32, FP16, BF16)

**输出要求**：
```
性能对比表:
| 实现版本 | 4096×4096 TFLOPS | 相对于 cuBLAS |
|---------|-----------------|--------------|
| Naive   | 0.5             | 1%           |
| Shared  | 5.0             | 10%          |
| Register| 25.0            | 50%          |
| Vectorized| 30.0          | 60%          |
| Double Buffer| 35.0       | 70%          |
| WMMA    | 150.0           | 90%          |
| cuBLAS  | 165.0           | 100%         |
```

---

## 评分标准

| 练习 | 正确性 (40%) | 性能 (30%) | 代码质量 (20%) | 文档 (10%) |
|-----|--------------|-----------|---------------|-----------|
| 1-2 (初级) | 能正确计算结果 | 达到基准 | 结构清晰 | 有基本注释 |
| 3-5 (中级) | 通过 correctness check | 接近理论峰值 50% | 模块化设计 | 完整的设计说明 |
| 6-8 (高级) | 处理所有边界情况 | 接近理论峰值 70% | 高效简洁 | 详细的优化分析 |
| 9-10 (综合) | 框架完整可用 | 自动找到较优参数 | 可扩展架构 | 完整的测试报告 |

---

## 参考答案位置

各练习的参考答案可参考项目中的实现：
- 练习 1-2: `src/sgemm_naive.cu`
- 练习 3: `src/sgemm_shared.cu`
- 练习 4: `src/sgemm_register_bank_conflict.cu`
- 练习 5: `src/sgemm_register.cu`
- 练习 6: `src/sgemm_register_v2.cu`
- 练习 7: `src/sgemm_register_v3.cu`
- 练习 8: `src/sgemm_wmma.cu`, `src/sgemm_wmma_v2.cu`

---

*文档生成时间：2026年3月19日*

# GEMM 基础概念面试题

本文档涵盖 GEMM (General Matrix Multiply) 的基础概念面试题，适合考察候选人对矩阵乘法基本原理和 CUDA 编程基础的理解。

---

## 初级问题

### 问题 1：什么是 GEMM？请写出标准 GEMM 的数学公式。

**参考答案**：

GEMM (General Matrix Multiply) 是通用矩阵乘法的缩写，标准形式为：

$$C = \alpha \cdot A \times B + \beta \cdot C$$

其中：
- $A$ 是 $M \times K$ 矩阵
- $B$ 是 $K \times N$ 矩阵
- $C$ 是 $M \times N$ 矩阵
- $\alpha$ 和 $\beta$ 是标量系数

**计算复杂度**：$O(M \times N \times K)$，总共需要 $M \times N \times K$ 次乘法和 $M \times N \times K$ 次加法。

---

### 问题 2：为什么矩阵乘法需要优化？直接的三重循环实现有什么问题？

**参考答案**：

**直接实现的伪代码**：
```cpp
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

**主要问题**：
1. **内存访问模式低效**：
   - A 按行访问（连续）✓
   - B 按列访问（跨步 $N$）✗ - 导致缓存未命中
   - C 每个元素被重复读写 $K$ 次

2. **计算强度低**：每次乘法只伴随 2 次内存访问（读取 A 和 B）

3. **无法利用并行性**：顺序执行，没有利用现代硬件的并行能力

4. **全局内存瓶颈**：GPU 全局内存带宽有限，直接实现会受限于内存带宽而非计算能力

---

### 问题 3：解释矩阵乘法中的"计算强度"(Arithmetic Intensity)概念。

**参考答案**：

**定义**：计算强度 = 浮点运算次数 / 内存访问字节数

**Naive GEMM 的计算强度**：
```
浮点运算 = 2 * M * N * K (乘加各一次)
内存访问 = 4 * (M*K + K*N + M*N) bytes (假设 float32)

计算强度 ≈ (2*MNK) / (4*(MK + KN + MN))
         ≈ MNK / (2*(MK + KN + MN))
```

当 $M=N=K$ 时：
```
计算强度 ≈ N³ / (6N²) = N/6
```

**意义**：
- 计算强度 >  Roofline 模型的"ridge point" → 计算受限
- 计算强度 <  Roofline 模型的"ridge point" → 内存受限

**优化目标**：提高计算强度，让 kernel 从内存受限转变为计算受限。

---

### 问题 4：什么是"行主序"(Row-major)和"列主序"(Column-major)？cuBLAS 使用哪种？

**参考答案**：

**行主序 (Row-major)**：
- 同一行的元素在内存中连续存储
- 地址计算：`addr = base + i * N + j` (对于 $M \times N$ 矩阵)
- C/C++ 默认使用

**列主序 (Column-major)**：
- 同一列的元素在内存中连续存储
- 地址计算：`addr = base + j * M + i`
- Fortran、MATLAB、cuBLAS 使用

**cuBLAS 说明**：
- cuBLAS 使用列主序（为了与 Fortran BLAS 兼容）
- 在 C/CUDA 代码中调用 cuBLAS 时需要注意转置问题
- 常用技巧：将 A 视为转置来避免数据重排

---

## 中级问题

### 问题 5：解释 CUDA 中实现 GEMM 时的"分块"(Tiling)策略。

**参考答案**：

**核心思想**：
将大矩阵分解成小矩阵块，使每个 Block 处理一个输出子矩阵，利用 Shared Memory 缓存数据。

**具体策略**：

1. **输出分块**：
   - 每个 CUDA Block 计算 $C$ 的一个 $BM \times BN$ 子矩阵
   - Block 网格：$(M/BM) \times (N/BN)$

2. **K 维分块**：
   - 沿 K 维度划分为 $BK$ 大小的块
   - 每次迭代加载 $A$ 的 $BM \times BK$ 块和 $B$ 的 $BK \times BN$ 块到 Shared Memory

3. **计算流程**：
```cpp
for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // 1. 协作加载 A/B 分块到 Shared Memory
    load_A_to_shared_memory();
    load_B_to_shared_memory();
    __syncthreads();
    
    // 2. 计算当前分块的贡献
    for (int k = 0; k < BK; k++) {
        accum += sA[local_i][k] * sB[k][local_j];
    }
    __syncthreads();
}
```

**优势**：
- Shared Memory 带宽比 Global Memory 高 10-100 倍
- 数据重用：每个加载的元素用于多个计算

---

### 问题 6：Bank Conflict 是什么？如何在 GEMM 中避免？

**参考答案**：

**Bank Conflict 定义**：
- GPU Shared Memory 被划分为 32 个 Bank（对应一个 Warp 的 32 线程）
- 理想情况：Warp 内 32 个线程同时访问不同 Bank，可并行执行
- Bank Conflict：多个线程同时访问同一 Bank，导致访问串行化

**GEMM 中的典型场景**：
```cpp
// 容易导致 Bank Conflict 的代码
__shared__ float sA[128][8];
// 线程 i 访问 sA[threadIdx.x][0]
// 线程 0 和 线程 16 都访问 Bank 0 → Conflict！
```

**避免方法**：

1. **Padding（填充）**：
```cpp
// 将列数从 8 填充到 8+1=9，使访问错开 Bank
__shared__ float sA[128][8 + 1];  // +1 padding
```

2. **转置存储**：
```cpp
// A 按行访问，B 按列访问，两者至少一个不会 Conflict
__shared__ float sA[BM][BK];  // A 行访问
__shared__ float sB[BN][BK];  // B 转置后行访问（实际访问原矩阵列）
```

3. **向量化访问**：
```cpp
// 使用 float4 加载，一次访问 4 个连续元素
float4 vec = reinterpret_cast<float4*>(sA[row])[col / 4];
```

---

### 问题 7：解释"双缓冲"(Double Buffering)在 GEMM 中的作用。

**参考答案**：

**核心思想**：
使用两个共享内存缓冲区，在计算当前分块的同时，预取下一个分块的数据。

**传统实现的问题**：
```cpp
for (int k = 0; k < K; k += BK) {
    load_data();      // 加载阶段（访存受限，计算单元空闲）
    __syncthreads();
    compute();        // 计算阶段（计算受限，访存单元空闲）
    __syncthreads();
}
// 加载和计算无法重叠！
```

**双缓冲实现**：
```cpp
__shared__ float sA[2][BM][BK];  // 两个缓冲
__shared__ float sB[2][BK][BN];

int load_idx = 0, compute_idx = 1;

// 预加载第一块
load_to_buffer(load_idx);
__syncthreads();

for (int k = BK; k < K; k += BK) {
    // 1. 预加载下一块到另一个缓冲（异步）
    load_to_buffer(load_idx);
    
    // 2. 计算当前块（与加载并行）
    compute_with_buffer(compute_idx);
    
    __syncthreads();
    
    // 3. 交换缓冲
    swap(load_idx, compute_idx);
}

// 计算最后一块
compute_with_buffer(compute_idx);
```

**优势**：
- 隐藏全局内存延迟
- 提高计算单元利用率
- 通常可提升 10-20% 性能

---

### 问题 8：如何计算 GEMM 的理论峰值性能？

**参考答案**：

**理论峰值公式**：
```
峰值 TFLOPS = 2 × SM数量 × 每SM核心数 × 核心频率(GHz)
```

**以 RTX 4090 (Ada架构) 为例**：
```
SM 数量: 128
每SM FP32 核心: 128
核心频率: 2.52 GHz (Boost)

FP32 峰值 = 2 × 128 × 128 × 2.52
         = 82,575 GFLOPS
         ≈ 82.6 TFLOPS

FP16 (Tensor Core) 峰值 ≈ 330-660 TFLOPS
```

**实际影响因素**：
1. 指令发射效率（Warp 调度）
2. 内存带宽瓶颈
3. Occupancy 限制
4. Bank Conflict
5. 分支发散

**实际可达成性能**：
- 优化良好的 SGEMM：可达理论峰值的 70-90%
- cuBLAS：通常 80-95%

---

## 高级问题

### 问题 9：解释 Roofline 模型及其在 GEMM 优化中的应用。

**参考答案**：

**Roofline 模型**：
```
性能 = min(峰值算力, 内存带宽 × 计算强度)
                    ↑                    ↑
                 平顶部分              斜坡部分
              (Compute Bound)      (Memory Bound)
```

**关键参数**：
- **Ridge Point**：平顶与斜坡的交点，决定计算强度阈值
- **计算强度 > Ridge Point**：计算受限
- **计算强度 < Ridge Point**：内存受限

**GEMM 优化策略**：

1. **提高计算强度**：
   - 分块策略（Tiling）
   - 寄存器累加（每个线程计算多个输出元素）

2. **优化内存访问**：
   - 向量化加载（float4）
   - 共享内存缓存
   - 预取和双缓冲

3. **优化计算**：
   - 循环展开
   - 指令级并行

**可视化**：
```
性能 (GFLOPS)
    │
900 ┤──────────┐  ← 理论峰值
    │          │
600 ┤          │
    │          │
300 ┤    ╱     │  ← 带宽限制斜坡
    │  ╱       │
  0 ┼──┬──┬──┬─┴──→ 计算强度 (FLOP/Byte)
       1   10  100
       ↑
    Ridge Point
```

---

### 问题 10：设计一个针对特定 GPU 的 GEMM 优化方案需要考虑哪些因素？

**参考答案**：

**1. 硬件特性分析**：
```
- SM 数量、核心数、频率
- 共享内存大小和 Bank 数量
- 寄存器文件大小
- 内存带宽
- Tensor Core 支持情况
```

**2. 矩阵维度特征**：
```
- M, N, K 的大小（大矩阵 vs 小矩阵）
- 是否为方阵
- 是否有对齐要求
```

**3. 分块参数选择**：
```
- Block 大小 (BM, BN, BK)
- 每线程计算量 (TM, TN)
- 需平衡：Occupancy、寄存器使用、共享内存使用
```

**4. 数据类型考虑**：
```
- FP32 (最通用)
- FP16 (使用 Tensor Core)
- BF16 (AI 训练)
- INT8 (推理加速)
```

**5. 特殊优化技巧**：
```
- 对 K 的padding（对齐到分块边界）
- Strided/Batched GEMM 支持
- 边界处理（非对齐维度）
```

**6. 调优流程**：
```
1. 从理论分析确定参数范围
2. 实现基础版本
3. 使用 profiler (Nsight Compute) 分析瓶颈
4. 迭代优化参数
5. 对比理论峰值验证优化效果
```

---

## 总结

| 难度级别 | 考察重点 | 典型问题 |
|---------|---------|---------|
| 初级 | 基础概念 | 公式、复杂度、存储格式 |
| 中级 | 实现细节 | 分块策略、Bank Conflict、双缓冲 |
| 高级 | 系统设计 | Roofline、参数选择、调优流程 |

---

*文档生成时间：2026年3月19日*

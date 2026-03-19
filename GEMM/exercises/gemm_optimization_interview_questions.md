# GEMM 优化技巧面试题

本文档涵盖 GEMM 性能优化的深度面试题，包括寄存器优化、共享内存优化、向量化等高级技巧。

---

## 寄存器优化

### 问题 1：什么是"寄存器分块"(Register Tiling)？为什么它比单纯的 Shared Memory Tiling 更高效？

**参考答案**：

**寄存器分块定义**：
在 Shared Memory Tiling 的基础上，进一步让每个线程使用寄存器累加多个输出元素，减少对 Shared Memory 的访问次数。

**层次结构对比**：

| 优化级别 | 数据位置 | 访问速度 | 重用粒度 |
|---------|---------|---------|---------|
| Naive | Global Memory | 慢 | 无 |
| Shared Tiling | Shared Memory | 快 | Block 级别 |
| Register Tiling | Register File | 最快 | 线程级别 |

**为什么更高效**：

1. **减少 Shared Memory 访问**：
```cpp
// Shared Tiling：每个元素从 Shared Memory 读取
for (int k = 0; k < BK; k++) {
    float a = sA[ty][k];  // Shared Memory 读取
    float b = sB[k][tx];  // Shared Memory 读取
    accum += a * b;
}
// 总访问次数: 2 * BK (每次迭代 2 次 Shared Memory 访问)
```

```cpp
// Register Tiling：先加载到寄存器，复用多次
float frag_a[TM], frag_b[TN];
for (int k = 0; k < BK; k++) {
    // 加载到寄存器（TM+TN 次 Shared Memory 访问）
    for (int i = 0; i < TM; i++) frag_a[i] = sA[ty*TM+i][k];
    for (int j = 0; j < TN; j++) frag_b[j] = sB[k][tx*TN+j];
    
    // 寄存器计算（TM*TN 次运算，0 次 Shared Memory 访问）
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            accum[i][j] += frag_a[i] * frag_b[j];
}
// 总访问次数: (TM+TN) * BK ( amortized 后更少)
```

2. **更高的计算强度**：
   - 每轮迭代执行 $TM \times TN$ 次 FMA
   - 仅需 $TM + TN$ 次数据加载
   - 计算/加载比 = $(TM \times TN) / (TM + TN)$

3. **典型配置收益**：
   - TM=8, TN=8：计算/加载比 = 64/16 = 4
   - TM=16, TN=8：计算/加载比 = 128/24 ≈ 5.3

---

### 问题 2：寄存器使用过多会有什么副作用？如何在性能和 Occupancy 之间权衡？

**参考答案**：

**寄存器过多的副作用**：

1. **Occupancy 下降**：
```
每 SM 寄存器预算: 256 KB = 65,536 个 32-bit 寄存器

场景 A: 每线程 64 寄存器, 256 线程/Block
- 每 Block: 256 × 64 = 16,384 寄存器
- 最大 Block 数: 65,536 / 16,384 = 4
- Occupancy: (4 × 8 Warps) / 64 = 50%

场景 B: 每线程 128 寄存器, 256 线程/Block
- 每 Block: 256 × 128 = 32,768 寄存器
- 最大 Block 数: 65,536 / 32,768 = 2
- Occupancy: (2 × 8 Warps) / 64 = 25%
```

2. **延迟隐藏能力下降**：
   - 活跃 Warp 数减少 → 无法有效隐藏全局内存延迟
   - 可能从计算受限变为延迟受限

3. **编译器 spill**：
   - 超过每线程 255 寄存器限制
   - 编译器将多余变量存入 Local Memory（更慢）

**权衡策略**：

1. **Sweet Spot 分析**：
```
目标: 25-50% Occupancy + 足够每线程计算量

推荐配置:
- Block: 16×16 = 256 线程
- TM×TN: 8×8 到 16×8
- 寄存器: 64-128 个/线程
- Occupancy: 25-50%
```

2. **渐进式优化**：
```cpp
// 步骤 1: 基础版本 (低寄存器, 高 Occupancy)
TM=4, TN=4, 寄存器≈32, Occupancy=50%

// 步骤 2: 增加计算量
TM=8, TN=8, 寄存器≈80, Occupancy=25%
性能提升: 2-3x

// 步骤 3: 进一步优化 (谨慎)
TM=16, TN=8, 寄存器≈144, Occupancy=25%
如果代码效率提升 > 寄存器压力，则继续优化
```

3. **实际测试验证**：
   - 不要仅看理论分析
   - 使用 Nsight Compute 确认实际 Occupancy
   - 检查是否有寄存器 spill

---

## 向量化与内存优化

### 问题 3：什么是向量化加载/存储？在 GEMM 中如何实现？

**参考答案**：

**向量化定义**：
一次指令加载/存储多个连续的数据元素，减少指令数量和内存事务开销。

**CUDA 向量化类型**：
- `float4`：4 个 float (128 bit)
- `float2`：2 个 float (64 bit)
- `int4`, `int2` 等

**GEMM 实现**：

1. **加载 A 矩阵**：
```cpp
// 非向量化：4 次 32-bit 加载
float a0 = A[row * K + col + 0];
float a1 = A[row * K + col + 1];
float a2 = A[row * K + col + 2];
float a3 = A[row * K + col + 3];

// 向量化：1 次 128-bit 加载
float4 vec_a = reinterpret_cast<float4*>(A + row * K + col)[0];
// vec_a.x, vec_a.y, vec_a.z, vec_a.w
```

2. **协作向量化加载（Shared Memory 版本）**：
```cpp
// 128 个线程协作加载 128×8 的 A 分块
// 每个线程加载 8 个 float = 2 个 float4

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__shared__ float sA[128][8 + 1];  // padding 避免 conflict

int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0-127
int load_row = tid / 2;      // 64 行
int load_col = (tid % 2) * 4; // 0 或 4

// 每个线程加载 4 个元素
float4 vec = FETCH_FLOAT4(A[(by*128 + load_row) * K + load_col]);
sA[load_row][load_col + 0] = vec.x;
sA[load_row][load_col + 1] = vec.y;
sA[load_row][load_col + 2] = vec.z;
sA[load_row][load_col + 3] = vec.w;
```

3. **优势分析**：
```
带宽利用率:
- 标量加载: 约 50-70% 理论带宽
- 向量化加载: 可达 80-95% 理论带宽

指令效率:
- 4 次标量加载 → 4 条指令
- 1 次 float4 加载 → 1 条指令
```

**注意事项**：
- 地址必须 16-byte 对齐
- 可能需要 padding 来确保对齐
- 边界处理要注意（剩余不足 4 个元素时）

---

### 问题 4：解释"合并访问"(Coalesced Access)及其在 GEMM 中的重要性。

**参考答案**：

**合并访问定义**：
Warp 内的 32 个线程访问连续的内存地址，使得硬件可以将这些访问合并为少量内存事务。

**理想模式**：
```
线程 0: 访问地址 0x1000
线程 1: 访问地址 0x1004
线程 2: 访问地址 0x1008
...
线程 31: 访问地址 0x107C

合并为 1-2 个 128-byte 事务
```

**GEMM 中的应用**：

1. **A 矩阵访问（行主序）**：
```cpp
// ✅ 合并访问：每行内连续
int row = blockIdx.y * BM + threadIdx.y;
for (int k = 0; k < K; k++) {
    float val = A[row * K + k];  // 线程 (0,1,2...) 访问连续地址
}
```

2. **B 矩阵访问（行主序下的列访问）**：
```cpp
// ❌ 非合并：跨行访问，步长为 N
for (int k = 0; k < K; k++) {
    float val = B[k * N + col];  // 线程间地址差 N*sizeof(float)
}

// ✅ 解决方案：转置 B 或使用 Shared Memory
__shared__ float sB[BK][BN];
// 协作加载时让线程 0-31 加载 sB[k][0-31]（连续）
```

3. **C 矩阵写回**：
```cpp
// ✅ 合并写入
C[row * N + col] = accum;
```

**性能影响**：
```
全局内存带宽利用率:
- 完全合并: ~90%
- 部分合并: ~50%
- 完全分散: ~10-20%
```

---

## 同步与流水线

### 问题 5：`__syncthreads()` 在 GEMM 中有什么作用？是否可以减少同步次数？

**参考答案**：

**__syncthreads() 的作用**：

1. **确保数据可见性**：
```cpp
// 线程加载数据到 Shared Memory
sA[tid] = A[global_idx];
__syncthreads();  // 确保所有线程都完成加载，其他线程才能读取

// 线程计算（读取其他线程写入的 Shared Memory）
float val = sA[other_tid];
```

2. **防止读写竞争**：
```cpp
// 没有同步会导致问题
sA[tid] = load();       // 线程 A 写入
// ... 没有 __syncthreads() ...
compute(sA[tid + 1]);   // 线程 B 读取（可能读到旧值）
```

**GEMM 中的同步点**：

```cpp
for (int k = 0; k < K; k += BK) {
    // 同步点 1: 加载前（确保上一轮计算完成，可以覆盖数据）
    load_to_shared(A, B);
    
    // 同步点 2: 计算前（确保数据全部加载完成）
    __syncthreads();
    
    compute_with_shared_data();
    
    // 同步点 3: 下一轮加载前（确保计算完成，可以覆盖数据）
    __syncthreads();
}
```

**减少同步的策略**：

1. **双缓冲减少同步**：
```cpp
__shared__ float sA[2][BM][BK];
int buf = 0;

// 第一轮
load_to_buffer(buf);
__syncthreads();

for (int k = BK; k < K; k += BK) {
    compute_with_buffer(buf);  // 计算
    load_to_buffer(1 - buf);   // 同时加载到另一缓冲（不需要同步）
    __syncthreads();           // 只需要最后 1 次同步
    buf = 1 - buf;
}
```

2. **寄存器分块减少 Shared Memory 访问**：
```cpp
// 更多计算在寄存器中，减少对 Shared Memory 的同步依赖
float frag_a[TM], frag_b[TN];
// 加载到寄存器后，计算不需要同步
```

**同步开销**：
```
- __syncthreads() 通常需要 10-30 个时钟周期
- 在计算密集型 kernel 中占比不大
- 在访存密集型 kernel 中可能成为瓶颈
```

---

### 问题 6：如何实现异步数据加载（Async Copy）来进一步优化 GEMM？

**参考答案**：

**异步拷贝 API** (Compute Capability 8.0+，即 Ampere 及更新架构)：
```cpp
#include <cuda_pipeline.h>

// 异步从 Global Memory 加载到 Shared Memory
__pipeline_memcpy_async(&sA[local_idx], &A[global_idx], sizeof(float4));
```

**优化原理**：

1. **传统同步加载**：
```
线程 0: 加载 → 完成
线程 1: 加载 → 完成
...
线程 31: 加载 → 完成
__syncthreads()  // 等待所有线程
```

2. **异步加载**：
```
所有线程: 发起异步加载（立即返回）
__pipeline_commit();  // 提交所有异步操作
__pipeline_wait_prior(0);  // 等待完成
__syncthreads();  // 可选，确保全部完成
```

**GEMM 中的应用**：

```cpp
__shared__ float sA[BM][BK];
__shared__ float sB[BK][BN];

// 预加载第一块
for (int i = tid; i < BM*BK; i += blockDim.x) {
    __pipeline_memcpy_async(&sA[0][i], &A[global_i], sizeof(float));
}
__pipeline_commit();
__pipeline_wait_prior(0);
__syncthreads();

for (int k = BK; k < K; k += BK) {
    // 计算当前块
    compute(sA, sB);
    
    // 异步预加载下一块（与计算重叠）
    for (int i = tid; i < BM*BK; i += blockDim.x) {
        __pipeline_memcpy_async(&sA[0][i], &A[next_global_i], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);  // 等待加载完成
    __syncthreads();
}
```

**性能收益**：
```
- 隐藏加载延迟
- 减少线程等待时间
- 通常提升 5-15% 性能
```

---

## 实战分析

### 问题 7：如何分析一个 GEMM kernel 的性能瓶颈？

**参考答案**：

**分析工具链**：

1. **Nsight Compute 关键指标**：
```
- SOL (Speed of Light) 内存/计算利用率
- Achieved Occupancy vs Theoretical Occupancy
- 寄存器使用量和 Spill
- Shared Memory Bank Conflict
- 指令发射效率
```

2. **瓶颈识别流程**：
```
步骤 1: 检查 SOL 利用率
- 如果 SOL Memory > 60% → 内存受限
- 如果 SOL Compute > 60% → 计算受限
- 两者都低 → 可能是延迟或指令问题

步骤 2: 检查 Occupancy
- Achieved Occupancy << Theoretical → 检查寄存器/共享内存

步骤 3: 检查内存指标
- L2 Cache Hit Rate
- Global Memory 带宽利用率
- Shared Memory Bank Conflict

步骤 4: 检查指令效率
- 指令发射率
- Warp Stall 原因
- 分支发散比例
```

3. **常见瓶颈及解决方案**：

| 瓶颈 | 指标特征 | 解决方案 |
|-----|---------|---------|
| 内存带宽受限 | SOL Memory > 80%, SOL Compute < 50% | 增加分块大小，提高重用率 |
| 寄存器压力 | 寄存器 Spill, Occupancy 低 | 减少每线程寄存器使用 |
| Bank Conflict | Shared Load/Store 效率低 | Padding 或改变访问模式 |
| Warp Stall | Warp Stall 比例高 | 增加活跃 Warp 数，检查同步 |
| 分支发散 | 发散比例 > 5% | 优化边界处理逻辑 |

---

## 总结

| 优化技术 | 适用场景 | 预期收益 | 复杂度 |
|---------|---------|---------|--------|
| Shared Memory Tiling | 所有 GEMM | 5-10x | 低 |
| Register Tiling | 中大矩阵 | 2-3x | 中 |
| 向量化加载 | 对齐的矩阵 | 10-20% | 低 |
| 双缓冲 | 计算受限场景 | 10-20% | 中 |
| 异步拷贝 | Ampere+ 架构 | 5-15% | 高 |
| Bank Conflict 消除 | Shared Memory 访问频繁 | 5-15% | 中 |

---

*文档生成时间：2026年3月19日*

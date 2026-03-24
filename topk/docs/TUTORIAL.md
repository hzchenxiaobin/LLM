# CUDA Top-K 算子性能优化教程

> **针对 RTX 5090 (Blackwell 架构) 的极致优化指南**

---

## 📋 目录

1. [背景与硬件环境](#0-背景与硬件环境)
2. [V0: Thrust 基线版](#v0-thrust-基线版--全局排序)
3. [V1: 线程级局部 Top-K](#v1-线程级局部-top-k)
4. [V2: 共享内存优化](#v2-共享内存优化)
5. [V3: Warp Shuffle 原语](#v3-warp-shuffle-原语)
6. [V4: Radix Select 算法](#v4-radix-select-算法)
7. [V5: Blackwell 架构专属优化](#v5-blackwell-架构专属优化)
8. [学习路径总结](#学习路径总结)

---

## 0. 背景与硬件环境

### 问题定义

**目标**：给定长度为 $N$ 的无序浮点数组，找出最大的 $K$ 个元素（及其索引）。

**典型场景**：
- $N = 10^7$（千万级元素）
- $K \" | 256（LLM 推理采样、检索召回）

### RTX 5090 (Blackwell) 性能画像

| 特性 | 规格 | 优化意义 |
|------|------|----------|
| 计算单元 | 海量 SM | 恐怖并发能力，适合大规模并行 |
| 内存带宽 | GDDR7 ~1.8 TB/s | 需最大化利用显存带宽 |
| L2 缓存 | 巨大容量 | 中等规模数据可全缓存命中 |
| 新特性 | Thread Block Clusters, DSMEM, TMA | 跨 Block 协作、异步传输 |

---

## V0: Thrust 基线版 — 全局排序

### 思路

最简单直接：全局降序排序，取前 $K$ 个。

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

void topK_v0(float* d_in, int N, int K, float* d_out) {
    // 拷贝数据（避免破坏原数组）
    thrust::device_vector<float> d_vec(d_in, d_in + N);
    
    // 全局降序排序: O(N log N)
    thrust::sort(thrust::device, d_vec.begin(), d_vec.end(), 
                 thrust::greater<float>());
    
    // 取前 K 个
    thrust::copy(d_vec.begin(), d_vec.begin() + K, d_out);
}
```

### ⚠️ 性能痛点

| 指标 | 数值 | 分析 |
|------|------|------|
| 时间复杂度 | $O(N \" log \" N)$ | 完全排序造成巨大浪费 |
| 内存带宽 | 高浪费 | 只需 Top-K，却排序全部元素 |
| 实际表现 | 慢 | 千万级数据下性能差 |

> **关键洞察**：我们只需要前 $K$ 个，却对所有 $N$ 个元素完全排序。

---

## V1: 线程级局部 Top-K

### 核心思想：Map-Reduce

采用**分治法**：
1. **Map**：每个线程维护局部 Top-K，遍历数据子集
2. **Reduce**：合并所有线程结果，得到全局 Top-K

```cpp
template <int K>
__global__ void topK_v1_kernel(const float* input, int N, float* block_tops) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // 寄存器中的局部 Top-K
    float local_top[K];
    for (int i = 0; i < K; ++i) local_top[i] = -1e20f;
    
    // Grid-Stride Loop 遍历数据
    for (int i = tid; i < N; i += stride) {
        float val = input[i];
        // 插入排序逻辑
        if (val > local_top[K - 1]) {
            int j = K - 1;
            while (j > 0 && val > local_top[j - 1]) {
                local_top[j] = local_top[j - 1];
                j--;
            }
            local_top[j] = val;
        }
    }
    
    // 写入全局内存供后续规约
    int out_offset = tid * K;
    for (int i = 0; i < K; ++i) {
        block_tops[out_offset + i] = local_top[i];
    }
}
```

### 性能分析

| 方面 | 表现 | 说明 |
|------|------|------|
| 时间复杂度 | $O(N \" K)$ | 单遍扫描，无排序 |
| 改进 | ✅ | 避免全局排序 |
| **痛点 1** | ⚠️ 寄存器溢出 | K > 16 时溢出到 Local Memory，延迟飙升 |
| **痛点 2** | ⚠️ 全局内存风暴 | 每个线程写 $K$ 个数据，中间数据量巨大 |

---

## V2: 共享内存优化

### 优化目标

解决 V1 的"全局内存写操作过多"问题。

### 核心策略

1. 每个线程计算局部 Top-K（寄存器）
2. **Block 内写入 Shared Memory**（而非全局内存）
3. **Shared Memory 中树形规约**，每个 Block 只输出 $K$ 个值
4. 全局内存写入量：`num_threads * K` → `num_blocks * K`

```cpp
template <int K>
__global__ void topK_v2_kernel(const float* input, int N, float* grid_tops) {
    extern __shared__ float smem[];  // 共享内存声明
    
    int tid = threadIdx.x;
    int global_id = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // 1. 线程级局部 Top-K
    float local_top[K];
    // ... 计算 local_top ...
    
    // 2. 写入 Shared Memory
    for (int i = 0; i < K; ++i) {
        smem[tid * K + i] = local_top[i];
    }
    __syncthreads();
    
    // 3. Block 内树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            merge_topK_in_smem(smem, tid * K, (tid + s) * K, K);
        }
        __syncthreads();
    }
    
    // 4. 线程 0 写入全局内存
    if (tid == 0) {
        for (int i = 0; i < K; ++i) {
            grid_tops[blockIdx.x * K + i] = smem[i];
        }
    }
}
```

### 性能飞跃

| 指标 | V1 | V2 | 提升 |
|------|-----|-----|------|
| 全局内存写入 | `num_threads * K` | `num_blocks * K` | **32x 减少**（假设 256 线程/Block）|
| 显存带宽压力 | 高 | 显著降低 | ✅ |
| Shared Memory 速度 | - | ~L1 Cache 级别 | ✅ |

---

## V3: Warp Shuffle 原语

### 进一步优化方向

V2 使用 Shared Memory 规约仍需 `__syncthreads()` 和访存指令。

**Warp Shuffle 突破**：Warp 内（32 线程）可直接读取其他线程寄存器，无需经过共享内存！

### 优化策略

| 步骤 | 操作 | 效果 |
|------|------|------|
| 1 | 线程在寄存器求局部 Top-K | 基础计算 |
| 2 | **Warp 级规约**：`__shfl_down_sync` 归并 | 每 Warp 只 Lane 0 有效 |
| 3 | **Lane 0 写 Shared Memory** | 数据量减少 32 倍 |
| 4 | **Warp 0 完成最终合并** | 最少同步开销 |

```cpp
__device__ void warp_merge_topK(float* local_top) {
    for (int offset = 16; offset > 0; offset /= 2) {
        // 直接读取其他线程寄存器
        float other_val = __shfl_down_sync(0xffffffff, local_top[k], offset);
        // ... 合并逻辑 ...
    }
}
```

### 🚀 Blackwell 加成

Blackwell 架构对寄存器堆带宽和调度做了极大优化，**Warp-level 原语**是释放其计算密度的核心密钥。

---

## V4: Radix Select 算法

### 问题背景

当 $K=256$ 甚至 $1024$ 时，维护寄存器数组会导致"寄存器溢出"。

工业界方案（NVIDIA CUB, xformers）：**位级直方图统计（Radix Select）**

### 核心思想

不保存具体 $K$ 个元素，通过统计浮点数二进制分布来"猜"阈值：

```
┌─────────────────────────────────────────────────────┐
│  Radix Select 流程                                  │
├─────────────────────────────────────────────────────┤
│  Pass 1: 扫描第 31-24 位 → 统计 Histogram           │
│          ↓                                          │
│  Pass 2: 计算前 K 个元素落在哪个前缀区间          │
│          ↓                                          │
│  Pass 3-4: 逐层往下扫描（类似基数排序）             │
│          ↓                                          │
│  Final: 提取大于等于阈值的元素                      │
└─────────────────────────────────────────────────────┘
```

### 优势

- ✅ 消除分支预测失败（Branch Divergence）
- ✅ 无数组维护开销
- ✅ GPU 大 $K$ 值 Top-K 的最优解

---

## V5: Blackwell 架构专属优化

### 已优化内核 → 5090 更进一步

#### 1. 分布式共享内存 (DSMEM)

| 特性 | Hopper/Blackwell 新特性 |
|------|------------------------|
| Thread Block Clusters | 多 Block 组成 Cluster |
| 跨 Block 访问 | 直接访问彼此 Shared Memory |
| 优化点 | Grid 级合并无需写回全局显存 |

#### 2. L2 缓存持久化

```cpp
// CUDA 11.8+ 设置 L2 驻留
cudaStreamAttrValue policy;
policy.accessPolicyWindow.base_ptr = d_input;
policy.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &policy);
```

- 适用于多次 Pass 的 Radix Select
- 数据常驻 L2，访问速度起飞

#### 3. 异步 TMA (Tensor Memory Accelerator)

- 异步全局内存 → Shared Memory 传输
- CPU 指令发射无需等待
- 适合批量 Top-K（如 Transformer 注意力路由）

---

## 学习路径总结

| 阶段 | 版本 | 重点学习 | 目标 |
|------|------|----------|------|
| **入门** | V1 + V2 | `__syncthreads()`, Shared Memory Bank Conflict | 理解 GPU 内存层次 |
| **进阶** | V3 | Warp Shuffle, 寄存器级通信 | 体会极致速度 |
| **精通** | V4 | Radix Select, 位级操作 | 掌握工业级算法 |
| **实战** | V5 | DSMEM, L2 Persistence, TMA | Blackwell 专属优化 |

### 生产环境建议

> **永远不要在生产环境造轮子，除非你能比 CUB 写得快。**

```cpp
#include <cub/cub.cuh>

// 生产环境推荐
cub::DeviceSelect::TopK(d_temp_storage, temp_storage_bytes, 
                        d_in, d_out, N, K);
```

---

## 📚 附录

### 版本对比速查

| 版本 | 核心优化 | 适用场景 | 性能等级 |
|------|----------|----------|----------|
| V0 | Thrust 排序 | 快速原型 | ⭐ |
| V1 | Map-Reduce | 理解基础 | ⭐⭐ |
| V2 | Shared Memory | 减少全局写入 | ⭐⭐⭐ |
| V3 | Warp Shuffle | 寄存器级速度 | ⭐⭐⭐⭐ |
| V4 | Radix Select | 大 K 值最优 | ⭐⭐⭐⭐⭐ |
| V5 | Blackwell 特性 | 5090 极致性能 | ⭐⭐⭐⭐⭐ |

---

**祝你在 RTX 5090 的算力海洋里冲浪愉快！** 🏄‍♂️

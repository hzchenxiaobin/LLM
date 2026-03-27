# FlashAttention 基础知识手册

## 目录

1. [注意力机制基础](#1-注意力机制基础)
2. [GPU 架构基础](#2-gpu-架构基础)
3. [算法复杂度分析](#3-算法复杂度分析)
4. [Softmax 算法详解](#4-softmax-算法详解)
5. [CUDA 编程入门](#5-cuda-编程入门)
6. [FlashAttention 核心概念](#6-flashattention-核心概念)

---

## 1. 注意力机制基础

### 1.1 什么是注意力机制

注意力（Attention）机制是深度学习中的核心技术，灵感来源于人类视觉注意力的选择性聚焦特性。在神经网络中，它允许模型在处理序列数据时，动态地关注输入的不同部分。

**直观理解**：就像阅读文章时，你会根据当前问题重点关注某些关键词句，而忽略其他内容。

### 1.2 自注意力（Self-Attention）

自注意力是 Transformer 架构的核心，其数学定义为：

```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

其中：
- **Q (Query)**：查询向量，表示"我要查什么"
- **K (Key)**：键向量，表示"我有什么信息"
- **V (Value)**：值向量，表示"信息的具体内容"
- **d**：注意力头的维度

### 1.3 多头注意力（Multi-Head Attention）

将 Q、K、V 投影到多个子空间，并行计算多组注意力：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 1.4 计算过程拆解

标准注意力的计算分为三步：

1. **计算相似度**：$S = QK^T$ （得到 $N \times N$ 的分数矩阵）
2. **Softmax 归一化**：$P = \text{softmax}(S / \sqrt{d})$ （得到注意力权重）
3. **加权求和**：$O = PV$ （得到最终输出）

**问题**：当序列长度 $N$ 很大时，$N \times N$ 的矩阵需要巨大的内存存储。

---

## 2. GPU 架构基础

### 2.1 GPU 与 CPU 的区别

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数量 | 少（几十个） | 多（成千上万个） |
| 核心复杂度 | 复杂，功能强大 | 简单，专精计算 |
| 擅长任务 | 顺序执行、复杂逻辑 | 并行计算、大规模数据 |
| 内存带宽 | 相对较小 | 非常大 |

### 2.2 GPU 内存层次结构

现代 GPU 具有复杂的内存层级，理解它们是优化 CUDA 程序的关键：

```
┌─────────────────────────────────────┐
│           寄存器 (Registers)          │  ← 最快，每个线程私有
│        ~256 KB / SM，延迟 ~1 周期      │
├─────────────────────────────────────┤
│          共享内存 (Shared Memory)       │  ← 极快，同一块内线程共享
│       ~100-256 KB / SM，~19 TB/s      │
├─────────────────────────────────────┤
│        L2 缓存 (L2 Cache)              │  ← 快，所有 SM 共享
│          几 MB，带宽较高               │
├─────────────────────────────────────┤
│      全局内存 / HBM (Global Memory)     │  ← 较慢，容量大
│     40-80 GB，带宽 ~1.5-3 TB/s        │
└─────────────────────────────────────┘
```

#### 关键概念

- **HBM (High Bandwidth Memory)**：高带宽内存，即全局内存
- **SRAM (Static RAM)**：静态随机存取存储器，即共享内存
- **SM (Streaming Multiprocessor)**：流式多处理器，GPU 的计算单元

### 2.3 CUDA 执行模型

```
Grid（网格）
├── Block（线程块）
│   ├── Warp（线程束，32 个线程）
│   │   ├── Thread（线程）
│   │   └── Thread
│   └── Warp
└── Block
```

**重要参数**：
- **线程 (Thread)**：最基本的执行单元
- **线程束 (Warp)**：32 个线程为一组，同步执行
- **线程块 (Block)**：一组线程，共享共享内存
- **网格 (Grid)**：所有线程块的集合

### 2.4 内存访问模式

**合并访存 (Coalesced Memory Access)**：
- GPU 喜欢连续的内存访问模式
- 当一个 Warp 中的线程访问连续的内存地址时，可以合并为一次内存事务
- 随机访问会导致性能急剧下降

---

## 3. 算法复杂度分析

### 3.1 大 O 记法

用于描述算法随输入规模增长的时间/空间需求：

| 复杂度 | 名称 | 示例 |
|--------|------|------|
| $O(1)$ | 常数 | 数组索引访问 |
| $O(\log N)$ | 对数 | 二分查找 |
| $O(N)$ | 线性 | 遍历数组 |
| $O(N \log N)$ | 线性对数 | 快速排序 |
| $O(N^2)$ | 平方 | 双重循环、矩阵乘法 |
| $O(2^N)$ | 指数 | 穷举所有子集 |

### 3.2 标准注意力的复杂度

**时间复杂度**：$O(N^2 \cdot d)$
- $QK^T$ 矩阵乘法：$O(N \cdot d \cdot N) = O(N^2 d)$
- Softmax 计算：$O(N^2)$
- 与 V 相乘：$O(N \cdot N \cdot d) = O(N^2 d)$

**空间复杂度**：$O(N^2)$
- 需要存储 $N \times N$ 的注意力分数矩阵
- 当 $N = 100K$ 时，$N^2 = 10^{10}$ 个元素！

### 3.3 内存带宽瓶颈

**算术强度 (Arithmetic Intensity)**：
```
算术强度 = 浮点运算次数 / 内存访问字节数
```

- 当算术强度低时，运算被内存带宽限制 (**Memory-bound**)
- 当算术强度高时，运算被计算能力限制 (**Compute-bound**)

标准注意力是典型的 **Memory-bound** 操作。

---

## 4. Softmax 算法详解

### 4.1 Softmax 定义

Softmax 将任意实数向量转换为概率分布：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

### 4.2 数值稳定性问题

直接计算 $e^x$ 会导致数值溢出（当 $x$ 很大时）。

**解决方案 - 数值稳定的 Softmax**：
```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

减去最大值保证指数的最大值为 0，避免溢出。

### 4.3 Online Softmax

传统 Softmax 需要两次遍历数据：
1. 找到最大值
2. 计算指数和并归一化

**Online Softmax** 可以在一次遍历中完成，适合流式处理：

对于分块输入 $[x^{(1)}, x^{(2)}]$：
```
m_new = max(m_old, m_current)
l_new = l_old * exp(m_old - m_new) + Σ exp(x_current - m_new)
```

通过维护运行时的最大值和指数和，可以逐步更新 Softmax 结果。

---

## 5. CUDA 编程入门

### 5.1 基本结构

```cuda
// 核函数定义
__global__ void kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

// 主机代码调用
int main() {
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // 启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    
    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
}
```

### 5.2 共享内存使用

```cuda
__global__ void sharedMemoryExample(float* input, float* output) {
    __shared__ float sharedData[256];  // 声明共享内存
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sharedData[tid] = input[gid];
    __syncthreads();  // 同步块内所有线程
    
    // 使用共享内存进行计算
    // ...
    
    __syncthreads();  // 再次同步
    output[gid] = sharedData[tid];
}
```

### 5.3 关键优化技巧

1. **合并访存**：确保线程访问连续内存地址
2. **避免 bank conflict**：共享内存访问模式优化
3. **循环展开**：减少循环开销
4. **指令级并行**：隐藏指令延迟

---

## 6. FlashAttention 核心概念

### 6.1 核心问题

标准注意力的瓶颈：
- **内存**：$O(N^2)$ 的中间矩阵存储
- **IO**：频繁的 HBM 读写
- **利用率**：算力闲置，等待数据

### 6.2 FlashAttention 解决思路

**核心思想**：IO-Aware 算法设计

```
┌─────────────────────────────────────────────────────┐
│  标准注意力                    FlashAttention       │
│  ───────────                   ─────────────      │
│                                                     │
│  HBM → SRAM → 计算 → HBM       HBM → SRAM          │
│       ↑                              ↓              │
│       └──────── 多次往返 ────────────┘              │
│                              一次性计算完成         │
└─────────────────────────────────────────────────────┘
```

### 6.3 关键技术

#### Tiling（分块）

将大矩阵分解为适合 SRAM 的小块：

```
原始矩阵: N × d          分块后:
┌─────────────┐          ┌─────┬─────┬─────┐
│             │          │ Q_1 │ Q_2 │ ... │  (Br × d)
│    N × d    │    →     ├─────┼─────┼─────┤
│             │          │ ... │ ... │ ... │
└─────────────┘          └─────┴─────┴─────┘
```

#### Kernel Fusion（核函数融合）

将多个操作合并为一个 CUDA 核函数：
- 矩阵乘法
- Softmax
- 掩码应用
- 与 V 相乘

**优势**：减少中间结果的 HBM 往返

#### Recomputation（重计算）

训练时不在 HBM 保存中间矩阵，反向传播时重新计算：
- 仅保存 $O(N)$ 的统计量（最大值、指数和）
- 用更多计算换取更少内存带宽

### 6.4 版本演进

| 版本 | 核心改进 | GPU 架构 | 利用率 |
|------|----------|----------|--------|
| FA-1 | Tiling + Online Softmax | V100/A100 | 25-40% |
| FA-2 | Warp 级并行优化 | A100 | 50-73% |
| FA-3 | 异步 TMA + WGMMA | H100 | ~740 TFLOPS |
| FA-4 | TMEM + UMMA + CuTeDSL | B200 | ~1613 TFLOPS |

### 6.5 Flash-Decoding

针对推理阶段（Decode phase）的优化：
- 问题：Query 是单个向量，并行度极低
- 解决：在 KV Cache 序列维度上并行拆分
- 效果：长序列推理速度提升 8 倍

---

## 7. 关键术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| 注意力头 | Attention Head | 并行的注意力计算单元 |
| 查询/键/值 | Query/Key/Value | 注意力机制的三个输入矩阵 |
| HBM | High Bandwidth Memory | GPU 全局内存 |
| SRAM | Static RAM | GPU 共享内存，速度快容量小 |
| SM | Streaming Multiprocessor | GPU 流式多处理器 |
| Warp | Warp | 32 个线程的执行单元 |
| Tiling | Tiling | 将大矩阵分解为小块的技术 |
| Kernel Fusion | Kernel Fusion | 合并多个 CUDA 核函数 |
| Online Softmax | Online Softmax | 流式计算的 Softmax |
| Recomputation | Recomputation | 反向传播时重新计算中间结果 |
| TMA | Tensor Memory Accelerator | 张量内存加速器（Hopper） |
| WGMMA | Warpgroup MMA | Warp 组矩阵乘加（Hopper） |
| TMEM | Tensor Memory | 张量内存（Blackwell） |
| UMMA | Unified MMA | 统一矩阵乘加（Blackwell） |

---

## 8. 学习路径建议

### 前置知识
1. 线性代数（矩阵运算）
2. Python 和 PyTorch 基础
3. 深度学习基础（Transformer 架构）

### 进阶学习
1. CUDA C/C++ 编程
2. GPU 架构深入理解
3. 性能分析与调优（Nsight Compute/Systems）

### 推荐阅读
- 《CUDA C Programming Guide》
- 《Programming Massively Parallel Processors》
- 《Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM》

---

*本文档是 FlashAttention 教程的配套基础知识手册，建议结合 [TUTORIAL.md](./TUTORIAL.md) 阅读。*

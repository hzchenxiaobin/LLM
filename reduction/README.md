# CUDA Reduction 算子性能优化教程：从零到 RTX 5090 极致榨能

## 前言与背景

在并行计算中，Reduction（归约）是指将一个数组通过某种满足结合律的操作（如加法、乘法、最大值、最小值）缩减为一个单一标量值的过程。  
本教程以**数组求和 (Sum Reduction)** 为例。

**性能评估指标**：  
由于 Reduction 是访存密集型（Memory-Bound）任务，我们通常不看 TFLOPS，而是看**有效显存带宽 (Effective Bandwidth)**。  
公式为：`Bandwidth = (N * sizeof(type)) / Time`。  
在 RTX 5090 上，我们的目标是让这个数值尽可能逼近其理论显存带宽。

## 核心概念：两阶段归约

由于跨 Block（线程块）的同步在 CUDA 中开销极大，标准的做法是分为两步（或使用原子操作）：

1. **Block-level Reduction**：每个 Block 负责处理数组的一部分，将其归约为了一个局部结果（Local Sum）。
2. **Global-level Reduction**：将所有 Block 的局部结果相加，得到最终结果。在现代 GPU（包括你的 RTX 5090）上，由于 L2 Cache 极大且原子操作极快，我们通常在内核末尾直接使用 `atomicAdd` 将 Block 的结果累加到全局内存中。

接下来，我们将重点优化 **Block-level Reduction**。

## 第 1 步：朴素版本 (Interleaved Addressing)

这是最直观的树状归约。我们使用 Shared Memory（共享内存）来存储线程块内的数据。

**逻辑**：  
第一轮，跨度为 1，线程 0 加上线程 1，线程 2 加上线程 3...  
第二轮，跨度为 2，线程 0 加上线程 2，线程 4 加上线程 6...

```
__global__ void reduce_v1(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    // 每个线程加载一个元素到共享内存
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads(); // 等待所有线程加载完毕

    // 在共享内存中进行树状归约
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 线程 0 将当前 block 的结果写入全局内存（或者使用原子操作累加）
    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

```

🔴 **致命问题 (Warp Divergence)**：  
在 `if (tid % (2 * s) == 0)` 这行代码中。GPU 是以 Warp (32个线程) 为单位执行指令的。在第一轮，只有偶数线程工作，奇数线程空闲。这导致同一个 Warp 发生严重的**分支发散 (Warp Divergence)**，执行效率直接减半。

## 第 2 步：解决分支发散 (Strided Index)

为了解决 Warp Divergence，我们改变归约的步长和线程的映射方式，让连续的线程去处理数据。

**逻辑**：  
第一轮：跨度为 1，线程 0 处理索引 0,1；线程 1 处理索引 2,3...  
改为：  
线程 `tid` 总是活动的，它去访问 `2 * s * tid`。

```
__global__ void reduce_v2(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // 改变步长策略，保证前面的线程总是满载的
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

```

🟢 **改进**：消除了早期的 Warp Divergence。  
🔴 **新问题 (Bank Conflict)**：当线程访问 Shared Memory 时，步长变成了 2、4、8... 这会导致严重的**共享内存冲突 (Bank Conflict)**，因为多个线程试图同时访问同一个内存 Bank 中的不同地址。

## 第 3 步：解决 Bank Conflict (Sequential Addressing)

我们将步长从大到小排列（反向归约）。

**逻辑**：  
假设 Block 大小为 256。  
第一轮：前 128 个线程工作。线程 0 加上线程 128；线程 1 加上线程 129...（步长 128）  
第二轮：前 64 个线程工作。线程 0 加上线程 64...（步长 64）

```
__global__ void reduce_v3(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // 反向步长遍历
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

```

🟢 **改进**：完美解决了 Bank Conflict，且无 Warp Divergence（直到活跃线程数小于32）。性能会有质的飞跃。

## 第 4 步：提高指令吞吐与隐藏延迟 (First Add During Load)

在之前的版本中，当一半的线程退出时（例如 256 个线程启动后，立马只有 128 个参与归约），另一半线程完全白白浪费了。  
我们可以**在从全局内存 (Global Memory) 读入数据到共享内存时，顺便做一次加法**。这直接将 Block 的数量减半，大幅减少了 Kernel 启动和调度的开销。

```
__global__ void reduce_v4(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // 注意这里：每个线程现在负责 2 个元素
    // blockDim.x * 2 是因为每个 block 处理双倍的数据
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 加载时做第一次加法
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();

    // 保持和 v3 相同的 shared memory 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

```

## 第 5 步：现代 GPU 杀手锏 —— Warp Shuffle (终结 Shared Memory)

当归约进行到最后阶段（`s <= 16` 时），所有的活跃线程都在同一个 Warp (前32个线程) 内。  
此时还在使用 Shared Memory 和 `__syncthreads()` 是巨大的浪费。RTX 5090 支持极速的 **Warp 级寄存器通信 (Warp Shuffle)**。我们可以让线程直接读取其他线程的寄存器值！

```
// 辅助函数：Warp内归约
__inline__ __device__ float warpReduceSum(float val) {
    // __shfl_down_sync 允许当前线程获取 ID 为 (当前线程ID + offset) 的线程的寄存器值
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v5_warp_shuffle(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 1. 全局内存加载并做第一次归约
    float sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];
    
    sdata[tid] = sum;
    __syncthreads();

    // 2. 共享内存归约，直到剩余 32 个元素 (1 个 Warp)
    // 假设 blockDim.x = 256
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 将最后 32 个元素交给 Warp 0 进行纯寄存器归约
    if (tid < 32) {
        // 先加上 s=32 那一轮的值
        if (blockDim.x >= 64) sum = sdata[tid] + sdata[tid + 32];
        else sum = sdata[tid];
        
        // 纯寄存器级别的极致规约，不需要任何 __syncthreads()
        sum = warpReduceSum(sum);
        
        // 写入结果
        if (tid == 0) atomicAdd(g_odata, sum);
    }
}

```

## 第 6 步：终极带宽榨取 —— 向量化访存 (Vectorized Memory Access)

在 RTX 5090 这样的怪兽上，单次读取一个 `float` (4 Bytes) 根本无法塞满其显卡总线。我们要一次性读取 4 个 float，即使用 `float4` (16 Bytes)，这使得 L2 Cache 的命中和总线利用率达到最高。

这是目前手写 CUDA Reduction 最接近极限性能的做法。

```
// 注意：处理的数组长度 n 最好是 4 的整数倍
__global__ void reduce_v6_vectorized(float *g_idata, float *g_odata, unsigned int n) {
    // 强转指针以支持 float4 读取
    float4 *g_idata_f4 = reinterpret_cast<float4*>(g_idata);
    
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // 每个线程现在加载一个 float4（4个float），相当于原来4个线程的工作量
    // 加上 grid-stride loop 思想，让一个 block 可以处理多个数据块
    float sum = 0.0f;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop: 处理大数组
    while (i < n / 4) {
        float4 vec = g_idata_f4[i];
        sum += vec.x + vec.y + vec.z + vec.w;
        i += stride;
    }
    // 处理剩余的不满 4 个的元素 (省略边界判断代码以保持清晰...)

    // 将局部 sum 存入 shared memory (或者直接开始跨线程规约)
    sdata[tid] = sum;
    __syncthreads();

    // --- 之后的部分与 V5 完全一致 ---
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockDim.x >= 64) sum = sdata[tid] + sdata[tid + 32];
        else sum = sdata[tid];
        
        // __shfl_down_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) atomicAdd(g_odata, sum);
    }
}

```

## 结语与下一步建议

通过以上 6 步，你的代码性能将会有百倍以上的提升，并且能够跑满 RTX 5090 的大部分显存带宽。

**工程实践提示**：  
虽然手写 Reduction 对于学习非常有帮助，但在实际的项目工程中（如深度学习、科学计算），强烈建议**不要自己造轮子**，而是使用 NVIDIA 官方的 **CUB 库 (CUDA Unbound)**。CUB 内部使用了 `Cooperative Groups`、高度模板化的展开以及针对不同架构（包括 Blackwell）的微调，性能比手写还要高出 5%~10%。

希望这个教程能帮助你驾驭你的 RTX 5090！
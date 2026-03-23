# CUDA Softmax 性能优化教程（面向 RTX 5090）

## 0. 背景知识与数学公式

Softmax 通常应用于二维张量 `[M, N]`，其中：
- `M` 可以看作 `batch_size * num_heads`
- `N` 是序列长度（sequence_length）或词表大小（vocab_size）

我们对每一行独立计算 Softmax。

为了保证数值稳定性（防止 $e^x$ 溢出），标准的 Safe Softmax 公式分为三步：

**求最大值：**
$$
m = \max_{i=1}^N (x_i)
$$

**求指数和：**
$$
d = \sum_{i=1}^N \exp(x_i - m)
$$

**归一化：**
$$
y_i = \frac{\exp(x_i - m)}{d}
$$

---

## V1：最朴素的实现（Naive Softmax）

最直观的做法是严格按照公式，写三个 Kernel：

1. **Kernel_Max**：遍历数据找最大值，写回显存
2. **Kernel_Sum**：读取数据和最大值，计算 exp 并求和，写回显存
3. **Kernel_Div**：读取数据、最大值和和，计算最终结果，写回显存

### 性能分析

这是最糟糕的做法。对于每个元素：
- 需要从全局内存（Global Memory）读取 **3 次**
- 需要写入 **3 次** 中间结果

RTX 5090 虽然有极高的 GDDR7 带宽，但也经不起这样挥霍。

---

## V2：算子融合与 Shared Memory（Block-level）

为了减少全局内存访存，我们将三个步骤**融合**（Kernel Fusion）到一个 Kernel 中。

**策略**：让一个 Thread Block 处理矩阵的一行。将整行数据加载到共享内存（Shared Memory）中，在 Block 内部完成归约（Reduction）。

```cuda
// 假设 blockDim.x = 256，一个 Block 处理一行
__global__ void softmax_v2_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x; // 当前处理的行
    if (row >= M) return;

    int tid = threadIdx.x;
    const float* x = input + row * N;
    float* y = output + row * N;

    // 分配共享内存用于 Block 内部规约
    extern __shared__ float sdata[];

    // 1. 寻找局部最大值并存入寄存器
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Block 级别归约求全局最大值 (简化的归约逻辑)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // 2. 计算指数并求局部和
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(x[i] - row_max);
        local_sum += val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block 级别归约求全局和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // 3. 归一化并写回全局内存
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}
```

### 进步

- 全局内存只被读取 **2 次**（找最大值/求和读取 1 次，最后写回读取 1 次）
- 避免了中间结果的写回

### 缺点

- `__syncthreads()` 开销较大
- 如果 N 比较小（比如 128 或 256），一个 Block 处理一行会导致大量的 SM 资源闲置

---

## V3：Warp 级别规约（Warp-level Reduction）

对于大语言模型（LLM）等场景，有时候 N 不大，或者为了极致的性能，我们摒弃 Shared Memory，转而使用 NVIDIA 硬件提供的 **Warp Shuffle** 指令。

**策略**：让一个 Warp（32个线程）处理一行。这样根本不需要 `__syncthreads()`，线程间通信直接通过寄存器完成！

```cuda
// 辅助函数：Warp 内求最大值
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 辅助函数：Warp 内求和
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 一个 Warp 处理一行，假设 blockDim.x = 128 (包含 4 个 Warp)
__global__ void softmax_v3_warp_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    int row = warp_id;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    // 1. 求最大值
    float local_max = -INFINITY;
    for (int i = lane_id; i < N; i += 32) {
        local_max = fmaxf(local_max, x[i]);
    }
    float row_max = warpReduceMax(local_max);
    // 将 Warp 的第 0 个线程的结果广播给所有线程
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // 2. 求指数和
    float local_sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        local_sum += expf(x[i] - row_max);
    }
    float row_sum = warpReduceSum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // 3. 写回
    for (int i = lane_id; i < N; i += 32) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}
```

### 进步

- 消除了 Shared Memory 的分配和 `__syncthreads()` 的同步开销
- 利用寄存器通信，速度极快
- RTX 5090 的每个 SM 有大量的寄存器，非常适合这种写法

---

## V4：向量化访存（Vectorized Memory Access）

RTX 5090 拥有惊人的显存带宽，但如果你每次只取一个 float（32 bits/4 Bytes），指令发射的开销会成为瓶颈，你无法打满带宽。

**策略**：使用 `float4`，每次读取 128 bits（16 Bytes）。这被称为向量化访存，是压榨显存带宽的标配操作。

```cuda
// 将指针强转为 float4 进行操作
__global__ void softmax_v4_vectorized_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float4* x_vec = reinterpret_cast<const float4*>(input + row * N);
    float4* y_vec = reinterpret_cast<float4*>(output + row * N);

    // 注意：N 必须是 4 的倍数，这里假设 N 能被 4 整除
    int N_vec = N / 4;

    // ... 后续逻辑与 V3 类似，只是每次处理 float4 的 x, y, z, w 四个分量
    // 例如求最大值：
    float local_max = -INFINITY;
    for (int i = lane_id; i < N_vec; i += 32) {
        float4 val = x_vec[i];
        local_max = fmaxf(local_max, val.x);
        local_max = fmaxf(local_max, val.y);
        local_max = fmaxf(local_max, val.z);
        local_max = fmaxf(local_max, val.w);
    }
    // ... Warp Reduce 等同 V3
}
```

### 进步

- 大幅提升显存读取和写入吞吐量（Throughput）
- 减少内存指令发射数量

---

## V5：终极优化 —— Online Softmax (FlashAttention 的基石)

前面的所有版本，我们都需要遍历数组 **两次**：
1. 第一次找 max
2. 第二次算 sum

这导致了至少 2 次全局内存读取。

**有没有办法只遍历一次全局数据？**

答案是 **Online Softmax**，由 FlashAttention 的作者推广。它通过数学变换，在一次遍历中同时更新 max 和 sum。

### 数学推导

假设我们遍历到第 $i$ 个元素，当前的局部最大值是：
$$
m_{new} = \max(m_{old}, x_i)
$$

那么之前的局部和 $d_{old}$ 该怎么修正呢？只需乘以一个修正系数：
$$
d_{new} = d_{old} \times \exp(m_{old} - m_{new}) + \exp(x_i - m_{new})
$$

```cuda
// Online Softmax (单次遍历更新 max 和 sum)
__global__ void softmax_v5_online_kernel(const float* input, float* output, int M, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // 单次遍历数据
    for (int i = lane_id; i < N; i += 32) {
        float val = x[i];
        float new_max = fmaxf(local_max, val);
        // 修正之前的 sum
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }

    // Warp 内跨线程规约合并 local_max 和 local_sum
    // 这里的规约稍微复杂一点，因为 max 和 sum 需要一起规约
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);

        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max) + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    float row_max = __shfl_sync(0xffffffff, local_max, 0);
    float row_sum = __shfl_sync(0xffffffff, local_sum, 0);

    // 第二次遍历写回结果（此时只需要算除法）
    // 注意：这里的读取可能命中 RTX 5090 巨大的 L2 Cache，实际并不触发全局内存 IO
    for (int i = lane_id; i < N; i += 32) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}
```

### 终极形态

结合 **V4** (float4 向量化) + **V5** (Online Softmax)。

由于第一次遍历的读取会被 RTX 5090 超大的 L2 Cache 缓存，第二次遍历写回时的读取几乎全是 Cache Hit（只要 N 不是大得离谱）。这就把 Softmax 的显存读写瓶颈逼到了物理极限！

---

## 总结

在 RTX 5090 上优化 Softmax 的路径是：

### 算法级
标准三趟 → Block 融合 → Warp 级别规约 → Online Softmax

### 访存级
单精度浮点读取 (float) → 向量化读取 (float4)

### 架构级
利用寄存器（Warp Shuffle）替代 Shared Memory 同步

---

## 编译与运行

你可以将这些代码放入 `.cu` 文件中，使用以下命令编译：

```bash
nvcc -O3 -arch=sm_90a -o softmax_benchmark softmax.cu
```

> **注意**：RTX 5090 属于 Blackwell 架构，请根据实际安装的 CUDA toolkit 版本选择最新的 `-arch` 参数。

利用 `ncu` (Nsight Compute) 观察 Memory Throughput，你会发现 V5 版本的带宽利用率会贴近显卡的理论上限。

# CUDA Scan (前缀和) 性能优化进阶指南

前缀和（Scan）算法分为**包含型（Inclusive）**和**排他型（Exclusive）**。排他型通常更常用（例如用于数组压缩、基数排序等）。本文以**排他型前缀和 (Exclusive Scan)** 为例。

给定输入数组 `[3, 1, 7, 0, 4, 1, 6, 3]`，排他型前缀和的结果为 `[0, 3, 4, 11, 11, 15, 16, 22]`。

---

## 优化第 1 步：朴素的并行扫描 (Hillis-Steele 算法)

最直观的想法是基于步骤的并行计算。

在第 $d$ 步，每个线程将其当前元素与距离它 $2^{d-1}$ 的元素相加。

### 核心思想

- **步骤 1**：间隔为 1 相加
- **步骤 2**：间隔为 2 相加
- **步骤 3**：间隔为 4 相加...

```cuda
// 伪代码与简单实现 (Inclusive Scan)
__global__ void naive_scan_kernel(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[]; // 分配两倍的 shared memory 用于双缓冲
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    // 将数据从 Global Memory 读入 Shared Memory
    temp[pout * n + thid] = (thid < n) ? g_idata[thid] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // 交换双缓冲
        pin  = 1 - pout;

        if (thid >= offset) {
            temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
        } else {
            temp[pout * n + thid] = temp[pin * n + thid];
        }
        __syncthreads();
    }

    // 写回 Global Memory
    g_odata[thid] = temp[pout * n + thid];
}
```

### 痛点分析

**工作效率低下 (Work-Inefficient)**：串行 Scan 只需要 $O(N)$ 次加法。而这个算法执行了 $O(N \log N)$ 次加法。在 RTX 5090 这样拥有巨量计算单元的卡上，虽然算力充沛，但多余的指令意味着多余的能耗和寄存器占用。

---

## 优化第 2 步：工作高效的扫描 (Blelloch 算法)

为了将工作复杂度降到 $O(N)$，我们引入树状归约的思想，分为两步：

- **Up-Sweep (归约阶段)**：自底向上构建部分和，类似 Reduce
- **Down-Sweep (分发阶段)**：自顶向下计算前缀和。将根节点设为 0，然后向下传递并交换相加

```cuda
__global__ void blelloch_scan_kernel(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1;

    // 每个线程加载两个元素到 Shared Memory
    temp[2*thid] = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];

    // 1. Up-Sweep 阶段
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // 将最后一个元素设为 0 (Exclusive Scan 的关键)
    if (thid == 0) { temp[n - 1] = 0; }

    // 2. Down-Sweep 阶段
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // 写回 Global Memory
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}
```

### 痛点分析

**Shared Memory Bank Conflicts (显存组冲突)**：CUDA 的 Shared Memory 被分为 32 个 Bank。在 Up-Sweep 和 Down-Sweep 的循环中，我们的访问步长（stride）会翻倍（2, 4, 8...），这会导致严重的 Bank 冲突，使得 Shared Memory 的带宽急剧下降。

---

## 优化第 3 步：消除 Bank Conflicts (Padding 技巧)

为了打破步长为 $2^n$ 导致的 Bank 冲突，我们在计算 Shared Memory 索引时，人为插入空隙（Padding）。

一个宏观的经验法则是：每遇到 **NUM_BANKS**（通常是 32）个元素，就跳过一个位置。

```cuda
#define NUM_BANKS 32
#define CONFLICT_FREE_OFFSET(n) ((n) >> 5) // 等价于 n / 32

__global__ void bank_free_scan_kernel(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;

    // 计算无冲突的索引
    int ai = thid;
    int bi = thid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    // 后续的 Up-sweep 和 Down-sweep 代码中
    // 每次计算出的 index 都需要加上 CONFLICT_FREE_OFFSET(index)
    // ... (代码略，逻辑与第二步相同，仅索引改变)
}
```

这一步能大幅度提升 Block 内部的执行速度。

---

## 优化第 4 步：极致榨干 RTX 5090 (Warp-Level Primitives)

上述优化依赖于 `__syncthreads()` 同步整个 Block（通常包含 256 或 512 个线程）。

但现代 NVIDIA 显卡（特别是 RTX 5090 的 Blackwell 架构）在 **Warp**（32个线程为一组）内部具有极高带宽的寄存器级通信指令，比如 `__shfl_up_sync`。

利用 **Warp 原语**，我们可以完全抛弃 Shared Memory，在寄存器层面完成 32 个元素的 Scan，这被称为 **Warp Scan**。速度快到飞起！

```cuda
// 寄存器级别的 Warp Inclusive Scan
template <typename T>
__device__ __forceinline__ T warp_scan_inclusive(T val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        // 从前面的线程获取值
        T n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += n;
        }
    }
    return val;
}

// 利用 Warp Scan 构建 Block Scan
template <typename T>
__device__ __forceinline__ T block_scan(T val, T* shared_mem) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 1. 在每个 Warp 内部进行 Scan
    T warp_sum = warp_scan_inclusive(val);

    // 2. 将每个 Warp 的总和写入 Shared Memory
    if (lane_id == 31) {
        shared_mem[warp_id] = warp_sum;
    }
    __syncthreads();

    // 3. 用第一个 Warp 对前面得到的 warp_sums 进行 Scan
    if (warp_id == 0) {
        T sum = (lane_id < (blockDim.x / 32)) ? shared_mem[lane_id] : 0;
        sum = warp_scan_inclusive(sum);
        shared_mem[lane_id] = sum;
    }
    __syncthreads();

    // 4. 将之前的 Warp Scan 结果加上前置 Warp 的总和
    T block_prefix = (warp_id == 0) ? 0 : shared_mem[warp_id - 1];
    return warp_sum + block_prefix;
}
```

在这个阶段，单个 Block（通常最大 1024 个线程）内部的 Scan 性能已经接近硬件极限。

---

## 优化第 5 步：处理任意长度数组与单趟扫描 (Decoupled Look-back)

前面的步骤只能处理一个 Block 内的数据。如果有一个 10 亿元素的数组（这对 RTX 5090 是小菜一碟），需要多个 Block。

### 传统的 3-pass 方法

1. 每个 Block 算自己的 Scan，并把总和写入全局内存的一个数组 `block_sums`
2. 启动一个小的 Kernel 对 `block_sums` 进行 Scan
3. 再启动一个 Kernel，把算好的 `block_sums` 加回到每个 Block 的元素上

> **缺点**：频繁访问 Global Memory，读写了多次

### 终极杀器：Decoupled Look-back (单趟扫描 Single-pass Scan)

这是 Nvidia CUB 库中使用的前沿算法。RTX 5090 带宽极高，我们希望数据从全局内存只读入一次，写出一次。

**思路简介**：

每个 Block 在计算完自己的局部 Scan 后，动态地向 Global Memory 中发布自己的状态（未就绪、仅有局部总和、已有全局前缀和）。

紧接着，它会往回「看」（Look-back）前面 Block 的状态：

- 如果前面的 Block 算完了全局前缀和，它直接拿过来加上自己的局部结果并输出
- 如果前面的 Block 只有局部和，它就累加局部和并继续往前看

由于涉及复杂的**原子操作（Atomic Operations）**和**内存一致性模型（Memory Fences）**，实现起来非常复杂。

---

## 总结与给你的建议

既然你在学习阶段，并且手握 RTX 5090，我建议的学习路径：

1. 先手写并跑通 **Blelloch (Bank conflict-free)** 版本，理解 Shared Memory 和线程同步的开销
2. 重点学习和手写 **Warp-Level Primitives** 版本，这是现代 CUDA 优化的核心基石

> **在生产环境中，千万不要自己手写 Scan！** 直接调用 Nvidia 的开源库 **CUB** (`cub::DeviceScan::ExclusiveSum`)。CUB 内部使用了 Decoupled Look-back 算法，结合了 Warp 原语和极致的向量化访存（`float4` 加载），能够跑满 RTX 5090 超过 1000 GB/s 的实际有效显存带宽。

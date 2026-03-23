# Softmax V2: 共享内存优化详解

> CUDA 单核函数 + 共享内存归约优化版本
> 源文件：`src/softmax_v2_shared_memory.cu`

---

## 一、与 V1 的核心对比

| 特性 | V1 朴素版本 | V2 共享内存版本 |
|------|------------|----------------|
| **核函数数量** | 3 个 | **1 个** (融合) |
| **每行线程数** | 1 个 | **512 个** (协作) |
| **共享内存使用** | ❌ 无 | ✅ 有 |
| **内存访问次数** | 每个元素 6 次 | 每个元素 **2 次** |
| **Kernel Launch 次数** | 3 次 | **1 次** |
| **归约方式** | 串行 for 循环 | **并行 tree reduction** |

---

## 二、线程组织方式

### V1 vs V2 对比

```
V1: 一维线程网格，每行一个线程
┌─────────────────────────────────────────┐
│ Block 0 (512 threads)                   │
│ ┌────┬────┬────┬────┐                   │
│ │ T0 │ T1 │ T2 │ T3 │  ... T511          │
│ │ R0 │ R1 │ R2 │ R3 │     R511           │
│ └────┴────┴────┴────┘                   │
│ 每线程串行处理整行 (for i=0; i<N; i++)   │
└─────────────────────────────────────────┘

V2: 每行一个 Block，多线程协作
┌─────────────────────────────────────────┐
│ Block 0 (处理 Row 0)                     │
│ ┌────┬────┬────┬────┬────┬────┐         │
│ │ T0 │ T1 │ T2 │... │T510│T511│         │
│ └────┴────┴────┴────┴────┴────┘         │
│  512 线程协作处理 Row 0 的所有元素      │
│  每个线程处理 N/512 个元素              │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ Block 1 (处理 Row 1)                     │
│  ... 同上结构                            │
└─────────────────────────────────────────┘
```

### V2 线程索引计算

```cuda
// 每个 block 负责一行
int row = blockIdx.x;           // 行索引 = block 索引
int tid = threadIdx.x;          // 线程在 block 内的索引 (0-511)

// 数据指针偏移到当前行
const float* x = input + row * N;
float* y = output + row * N;
```

---

## 三、Strided 循环访问模式

### 数据分配方式

当 N=4096，threads=512 时，每个线程处理 8 个元素：

```
Row 0 的数据分布 (4096 个元素):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ T0  │ T1  │ T2  │ T3  │ ... │T509 │T510 │T511│  ← 第一轮
│ 0   │ 1   │ 2   │ 3   │     │ 509 │ 510 │ 511│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ T0  │ T1  │ T2  │ T3  │ ... │T509 │T510 │T511│  ← 第二轮 (i += 512)
│ 512 │ 513 │ 514 │ 515 │     │1021 │1022 │1023│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
       ... 继续直到覆盖全部 N 个元素 ...
```

### 代码实现

```cuda
// Strided 循环：每个线程从 tid 开始，每次增加 blockDim.x
for (int i = tid; i < N; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
}
```

**优点**：
- 内存访问**合并**（coalesced）- 相邻线程访问相邻内存
- 负载均衡 - 自动处理 N 不是线程数倍的情况

---

## 四、共享内存归约 (Shared Memory Reduction)

### Tree Reduction 算法

求 max 或 sum 的过程使用经典的二分归约：

```
初始：512 个线程各自持有 local_max
      ┌────┬────┬────┬────┬────┬────┬────┬────┐
      │ M0 │ M1 │ M2 │ M3 │... │M510│M511│  (写入 sdata[0..511])
      └────┴────┴────┴────┴────┴────┴────┴────┘
      
Step 1: stride=256, tid<256 的线程比较 sdata[tid] vs sdata[tid+256]
        ┌─────────┬─────────┬─────────┬─────────┐
        │max(M0,  │max(M1,  │max(M2,  │  ...   │
        │  M256)  │  M257)  │  M258)  │         │
        └─────────┴─────────┴─────────┴─────────┘
        剩余 256 个最大值
        
Step 2: stride=128, tid<128 的线程继续比较
        ┌─────────────────┬─────────────────┐
        │  max(前两个)    │  max(后两个)    │  ...
        └─────────────────┴─────────────────┘
        剩余 128 个最大值
        
Step 3: stride=64 ... 继续折半

...

Step 9: stride=1
        ┌─────────┐
        │ row_max │  ← sdata[0] 保存最终结果
        └─────────┘
```

### 归约代码详解

```cuda
// 1. 每个线程写自己的 local_max 到共享内存
sdata[tid] = local_max;
__syncthreads();  // 确保全部写入完成

// 2. 二分归约：stride 从 256, 128, 64, ... 到 1
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        // 当前线程比较自己与对面线程的值
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();  // 每轮都需要同步
}

// 3. 结果在 sdata[0]
float row_max = sdata[0];
__syncthreads();
```

### Sum 归约（类似过程）

```cuda
// 写 local_sum 到共享内存
sdata[tid] = local_sum;
__syncthreads();

// 二分累加
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];  // 这次是加法
    }
    __syncthreads();
}

float row_sum = sdata[0];
__syncthreads();
```

---

## 五、完整执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ Block 0 处理 Row 0 (512 threads, shared_mem = 512 * 4 bytes)    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: 并行求每行最大值                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ T0-T511 并行读取各自负责的元素 (strided loop)            │  │
│  │ 每个线程计算 local_max                                    │  │
│  │                                                          │  │
│  │ sdata: [M0, M1, M2, ..., M511]                          │  │
│  │           ↓                                              │  │
│  │ Tree Reduction (stride: 256→128→64→...→1)                 │  │
│  │           ↓                                              │  │
│  │ sdata[0] = row_max (整行的最大值)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Phase 2: 并行求 exp(x-max) 之和                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ T0-T511 再次读取各自负责的元素                            │  │
│  │ 每个线程计算 exp(x[i] - row_max) 并累加                    │  │
│  │                                                          │  │
│  │ sdata: [S0, S1, S2, ..., S511]                          │  │
│  │           ↓                                              │  │
│  │ Tree Reduction (累加)                                    │  │
│  │           ↓                                              │  │
│  │ sdata[0] = row_sum (整行的 exp 之和)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Phase 3: 并行归一化输出                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ T0-T511 最后一次读取各自负责的元素                          │  │
│  │ y[i] = exp(x[i] - row_max) / row_sum                     │  │
│  │ 直接写入全局内存 output                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

其他 Block (1, 2, ..., M-1) 同时执行相同流程处理各自对应的行
```

---

## 六、内存访问分析

### 相比 V1 的改进

```
V1 内存访问模式 (3 kernels):
┌─────────────────────────────────────────────────────────────┐
│ Kernel 1: 读 input, 写 max_vals                            │
│ Kernel 2: 读 input, 读 max_vals, 写 sum_vals               │
│ Kernel 3: 读 input, 读 max_vals, 读 sum_vals, 写 output    │
│                                                             │
│ 总计：input 读 3 次, 中间结果读写 4 次, output 写 1 次      │
│ 每个元素 6 次全局内存访问                                    │
└─────────────────────────────────────────────────────────────┘

V2 内存访问模式 (1 kernel):
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 读 input (求 max)                                   │
│ Phase 2: 读 input (求 sum)                                   │
│ Phase 3: 读 input, 写 output (归一化)                        │
│                                                             │
│ 总计：input 读 3 次 (但可缓存), output 写 1 次             │
│ 中间结果用 shared memory，无需全局内存读写                  │
│ 实际有效带宽：input 读 1 次 + output 写 1 次 = 2 次         │
└─────────────────────────────────────────────────────────────┘
```

### 内存指标对比

| 指标 | V1 | V2 | 改进 |
|------|-----|-----|------|
| 全局内存读取 (每元素) | 3 | **1** | 3x |
| 全局内存写入 (每元素) | 1 | **1** | 1x |
| 中间结果存储 | 全局内存 | **共享内存** | N/A |
| Kernel Launch 开销 | 3 次 | **1 次** | 3x |

---

## 七、核心代码解析

### 共享内存声明

```cuda
__global__ void softmax_v2_kernel(const float* input, float* output, int M, int N) {
    // 动态共享内存声明 - 大小由调用时指定
    extern __shared__ float sdata[];
    // 实际分配: threads * sizeof(float) = 512 * 4 = 2048 bytes
}
```

### 核函数调用

```cuda
void softmax_v2(const float* d_input, float* d_output, int M, int N) {
    int threads = 512;
    int blocks = M;                          // 每个 block 处理一行
    size_t shared_mem = threads * sizeof(float);  // 2048 bytes
    
    softmax_v2_kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

### 完整的 Phase 实现

```cuda
// ===== Phase 1: Find Max =====
float local_max = -INFINITY;
for (int i = tid; i < N; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
}
sdata[tid] = local_max;
__syncthreads();

// Tree reduction for max
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
}
float row_max = sdata[0];
__syncthreads();

// ===== Phase 2: Compute Sum =====
float local_sum = 0.0f;
for (int i = tid; i < N; i += blockDim.x) {
    local_sum += expf(x[i] - row_max);
}
sdata[tid] = local_sum;
__syncthreads();

// Tree reduction for sum
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}
float row_sum = sdata[0];
__syncthreads();

// ===== Phase 3: Normalize =====
for (int i = tid; i < N; i += blockDim.x) {
    y[i] = expf(x[i] - row_max) / row_sum;
}
```

---

## 八、同步点分析

```
执行流程中的 __syncthreads() 位置:

1. 求 local_max 后: sdata[tid] = local_max;
   __syncthreads();  ← 确保所有线程写完
   
2. 每轮归约后: 
   for (stride = 256; stride > 0; stride >>= 1) {
       if (tid < stride) sdata[tid] = fmaxf(...);
       __syncthreads();  ← 每轮都需要！
   }
   
3. 读取 row_max 前:
   float row_max = sdata[0];
   __syncthreads();  ← 防止下阶段覆盖 sdata
   
4. 同样过程重复 3 次 (max归约, sum归约, 归一化前)
   
总计: 约 2 * (1 + log2(512)) + 1 ≈ 20 次同步
```

**注意**: `__syncthreads()` 是 block 级别的同步，只同步当前 block 内的线程。

---

## 九、优缺点总结

### ✅ 优点

1. **核函数融合** - 1 次 kernel launch，减少启动开销
2. **共享内存加速** - 中间结果存于 shared memory，延迟低 (~20-30 cycles vs ~400 cycles)
3. **并行归约** - Tree reduction 比串行更高效 (O(logN) vs O(N))
4. **内存访问减少** - 避免中间结果的全局内存读写
5. **协作式处理** - 多线程处理一行，适合 N 较大的情况

### ❌ 缺点

1. **同步开销** - 需要多次 `__syncthreads()`，warp 发散时效率降低
2. **共享内存限制** - 每个 block 使用 2KB，限制同时运行的 block 数
3. **线程利用率** - 归约后期很多线程空闲 (tid >= stride 的线程不参与)
4. **bank conflict** - 如果访问模式不当可能发生共享内存 bank conflict
5. **N 较小的情况** - 当 N < threads 时，部分线程完全空闲

---

## 十、适用场景

### 推荐使用场景

- **N 较大**（如 1024, 4096, 10000+）- 能充分发挥多线程协作优势
- **M 中等**（几百到几千行）- 需要足够的 block 来填充 GPU
- 共享内存充足的 GPU（如 RTX 5090）

### 不推荐使用场景

- **N 很小**（如 N < 128）- 大量线程空闲，归约开销占比大
- **M 很小**（如 M < 10）- 无法填满 GPU，并行度不足
- 共享内存受限的旧 GPU

---

## 十一、进一步优化方向

### V2 → V3+ 可能的优化点

1. **Warp Shuffle** - 使用 `__shfl_down_sync` 替代 shared memory 进行 warp 级归约
2. **向量化访存** - 使用 float4 一次读取 4 个 float
3. **寄存器优化** - 减少 shared memory 使用，改用 warp register
4. **持久化线程块** - 一个 block 处理多行，减少 block 总数
5. **计算与通信重叠** - 在读取数据时同时计算

---

*文档生成时间：2026-03-23*

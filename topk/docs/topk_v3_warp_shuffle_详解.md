# TopK V3: Warp Shuffle 版本详解

## 概述

这是 TopK 算法的**工业级优化版本**，采用 **Warp-per-Row** 策略：一个 Warp（32 线程）协作处理一行数据，利用 **Warp Shuffle 原语** 实现高效的线程间数据交换，无需共享内存即可完成归并。

---

## 算法核心思想

```
┌─────────────────────────────────────────────────────────┐
│                    输入数据矩阵                          │
│              (Batch 行 × N 列的 float 数组)               │
├─────────────────────────────────────────────────────────┤
│  Row 0: [ 3.5, 1.2, 7.8, 2.1, 9.0, 4.5, ... ]           │
│  Row 1: [ 6.2, 8.1, 2.3, 5.7, 1.9, 3.4, ... ]           │
│  Row 2: [ 4.1, 9.5, 3.2, 7.6, 2.8, 6.3, ... ]           │
│     ...                                                 │
│  Row Batch-1: [ ... ]                                   │
└─────────────────────────────────────────────────────────┘
                          ↓ 每个 Warp (32线程) 处理一行
┌─────────────────────────────────────────────────────────┐
│              Warp 0 (Row 0)  Warp 1 (Row 1)             │
│           ┌─────────────┐    ┌─────────────┐             │
│           │ Lane0-31    │    │ Lane0-31    │             │
│           │ 协作处理     │    │ 协作处理     │             │
│           └─────────────┘    └─────────────┘             │
│                  ↓                    ↓                 │
│           ┌─────────────┐    ┌─────────────┐             │
│           │ Top K 结果  │    │ Top K 结果  │             │
│           └─────────────┘    └─────────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## CUDA 线程组织 (2D 配置)

```
线程网格配置 (1D Grid / 2D Block)

Grid:  ┌─────┬─────┬─────┬─────┬─────┐
       │Block│Block│Block│ ... │Block│
       │  0  │  1  │  2  │     │  M  │
       └─────┴─────┴─────┴─────┴─────┘
              ↓ 每个 Block: 4 Warps × 32 Lanes
Block: ┌──────────────────────────────┐
       │  Warp 0 (threadIdx.y=0)      │ ← 处理行 0, 4, 8...
       │  Lane: 0  1  2  ...  31      │
       ├──────────────────────────────┤
       │  Warp 1 (threadIdx.y=1)      │ ← 处理行 1, 5, 9...
       │  Lane: 0  1  2  ...  31      │
       ├──────────────────────────────┤
       │  Warp 2 (threadIdx.y=2)      │ ← 处理行 2, 6, 10...
       ├──────────────────────────────┤
       │  Warp 3 (threadIdx.y=3)      │ ← 处理行 3, 7, 11...
       └──────────────────────────────┘

dim3 blockDim(32, 4);  // 32 lanes × 4 warps = 128 threads/block
dim3 gridDim((Batch + 4 - 1) / 4);

线程索引计算:
    row = blockIdx.x * blockDim.y + threadIdx.y
        = blockIdx.x * 4 + threadIdx.y
```

---

## 代码逐行解析

### 1. 内核函数签名与线程索引

```cuda
__global__ void topk_v3_kernel(
    const float* input,    // 输入数据: [Batch × N]
    float* out_vals,       // 输出值: [Batch × K]
    int* out_inds,         // 输出索引: [Batch × K]
    int Batch, int N, int K
) {
    // 2D 线程索引计算
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // 行号由 y 维度决定
    if (row >= Batch) return;
    
    int lane_id = threadIdx.x;  // 0~31, 当前 warp 内的 lane 编号
    const float* row_input = input + row * N;  // 当前行起始地址
```

**关键设计**: 
- `threadIdx.x` (0-31) 表示 Warp 内的 Lane ID
- `threadIdx.y` (0-3) 表示 Warp ID 在一个 Block 内
- 这样 4 个 Warps 可以同时处理 4 行数据

---

### 2. 阶段一：线程局部 Top-K

```cuda
    // 每个线程维护自己的局部 Top K
    float local_vals[MAX_K];
    int local_inds[MAX_K];
    for (int i = 0; i < K; ++i) { 
        local_vals[i] = -1e20f; 
        local_inds[i] = -1; 
    }

    // Warp 内 32 个线程交错读取数据 (完美合并访存)
    for (int i = lane_id; i < N; i += 32) {
        float val = row_input[i];
        if (val > local_vals[K - 1]) {
            // 插入排序维护局部 Top K
            int p = K - 1;
            while (p > 0 && val > local_vals[p - 1]) {
                local_vals[p] = local_vals[p - 1];
                local_inds[p] = local_inds[p - 1];
                p--;
            }
            local_vals[p] = val;
            local_inds[p] = i;
        }
    }
```

**交错访问模式图解**:

```
Warp 读取一行数据 (N=128) 的交错模式:

Lane 0:  读取索引 0, 32, 64, 96
Lane 1:  读取索引 1, 33, 65, 97
Lane 2:  读取索引 2, 34, 66, 98
...      
Lane 31: 读取索引 31, 63, 95, 127

内存布局: [0][1][2]...[31][32][33]...[63]...
              ↓ 合并访存（一条指令读取 32 个连续 float）
         完美利用 GPU 内存带宽！

对比 V1 (线程每行):
┌──────────────────────────────────────────┐
│  V1: Thread 0 读全行                      │
│      Thread 1 读全行 (地址跳跃大)          │
│      → 非合并访存，性能差                   │
├──────────────────────────────────────────┤
│  V3: Warp 内 32 线程交错读取               │
│      → 连续地址，完美合并                   │
└──────────────────────────────────────────┘
```

---

### 3. 阶段二：Warp 级归并 (核心创新)

```cuda
    int local_ptr = 0;
    float current_val = local_vals[0];
    int current_idx = local_inds[0];

    float* row_out_vals = out_vals + row * K;
    int* row_out_inds = out_inds + row * K;

    for (int k = 0; k < K; ++k) {
        float max_val = current_val;
        int max_idx = current_idx;
        int max_lane = lane_id;

        // Warp Reduction: 使用 __shfl_down_sync 在寄存器间交换
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            int other_lane = __shfl_down_sync(0xffffffff, max_lane, offset);

            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
                max_lane = other_lane;
            }
        }
```

#### Warp Shuffle 原理解析

```
__shfl_down_sync(mask, var, offset) 工作原理:

Warp 内 32 个线程的寄存器数据交换:

Lane:    0     1     2     ...    15    16    ...   31
         │     │     │           │     │           │
         ├─────┼─────┼─────┬─────┼─────┼─────┬─────┤
         │     │     │     │     │     │     │     │
         ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
        [======= offset=16 ======]
         0←16  1←17  2←18  ...   15←31

offset=16: Lane i 从 Lane (i+16) 读取数据
offset=8:  Lane i 从 Lane (i+8)  读取数据
offset=4:  Lane i 从 Lane (i+4)  读取数据
offset=2:  Lane i 从 Lane (i+2)  读取数据
offset=1:  Lane i 从 Lane (i+1)  读取数据
```

#### 归约树 (Reduction Tree) 图解

```
第一次迭代 (offset=16): 两两比较

Lane 0-15 持有较大的局部最大值
Lane 16-31 的数据被比较后淘汰

   0───────8───────4───────2───────1 (offset变化)
   │       │       │       │       │
  ┌┴┐     ┌┴┐     ┌┴┐     ┌┴┐     ┌┴┐
  │0│←───→│8│     │4│     │2│     │1│
  │1│     │9│←───→│5│     │3│     │2│
  │2│     │10│    │6│←───→│4│     │3│
  │3│     │11│    │7│     │5│←───→│4│
  │4│     │12│    │0│     │6│     │5│←───→│5│
  │5│     │13│    │1│     │7│     │6│←───→│6│
  │6│     │14│    │2│     │0│     │7│←───→│7│
  │7│     │15│    │3│     │1│     │0│←───→│0│
  └┬┘     └┬┘     └┬┘     └┬┘     └┬┘
   │       │       │       │       │
  [======= 最终 Lane 0 持有全局最大值 =====]

经过 5 轮 (__shfl_down_sync)，所有线程都知道了最大值！
```

---

### 4. 阶段三：广播与更新

```cuda
        // Lane 0 持有真正的最大值，广播给所有线程
        max_lane = __shfl_sync(0xffffffff, max_lane, 0);

        if (lane_id == 0) {
            row_out_vals[k] = max_val;
            row_out_inds[k] = max_idx;
        }

        // 胜出的线程更新它的候选值 (弹出栈顶)
        if (lane_id == max_lane) {
            local_ptr++;
            if (local_ptr < K) {
                current_val = local_vals[local_ptr];
                current_idx = local_inds[local_ptr];
            } else {
                current_val = -1e20f;  // 已耗尽，设为极小值
            }
        }
    }
}
```

**工作原理图解**:

```
假设 K=3，3 个线程的局部 Top K 如下:

Local Top Ks:        归约找全局第1大:
Lane 0: [9, 5, 2]           ┌──────────────────┐
Lane 1: [8, 4, 1]           │ 比较 9, 8, 7    │
Lane 2: [7, 3, 0]           │ 最大=9 (Lane 0) │
                            └──────────────────┘
                              ↓
                     Lane 0 提供 9，弹出栈
                     输出: [9, ...]
                     
第二轮 (找第2大):    
Lane 0: [5, 2]              ┌──────────────────┐
Lane 1: [8, 4, 1]           │ 比较 5, 8, 7    │
Lane 2: [7, 3, 0]           │ 最大=8 (Lane 1) │
                            └──────────────────┘
                              ↓
                     Lane 1 提供 8，弹出栈
                     输出: [9, 8, ...]
                     
第三轮 (找第3大):
Lane 0: [5, 2]
Lane 1: [4, 1]
Lane 2: [7, 3, 0]           │ 比较 5, 4, 7    │
                            │ 最大=7 (Lane 2) │
                     
最终输出: [9, 8, 7]
```

---

## Warp Shuffle 指令详解

### `__shfl_down_sync(mask, var, offset)`

```
功能: Lane i 从 Lane (i + offset) 读取 var

示意图 (offset=4):
Lane:     0     1     2     3     4     5     6     7
          │     │     │     │     │     │     │     │
          └─────┴─────┴─────┴──┬──┘     │     │     │
                             ┌─┴────────┴─────┴─────┘
                             ↓
                          读取值
          
Lane 0 得到 Lane 4 的值
Lane 1 得到 Lane 5 的值
... (超出范围返回自身值)
```

### `__shfl_sync(mask, var, srcLane)`

```
功能: 所有线程从 srcLane 读取 var

示意图 (srcLane=0):
Lane 0:  value = 42

         ┌──────────────────────────────────────┐
         │        Broadcast from Lane 0         │
         └──────────────────────────────────────┘
              ↓        ↓        ↓        ↓
Lane 0:  42   Lane 1:  42   Lane 2:  42   Lane 3:  42
Lane 4:  42   Lane 5:  42   Lane 6:  42   Lane 7:  42
... (所有线程都得到 42)
```

---

## 复杂度分析

### 时间复杂度

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| 局部 Top K | O(N/32 × K) | 32 线程交错读取，每线程处理 N/32 元素 |
| Warp 归约 | O(K × log(32)) | K 轮，每轮 5 次 shuffle |
| **总计** | **O(N×K/32 + K×5)** | 比 V1 快约 32 倍（第一阶段） |

### 空间复杂度

- **寄存器**: O(K) 每个线程
- **共享内存**: **0！** 不使用 shared memory
- **全局内存**: O(Batch × K) 输出

---

## 与 V1/V2 的对比

```
┌────────────────────────────────────────────────────────────┐
│                    三种实现对比                              │
├─────────────┬──────────────┬──────────────┬────────────────┤
│   特性       │     V1       │     V2       │      V3        │
├─────────────┼──────────────┼──────────────┼────────────────┤
│ 并行单位     │  Thread      │  Block       │     Warp       │
│ 每行线程数   │     1        │   128/256    │      32        │
│ 协作方式     │   无协作      │ Shared Mem   │  Warp Shuffle  │
│ 共享内存     │    不用       │   需要       │     不用       │
│ 内存访问     │  非合并       │   合并        │    合并        │
│ 同步开销     │     无        │  __syncthreads│    隐式同步    │
│ 实现复杂度   │     低        │    中         │     中         │
│ 适用 K 值    │    小 K       │   中小 K      │    中小 K      │
└─────────────┴──────────────┴──────────────┴────────────────┘
```

---

## 优势与适用场景

### 核心优势

1. **无需共享内存**: Warp shuffle 直接在寄存器间交换数据
2. **隐式同步**: Warp 内线程天然同步，无需 `__syncthreads()`
3. **更高带宽利用率**: 交错读取实现完美合并访存
4. **较低延迟**: 寄存器间传输比共享内存更快
5. **更高的 Occupancy**: 不使用 shared memory 减少资源竞争

### 适用场景

- **K <= 32**: 适合寄存器存储，性能最佳
- **任意 Batch 大小**: 可灵活配置 Block 内 Warp 数量
- **延迟敏感场景**: 需要最低延迟的 TopK 计算
- **共享内存受限**: 当 shared memory 已被其他操作占满时

---

## 完整调用示例

```cpp
void launch_topk_v3(const float* d_input, float* d_out_vals, int* d_out_inds,
                    int Batch, int N, int K) {
    // 2D 线程配置: 32 lanes × 4 warps = 128 threads
    dim3 blockDim(32, 4);  // x=32 (warp size), y=4 (warps per block)
    dim3 gridDim((Batch + 4 - 1) / 4);  // 4 行每 block
    
    topk_v3_kernel<<<gridDim, blockDim>>>(
        d_input, d_out_vals, d_out_inds, Batch, N, K
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
}
```

---

## 关键优化总结

```
V3 Warp Shuffle 版本的核心优化点:

┌────────────────────────────────────────────────────────┐
│  1. 交错内存访问 → 完美合并访存，带宽利用最大化          │
│                                                          │
│  2. Warp Shuffle 归约 → 无需共享内存，减少同步开销       │
│                                                          │
│  3. 隐式 Warp 同步 → 无需 __syncthreads()               │
│                                                          │
│  4. 寄存器级数据交换 → 比 shared memory 更快            │
│                                                          │
│  5. 灵活 2D 配置 → 可调节 Block 内 Warp 数量适配不同 GPU  │
└────────────────────────────────────────────────────────┘
```

---

## 性能提示

1. **Warp 数量**: 通常每 Block 4-8 个 Warps 效果最佳
2. **K 值限制**: 虽然支持到 32，但 K <= 16 时寄存器压力更小
3. **Occupancy**: 不使用 shared memory 有助于提高 SM 占用率
4. **兼容性**: Warp shuffle 需要 Compute Capability >= 3.0 (Kepler+)

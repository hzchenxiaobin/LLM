# Top-K V4: Radix Select 详解

## 核心思想

V4版本采用**Radix-Style**的选择策略，是基于分治思想的Top-K算法，专门设计用于处理各种规模的K值。

**特点**：
- 1D 输入（无 Batch 轴）
- 只输出 Top-K 值，不输出索引
- 两阶段算法：粗筛选 + 精排序

---

## 算法流程图

```
┌─────────────────────────────────────────────────────────┐
│                    V4 Radix Select                      │
│                   (1D输入，只输出值)                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────┐ │
│  │  Phase 1:    │ --> │  Phase 2:    │ --> │ Output  │ │
│  │  粗筛选       │     │  精排序       │     │ Top-K   │ │
│  │  (并行)       │     │  (分层归约)    │     │ (值数组) │ │
│  └──────────────┘     └──────────────┘     └─────────┘ │
│                                                         │
│  每个线程:              Warp/Block级别:                   │
│  - 读取数据             - 分层归约找出全局Top-K          │
│  - 本地插入排序         - 逐轮输出最大/次大/...          │
│  - 维护Local Top-K      - 获胜线程推进指针               │
│  (只存值，不存索引)                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 关键技术细节

### 1. 辅助函数：插入排序（只存值）

```cuda
__device__ void radix_insert_topk(float* vals, int K, float new_val) {
    // 快速拒绝：比当前最小值小
    if (new_val <= vals[K - 1]) return;

    // 找到插入位置（降序）
    int pos = K - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
}
```

**注意**：
- 只维护值的数组，不维护索引
- 快速拒绝减少不必要的比较

---

### 2. 两阶段算法

#### Phase 1: 粗筛选 (Coarse Selection)

```cuda
// Warp交错读取数据
for (int i = lane_id; i < N; i += WARP_SIZE) {
    float val = input[i];
    radix_insert_topk(local_vals, K, val);  // 只存值
}
```

每个线程：
1. 读取 `N / 32` 个元素（交错读取，合并访存）
2. 使用插入排序维护Local Top-K（只存值）
3. 只保留可能比当前Local最小值大的元素

**复杂度**：O(N/32 × K)，但由于是Local操作，完全并行无冲突。

#### Phase 2: 精排序 (Fine-grained Selection)

```cuda
for (int k = 0; k < K; ++k) {
    // 1. 找出当前全局最大值
    float max_val = warp_reduce_max(...);

    // 2. 输出到全局内存 (只写值)
    if (lane_id == 0) out_vals[k] = max_val;

    // 3. 获胜线程前进指针
    if (my_val == max_val) local_ptr++;
}
```

**巧妙之处**：
- 不是一次性排序所有 `32 × K` 个候选
- 而是逐轮"选举"，每轮找出一个最大值
- 获胜线程从自己的Local数组中移除已输出的值
- 下一轮自动找次大值

**复杂度**：O(K × log(32))，使用Warp Shuffle的Tree Reduction。

---

### 3. Block级别扩展

当N较大时（如N > 1024），需要使用Block级别版本：

```cuda
__global__ void topk_v4_kernel(const float* input, float* out_vals, int N, int K)
```

**关键差异**：
- 使用Shared Memory存储所有Warp的中间结果
- 需要`__syncthreads()`进行Block级别同步
- 第一个Warp负责最终归约

**Shared Memory 布局**：
```
┌─────────────────────────────────────────────────────────┐
│  Shared Memory 布局 (num_warps × K 个 float)             │
├─────────────────────────────────────────────────────────┤
│  Warp 0 Local Top-K: [val0, val1, ..., valK-1]         │
│  Warp 1 Local Top-K: [val0, val1, ..., valK-1]         │
│  ...                                                    │
│  Warp N Local Top-K: [val0, val1, ..., valK-1]         │
└─────────────────────────────────────────────────────────┘
```

---

## 性能分析

### 理论性能对比

| 指标 | V3 (Warp) | V4 Warp (N≤1024) | V4 Block (N>1024) |
|------|-----------|------------------|-------------------|
| **输入处理** | 1 Warp | 1 Warp | 多 Warp |
| **Shared Memory** | 0 | 0 | num_warps × K × 4 bytes |
| **同步开销** | 0 | 0 | 需要 __syncthreads() |
| **适合N范围** | 任意 | N ≤ 1024 | N > 1024 |
| **带宽利用率** | ~90% | ~90% | ~85% |

### 实际使用建议

```cpp
// 调用策略
if (N <= 1024) {
    // 小数据量：单Warp即可
    dim3 block(32);
    dim3 grid(1);
    topk_v4_warp_kernel<<<grid, block>>>(input, out_vals, N, K);
} else {
    // 大数据量：Block级别
    int block_size = 256;
    int num_warps = block_size / 32;
    size_t smem = num_warps * K * sizeof(float);  // 只存值
    
    dim3 block(block_size);
    dim3 grid(1);
    topk_v4_kernel<<<grid, block, smem>>>(input, out_vals, N, K);
}
```

---

## 代码亮点

### 1. 精简的插入排序

```cuda
__device__ void radix_insert_topk(float* vals, int K, float new_val) {
    // 快速拒绝：比当前最小值小
    if (new_val <= vals[K - 1]) return;

    // 线性查找插入位置
    int pos = K - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
}
```

相比带索引的版本，省去了索引数组的操作，更加高效。

### 2. Warp Shuffle Tree Reduction

```cuda
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_down_sync(0xffffffff, candidate_val, offset);

    if (other_val > candidate_val) {
        candidate_val = other_val;
    }
}
```

**优势**：
- 5轮比较即可在32线程中找到最大值
- 完全在寄存器层面操作，无内存访问
- 只传递值，不传递索引，更高效

### 3. 分层归约策略

```
Phase 1: 每个线程独立处理数据
         ↓
Phase 2: Warp内归约（Shuffle）
         ↓
Phase 3: Warp间归约（Shared Memory）
         ↓
Phase 4: 第一个Warp最终归约
         ↓
输出结果
```

---

## 调试技巧

### 1. 验证Local Top-K正确性

```cuda
// 在kernel中添加调试代码
if (lane_id == 0) {
    printf("After phase 1, local top 3: %f, %f, %f\n",
           local_vals[0], local_vals[1], local_vals[2]);
}
```

### 2. 检查Warp Shuffle结果

```cuda
// 验证每轮的global max
if (lane_id == 0) {
    printf("Round %d: global max = %f\n", k, global_max_val);
}
```

---

## 进一步优化方向

1. **向量化访存**：使用`float4`一次性读取4个float
2. **异步内存拷贝**：使用`cp.async`将数据预取到Shared Memory
3. **BF16/FP8支持**：在RTX 5090上使用半精度，带宽翻倍
4. **多Pass策略**：当K > 256时，分多轮筛选，每轮淘汰一部分

---

## 参考实现

完整代码参见：`topk/src/topk_v4_radix_select.cu`

关键Kernel：
- `topk_v4_warp_kernel`: Warp级别，适合N ≤ 1024
- `topk_v4_kernel`: Block级别，支持更大N

### 调用示例

```cpp
// 小数据量 (N <= 1024)
void launch_topk_v4_small(const float* d_input, float* d_out_vals,
                          int N, int K) {
    dim3 block(32);
    dim3 grid(1);
    topk_v4_warp_kernel<<<grid, block>>>(d_input, d_out_vals, N, K);
}

// 大数据量 (N > 1024)
void launch_topk_v4_large(const float* d_input, float* d_out_vals,
                          int N, int K) {
    int block_size = 256;
    int num_warps = block_size / 32;
    size_t smem = num_warps * K * sizeof(float);
    
    dim3 block(block_size);
    dim3 grid(1);
    topk_v4_kernel<<<grid, block, smem>>>(d_input, d_out_vals, N, K);
}
```

---

## 版本对比总结

| 版本 | 适用场景 | 调用方式 | 特点 |
|------|----------|----------|------|
| V4 Warp | N ≤ 1024 | `<<<1, 32>>>` | 单Warp，无Shared Memory |
| V4 Block | N > 1024 | `<<<1, 256, smem>>>` | 多Warp协作，使用Shared Memory |

**输出**：都只输出Top-K值，不输出索引。

---

*文档版本: 2.0 | 更新说明: 已修改为1D输入，无Batch轴，只输出值（无索引）*

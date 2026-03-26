# Top-K V4: Radix Select 详解

## 核心思想

V4版本采用**Radix-Style**的选择策略，是V3 Warp-per-Row的扩展版本，专门设计用于**K > 32**的场景。

### 为什么需要V4？

当 `K > 32` 时：
- V3的每个线程寄存器空间不足（需要存储K个值和索引）
- 寄存器溢出到Local Memory，性能暴跌
- 必须使用Shared Memory辅助存储

### 算法流程图

```
┌─────────────────────────────────────────────────────────┐
│                    V4 Radix Select                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────┐ │
│  │  Phase 1:    │ --> │  Phase 2:    │ --> │ Output  │ │
│  │  粗筛选       │     │  精排序       │     │ Top-K   │ │
│  └──────────────┘     └──────────────┘     └─────────┘ │
│                                                         │
│  每个线程:              Warp/Block级别:                   │
│  - 读取数据             - 分层归约找出全局Top-K          │
│  - 本地插入排序         - 逐轮输出最大/次大/...          │
│  - 维护Local Top-K      - 获胜线程推进指针               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 关键技术细节

### 1. 浮点数键值转换

这是实现基于位操作排序的关键技巧：

```cuda
__device__ __inline__ uint32_t float_to_sortable_uint(float val) {
    uint32_t u = __float_as_uint(val);
    // IEEE 754 浮点数位布局：
    // 正数: 0 | exponent(8) | mantissa(23)
    // 负数: 1 | exponent(8) | mantissa(23)
    //
    // 技巧：翻转符号位后，负数的补码表示使得
    // -0.1 (接近0的大负数) > -1.0 (远离0的小负数)
    return (u & 0x80000000) ? (~u) : (u | 0x80000000);
}
```

**为什么这样做？**
- 正数：设置符号位为1，数值越大uint越大
- 负数：按位取反，使得-0.1 > -1.0（符合浮点数大小顺序）

### 2. 两阶段算法

#### Phase 1: 粗筛选 (Coarse Selection)

```cuda
// Warp交错读取数据
for (int i = lane_id; i < N; i += WARP_SIZE) {
    float val = row_input[i];
    radix_insert_topk(local_vals, local_inds, K, val, i);
}
```

每个线程：
1. 读取 `N / 32` 个元素
2. 使用插入排序维护Local Top-K
3. 只保留可能比当前Local最小值大的元素

**复杂度**：O(N/32 × K)，但由于是Local操作，完全并行无冲突。

#### Phase 2: 精排序 (Fine-grained Selection)

```cuda
for (int k = 0; k < K; ++k) {
    // 1. 找出当前全局最大值
    float max_val = warp_reduce_max(...);

    // 2. 输出到全局内存
    if (lane_id == 0) row_out_vals[k] = max_val;

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

### 3. Block级别扩展

当K更大时（如K > 32），需要使用Block级别版本：

```cuda
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void topk_v4_block_kernel(...)
```

**关键差异**：
- 使用Shared Memory存储所有线程的Local Top-K
- 需要`__syncthreads()`进行Block级别同步
- 使用Shared Memory树形归约找出全局最大值

## 性能分析

### 理论性能对比

| 指标 | V3 (K≤32) | V4 Warp (K≤32) | V4 Block (K>32) |
|------|-----------|----------------|-----------------|
| **寄存器/线程** | 2×K words | 2×K words | 2×K words |
| **Shared Memory** | 0 | 0 | BLOCK_SIZE×K×8 bytes |
| **同步开销** | 0 | 0 | K × __syncthreads() |
| **适合K范围** | 1-32 | 1-32 (更灵活) | 32-256+ |
| **带宽利用率** | ~90% | ~90% | ~85% |

### 实际使用建议

```cpp
// 调用策略
if (K <= 32) {
    // 使用V3或V4 Warp版本
    dim3 block(32, 4);  // 4 warps per block
    dim3 grid((Batch + 3) / 4);
    topk_v4_kernel<<<grid, block>>>(...);
} else {
    // 使用V4 Block版本
    dim3 block(256);
    dim3 grid(Batch);
    size_t smem = 256 * K * sizeof(float) * 2;  // vals + inds
    topk_v4_block_kernel<256, 4><<<grid, block, smem>>>(...);
}
```

## 代码亮点

### 1. 无分支插入排序

```cuda
__device__ void radix_insert_topk(float* vals, int* inds, int K,
                                   float new_val, int new_idx) {
    // 快速拒绝：比当前最小值小
    if (new_val <= vals[K - 1]) return;

    // 二分/线性查找插入位置
    int pos = K - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        inds[pos] = inds[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
    inds[pos] = new_idx;
}
```

### 2. Warp Shuffle Tree Reduction

```cuda
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_down_sync(0xffffffff, candidate_val, offset);
    int other_idx = __shfl_down_sync(0xffffffff, candidate_idx, offset);

    if (other_val > candidate_val) {
        candidate_val = other_val;
        candidate_idx = other_idx;
    }
}
```

**优势**：
- 5轮比较即可在32线程中找到最大值
- 完全在寄存器层面操作，无内存访问
- 比Shared Memory Reduction快2-3倍

### 3. __match_any_sync 优化

```cuda
// 找到所有具有相同最大值的线程
uint32_t match_mask = __match_any_sync(0xffffffff, global_max_val);
```

当存在多个相同值时，所有匹配的线程都推进指针，避免数据丢失。

## 调试技巧

### 1. 验证Local Top-K正确性

```cuda
// 在kernel中添加调试代码
if (row == 0 && lane_id == 0) {
    printf("After phase 1, local top 3: %f, %f, %f\n",
           local_vals[0], local_vals[1], local_vals[2]);
}
```

### 2. 检查Warp Shuffle结果

```cuda
// 验证每轮的global max
if (lane_id == 0) {
    printf("Round %d: global max = %f at idx %d\n",
           k, global_max_val, global_max_idx);
}
```

## 进一步优化方向

1. **向量化访存**：使用`float4`一次性读取4个float
2. **异步内存拷贝**：使用`cp.async`将数据预取到Shared Memory
3. **BF16/FP8支持**：在RTX 5090上使用半精度，带宽翻倍
4. **多Pass策略**：当K > 256时，分多轮筛选，每轮淘汰一部分

## 参考实现

完整代码参见：`topk/src/topk_v4_radix_select.cu`

关键Kernel：
- `topk_v4_kernel`: Warp级别，适合K ≤ 32
- `topk_v4_block_kernel<256, 4>`: Block级别，支持更大K

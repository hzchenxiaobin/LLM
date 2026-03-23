# Softmax V5: Online Softmax 详解

> CUDA Online Softmax 优化版本 - 单次遍历同时计算 max 和 sum
> 基于 FlashAttention 的 Online Softmax 算法
> 源文件：`src/softmax_v5_online.cu`

---

## 一、核心思想

V5 使用 **Online Softmax** 算法，在**单次遍历**中同时计算最大值和指数和，减少内存访问次数。这是 FlashAttention 中的核心优化技术。

### 传统 vs Online Softmax

```
传统 Softmax (V3/V4):
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: 遍历所有元素求 max                                  │
│ Pass 2: 遍历所有元素求 exp(x-max) 的和                       │
│ Pass 3: 遍历所有元素计算最终 softmax 值                      │
│                                                             │
│ 内存访问: 3 次读取输入 (每个元素被读 3 次)                   │
└─────────────────────────────────────────────────────────────┘

Online Softmax (V5):
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: 同时更新 max 和 sum                                  │
│   for each x:                                               │
│     new_max = max(old_max, x)                                │
│     sum = sum * exp(old_max - new_max) + exp(x - new_max)   │
│                                                             │
│ Pass 2: 计算最终 softmax (需要 row_max 和 row_sum)          │
│                                                             │
│ 内存访问: 2 次读取输入 (减少 33%)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、Online Softmax 数学原理

### 核心公式推导

假设我们有一个部分和，基于当前的最大值 `m_old`：

```
部分和: sum_old = Σ exp(x_i - m_old)

当发现新的最大值 m_new > m_old 时，需要重新调整 sum：

exp(x_i - m_old) = exp(x_i - m_new + m_new - m_old)
                 = exp(x_i - m_new) * exp(m_new - m_old)

因此:
sum_old = Σ exp(x_i - m_new) * exp(m_new - m_old)
        = exp(m_new - m_old) * Σ exp(x_i - m_new)

新的部分和应该是:
sum_new = Σ exp(x_i - m_new)
        = sum_old * exp(m_old - m_new) + exp(x_new - m_new)
```

### Online 更新公式

```
对于每个新输入 x:

new_max = max(old_max, x)
new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)

数学意义:
• exp(old_max - new_max): 旧数据的缩放因子 (<1, 因为 new_max ≥ old_max)
• exp(x - new_max): 新数据的贡献 (≤1)
```

### 数值示例

```
输入序列: [2.0, 1.0, 3.0]

Step 1: x = 2.0
  new_max = max(-∞, 2.0) = 2.0
  new_sum = 0 * exp(-∞ - 2.0) + exp(2.0 - 2.0)
          = 0 + 1.0 = 1.0

Step 2: x = 1.0
  new_max = max(2.0, 1.0) = 2.0
  new_sum = 1.0 * exp(2.0 - 2.0) + exp(1.0 - 2.0)
          = 1.0 * 1.0 + exp(-1.0)
          = 1.0 + 0.368 = 1.368

Step 3: x = 3.0 (发现更大的值!)
  new_max = max(2.0, 3.0) = 3.0
  new_sum = 1.368 * exp(2.0 - 3.0) + exp(3.0 - 3.0)
          = 1.368 * exp(-1.0) + 1.0
          = 1.368 * 0.368 + 1.0
          = 0.503 + 1.0 = 1.503

最终结果: max = 3.0, sum = 1.503
验证: exp(2-3) + exp(1-3) + exp(3-3) = 0.368 + 0.135 + 1.0 = 1.503 ✓
```

---

## 三、代码实现详解

### 核心循环 (Online 更新)

```cuda
float local_max = -INFINITY;
float local_sum = 0.0f;

// Single pass: update max and sum together
for (int i = lane_id; i < N; i += 32) {
    float val = x[i];                              // 读取输入
    float new_max = fmaxf(local_max, val);         // 更新最大值
    
    // Online 更新公式
    local_sum = local_sum * expf(local_max - new_max)   // 旧数据的缩放
              + expf(val - new_max);                      // 新数据的贡献
    
    local_max = new_max;
}
```

### Warp 级 Online 归约

```cuda
// Warp reduce with online correction
for (int offset = 16; offset > 0; offset /= 2) {
    // 从相邻线程获取其 local_max 和 local_sum
    float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
    float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);

    // 合并两个部分结果 (使用相同的 online 公式)
    float new_max = fmaxf(local_max, other_max);
    local_sum = local_sum * expf(local_max - new_max) 
              + other_sum * expf(other_max - new_max);
    local_max = new_max;
}
```

---

## 四、完整执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│ Warp 0 处理 Row 0 (32 lanes)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Online Single-Pass (关键优化)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 初始化: local_max = -∞, local_sum = 0                  │  │
│  │                                                           │  │
│  │ for i = lane_id; i < N; i += 32:                        │  │
│  │   val = x[i]        ← 读取输入 (每个元素只读一次!)      │  │
│  │   new_max = max(local_max, val)                         │  │
│  │   local_sum = local_sum * exp(local_max - new_max)      │  │
│  │             + exp(val - new_max)                        │  │
│  │   local_max = new_max                                   │  │
│  │                                                           │  │
│  │ 结果: 每个 lane 有自己的 (local_max, local_sum)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Phase 2: Warp Online 归约                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ stride 16→8→4→2→1 的 tree reduction:                     │  │
│  │   从相邻 lane 获取 (other_max, other_sum)                │  │
│  │   new_max = max(local_max, other_max)                     │  │
│  │   merged_sum = local_sum * exp(local_max - new_max)       │  │
│  │              + other_sum * exp(other_max - new_max)       │  │
│  │   (完全相同的 online 更新公式!)                          │  │
│  │                                                           │  │
│  │ __shfl_sync 广播 row_max 和 row_sum                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Phase 3: 写入输出 (第二次遍历，但可能 L2 Cache Hit)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ for i = lane_id; i < N; i += 32:                        │  │
│  │   y[i] = expf(x[i] - row_max) / row_sum                 │  │
│  │   ↑ 这次 x[i] 可能从 L2 cache 读取，更快!                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、内存访问分析

### 相比 V3/V4 的改进

```
V3/V4 (三次遍历逻辑):
┌────────────────────────────────────────────────────────────┐
│ Pass 1 (Find Max):    读 input                            │
│ Pass 2 (Compute Sum): 读 input  ← 第 2 次读!               │
│ Pass 3 (Normalize):   读 input  ← 第 3 次读!               │
│                       写 output                             │
│                                                             │
│ 每个元素被读取 3 次                                         │
└────────────────────────────────────────────────────────────┘

V5 (Online Single-Pass):
┌────────────────────────────────────────────────────────────┐
│ Pass 1 (Online):      读 input                            │
│                       同时计算 max 和 sum                   │
│                       (只需 1 次读取!)                      │
│                                                             │
│ Pass 2 (Output):      读 input  ← 可能 L2 Cache Hit!        │
│                       写 output                             │
│                                                             │
│ 每个元素被读取 2 次 (减少 33%)                            │
│ 实际效果可能更好，因为第二次读取可能命中 L2 cache         │
└────────────────────────────────────────────────────────────┘
```

### 内存指标对比

| 指标 | V3/V4 | V5 Online | 改进 |
|------|-------|-----------|------|
| 输入读取次数 | 3 次 | **2 次** | **-33%** |
| 有效计算遍历 | 2 次 | **1 次** | 减少同步点 |
| L2 Cache 命中率 | 低 | **高** | 第 2 次读命中 |
| 全局内存带宽压力 | 高 | **降低** | 显著优化 |

---

## 六、核心代码完整解析

### 核函数实现

```cuda
__global__ void softmax_v5_online_kernel(const float* input, float* output, int M, int N) {
    // 继承 V3/V4 的 warp 组织方式
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    // Online 算法状态变量
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // ===== Phase 1: Single-Pass Online Update =====
    for (int i = lane_id; i < N; i += 32) {
        float val = x[i];
        float new_max = fmaxf(local_max, val);
        
        // Online Softmax 核心公式
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }

    // ===== Phase 2: Warp Online Reduction =====
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);

        float new_max = fmaxf(local_max, other_max);
        // 合并两个 online 状态
        local_sum = local_sum * expf(local_max - new_max) 
                  + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    // 广播最终结果
    float row_max = __shfl_sync(0xffffffff, local_max, 0);
    float row_sum = __shfl_sync(0xffffffff, local_sum, 0);

    // ===== Phase 3: Write Output =====
    for (int i = lane_id; i < N; i += 32) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}
```

### 调用配置

```cuda
void softmax_v5(const float* d_input, float* d_output, int M, int N) {
    int threads = 256;                    // 8 warps
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    
    softmax_v5_online_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

---

## 七、Online Warp 归约详解

### 归约过程可视化

```
初始: 32 个 lane，每个持有 (max, sum)
Lane:  0       1       2       ...    15      16      ...    31
      (m0,s0) (m1,s1) (m2,s2)       (m15,s15) (m16,s16)    (m31,s31)

Step 1: stride=16 (lane 0-15 与 lane 16-31 合并)
  Lane 0: 合并 (m0,s0) 和 (m16,s16)
    new_max = max(m0, m16)
    new_sum = s0 * exp(m0 - new_max) + s16 * exp(m16 - new_max)
  Lane 1: 合并 (m1,s1) 和 (m17,s17)
  ...
  结果: 16 个 (max, sum) 对

Step 2: stride=8
  16 → 8 个 (max, sum) 对

Step 3: stride=4
  8 → 4 个

Step 4: stride=2
  4 → 2 个

Step 5: stride=1
  2 → 1 个 (在 Lane 0)

广播: Lane 0 的 (row_max, row_sum) 广播给全部 32 lanes
```

---

## 八、优缺点总结

### ✅ 优点

1. **内存访问减少** - 输入只读 2 次而非 3 次，减少 33%
2. **L2 Cache 友好** - 第二次读取大概率命中 L2 cache
3. **计算与内存重叠** - online 更新隐藏部分延迟
4. **继承 V3 优势** - 保留 Warp Shuffle 低延迟归约
5. **FlashAttention 核心技术** - 现代 Attention 优化的标准做法

### ❌ 缺点

1. **更多寄存器使用** - 需要同时保存 max 和 sum
2. **更多计算量** - 每次迭代需要额外 exp 运算
3. **数值稳定性略低** - 多次 rescaling 可能累积误差（但通常可接受）
4. **代码复杂度** - online 算法逻辑较难理解

---

## 九、适用场景

### 推荐使用场景

- **内存带宽受限** - 任何需要减少内存访问的场景
- **大规模 N** - N 越大，减少的内存访问越多
- **L2 Cache 充足** - 第二次遍历能从 cache 受益
- **FlashAttention 类工作负载** - 这是其核心优化

### 与其他版本的选择

```
N 很大, 内存带宽紧张:     V5 (Online) ← 最佳选择
N 是 4 的倍数, 带宽充足:  V4 (Vectorized)
通用场景, 追求简单:       V3 (Warp Shuffle)
小 N, M 很大:             V3 或 V5
```

---

## 十、数值精度注意事项

### 潜在问题

```
Online Softmax 在连续 rescaling 时可能累积误差:

local_sum = local_sum * exp(local_max - new_max) + exp(val - new_max)
           ↑ 这部分可能引入微小数值误差

但在实际应用中:
• 误差通常在 1e-6 级别，对深度学习可接受
• FlashAttention 使用相同算法，已验证可靠性
• 可通过更高精度浮点数（如 FP64）缓解
```

### 误差对比

| 场景 | 传统 Softmax | Online Softmax | 差异 |
|------|-------------|----------------|------|
| 典型数值 | ~1e-7 | ~1e-6 | 可接受 |
| 极端数值 | ~1e-8 | ~1e-5 | 需注意 |

---

## 十一、进一步优化方向

### V5 → V6+ 可能的优化点

1. **向量化 + Online** - 结合 V4 的 float4 和 V5 的 online 算法
2. **Tensor Memory Accelerator (TMA)** - 使用 Hopper/Blackwell 特性
3. **持久化 Warp** - 一个 warp 处理多行提高数据复用
4. **异步拷贝** - 使用 `cp.async` 预取数据
5. **FP16/BF16** - 使用低精度计算进一步加速

---

*文档生成时间：2026-03-23*

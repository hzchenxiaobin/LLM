# Softmax V1: 朴素实现详解

> CUDA 三核函数朴素实现版本 · 适合初学者理解基础概念
> 源文件：`src/softmax_v1_naive.cu`

---

## 一、Softmax 数学公式

Softmax 函数将输入向量转换为概率分布，公式如下：

```
Softmax(xᵢ) = exp(xᵢ - max) / Σⱼ exp(xⱼ - max)
```

其中减去 max 是为了**数值稳定性**，防止指数爆炸。

---

## 一（补）、数值稳定性详解

### 为什么需要减去 max？

Softmax 的原始数学公式是：

```
Softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

但在实际计算中，这会遇到严重的**数值溢出问题**。

### 数值爆炸示例

假设输入向量中有较大的数值：

```
输入: [1000, 1000, 1000]

原始计算:
exp(1000) = 1.97 × 10^434  ← 超出 float32 范围（~3.4 × 10^38）
```

在 float32 中，`exp(1000)` 会立即溢出为 `inf`（无穷大），导致整个 softmax 计算失效。

### 数值稳定性技巧原理

通过减去最大值 `max(x)`，我们将指数运算的参数变为负数或零：

```
输入: [1000, 1000, 1000], max = 1000

变换后:
exp(1000 - 1000) = exp(0) = 1
exp(1000 - 1000) = exp(0) = 1
exp(1000 - 1000) = exp(0) = 1

Softmax = [1/3, 1/3, 1/3]  ✓ 正确结果
```

### 数学等价性证明

减去 max 不改变 softmax 的输出结果：

```
exp(xᵢ - max) / Σⱼ exp(xⱼ - max)
= exp(xᵢ) × exp(-max) / [Σⱼ exp(xⱼ) × exp(-max)]
= exp(xᵢ) / Σⱼ exp(xⱼ)           ← exp(-max) 被约掉
= 原始 Softmax 公式
```

**结论**：数值上更稳定，数学上完全等价。

### 更多对比示例

| 输入 | 原始方法 | 稳定方法 (减 max) | 结果 |
|------|----------|-------------------|------|
| `[1000, 1000, 1000]` | `inf / inf = NaN` ❌ | `[0.333, 0.333, 0.333]` ✓ | 避免溢出 |
| `[1000, 1001, 1002]` | 全部溢出 ❌ | `[0.090, 0.245, 0.665]` ✓ | 正确计算 |
| `[-1000, -1000, -1000]` | `exp(-1000) ≈ 0` (下溢) | `[0.333, 0.333, 0.333]` ✓ | 避免下溢 |

### 实现细节

在代码中，我们只需在每行计算前找到最大值：

```cuda
// Step 1: 找到该行最大值
float row_max = -INFINITY;
for (int i = 0; i < N; i++) {
    row_max = fmaxf(row_max, input[row * N + i]);
}

// Step 2: 计算 exp(x - max) 的累加和
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += expf(input[row * N + i] - row_max);  // 关键：减 max
}

// Step 3: 归一化
for (int i = 0; i < N; i++) {
    output[row * N + i] = expf(input[row * N + i] - row_max) / sum;
}
```

### 关键要点总结

1. **必须减 max**：否则大数输入必然溢出
2. **不影响结果**：数学上等价变换
3. **每行独立**：每行的 max 单独计算
4. **标准做法**：所有深度学习框架都采用此技巧

---

## 二、算法流程

```
输入矩阵 (M×N)          每行最大值          每行 exp 求和         输出概率矩阵
┌───┬───┬───┬───┐       ┌─────┐           ┌──────┐            ┌─────┬─────┬─────┬─────┐
│1.2│3.4│2.1│0.5│       │ 3.4 │           │ 1.44 │            │0.08 │0.70 │0.19 │0.04 │
├───┼───┼───┼───┤  →    ├─────┤    →      ├──────┤   →        ├─────┼─────┼─────┼─────┤
│5.6│2.3│4.1│1.8│       │ 5.6 │           │ 1.28 │            │0.78 │0.03 │0.14 │0.05 │
└───┴───┴───┴───┘       └─────┘           └──────┘            └─────┴─────┴─────┴─────┘
   ↑                        ↑                 ↑
每行一个线程           中间结果存储          最终输出
```

---

## 三、三核函数执行流程

### Kernel 1: kernel_max_v1 - 求每行最大值
- **功能**：每个线程负责一行的遍历，找出该行最大值
- **公式**：`local_max = max(input[row*N + i]) for i in [0, N)`

### Kernel 2: kernel_sum_v1 - 计算指数和
- **功能**：再次遍历每行，计算 `exp(x - max)` 的累加和
- **公式**：`local_sum += exp(input[row*N + i] - row_max) for i in [0, N)`

### Kernel 3: kernel_div_v1 - 归一化输出
- **功能**：最后一次遍历，计算最终 softmax 值
- **公式**：`output = exp(x - max) / sum`

---

## 四、线程与数据映射

### 一维线程网格配置

```
Block 0 (512 threads)
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │  ...  (T511)
│ R0 │ R1 │ R2 │ R3 │ R4 │ R5 │ R6 │ R7 │       (R511)
└────┴────┴────┴────┴────┴────┴────┴────┘

每个线程处理一整行数据（Sequential Row Processing）
```

### 线程索引计算

```cuda
int row = blockIdx.x * blockDim.x + threadIdx.x;  // 每行一个线程
if (row >= M) return;  // 边界检查
```

---

## 五、内存访问分析

| 指标 | 数值 | 说明 |
|------|------|------|
| 每个元素访问次数 | **6** | ⚠️ 过高 |
| 全局内存读取次数 | **3** | 每个 kernel 读一次 |
| 全局内存写入次数 | **1** | 仅 output |
| 共享内存利用率 | **0%** | ⚠️ 未使用 |

### 详细内存访问流程

```
输入矩阵 (Global)              Kernel 处理                    输出矩阵 (Global)
┌─────────────────┐           ┌────────────────┐            ┌─────────────────┐
│ input[row*N+i]  │ ──────→   │ Kernel 1: Max  │ ──────→    │  output[row*N]  │
│   读取 3 次     │           │  max_vals[row] │            │    写入 1 次    │
│                 │ ──────→   ├────────────────┤            │                 │
│                 │           │ Kernel 2: Sum  │            │                 │
│                 │ ──────→   │ sum_vals[row]  │            │                 │
└─────────────────┘           └────────────────┘            └─────────────────┘
```

⚠️ **性能问题**：每个输入元素被读取 3 次，且完全从全局内存访问，没有使用共享内存或寄存器优化。

---

## 六、核心代码解析

### Kernel 1: 求最大值

```cuda
__global__ void kernel_max_v1(const float* input, float* max_vals, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float local_max = -INFINITY;  // 初始化为负无穷
    for (int i = 0; i < N; i++) {
        local_max = fmaxf(local_max, input[row * N + i]);  // 遍历整行
    }
    max_vals[row] = local_max;  // 写回全局内存
}
```

### Kernel 2: 求指数和

```cuda
__global__ void kernel_sum_v1(const float* input, const float* max_vals, 
                              float* sum_vals, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float row_max = max_vals[row];  // 读取 Kernel 1 的结果
    float local_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        local_sum += expf(input[row * N + i] - row_max);  // 减 max 防溢出
    }
    sum_vals[row] = local_sum;
}
```

### Kernel 3: 归一化

```cuda
__global__ void kernel_div_v1(const float* input, const float* max_vals, 
                             const float* sum_vals, float* output, 
                             int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float row_max = max_vals[row];
    float row_sum = sum_vals[row];
    for (int i = 0; i < N; i++) {
        // 计算最终 softmax 值
        output[row * N + i] = expf(input[row * N + i] - row_max) / row_sum;
    }
}
```

---

## 七、优缺点总结

### ✅ 优点

- **代码简单易懂** - 逻辑清晰，适合初学者
- **并行度高** - 行间完全并行 (M 行同时处理)
- **无同步问题** - 行间无数据依赖
- **易于调试** - 三阶段分离，便于验证

### ❌ 缺点

- **内存访问冗余** - 每个元素读取 3 次
- **无共享内存优化** - 完全依赖全局内存
- **核函数启动开销** - 3 次 kernel launch
- **线程负载不均** - 每线程处理整行，N 大时串行长
- **无向量化访存** - 未利用 float4 等优化

---

## 八、适用场景

### 推荐使用场景

- 行数 M 很大（如 > 10000），每行长度 N 较小（如 < 100）
- 教学演示，理解 Softmax 的基本并行化思路
- 作为基准版本，用于对比后续优化版本

### 不推荐使用场景

- 每行长度 N 很大（如 > 1000），此时每线程串行处理太慢
- 对延迟敏感的生产环境（有 kernel launch 开销）
- 带宽受限的场景（需要减少全局内存访问）

---

## 九、性能参考数据

在 RTX 5090 上的典型性能（仅供参考）：

| 指标 | 数值 |
|------|------|
| 内存带宽利用率 | ~50-100 GB/s |
| 理论峰值性能 | ~5-10% |
| 计算密度 | 低 |

> 注：实际性能取决于矩阵大小 (M, N)。后续优化版本（V2+）可提升数倍至数十倍性能。

---

## 十、调用方式

```cuda
// 线程配置 (针对 RTX 5090 优化)
int threads = 512;  // 从 256 增加到 512，利用更多 SMs
int blocks = (M + threads - 1) / threads;

// 依次启动三个核函数
kernel_max_v1<<<blocks, threads>>>(d_input, d_max, M, N);
kernel_sum_v1<<<blocks, threads>>>(d_input, d_max, d_sum, M, N);
kernel_div_v1<<<blocks, threads>>>(d_input, d_max, d_sum, d_output, M, N);
cudaDeviceSynchronize();
```

---

*文档生成时间：2026-03-23*

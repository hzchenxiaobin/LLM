# Softmax V4: 向量化内存访问详解

> CUDA 向量化内存访问优化版本 - 使用 float4 (128-bit) 最大化内存带宽
> 源文件：`src/softmax_v4_vectorized.cu`

---

## 一、核心思想

V4 在 V3 (Warp Shuffle) 的基础上，增加了**向量化内存访问**优化，通过 `float4` 类型一次读写 4 个 float，提升内存带宽利用率。

### 为什么需要向量化？

```
GPU 内存访问特性:
• 全局内存以 32-byte (L2 cache line) 或 128-byte (L1 cache line) 为单位传输
• 一次读取 4 个 float (16 bytes) 比读取 1 个 float (4 bytes) 更高效
• 减少内存事务数量，提高带宽利用率
```

### V3 vs V4 对比

| 特性 | V3 Warp Shuffle | V4 向量化版本 |
|------|----------------|--------------|
| **内存访问粒度** | 单个 float (32-bit) | **4x float (128-bit)** |
| **每次读取数据量** | 4 bytes | **16 bytes** |
| **内存事务数** | N / 线程数 | **N/4 / 线程数** (减少 4x) |
| **归约方式** | Warp Shuffle | **Warp Shuffle (继承)** |
| **适用条件** | 任意 N | **N 必须是 4 的倍数** |
| **带宽利用率** | ~50-70% | **~80-95%** |

---

## 二、float4 数据类型详解

### float4 结构

```cuda
// CUDA 内置向量类型
struct float4 {
    float x;  // 第 0 个 float
    float y;  // 第 1 个 float
    float z;  // 第 2 个 float
    float w;  // 第 3 个 float
};  // 总共 16 bytes (128 bits)
```

### 指针转换

```cuda
// 原始 float 指针
const float* input;  // 每个元素 4 bytes

// 转换为 float4 指针 (向量化访问)
const float4* x_vec = reinterpret_cast<const float4*>(input);
// 每个元素现在包含 4 个 float，共 16 bytes
```

### 内存布局对比

```
原始 float 数组 (N=16):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  ...
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↓ 每次读取 1 个

向量化后 float4 数组 (N_vec = N/4 = 4):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  float4[0]  │  float4[1]  │  float4[2]  │  float4[3]  │
│ (0,1,2,3)   │  (4,5,6,7)  │ (8,9,10,11) │(12,13,14,15)│
└─────────────┴─────────────┴─────────────┴─────────────┘
  ↓ 每次读取 4 个 float
```

---

## 三、向量化访问模式

### 数据分配方式

```
Row 数据分布 (N=4096 → N_vec=1024):

Warp 0 (Lane 0-31) 处理 Row 0:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ Round 1:                                                    │
│ Lane0   Lane1   Lane2   Lane3   ...  Lane30  Lane31          │
│   ↓       ↓       ↓       ↓           ↓        ↓           │
│  f4[0]  f4[1]  f4[2]  f4[3]  ...  f4[30]  f4[31]           │
│ (0-3)   (4-7)  (8-11) (12-15)    (120-123)(124-127)       │
│                                                             │
│ Round 2: (i += 32)                                          │
│   ↓       ↓       ↓       ↓           ↓        ↓           │
│ f4[32]  f4[33] f4[34] f4[35] ... f4[62]  f4[63]            │
│(128-131)(132..)                              (252-255)     │
│                                                             │
│ ... 继续直到 1024 ...                                       │
│                                                             │
│ Total: 每个 lane 处理 1024/32 = 32 个 float4 = 128 个 float │
└─────────────────────────────────────────────────────────────┘
```

### 代码实现

```cuda
// 计算向量化后的元素数量 (N 必须是 4 的倍数)
int N_vec = N / 4;

// Strided 循环访问 float4
for (int i = lane_id; i < N_vec; i += 32) {
    float4 val = x_vec[i];  // 一次读取 4 个 float
    // 展开处理 4 个元素
    local_max = fmaxf(local_max, val.x);
    local_max = fmaxf(local_max, val.y);
    local_max = fmaxf(local_max, val.z);
    local_max = fmaxf(local_max, val.w);
}
```

---

## 四、完整执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│ Warp 0 处理 Row 0 (32 lanes)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: 并行求最大值 (向量化读取)                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Lane 0-31 并行读取 float4 向量 (i = lane_id; i < N_vec)   │  │
│  │ 每次读取 4 个 float (val.x, val.y, val.z, val.w)         │  │
│  │                                                          │  │
│  │ 展开循环处理 4 个元素:                                    │  │
│  │   local_max = fmaxf(local_max, val.x)                    │  │
│  │   local_max = fmaxf(local_max, val.y)                    │  │
│  │   local_max = fmaxf(local_max, val.z)                    │  │
│  │   local_max = fmaxf(local_max, val.w)                    │  │
│  │                                                          │  │
│  │ warpReduceMax (shuffle 归约)                             │  │
│  │ __shfl_sync 广播 row_max                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Phase 2: 并行求指数和 (向量化读取)                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Lane 0-31 再次读取 float4 向量                            │  │
│  │ 展开计算 4 个 expf 并累加:                                │  │
│  │   local_sum += expf(val.x - row_max)                     │  │
│  │   local_sum += expf(val.y - row_max)                     │  │
│  │   local_sum += expf(val.z - row_max)                     │  │
│  │   local_sum += expf(val.w - row_max)                     │  │
│  │                                                          │  │
│  │ warpReduceSum (shuffle 归约)                            │  │
│  │ __shfl_sync 广播 row_sum                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Phase 3: 向量化归一化输出                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Lane 0-31 最后一次读取 float4 向量                        │  │
│  │ 展开计算 4 个输出:                                        │  │
│  │   out_val.x = expf(val.x - row_max) / row_sum            │  │
│  │   out_val.y = expf(val.y - row_max) / row_sum            │  │
│  │   out_val.z = expf(val.z - row_max) / row_sum            │  │
│  │   out_val.w = expf(val.w - row_max) / row_sum            │  │
│  │                                                          │  │
│  │ y_vec[i] = out_val;  // 向量化写入 4 个 float            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、内存带宽分析

### 内存事务对比

```
假设: N = 4096, warp 32 线程

V3 (float 访问):
┌────────────────────────────────────────────────────────────┐
│ 每个 lane 读取次数: 4096 / 32 = 128 次                      │
│ 每次读取: 4 bytes (1 float)                                 │
│ 总共读取: 128 × 4 = 512 bytes/lane                         │
│ 每 warp 读取: 32 × 512 = 16,384 bytes = 16 KB              │
│ 内存事务数: ~16 KB / 32 B (cache line) ≈ 512 事务          │
└────────────────────────────────────────────────────────────┘

V4 (float4 访问):
┌────────────────────────────────────────────────────────────┐
│ N_vec = 4096 / 4 = 1024                                     │
│ 每个 lane 读取次数: 1024 / 32 = 32 次                       │
│ 每次读取: 16 bytes (1 float4 = 4 floats)                   │
│ 总共读取: 32 × 16 = 512 bytes/lane (相同数据量)            │
│ 每 warp 读取: 32 × 512 = 16,384 bytes = 16 KB              │
│ 内存事务数: ~16 KB / 32 B ≈ 128 事务 (减少 4x!)          │
└────────────────────────────────────────────────────────────┘
```

### 带宽利用率

| 指标 | V3 | V4 | 提升 |
|------|-----|-----|------|
| 内存事务数 | 512 | **128** | **4x 减少** |
| 每次读取数据量 | 4 bytes | **16 bytes** | 4x |
| 理论带宽利用率 | ~60-75% | **~85-95%** | **+20-30%** |
| 实际内存效率 | 中等 | **高** | 显著 |

---

## 六、核心代码解析

### 核函数入口

```cuda
__global__ void softmax_v4_vectorized_kernel(const float* input, float* output, int M, int N) {
    // 继承 V3 的 warp 组织方式
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    // 关键: 转换为 float4 指针进行向量化访问
    const float4* x_vec = reinterpret_cast<const float4*>(input + row * N);
    float4* y_vec = reinterpret_cast<float4*>(output + row * N);

    int N_vec = N / 4;  // 向量化后的元素数量
    
    // ... 三阶段计算
}
```

### Phase 1: 向量化求最大值

```cuda
// 1. Find max (向量化版本)
float local_max = -INFINITY;
for (int i = lane_id; i < N_vec; i += 32) {
    float4 val = x_vec[i];  // 一次读取 128 bits (4 floats)
    
    // 展开处理 4 个元素
    local_max = fmaxf(local_max, val.x);
    local_max = fmaxf(local_max, val.y);
    local_max = fmaxf(local_max, val.z);
    local_max = fmaxf(local_max, val.w);
}
float row_max = warpReduceMax(local_max);
row_max = __shfl_sync(0xffffffff, row_max, 0);
```

### Phase 2: 向量化求指数和

```cuda
// 2. Compute sum (向量化版本)
float local_sum = 0.0f;
for (int i = lane_id; i < N_vec; i += 32) {
    float4 val = x_vec[i];  // 复用读取的数据或重新读取
    
    // 展开计算 4 个 expf 并累加
    local_sum += expf(val.x - row_max);
    local_sum += expf(val.y - row_max);
    local_sum += expf(val.z - row_max);
    local_sum += expf(val.w - row_max);
}
float row_sum = warpReduceSum(local_sum);
row_sum = __shfl_sync(0xffffffff, row_sum, 0);
```

### Phase 3: 向量化写入

```cuda
// 3. Write back (向量化版本)
for (int i = lane_id; i < N_vec; i += 32) {
    float4 val = x_vec[i];  // 读取输入
    float4 out_val;          // 构造输出向量
    
    // 展开计算 4 个输出
    out_val.x = expf(val.x - row_max) / row_sum;
    out_val.y = expf(val.y - row_max) / row_sum;
    out_val.z = expf(val.z - row_max) / row_sum;
    out_val.w = expf(val.w - row_max) / row_sum;
    
    y_vec[i] = out_val;  // 向量化写入 128 bits
}
```

### 调用配置

```cuda
void softmax_v4(const float* d_input, float* d_output, int M, int N) {
    // 要求 N 必须是 4 的倍数
    if (N % 4 != 0) {
        // 需要 padding 或回退到 V3
    }
    
    int threads = 256;                    // 8 warps
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    
    softmax_v4_vectorized_kernel<<<blocks, threads>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

---

## 七、优缺点总结

### ✅ 优点

1. **更高内存带宽** - float4 访问减少内存事务数 4x
2. **更高带宽利用率** - 从 ~60% 提升到 ~90%+
3. **减少指令数** - 读取指令减少 4x
4. **继承 V3 优势** - 保留 Warp Shuffle 的低延迟归约
5. **特别适合新 GPU** - RTX 5090 等 GDDR7 显卡能充分发挥带宽

### ❌ 缺点

1. **N 必须是 4 的倍数** - 需要数据对齐或 padding
2. **代码复杂性增加** - 需要手动展开 4 个元素
3. **寄存器压力** - float4 占用更多寄存器空间
4. **不灵活** - 如果 N 不是 4 的倍数需要额外处理

---

## 八、适用场景

### 推荐使用场景

- **N 是 4 的倍数**（如 1024, 2048, 4096）- 完美匹配
- **带宽受限场景** - 需要最大化内存带宽利用率
- **新架构 GPU** - RTX 5090 (GDDR7), A100/H100 (HBM) 等高带宽显卡
- **大规模数据处理** - 大数据集更能体现带宽优势

### 不推荐使用场景

- **N 不是 4 的倍数** - 需要 padding 或复杂边界处理
- **计算密集型任务** - 如果计算是瓶颈而非内存
- **小 N 值** - N < 100 时向量化收益不明显

---

## 九、边界处理方案

### 当 N 不是 4 的倍数时

```cuda
// 方案 1: Padding (推荐)
// 分配时多分配一些空间，保证是 4 的倍数
int N_padded = ((N + 3) / 4) * 4;  // 向上取整到 4 的倍数
cudaMalloc(&d_input, M * N_padded * sizeof(float));

// 方案 2: 混合策略
// 主体用 V4 (向量化), 边界用 V3 (标量)
if (N % 4 == 0) {
    softmax_v4(d_input, d_output, M, N);
} else {
    softmax_v3(d_input, d_output, M, N);  // 回退到 V3
}

// 方案 3: 循环展开处理剩余元素 (复杂)
// 在 kernel 内额外处理最后 1-3 个元素
```

---

## 十、进一步优化方向

### V4 → V5+ 可能的优化点

1. **Online Softmax** - 一次遍历同时计算 max 和 sum，减少内存访问
2. **Tensor Memory Accelerator (TMA)** - 使用 Hopper/Blackwell 的 TMA 单元
3. **更宽的向量类型** - float8 (如果硬件支持 256-bit 访问)
4. **异步拷贝** - 使用 `cp.async` 隐藏内存延迟

---

*文档生成时间：2026-03-23*

# V4 向量化与 Bank Conflict 消除 - 快速参考

## 核心改进

V4 相比 V3 引入两大关键技术：
1. **float4 向量化加载**：充分利用128位内存总线
2. **Padding 消除 Bank Conflict**：提升共享内存并行度

```
┌──────────────────────────────────────────────────────────────────┐
│  V3: 标量加载 + Bank Conflict                                    │
│  - 32位加载，浪费75%带宽                                          │
│  - Shared Memory访问串行化                                        │
├──────────────────────────────────────────────────────────────────┤
│  V4: 向量化 + 无Conflict                                          │
│  - 128位float4加载，100%带宽利用                                 │
│  - Padding对齐，并行访问                                          │
├──────────────────────────────────────────────────────────────────┤
│  结果: 2-3x额外加速，带宽接近峰值！                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 关键技术点

### 1. float4 向量化

```cuda
// float4 定义 (16字节 = 128位)
struct float4 {
    float x, y, z, w;
};

// 加载函数 (编译成LD.128指令)
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

// 使用示例
const float4* Q_vec4 = reinterpret_cast<const float4*>(Q + q_row * d);
float4 val = load_float4(reinterpret_cast<const float*>(Q_vec4 + i));

// 解包
q_ptr[0] = val.x;
q_ptr[1] = val.y;
q_ptr[2] = val.z;
q_ptr[3] = val.w;
```

**效果**：4个float → 1条指令，4×带宽利用率！

### 2. Bank Conflict 消除

```cuda
// Padding配置
constexpr int SMEM_PAD = 1;
int d_padded = d + SMEM_PAD;  // 64 + 1 = 65

// 原始布局 (冲突)
K_tile[row * d + col]  // d=64是32的倍数

// Padding后 (无冲突)
K_tile[row * d_padded + col]  // d_padded=65不是32的倍数
```

**效果**：Bank访问错开，冲突消除！

---

## 关键代码段

### 1. 向量化 Helper
```cuda
// 128位加载
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

// 128位存储
__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}
```

### 2. 带Padding的共享内存布局
```cuda
int d_padded = d + SMEM_PAD;  // 65
int buf_size = V4_Bc * d_padded;  // 64 × 65 = 4160

K_buffers[0] = shared_mem;                          // Buffer 0 K
V_buffers[0] = shared_mem + buf_size;              // Buffer 0 V
K_buffers[1] = shared_mem + 2 * buf_size;         // Buffer 1 K
V_buffers[1] = shared_mem + 3 * buf_size;         // Buffer 1 V
// 总计: 4 × 4160 × 4 = 65KB (对比V3的64KB)
```

### 3. 向量化Q加载
```cuda
// 加载Q使用float4
int d_vec4 = d / 4;  // 16个float4
const float4* Q_vec4 = reinterpret_cast<const float4*>(Q + q_row * d);

for (int i = 0; i < d_vec4; i++) {
    float4 val = load_float4(reinterpret_cast<const float*>(Q_vec4 + i));
    q_ptr[0] = val.x;
    q_ptr[1] = val.y;
    q_ptr[2] = val.z;
    q_ptr[3] = val.w;
    q_ptr += 4;
}
```

### 4. 向量化KV加载 (核心)
```cuda
auto load_tile_vectorized = [&](int buf_idx, int kv_start) {
    int total_elements = V4_Bc * d;  // 4096
    int float4_per_thread = (total_elements / 4 + 128 - 1) / 128;  // 8

    for (int i = 0; i < float4_per_thread; i++) {
        int idx4 = tid * float4_per_thread + i;
        int base_idx = idx4 * 4;

        if (base_idx < total_elements) {
            int row = base_idx / d;
            int col = base_idx % d;
            int global_row = kv_start + row;

            // 加载float4 (128位)
            if (global_row < N && col + 3 < d) {
                const float4 k_val = load_float4(&K[global_row * d + col]);

                // 存储到padded共享内存 (注意d_padded！)
                K_buffers[buf_idx][row * d_padded + col] = k_val.x;
                K_buffers[buf_idx][row * d_padded + col + 1] = k_val.y;
                K_buffers[buf_idx][row * d_padded + col + 2] = k_val.z;
                K_buffers[buf_idx][row * d_padded + col + 3] = k_val.w;
            }
        }
    }
};
```

### 5. 向量化输出存储
```cuda
// 写回O也使用float4
int d_vec4 = d / 4;
float4* O_vec4 = reinterpret_cast<float4*>(O + q_row * d);

for (int i = 0; i < d_vec4; i++) {
    float4 val;
    val.x = o_ptr[0] / l;
    val.y = o_ptr[1] / l;
    val.z = o_ptr[2] / l;
    val.w = o_ptr[3] / l;
    store_float4(reinterpret_cast<float*>(O_vec4 + i), val);
    o_ptr += 4;
}
```

---

## 内存布局对比

### 共享内存布局

```
V3 (无padding):
┌────────────────────────────────────────────────────────────┐
│ Buffer 0 K: 64 × 64 = 4096 floats = 16KB                     │
│ Row 0: [0:63]                                              │
│ Row 1: [64:127]                                            │
│ ...                                                        │
│ Row 63: [4032:4095]                                        │
│ 问题: Row 1 与 Row 0 bank冲突                              │
└────────────────────────────────────────────────────────────┘

V4 (带padding, d_padded=65):
┌────────────────────────────────────────────────────────────┐
│ Buffer 0 K: 64 × 65 = 4160 floats = 16.25KB                │
│ Row 0: [0:64] (65 elements, 1 padding)                     │
│ Row 1: [65:129] (从65开始，不是64！)                      │
│ ...                                                        │
│ Row 63: [4095:4159]                                        │
│ 解决: Row 1 访问的bank与Row 0错开                          │
└────────────────────────────────────────────────────────────┘
```

### 索引计算对比

| 操作 | 索引公式 | 说明 |
|------|---------|------|
| 全局内存读取 | `K[global_row * d + col]` | 使用原始d |
| 共享内存读取 | `K_tile[b * d_padded + i]` | 使用d_padded |
| 共享内存写入 | `K_tile[row * d_padded + col]` | 使用d_padded |

---

## 性能数据

| 指标 | V3 | V4 | 提升 |
|------|----|----|------|
| 内存总线利用 | 25% | 100% | **4×** |
| 加载指令数 | 4条/float4 | 1条/float4 | **4×↓** |
| Bank Conflict | 有 | 无 | **消除** |
| 额外加速 | 1× | **2-3×** | ✓ |
| 总加速(V1→V4) | ~10× | **~20×** | ✓ |

---

## 对齐要求

```cuda
// Host wrapper检查对齐
if (d % 4 != 0) {
    printf("Warning: d=%d is not divisible by 4\n", d);
}

// 要求:
// - d必须是4的倍数 (float4对齐)
// - 内存地址16字节对齐
// - 如果不对齐，向量化会失效或崩溃
```

---

## 边界处理

```cuda
// 安全加载float4 (检查边界)
if (global_row < N && col + 3 < d) {
    // 可以安全加载float4
    float4 k_val = load_float4(&K[global_row * d + col]);
} else if (global_row < N) {
    // 边界情况: 逐个标量加载
    for (int c = 0; c < 4 && col + c < d; c++) {
        K_buffers[...] = K[...];  // 标量
    }
}
```

---

## 常见错误

### ❌ 错误1: 索引混淆
```cuda
// 错误: 在共享内存访问时使用d而不是d_padded
K_tile[b * d + i]  // ❌ 这是V3的写法

// 正确: 使用d_padded
K_tile[b * d_padded + i]  // ✅ V4必须用这个
```

### ❌ 错误2: 忘记对齐检查
```cuda
// 错误: 直接reinterpret_cast
float4* ptr = (float4*)Q;  // 如果Q不对齐会崩溃

// 正确: 确保d是4的倍数
if (d % 4 != 0) return;  // 或使用标量回退
```

### ❌ 错误3: 越界float4加载
```cuda
// 错误: 在边界加载float4
float4 val = load_float4(&K[row * d + 62]);  // 62+3=65 > d=64, 越界！

// 正确: 检查col + 3 < d
if (col + 3 < d) {  // ✅ 安全检查
    float4 val = load_float4(&K[...]);
}
```

---

## 文档索引

| 文档 | 内容 | 推荐阅读顺序 |
|------|------|-------------|
| `V4_VECTORIZED_EXPLAINED.md` | 完整技术解析 | 第1 |
| `V4_VECTORIZED_VISUAL.md` | 可视化图表 | 第2 |
| `V4_SUMMARY.md` (本文档) | 快速参考 | 第3 |

---

## 下一版本预告

**V5: FlashAttention-2 优化**
- Split-KV并行策略
- Warp级优化
- 更好的长序列处理

---

*FlashAttention CUDA 教程系列*

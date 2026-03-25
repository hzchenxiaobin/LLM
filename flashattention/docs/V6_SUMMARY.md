# FlashAttention V6: Tensor Core (WMMA) 快速参考

## 一句话总结

**V6 是教育演示版本，展示如何使用 NVIDIA Tensor Core 和 WMMA API 进行矩阵运算。这是简化的 API 演示，不包含完整 FlashAttention 实现。**

---

## 核心概念

| 概念 | 说明 |
|------|------|
| **Tensor Core** | GPU专用矩阵运算单元，执行 16×16×16 矩阵乘加 |
| **WMMA API** | Warp Matrix Multiply Accumulate，CUDA访问Tensor Core的接口 |
| **Fragment** | 16×16矩阵的分布式存储，由warp的32线程共同持有 |
| **混合精度** | FP16输入 + FP32累加，平衡速度与精度 |

---

## 关键技术点

### 1. WMMA 三步操作

```
加载 → 计算 → 存储

load_matrix_sync() → mma_sync() → store_matrix_sync()
     ↓                   ↓                ↓
  Q/K → fragment    D=A×B+C         fragment → S/O
```

### 2. Fragment 类型

```cuda
fragment<matrix_a, 16, 16, 16, __half, row_major> q_frag;
fragment<matrix_b, 16, 16, 16, __half, col_major> k_frag;
fragment<accumulator, 16, 16, 16, float> s_frag;
```

### 3. 配置常量

```
V6_Br = 64          // Query tile行数
V6_Bc = 64          // KV tile大小
V6_THREADS = 128    // 每block线程数
WMMA_M/N/K = 16     // Fragment固定16×16×16
```

---

## 完整 FlashAttention + Tensor Core 的挑战

```
为什么生产环境用 CUTLASS/cuDNN？

1. Online Softmax 不是矩阵运算
   → Tensor Core帮不上忙

2. 需要复杂的双重 Tiling
   → Q tiling + K/V tiling + Fragment tiling

3. 精度管理复杂
   → FP16输入可能溢出
   → 需要 careful scaling

4. Warp 协调困难
   → 多warp需要同步fragment
   → 共享内存管理复杂
```

---

## 代码结构

```
v6_tensor_core.cu
├── 协作加载Q tile (FP16)
├── 声明WMMA fragments
├── 初始化累加器 (fill_fragment)
├── 【简化】演示WMMA load/store
└── Host wrapper (调用kernel)
    └── 教育提示：建议用CUTLASS/cuDNN
```

---

## 与之前版本对比

| 特性 | V5 (FlashAttention-2) | V6 (Tensor Core演示) |
|------|----------------------|----------------------|
| 核心优化 | Split-KV并行 | Tensor Core API展示 |
| 精度 | FP32 | FP16 (混合精度) |
| 共享内存 | 100%利用 | FP16节省50% |
| 完整度 | 完整可运行 | 教育演示 |
| 生产可用 | ✅ 是 | ⚠️ 建议用CUTLASS |
| 代码复杂度 | 高 | 中等 |

---

## RTX 5090 (Blackwell) 新特性

```
5th Gen Tensor Cores:
├── FP8支持 (E4M3, E5M2)
├── 更高吞吐量 (~2000 TFLOPS)
├── TMA (Tensor Memory Accelerator)
└── 更大共享内存 (228KB/SM)
```

---

## 学习建议

1. **V1-V5** → 理解FlashAttention算法本质 ✅ 必须
2. **V6** → 了解Tensor Core API ⚠️ 演示性质
3. **生产** → 使用 CUTLASS / cuDNN ✅ 推荐

---

## 关键公式

```
WMMA核心: D = A × B + C

混合精度路径:
  FP16 input → FP32 multiply → FP32 accumulate → FP16 store

FlashAttention复杂度:
  完整Tensor Core FA需要: Q_tiling + KV_tiling + Fragment_tiling + Online_softmax
  = 极其复杂的实现 (3000+ 行代码)
```

---

## 参考资源

- **CUTLASS**: https://github.com/NVIDIA/cutlass
- **cuDNN**: `cudnnMultiHeadAttention`
- **FlashAttention官方**: https://github.com/Dao-AILab/flash-attention
- **NVIDIA Tensor Core文档**: CUDA C++ Programming Guide

---

*版本: 1.0 | 状态: 教育演示版 | 推荐: 学习V1-V5后了解API，生产用CUTLASS*

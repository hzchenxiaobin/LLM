# FlashAttention CUDA 教程 - 从基础到优化

专为 **RTX 5090 (Blackwell)** 优化的FlashAttention CUDA实现教程，从最朴素的版本开始，逐步添加优化直至接近FlashAttention-2性能。

## 仓库结构

```
flashattention/
├── Makefile                          # 编译脚本
├── README.md                         # 本教程
├── src/
│   ├── common.h                      # 公共头文件和工具
│   ├── benchmark.cu                  # 基准测试主程序
│   └── flashattention/
│       ├── kernels.h                 # 内核函数声明
│       ├── v1_naive.cu               # V1: 朴素实现
│       ├── v2_shared_kv.cu           # V2: 共享内存KV分块
│       ├── v3_q_tiling.cu            # V3: Q分块+双缓冲
│       ├── v4_vectorized.cu          # V4: 向量化+Bank Conflict消除
│       ├── v5_fa2.cu                 # V5: FlashAttention-2风格
│       ├── v6_tensor_core.cu         # V6: Tensor Core演示
│       └── reference.cu              # CPU/cuBLAS参考实现
```

## 快速开始

```bash
cd flashattention
make
./build/benchmark_flashattention
```

运行带参数的基准测试：
```bash
./build/benchmark_flashattention <batch> <seq_len> <head_dim> <iterations>
# 示例: B=2, N=2048, d=64, 20次迭代
./build/benchmark_flashattention 2 2048 64 20
```

## FlashAttention 核心概念

### 标准 Attention vs FlashAttention

**标准 Attention:**
```
S = Q @ K^T / sqrt(d)          # [N, N] 矩阵, O(N²) 内存
P = softmax(S)                 # [N, N] 矩阵
O = P @ V                      # [N, d] 输出
```

问题: 需要存储 N×N 的中间矩阵，当 N=4096 时，需要 64MB 内存！

**FlashAttention 核心思想:**
- **分块计算**: 不存储完整的 N×N 注意力矩阵
- **Online Softmax**: 增量计算 softmax，无需完整矩阵
- **IO-Aware**: 优化内存访问模式，减少 HBM 访问

### Online Softmax 算法

FlashAttention 的核心是 **online softmax**，它允许我们在不存储完整矩阵的情况下计算 softmax。

**标准 Softmax (3步):**
1. 找最大值: `m = max(x_i)`
2. 计算指数: `exp(x_i - m)`
3. 归一化: `sum(exp) / sum`

**Online Softmax (单步):**
维护两个运行统计量：
- `m`: 当前最大值
- `l`: 当前指数和

对于新值 `x`:
```
m_new = max(m_old, x)
l_new = l_old * exp(m_old - m_new) + exp(x - m_new)
```

这样可以在单次遍历中完成 softmax！

## 版本演进

### V1: Naive 实现

**核心特点:**
- 仅演示 online softmax 算法
- 所有访问都是全局内存 (最慢)
- 每个线程独立加载所有 K, V 行
- 教育用途，性能最差

**关键代码:**
```cuda
// 每个线程处理一个 Q 行
for (int k_idx = 0; k_idx < N; k_idx++) {
    // 直接从全局内存加载 K 和 V
    float qk = dot(q_vec, K[k_idx]);  // 全局内存访问
    // Online softmax 更新
    // 全局内存加载 V
}
```

**性能问题:**
- 无共享内存使用
- K/V 被重复加载 (每个线程都加载)
- 全局内存带宽瓶颈

---

### V2: 共享内存 KV 分块

**核心优化:**
- K 和 V 分块加载到共享内存
- 同一块中的线程共享 K/V 数据
- Bc 倍减少 K/V 全局内存访问

**技术细节:**
```cuda
// 协作加载 K 分块到共享内存
__shared__ float K_tile[Bc * d];
for (int tile = 0; tile < num_tiles; tile++) {
    // 所有线程一起加载 K[tile] 和 V[tile]
    load_tile_cooperatively(K_tile, K + tile * Bc * d);
    __syncthreads();

    // 所有线程使用共享内存中的相同 K/V
    for (int b = 0; b < Bc; b++) {
        float qk = dot(q_vec, K_tile + b * d);  // 共享内存访问
    }
}
```

**性能提升:**
- K/V 只加载一次 per block (不是 per thread)
- Bc 倍减少全局内存带宽 (~64x for Bc=64)
- 共享内存 ~20 cycles vs 全局内存 ~400 cycles

---

### V3: Q 分块 + 双缓冲

**核心优化:**
- 增加更多线程用于加载 (128 vs 64)
- 双缓冲: 计算当前分块时加载下一个分块
- 重叠计算和内存访问

**双缓冲技术:**
```cuda
// 两个缓冲区交替使用
__shared__ float K_buffer[2][Bc * d];
int current_buf = 0;

for (int tile = 0; tile < num_tiles; tile++) {
    // 使用 current_buf 计算
    compute_with_buffer(K_buffer[current_buf]);

    // 同时加载下一个分块到 1-current_buf
    load_next_tile_async(K_buffer[1 - current_buf]);

    __syncthreads();
    current_buf = 1 - current_buf;  // 交换
}
```

**性能提升:**
- 隐藏内存延迟
- 计算与加载重叠
- 10-20% 额外提升

---

### V4: 向量化加载 + Bank Conflict 消除

**核心优化:**
1. **float4 向量化**: 一次加载 4 个 float (128 位)
2. **Bank Conflict 消除**: Shared Memory padding

**向量化加载:**
```cuda
// 不使用向量化 (4 次 32 位加载)
for (int i = 0; i < 4; i++) {
    val[i] = global_mem[idx + i];  // 4 条指令
}

// 使用 float4 (1 次 128 位加载)
float4 val = reinterpret_cast<float4*>(global_mem)[idx / 4];  // 1 条指令
```

**Bank Conflict 消除:**
```cuda
// 原始布局: 冲突当 d % 32 == 0
__shared__ float K_tile[Bc][d];  // d=64 时，所有线程访问 bank 0, 1, 2...

// Padding 布局: 无冲突
__shared__ float K_tile[Bc][d + 1];  // stride=65，分散到不同 bank
```

**性能提升:**
- 4x 内存带宽利用
- 消除 bank conflict 序列化
- 显著性能提升

---

### V5: FlashAttention-2 风格

**核心思想 (来自 FA-2 论文):**
- **Split KV 而非 Split Q**: 在 warp 级别分割 KV 序列
- 更好的并行性 (特别是对于长序列)
- 共享 Q 分块，各 warp 处理不同 KV 分区

**线程组织:**
```
Block (128 threads = 4 warps)
├── Warp 0: 处理 KV tiles [0, N/4)
├── Warp 1: 处理 KV tiles [N/4, N/2)
├── Warp 2: 处理 KV tiles [N/2, 3N/4)
└── Warp 3: 处理 KV tiles [3N/4, N)
```

**关键优化:**
- Q 只加载一次 (共享)
- 各 warp 独立处理 KV 分区
- Warp-level reduction 优化

---

### V6: Tensor Core 演示

**说明:**
V6 是教育演示版本，展示 WMMA API 用法。完整的 Tensor Core FlashAttention 需要:
- FP16/BF16 输入
- 仔细的 fragment 管理
- 使用 CUTLASS 或 cuDNN 生产环境

**Tensor Core 要点:**
- WMMA API: `wmma::load_matrix_sync`, `wmma::mma_sync`
- 固定 fragment 大小 (16x16)
- 混合精度: FP16 输入，FP32 累加

---

## 性能对比 (RTX 4090/5090)

预期性能提升 (相对于 V1):

| 版本 | 主要优化 | 预期加速 | 关键瓶颈 |
|-----|---------|---------|---------|
| V1 | 无 | 1.0x | 全局内存带宽 |
| V2 | 共享内存 KV | 5-10x | 共享内存带宽 |
| V3 | 双缓冲 | 1.2x | 计算/内存平衡 |
| V4 | 向量化 + Bank-free | 2-3x | 指令吞吐 |
| V5 | FA-2 算法 | 1.5-2x | 线程利用率 |

**RTX 5090 特定优化:**
- Compute Capability 10.0 (Blackwell)
- FP8 Tensor Cores (用于未来扩展)
- TMA (Tensor Memory Accelerator) 异步拷贝

## 学习路径

### 阶段 1: 理解算法
1. 阅读 V1 代码，理解 online softmax
2. 对比标准 attention 和 FlashAttention 的内存使用

### 阶段 2: 内存优化
1. 阅读 V2，理解共享内存分块
2. 思考: 为什么 K/V 分块比 Q 分块更重要？

### 阶段 3: 并行优化
1. 阅读 V3 双缓冲
2. 理解计算与通信重叠

### 阶段 4: 细节优化
1. 阅读 V4 向量化和 bank conflict
2. 使用 Nsight Compute 分析内存访问

### 阶段 5: 算法优化
1. 阅读 V5 FA-2 实现
2. 理解 split-KV vs split-Q 的区别

## 调试与 profiling

### 使用 Nsight Compute
```bash
ncu -o profile.ncu-rep ./build/benchmark_flashattention
ncu-ui profile.ncu-rep
```

### 检查内存访问
```bash
# 检查全局内存合并
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct

# 检查共享内存 bank conflict
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
```

### 常见问题和解决

**问题: 数值不稳定 (NaN/inf)**
- 确保 softmax scale 正确: `1/sqrt(d)`
- 检查 online softmax 更新公式
- 初始化 `m = -INFINITY`

**问题: 精度不够**
- FlashAttention 使用 FP32 累加
- 检查中间计算是否在 FP32
- 对比 CPU 参考实现

**问题: 性能不达预期**
- 检查 occupancy: `ncu --metrics sm__occupancy`
- 检查 bank conflict: 使用 padding
- 检查向量化: d 必须是 4 的倍数

## 扩展阅读

### 论文
1. [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

### NVIDIA 资源
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - 优化的 GEMM 和 Attention

### 其他实现
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - 官方实现
- [cuDNN FlashAttention](https://docs.nvidia.com/deeplearning/cudnn/) - NVIDIA 官方

## 下一步

学完本教程后，可以探索:

1. **FP16/BF16 优化**: 使用半精度计算
2. **变长序列**: 支持不同长度的序列
3. **多查询注意力 (MQA/GQA)**: 共享 K/V 头
4. **因果 Mask**: Decoder-only 模型的因果注意力
5. **Softcap**: 如 Gemma 2 的 attention softcapping
6. **FP8**: RTX 5090 的 FP8 Tensor Core 支持

## 总结

FlashAttention 优化是一个循序渐进的过程:

1. **算法正确性** (V1): 先保证 online softmax 正确
2. **内存优化** (V2): 共享内存分块是关键
3. **并行优化** (V3): 双缓冲隐藏延迟
4. **微架构优化** (V4): 向量化，bank conflict
5. **算法改进** (V5): FA-2 的 split-KV
6. **专用硬件** (V6): Tensor Cores

每一步都建立在前一步的基础上，理解每一层优化的原理比记住代码更重要。

---

**Happy CUDA Programming!** 🚀

# RTX 5090 (Blackwell) 优化说明

本文档详细说明针对 NVIDIA RTX 5090 (Blackwell 架构) 做的特定优化。

## RTX 5090 硬件规格

| 规格 | 数值 |
|------|------|
| 架构 | Blackwell |
| 计算能力 | sm_100 |
| 显存 | GDDR7 |
| 理论带宽 | ~1792 GB/s |
| SMs | 更多 than Ada |
| L2 Cache | 增大 |
| 每SM寄存器 | 增加 |

## 代码优化详情

### 1. Makefile 优化

```makefile
# 默认使用 sm_100 (RTX 5090 Blackwell)
ARCH_FLAG := -arch=sm_100
```

- 默认编译目标改为 `sm_100`
- 保留手动覆盖选项 `ARCH=sm_XX`

### 2. V1: Naive (3 kernels)

**优化前:**
```cuda
int threads = 256;
```

**优化后 (RTX 5090):**
```cuda
int threads = 512;  // 2x for more SMs
```

**原因:** RTX 5090 有更多SM，增大block size提高occupancy

### 3. V2: Block Shared Memory

**优化前:**
```cuda
int threads = 256;
```

**优化后 (RTX 5090):**
```cuda
int threads = 512;  // 2x for better SM utilization
```

**原因:** 更大的block size更好地利用RTX 5090的共享内存带宽

### 4. V3: Warp-level Reduction

**优化前:**
```cuda
int threads = 128;  // 4 warps
```

**优化后 (RTX 5090):**
```cuda
int threads = 256;  // 8 warps for better parallelism
```

**原因:** 更多warps per block充分利用寄存器和warp调度器

### 5. V4: Vectorized (float4)

**优化前:**
```cuda
int threads = 128;
```

**优化后 (RTX 5090):**
```cuda
int threads = 256;  // 8 warps for GDDR7 bandwidth
```

**原因:** 256线程(8 warps)最大化GDDR7的128-bit访问效率

### 6. V5: Online Softmax

**优化前:**
```cuda
int threads = 128;
```

**优化后 (RTX 5090):**
```cuda
int threads = 256;  // 8 warps for optimal L2 cache usage
```

**原因:** 更大的L2 cache，256线程更好地利用cache line

### 7. V5+Vec: Ultimate

**优化前:**
```cuda
int threads = 128;
```

**优化后 (RTX 5090):**
```cuda
int threads = 256;  // 8 warps for ultimate bandwidth
```

**原因:** 结合Online算法和float4，256线程压榨GDDR7极限带宽

## 编译和运行

### 编译

```bash
# 默认编译为 sm_100 (RTX 5090)
make

# 显式指定 RTX 5090
make ARCH=sm_100

# 降级到 sm_80 测试 (如果没有 RTX 5090)
make ARCH=sm_80
```

### 运行

```bash
# 运行benchmark
./benchmark_all
```

## 预期性能

在 RTX 5090 上的预期带宽利用率:

| 版本 | 预期带宽利用率 | 说明 |
|------|---------------|------|
| V1 | ~5-10% | 内存瓶颈，多次读取 |
| V2 | ~15-25% | 融合kernel减少访问 |
| V3 | ~30-40% | Warp shuffle高效 |
| V4 | ~50-60% | float4向量化 |
| V5 | ~55-65% | Online减少读取 |
| V5+Vec | **~70-80%** | 最优组合 |

GDDR7 理论带宽 ~1792 GB/s，V5+Vec 版本预计可达 **1200-1400 GB/s**。

## 与其他GPU对比

| GPU | 架构 | 显存 | 理论带宽 | 最优版本预期 |
|-----|------|------|----------|--------------|
| RTX 5090 | Blackwell | GDDR7 | 1792 GB/s | ~1400 GB/s |
| RTX 4090 | Ada | GDDR6X | 1008 GB/s | ~800 GB/s |
| A100 | Ampere | HBM2e | 2039 GB/s | ~1600 GB/s |
| RTX 3090 | Ampere | GDDR6X | 936 GB/s | ~750 GB/s |

## 进一步优化建议

1. **使用异步拷贝 (TMA)** - Blackwell支持Tensor Memory Accelerator
2. **FP8/BF16支持** - RTX 5090原生支持FP8，可进一步加速
3. **PDL (Programmatic Dependent Launch)** - Blackwell特性，kernel间零开销启动
4. **更大的batch** - 充分利用RTX 5090的大显存(32GB)

## 参考文档

- [TUTORIAL.MD](docs/TUTORIAL.MD) - 优化教程
- [README.md](README.md) - 使用说明
- NVIDIA Blackwell Architecture Whitepaper

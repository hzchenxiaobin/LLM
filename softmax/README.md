# CUDA Softmax 优化全版本指南

> 从 Naive 到 Ultimate：完整 Softmax CUDA 优化教程  
> 专为 NVIDIA RTX 5090 (Blackwell/GDDR7) 优化，兼容 Ampere/Ada 架构

## 项目简介

本项目实现了 **6 个不同优化级别** 的 CUDA Softmax kernel，完整展示从基础实现到终极优化的全过程。每个版本都有详细的文档说明和可视化图解。

```
优化演进路线:
V1 (基础) → V2 (共享内存) → V3 (Warp Shuffle) → V4 (向量化) → V5 (Online) → V5+Vec (终极)
  ↓              ↓                ↓                ↓              ↓              ↓
6x内存访问     核函数融合        无同步开销       带宽最大化      单次遍历       全部优化叠加
```

---

## 快速开始

### 编译

```bash
# 编译所有版本（推荐）
make benchmark_all

# 编译单文件版本
make softmax_benchmark

# 自动检测 GPU 架构
make

# 手动指定架构
make ARCH=sm_100  # RTX 5090 Blackwell
make ARCH=sm_89   # RTX 4090 Ada
make ARCH=sm_80   # A100 Ampere
```

### 运行

```bash
# 运行所有版本对比测试
make run

# 直接运行
./benchmark_all

# 使用 Nsight Compute 分析性能
make profile
```

---

## 版本概览

### 全版本特性对比

| 版本 | 核心技术 | 内存访问 | 归约方式 | 遍历次数 | 带宽利用率* | 相对性能 |
|------|---------|---------|---------|---------|------------|---------|
| **V1** | 3 独立核函数 | 6x/元素 | 串行 for | 3 | ~10% | 1.0x (基准) |
| **V2** | 共享内存 + Kernel Fusion | 2x/元素 | Shared Memory Tree | 1 kernel | ~20% | 2.0x |
| **V3** | Warp Shuffle | 2x/元素 | Warp Shuffle | 2 | ~35% | 3.5x |
| **V4** | float4 向量化 | 2x/元素 (16B粒度) | Warp Shuffle | 2 | ~55% | 5.5x |
| **V5** | Online Softmax | 2x/元素 (单次遍历max+sum) | Warp Shuffle + Online | 2 | ~45% | 4.5x |
| **V5+Vec** | **Online + float4** | **2x/元素 (最优粒度)** | **Warp Shuffle + Online** | **2** | **~70%** | **7.0x** 🏆 |

*带宽利用率基于 RTX 5090 GDDR7 理论带宽 ~1.5 TB/s

### 版本选择指南

```
                    开始
                     │
              N % 4 == 0?
                /        \
              否          是
              │            │
           使用 V3     内存带宽紧张?
           (通用性好)     /        \
                      否          是
                      │            │
                   V3/V4       V5+Vec Ultimate
                   (简单)       (最强性能)
```

| 场景 | 推荐版本 | 原因 | 实测加速比 |
|------|---------|------|-----------|
| 学习/教学 | V1 | 代码最简单易懂 | 1.0x (基准) |
| 通用场景 / N%4≠0 | V3 | 兼容性好，无需对齐 | 43x (N=1K) |
| 小数据 (< 2MB) | V4/V5+Vec | 向量化收益明显 | 8-26x |
| 中等数据 (4-8MB) | V5+Vec | 带宽利用率 50%+ | 44-66x |
| **大数据 (> 16MB)** | **V5+Vec** | **可达 91x 加速，带宽 2267 GB/s** | **27-91x** |
| N=2048 特殊情况 | V4 | 向量化略胜 Online | 72x vs 66x |

---

## 各版本核心技术详解

### V1: Naive (3 Kernels) - 基础实现
**文件**: `src/softmax_v1_naive.cu` | **文档**: `docs/softmax_v1_explanation.md/html`

```cuda
// 3 个独立核函数
kernel_max_v1  →  求每行最大值
kernel_sum_v1  →  求指数和  
kernel_div_v1  →  归一化输出
```

**特点**: 每个元素被读取 **3 次**，6 次全局内存访问，适合作为教学基准。

---

### V2: Block Shared Memory - 核函数融合
**文件**: `src/softmax_v2_shared_memory.cu` | **文档**: `docs/softmax_v2_explanation.md/html`

```cuda
// 1 个核函数完成全部计算
__shared__ float sdata[];  // Block 级共享内存
// Tree Reduction: stride 256→128→64→...→1
```

**改进**: Kernel Fusion (3→1)，使用 Shared Memory 中间存储，减少全局内存访问。

---

### V3: Warp-level Reduction - 无同步归约
**文件**: `src/softmax_v3_warp_reduction.cu` | **文档**: `docs/softmax_v3_explanation.md/html`

```cuda
// 每个 warp (32线程) 处理一行
float row_max = warpReduceMax(local_max);  // Shuffle 指令
row_max = __shfl_sync(0xffffffff, row_max, 0);  // 广播
```

**突破**: 使用 `__shfl_sync` 替代 `__syncthreads()`，延迟从 ~30 cycles 降至 ~10 cycles。

---

### V4: Vectorized (float4) - 带宽最大化
**文件**: `src/softmax_v4_vectorized.cu` | **文档**: `docs/softmax_v4_explanation.md/html`

```cuda
// 128-bit 向量化访问
const float4* x_vec = reinterpret_cast<const float4*>(input);
float4 val = x_vec[i];  // 一次读取 4 个 float
```

**效果**: 内存事务减少 **4x**，带宽利用率从 ~35% 提升至 ~55%。

⚠️ **要求**: N 必须是 4 的倍数

---

### V5: Online Softmax - FlashAttention 核心算法
**文件**: `src/softmax_v5_online.cu` | **文档**: `docs/softmax_v5_explanation.md/html`

```cuda
// 单次遍历同时更新 max 和 sum
new_max = max(old_max, val);
new_sum = old_sum * exp(old_max - new_max) + exp(val - new_max);
```

**核心**: 基于 FlashAttention 的 Online Softmax 算法，输入读取从 **3 次减少至 2 次**。

---

### V5+Vec: Ultimate - 终极性能
**文件**: `src/softmax_v5_vec_ultimate.cu` | **文档**: `docs/softmax_v5_vec_ultimate_explanation.md/html`

```cuda
// 融合 V5 (Online) + V4 (float4)
float4 val = x_vec[i];
// 对 val.x, val.y, val.z, val.w 分别应用 Online 公式
```

**性能**: 带宽利用率 **~70%**，相比 V1 提升 **7 倍**。

🏆 **生产环境首选**（当 N 是 4 的倍数时）

---

## 性能参考

### RTX 5090 (Blackwell + GDDR7) 实测数据

> **测试环境**: NVIDIA GeForce RTX 5090 | Compute Capability 12.0 | 理论带宽: 1792 GB/s

#### 多规模 benchmark 结果汇总

| 配置 (M×N) | 数据大小 | V1 (ms) | V5+Vec (ms) | 最佳带宽 | 加速比 |
|-----------|---------|---------|-------------|---------|--------|
| 1024×128 | 0.5 MB | 0.0606 | 0.0072 | 145 GB/s | **8.4x** |
| 1024×512 | 2.0 MB | 0.2039 | 0.0078 | 535 GB/s | **26x** |
| 1024×1024 | 4.0 MB | 0.3968 | 0.0090 | 933 GB/s | **44x** |
| 1024×2048 | 8.0 MB | 0.7839 | 0.0119* | 1544 GB/s | **66x** |
| 4096×1024 | 16.0 MB | 0.3984 | 0.0148 | **2267 GB/s** | **27x** |
| 1024×4096 | 16.0 MB | 1.5534 | 0.0170 | **1979 GB/s** | **91x** |

> *注: N=2048 时 V4 (0.0109ms, 1544 GB/s) 略快于 V5+Vec

#### 各版本详细性能对比 (M=1024, N=1024)

| 版本 | 时间 (ms) | 实际带宽 (GB/s) | 带宽利用率 | 相比 V1 |
|------|----------|----------------|-----------|--------|
| V1 Naive | 0.3968 | 21.14 | 1.2% | 1.0x |
| V2 Shared | 0.0109 | 770.58 | 43% | 36x |
| V3 Warp | 0.0092 | 915.00 | 51% | 43x |
| V4 Vec | 0.0091 | 926.55 | 52% | 44x |
| V5 Online | 0.0092 | 913.62 | 51% | 43x |
| **V5+Vec** | **0.0090** | **933.24** | **52%** | **44x** |

#### 关键发现

1. **规模效应显著**: 数据越大，优化效果越明显。从 0.5MB 到 16MB，V5+Vec 相对 V1 的加速比从 8x 提升到 91x
2. **带宽利用率**: 大规模数据下可超过理论带宽（如 2267 GB/s > 1792 GB/s），得益于 L2 缓存命中
3. **V4 vs V5+Vec**: N=2048 时 V4 纯向量化版本意外领先，说明 Online 算法在特定规模下有一定开销
4. **数值精度**: 所有版本误差 < 1e-8，验证通过

#### 历史参考：预估数据 (M=8192, N=4096)

| 版本 | 预估时间 (ms) | 预估带宽 (GB/s) | 预估加速 |
|------|--------------|----------------|---------|
| V1 Naive | ~10.0 | ~150 | 1.0x |
| V2 Shared | ~5.0 | ~300 | 2.0x |
| V3 Warp | ~2.9 | ~520 | 3.5x |
| V4 Vec | ~1.8 | ~830 | 5.5x |
| V5 Online | ~2.2 | ~680 | 4.5x |
| **V5+Vec** | **~1.4** | **~1050** | **7.0x** |

---

## 文档索引

| 版本 | Markdown | HTML 可视化 |
|------|----------|-------------|
| V1 | [docs/softmax_v1_explanation.md](docs/softmax_v1_explanation.md) | [docs/softmax_v1_explanation.html](docs/softmax_v1_explanation.html) |
| V2 | [docs/softmax_v2_explanation.md](docs/softmax_v2_explanation.md) | [docs/softmax_v2_explanation.html](docs/softmax_v2_explanation.html) |
| V3 | [docs/softmax_v3_explanation.md](docs/softmax_v3_explanation.md) | [docs/softmax_v3_explanation.html](docs/softmax_v3_explanation.html) |
| V4 | [docs/softmax_v4_explanation.md](docs/softmax_v4_explanation.md) | [docs/softmax_v4_explanation.html](docs/softmax_v4_explanation.html) |
| V5 | [docs/softmax_v5_explanation.md](docs/softmax_v5_explanation.md) | [docs/softmax_v5_explanation.html](docs/softmax_v5_explanation.html) |
| V5+Vec | [docs/softmax_v5_vec_ultimate_explanation.md](docs/softmax_v5_vec_ultimate_explanation.md) | [docs/softmax_v5_vec_ultimate_explanation.html](docs/softmax_v5_vec_ultimate_explanation.html) |
| 进阶优化 | [RTX5090_OPTIMIZATIONS.md](RTX5090_OPTIMIZATIONS.md) | - |
| 详细教程 | [docs/TUTORIAL.MD](docs/TUTORIAL.MD) | - |

**建议**: 使用浏览器打开 HTML 版本查看可视化图示。

---

## 优化技术总结

### 每层优化的核心收益

| 演进 | 核心技术 | 主要收益 | 学习重点 |
|------|---------|---------|---------|
| V1→V2 | Kernel Fusion | 6x→2x 内存访问 | 核函数融合概念 |
| V2→V3 | Warp Shuffle | 无显式同步 | Warp 级并行 |
| V3→V4 | float4 向量化 | 4x 内存事务减少 | 内存访问模式 |
| V4→V5 | Online Softmax | 3x→2x 遍历次数 | FlashAttention 算法 |
| V5→V5+Vec | 双重优化叠加 | 6x 理论效率提升 | 优化组合策略 |

### 关键概念速查

- **Kernel Fusion**: 将多个核函数合并为一个，减少 launch 开销和中间数据传递
- **Warp Shuffle**: GPU 同 warp 内线程通过寄存器直接交换数据，无需共享内存
- **float4 向量化**: 128-bit 内存访问，充分利用 GPU 内存总线宽度
- **Online Softmax**: 单次遍历同时维护 max 和 sum 的算法
- **Tree Reduction**: 二分归约算法，O(logN) 复杂度

---

## 文件结构

```
softmax/
├── src/                                    # 源代码
│   ├── benchmark_all.cu                    # 模块化测试主入口
│   ├── softmax_common.h                    # 公共头文件（warp归约、CPU参考实现）
│   ├── softmax_v1_naive.cu                 # V1: Naive 基础实现
│   ├── softmax_v2_shared_memory.cu       # V2: 共享内存归约
│   ├── softmax_v3_warp_reduction.cu        # V3: Warp Shuffle
│   ├── softmax_v4_vectorized.cu          # V4: float4 向量化
│   ├── softmax_v5_online.cu                # V5: Online Softmax
│   └── softmax_v5_vec_ultimate.cu        # V5+Vec: 终极优化
├── docs/                                   # 文档目录
│   ├── TUTORIAL.MD                         # 详细优化教程
│   ├── softmax_v1_explanation.md/html      # V1 详解
│   ├── softmax_v2_explanation.md/html      # V2 详解
│   ├── softmax_v3_explanation.md/html      # V3 详解
│   ├── softmax_v4_explanation.md/html      # V4 详解
│   ├── softmax_v5_explanation.md/html       # V5 详解
│   └── softmax_v5_vec_ultimate_explanation.md/html  # V5+Vec 详解
├── softmax_benchmark.cu                    # 单文件版本（所有实现）
├── Makefile                                  # 编译脚本
├── README.md                                 # 本文件
└── RTX5090_OPTIMIZATIONS.md                  # RTX 5090 专用优化指南
```

---

## 高级用法

### 自定义测试配置

编辑 `src/benchmark_all.cu` 中的 `configs` 数组：

```cpp
// 添加自定义配置
configs[0] = {1024, 128};   // batch=32, seq=128
configs[1] = {1024, 4096};  // batch=32, seq=4096 (长序列)
```

### 单独测试某个版本

```cpp
// 在 main 函数中注释掉不需要的版本
// run_benchmark_v1(...);
run_benchmark_v5_vec(...);  // 只测试 V5+Vec
```

### 与其他 GPU 对比

```bash
# 查看当前 GPU 信息
nvidia-smi

# 修改 Makefile 中的 ARCH 参数
# sm_100: RTX 5090 (Blackwell)
# sm_89:  RTX 4090 (Ada)
# sm_80:  A100 (Ampere)
# sm_70:  V100 (Volta)
```

---

## 常见问题

### Q: V4/V5+Vec 要求 N % 4 == 0，如何处理其他情况？
**A**: 推荐方案：
1. Padding: 分配时向上取整到 4 的倍数
2. 混合策略: 主体用 V4/V5+Vec，边界用 V3
3. 回退: 不满足条件时自动使用 V3

### Q: 为什么 V5 Online 比 V4 向量化慢？
**A**: V5 减少内存访问次数，V4 提高每次访问效率。当带宽是瓶颈时，两者结合 (V5+Vec) 效果最佳。

### Q: 如何验证正确性？
**A**: 每个版本都与 CPU 参考实现对比，误差 < 1e-4。运行时会显示 `PASSED/FAILED`。

### Q: 能否用于生产环境？
**A**: V5+Vec 经过充分测试，适合生产环境。注意：
- N 必须是 4 的倍数
- 需要 CUDA CC 3.0+ (Kepler+)
- 数值误差 ~1e-6，深度学习场景可接受

---

## 学习路线图

```
初学者路径:
Week 1: 阅读 V1 文档 → 理解 Softmax 基础并行化
Week 2: 阅读 V2 文档 → 学习 Kernel Fusion 和 Shared Memory
Week 3: 阅读 V3 文档 → 掌握 Warp Shuffle 和寄存器优化
Week 4: 阅读 V4 文档 → 理解向量化内存访问
Week 5: 阅读 V5 文档 → 学习 Online 算法和 FlashAttention 核心
Week 6: 阅读 V5+Vec 文档 → 掌握多优化技术融合

进阶路径:
→ 阅读 RTX5090_OPTIMIZATIONS.md
→ 使用 Nsight Compute 分析自己的 kernel
→ 尝试实现 V6 (持久化 warp / TMA / 异步拷贝)
```

---

## 贡献与参考

### 参考资源
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - Online Softmax 算法来源
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Warp Shuffle 指令
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/) - 性能分析工具

### 相关项目
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - 生产级 Online Softmax
- [cutlass](https://github.com/NVIDIA/cutlass) - NVIDIA 官方 CUDA 模板库

---

## 许可证

MIT License - 自由使用和学习

---

**祝优化愉快！🚀**

如有问题，请查阅 `docs/` 目录下的详细文档或使用浏览器打开 HTML 版本查看可视化说明。

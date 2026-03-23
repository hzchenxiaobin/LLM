# Softmax Performance Benchmark for RTX 5090

CUDA Softmax 性能测试框架，**专为 NVIDIA RTX 5090 (Blackwell 架构) 优化**。

## RTX 5090 优化特性

| 参数 | 优化值 | 说明 |
|------|--------|------|
| 架构 | sm_100 | Blackwell |
| 线程数(V1/V2) | 512 | 充分利用更多SM |
| 线程数(V3-V5) | 256 | 8 warps/block |
| 内存 | GDDR7 | 理论带宽 1792 GB/s |
| 优化策略 | 更大block | 提高occupancy |

**两种代码组织方式：**
- **模块化版本**: 每个实现单独文件，便于单独测试和学习
- **单文件版本**: 所有实现在一个文件，便于整体理解

## 实现的版本

| 版本 | 文件名 | 名称 | 特点 |
|------|--------|------|------|
| V1 | `softmax_v1_naive.cu` | Naive (3 kernels) | 基础实现，3个独立kernel，6次全局内存访问 |
| V2 | `softmax_v2_shared_memory.cu` | Block Shared Memory | 使用 Shared Memory 进行 Block 级归约 |
| V3 | `softmax_v3_warp_reduction.cu` | Warp-level Reduction | 使用 Warp Shuffle 指令，无 Shared Memory |
| V4 | `softmax_v4_vectorized.cu` | Vectorized (float4) | 向量化访存，每次处理 128 bits |
| V5 | `softmax_v5_online.cu` | Online Softmax | 单次遍历同时更新 max 和 sum |
| V5+Vec | `softmax_v5_vec_ultimate.cu` | Ultimate | V5 + V4 结合，终极优化版本 |

**公共工具**: `softmax_common.h` - 错误检查、warp归约辅助函数、CPU参考实现

## 编译

### 模块化版本（推荐）
每个实现单独一个文件，适合学习和单独测试：

```bash
# 编译模块化版本
make benchmark_all

# 或编译所有
make all
```

### 单文件版本
所有实现在一个文件中：

```bash
# 编译单文件版本
make softmax_benchmark
```

### 指定 GPU 架构

```bash
# 自动检测 GPU 架构并编译 (默认 sm_100 for RTX 5090)
make

# 手动指定架构（RTX 5090 Blackwell）
make ARCH=sm_100

# 其他常见架构
# sm_90a: Blackwell/Ada (兼容模式)
# sm_89: RTX 4090 (Ada)
# sm_80: A100 (Ampere)
# sm_70: V100 (Volta)
```

## 运行

```bash
# 编译并运行模块化版本（默认）
make run
# 或
make run-all

# 编译并运行单文件版本
make run-mono

# 或直接运行已编译的程序
./benchmark_all
./softmax_benchmark
```

## 测试配置

默认测试以下 LLM 典型场景：

| M (rows) | N (cols) | 描述 |
|----------|----------|------|
| 1024 | 128 | batch=32, heads=32, seq=128 |
| 1024 | 512 | batch=32, heads=32, seq=512 |
| 1024 | 1024 | batch=32, heads=32, seq=1024 |
| 1024 | 2048 | batch=32, heads=32, seq=2048 |
| 4096 | 1024 | 大 batch 场景 |
| 1024 | 4096 | 超长序列 |

## 性能分析

使用 Nsight Compute 进行深入分析：

```bash
make profile

# 或手动运行
ncu -o report --metrics dram__bytes.sum,gpu__time_duration.avg ./softmax_benchmark
```

## 输出示例

```
===============================================
  Softmax Performance Benchmark
  Device: NVIDIA GeForce RTX 5090
  Compute Capability: 9.0
  Memory: 32.00 GB
  Theoretical Bandwidth: 1792.00 GB/s
===============================================

========================================
Configuration: M=1024, N=1024
========================================

Benchmarking V1: Naive (3 kernels)...
  Time: 1.2345 ms | Bandwidth: 6.78 GB/s | Max Error: 1.23e-05 | PASSED

Benchmarking V2: Block Shared Memory...
  Time: 0.5678 ms | Bandwidth: 14.73 GB/s | Max Error: 1.23e-05 | PASSED

...

--- Summary for M=1024, N=1024 ---
Version                                    Time(ms)     BW(GB/s)        Error
--------------------------------------------------------------------------------
V1: Naive (3 kernels)                         1.2345         6.78         1.23e-05 ✓
V2: Block Shared Memory                       0.5678        14.73         1.23e-05 ✓
...

Best performer: V5+Vec: Ultimate (Online + float4) (0.1234 ms)
```

## 代码结构

```
softmax/
├── src/                                  # 源代码目录
│   ├── benchmark_all.cu                  # 模块化版本主测试入口
│   ├── softmax_common.h                  # 公共头文件（工具函数、宏、CPU参考实现）
│   ├── softmax_v1_naive.cu               # V1: Naive (3 kernels)
│   ├── softmax_v2_shared_memory.cu     # V2: Block Shared Memory
│   ├── softmax_v3_warp_reduction.cu      # V3: Warp-level Reduction
│   ├── softmax_v4_vectorized.cu        # V4: Vectorized (float4)
│   ├── softmax_v5_online.cu              # V5: Online Softmax
│   └── softmax_v5_vec_ultimate.cu      # V5+Vec: Ultimate (Online + float4)
├── softmax_benchmark.cu                  # 单文件版本（所有实现在一个文件）
├── Makefile                              # 编译脚本
├── README.md                             # 本文件
└── docs/
    └── TUTORIAL.MD                       # 优化教程文档
```

## 优化要点总结

1. **V1 → V2**: Kernel Fusion，减少全局内存访问次数（6次 → 2次）
2. **V2 → V3**: 用 Warp Shuffle 替代 Shared Memory 同步，减少同步开销
3. **V3 → V4**: 向量化访存，提升带宽利用率（4x 内存指令减少）
4. **V4 → V5**: Online Softmax，单次遍历数据（理论最优）
5. **V5 → V5+Vec**: 向量化 + Online，结合所有优化

## 注意事项

- V4 和 V5+Vec 要求 N 能被 4 整除（float4 对齐）
- RTX 5090 使用 Blackwell 架构 (sm_90a)
- 实际带宽利用率取决于 GPU 型号和驱动版本

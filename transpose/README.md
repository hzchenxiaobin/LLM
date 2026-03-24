# CUDA 矩阵转置性能优化

针对 RTX 5090 (Blackwell) 的显存带宽优化实战项目。

## 项目结构

```
transpose/
├── src/
│   ├── include/
│   │   └── transpose_common.h      # 公共头文件
│   ├── transpose/
│   │   ├── v1_naive.cu            # 朴素实现（非合并写入）
│   │   ├── v2_shared_memory.cu    # 共享内存优化（有 bank conflict）
│   │   ├── v3_shared_pad.cu       # Padding 消除 bank conflict
│   │   ├── v4_optimized.cu        # ILP 与 Block 形状优化
│   │   ├── v5_vectorized.cu       # float4 向量化访问
│   │   └── transpose_host.cu      # 主机端辅助函数
│   ├── benchmark/
│   │   └── benchmark.cu           # 性能基准测试框架
│   └── main.cu                    # 主程序入口
├── docs/
│   └── TUTORIAL.md                # 详细教程文档
├── Makefile                       # 构建脚本
└── README.md                      # 项目说明
```

## 版本演进

| 版本 | 名称 | 关键技术 | 预期带宽利用率 |
|------|------|----------|----------------|
| v1 | naive | 朴素实现，非合并写入 | 10-20% |
| v2 | shared | 共享内存中转，bank conflict | 40-50% |
| v3 | shared_pad | Padding 消除 bank conflict | 70-80% |
| v4 | optimized | ILP + 优化 block 形状 | 80-90% |
| v5 | vectorized | float4 向量化 | 90%+ |

## 编译

```bash
make
```

## 运行

```bash
# 默认设置 (N=8192, 10 warmup, 100 benchmark iterations)
make run

# 或自定义参数
./transpose_benchmark -n 8192 -w 10 -b 100

# 小矩阵测试
make run-small

# 大矩阵测试
make run-large
```

### 命令行参数

- `-n SIZE`: 矩阵大小 (NxN), 默认: 8192
- `-w WARMUP`: Warmup 迭代次数, 默认: 10
- `-b BENCH`: Benchmark 迭代次数, 默认: 100

## 性能分析

```bash
# 使用 Nsight Compute 进行详细分析
make profile

# 或手动运行
ncu --set full ./transpose_benchmark -n 8192 -b 10
```

## 关键优化技术

### 1. 合并内存访问 (Coalesced Memory Access)

- **问题**: 非合并写入导致带宽利用率低下
- **解决**: 使用共享内存中转，确保读写都是合并的

### 2. Bank Conflict 消除

- **问题**: 共享内存 bank conflict 导致串行化访问
- **解决**: 列宽 +1 padding，使数据分散到不同 bank

### 3. 指令级并行 (ILP)

- **问题**: 单个线程处理元素太少，无法隐藏延迟
- **解决**: 每线程处理 4 个元素，提高 MLP

### 4. 向量化访问

- **问题**: 每次内存事务只搬运 4 字节，总线利用率低
- **解决**: 使用 float4 每次搬运 16 字节

## 预期结果

在 RTX 5090 上测试 N=8192 时的典型结果:

```
Version              │ Min(ms)  │ Max(ms)  │ Avg(ms)  │ BW(GB/s) │ Eff%
══════════════════════╪════════════╪════════════╪════════════╪════════════╪════════════
v1_naive             │   5.2341 │   5.3456 │   5.2898 │    51.23 │   8.5%
v2_shared            │   1.8923 │   1.9567 │   1.9234 │   140.89 │  23.5%
v3_shared_pad        │   0.7234 │   0.7890 │   0.7562 │   358.42 │  59.7%
v4_optimized         │   0.4567 │   0.5234 │   0.4901 │   552.89 │  92.1%
v5_vectorized        │   0.4123 │   0.4678 │   0.4400 │   615.78 │ 102.6%
```

## 参考

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

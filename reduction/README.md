# CUDA Reduction 性能优化项目

本项目包含 CUDA Reduction（归约）算子的多个优化版本实现，从朴素版本到向量化访存，完整展示了 GPU 性能优化的全过程。

## 项目结构

```
reduction/
├── README.md              # 本文件
├── TUTORIAL.md            # 详细优化教程（推荐阅读）
├── Makefile               # 编译脚本
├── requirements.txt       # Python 依赖
├── include/               # 头文件
├── kernels/               # CUDA kernel 实现
├── src/                   # 主程序
├── python/                # Python 脚本
└── scripts/               # Shell 脚本
```

## 快速开始

```bash
# 编译
make

# 运行测试
make test

# 完整测试
make run
```

## 优化版本

| 版本 | 文件名 | 优化内容 |
|------|--------|----------|
| v1 | reduce_interleaved.cu | 朴素版本 (Warp Divergence) |
| v2 | reduce_strided.cu | 解决分支发散 |
| v3 | reduce_sequential.cu | 解决 Bank Conflict |
| v4 | reduce_first_add.cu | 加载时相加 |
| v5 | reduce_warp_shuffle.cu | Warp Shuffle |
| v6 | reduce_vectorized.cu | 向量化访存 |
| v7 | reduce_cub.cu | NVIDIA CUB 库 (基准) |

## 详细教程

查看 [TUTORIAL.md](TUTORIAL.md) 获取完整的性能优化教程，包含每个版本的详细解释和优化思路。

## 性能基准

所有版本都以 NVIDIA CUB 库的性能作为基准进行对比测试。

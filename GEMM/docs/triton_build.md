# Triton SGEMM 使用说明

## 什么是 Triton？

**Triton** 是 OpenAI 开发的 Python DSL（领域特定语言），用于编写高效的 GPU kernel。

相比 CUDA C++，Triton 的优势：
- **更简洁**：用 Python 编写 GPU kernel，代码量减少 10x
- **自动优化**：自动处理 tiling、shared memory、指令级并行
- ** tile-based 编程模型**：每个 kernel 实例处理一个数据块，避免线程级编程复杂性
- **与 PyTorch 无缝集成**：直接在 PyTorch 张量上操作

## 安装依赖

```bash
# 安装 Triton 和 PyTorch
pip install triton torch

# 或者使用 conda
conda install pytorch triton -c pytorch
```

注意：Triton 需要 NVIDIA GPU（Compute Capability 7.0+）和 CUDA 11.4+。

## 快速开始

### 1. 运行简单测试

```bash
cd GEMM/triton_kernels
python sgemm_triton.py
```

### 2. 运行性能基准测试

```bash
python benchmark_triton.py
```

这将对比 Triton 实现与 PyTorch (cuBLAS) 的性能。

## Triton SGEMM 核心代码解析

### Kernel 定义

```python
@triton.jit
def sgemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
```

关键概念：
- `@triton.jit`: JIT 编译器装饰器，将 Python 函数编译为 GPU kernel
- `tl.constexpr`: 编译时常量（用于 tile 大小等静态参数）
- `tl.program_id`: 获取当前 thread block 的坐标

### Grouped Ordering (L2 Cache 优化)

```python
# 计算当前 block 坐标
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

Grouped ordering 是 Triton SGEMM 的关键优化：
- 默认的 row-major ordering 会导致 L2 cache miss
- Grouped ordering 让相邻的 blocks 在 M 维度上连续，提高 A 矩阵的 L2 cache 命中率

### 指针计算与内存访问

```python
# 计算当前 block 的数据偏移
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offs_k = tl.arange(0, BLOCK_SIZE_K)

# 创建指针块（向量化加载）
a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

- `tl.arange`: 创建连续的索引序列
- `[:, None]`, `[None, :]`: broadcasting 操作，创建 2D 索引网格
- 向量化内存访问：一次加载多个元素，提高带宽利用率

### Main Loop

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # 加载 A/B tiles（带边界 mask）
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

    # 矩阵乘累加（自动使用 Tensor Core 如果可用）
    accumulator += tl.dot(a, b)

    # 更新指针
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

- `tl.dot`: 矩阵乘法（自动优化为 MMA 指令或 FMA）
- `mask`: 处理边界情况
- 自动 shared memory tiling：Triton 自动管理 shared memory，无需手动处理

### Epilogue

```python
# 应用 alpha
accumulator = accumulator * alpha

# 处理 beta * C
if beta != 0.0:
    c = tl.load(c_ptrs, mask=c_mask, other=0.0)
    accumulator = accumulator + beta * c

# 存储结果
tl.store(c_ptrs, accumulator, mask=c_mask)
```

## 自动调优 (Autotuning)

Triton 提供了自动调优功能：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, ...}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, ...}, num_stages=4, num_warps=4),
        # ... 更多配置
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def sgemm_kernel_autotuned(...):
```

- `configs`: 候选配置列表（tile 大小、流水线 stage 数、warp 数）
- `key`: 用于缓存调优结果的参数（M, N, K）
- Triton 会自动运行所有配置，选择最快的

## 性能预期

在 NVIDIA A100/RTX 4090 上的典型性能：

| 矩阵大小 | cuBLAS (TFLOPS) | Triton (TFLOPS) | 效率 |
|---------|-----------------|-----------------|------|
| 1024³   | ~100            | ~80-90          | 80-90% |
| 2048³   | ~150            | ~120-140        | 80-93% |
| 4096³   | ~160            | ~140-155        | 88-97% |
| 8192³   | ~165            | ~150-160        | 91-97% |

Triton 通常能达到 cuBLAS 80-95% 的性能，代码量却少得多。

## 进阶优化

### 1. 使用 Tensor Core (FP16/BF16)

```python
# 将数据转换为 FP16
A = A.to(torch.float16)
B = B.to(torch.float16)

# Triton 会自动使用 Tensor Core
accumulator = tl.dot(a, b)  # 内部使用 MMA 指令
```

### 2. 双缓冲 (软件流水线)

Triton 自动处理双缓冲，通过 `num_stages` 参数控制：
- `num_stages=3`: 软件流水线，隐藏内存延迟
- `num_stages=4`: 更深流水线，适合更大的 tile

### 3. 混合精度训练

```python
# FP16 输入，FP32 累加
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
accumulator += tl.dot(a.to(tl.float16), b.to(tl.float16))
```

## 对比：CUDA vs Triton

| 特性 | CUDA C++ | Triton Python |
|-----|----------|---------------|
| 代码量 | ~300 行 | ~50 行 |
| 手动管理 shared memory | 是 | 否（自动） |
| 手动处理 bank conflict | 是 | 否（自动） |
| 手动处理边界 | 是 | 是（mask） |
| 自动调优 | 否 | 是 |
| 与 PyTorch 集成 | 需要 C++ 扩展 | 无缝 |
| 调试难度 | 高 | 低 |

## 参考资源

1. [Triton 官方教程 - Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
2. [Triton GitHub](https://github.com/openai/triton)
3. [Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html)
4. [OpenAI Triton Paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet.pdf)

## 故障排除

### 问题：Triton 安装失败

**解决**：确保 CUDA 版本兼容
```bash
# 检查 CUDA 版本
nvcc --version

# Triton 2.1+ 需要 CUDA 11.4+
# 如果 CUDA 版本太低，升级 CUDA 或使用旧版 Triton
pip install triton==2.0.0
```

### 问题：运行时 CUDA error

**解决**：检查 GPU Compute Capability
```python
import torch
print(torch.cuda.get_device_capability())  # 需要 >= (7, 0)
```

### 问题：性能不如预期

**解决**：尝试调整 tile 大小或使用 autotune
```python
# 对于小矩阵，使用更小的 tile
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32

# 对于大矩阵，使用更大的 tile
BLOCK_SIZE_M = 256
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 64
```

# CUDA Scan (前缀和) 性能优化实现

基于 TUTORIAL.md 文档实现的前缀和（Scan）算法各版本，包含完整的性能测试框架。

## 项目结构

```
scan/
├── include/scan/
│   └── scan.h                    # 公共头文件
├── src/scan/
│   ├── v1_hillis_steele.cu       # Hillis-Steele 朴素扫描
│   ├── v2_blelloch.cu            # Blelloch 工作高效扫描
│   ├── v3_bank_free.cu           # 消除 Bank Conflicts
│   ├── v4_warp_primitive.cu      # Warp 原语优化
│   └── benchmark.cu              # 性能测试框架
├── docs/
│   └── TUTORIAL.md               # 算法原理文档
├── CMakeLists.txt                # CMake 配置
├── Makefile                      # Makefile 配置
└── README.md                     # 项目说明
```

## 各版本实现说明

### V1: Hillis-Steele 朴素并行扫描

- **算法**: 基于步骤的并行计算，每个线程将当前元素与距离它 $2^{d-1}$ 的元素相加
- **复杂度**: $O(N \log N)$ 工作量，不是最优但实现简单
- **特点**: 使用双缓冲技术避免读写冲突

```cuda
// 核心思想：offset 每次翻倍 (1, 2, 4, 8...)
for (int offset = 1; offset < n; offset *= 2) {
    if (thid >= offset) {
        temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
    }
}
```

### V2: Blelloch 工作高效扫描

- **算法**: 树状归约，分为 Up-Sweep 和 Down-Sweep 两个阶段
- **复杂度**: $O(N)$ 工作量，理论最优
- **特点**: 通过两次遍历实现排他型前缀和

**Up-Sweep 阶段**:
```
Level 3: [A+B, C+D, E+F, G+H]
Level 2: [(A+B)+(C+D), (E+F)+(G+H)]
Level 1: [总和]
```

**Down-Sweep 阶段**:
- 根节点设为 0
- 向下传播并交换相加

### V3: 消除 Bank Conflicts

- **问题**: V2 中访问步长为 $2^n$ 会导致 Shared Memory Bank 冲突
- **解决方案**: 每 32 个元素插入一个 padding

```cuda
#define CONFLICT_FREE_OFFSET(n) ((n) >> 5)  // n / 32

// 访问索引时加上偏移量
int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
temp[ai + bankOffsetA] = g_idata[ai];
```

### V4: Warp-Level Primitives

- **核心**: 使用 `__shfl_up_sync` 在寄存器级别完成 32 个元素的扫描
- **优势**: 完全避免 Shared Memory，利用 Warp 内部高带宽寄存器通信
- **实现**: 三层结构
  1. **Warp Scan**: 32 线程内寄存器级扫描
  2. **Block Scan**: 使用少量 Shared Memory 存储 warp 总和
  3. **前缀和传播**: 将 block 前缀加到各线程结果

```cuda
// Warp 级别的包含型扫描
__device__ __forceinline__ T warp_scan_inclusive(T val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        T n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}
```

## 编译与运行

### 使用 Makefile (推荐)

```bash
# 编译项目
make

# 运行基准测试
make run

# 或手动运行
./build/benchmark

# 清理构建
make clean

# 调试构建（包含调试信息）
make debug
```

### 使用 CMake

```bash
mkdir build && cd build
cmake ..
make
./benchmark
```

## 性能测试框架

测试框架提供以下功能：

1. **正确性验证**: 与 CPU 串行实现对比结果
2. **性能计时**: 使用 CUDA Event 精确测量 GPU 时间
3. **带宽计算**: 自动计算有效显存带宽
4. **多规模测试**: 测试 256 到 4096 元素的性能

### 输出示例

```
================================================================================
                   CUDA Scan (Prefix Sum) Performance Benchmark
================================================================================

Device: NVIDIA GeForce RTX 4090
Compute Capability: 8.9
Shared Memory per Block: 48 KB
Max Threads per Block: 1024

Running benchmarks...
--------------------------------------------------------------------------------

Test Size: N = 256 elements
--------------------------------------------------------------------------------
  V1: Hillis-Steele (Naive)            N=   256  Time=  0.015 ms  BW= 68.23 GB/s  [PASS]
  V2: Blelloch (Work-Efficient)        N=   256  Time=  0.012 ms  BW= 85.33 GB/s  [PASS]
  V3: Bank-Free (No Conflicts)         N=   256  Time=  0.008 ms  BW=128.00 GB/s  [PASS]
  V4: Warp Primitives                  N=   256  Time=  0.005 ms  BW=204.80 GB/s  [PASS]
```

## 性能预期

在 RTX 4090/5090 等现代 GPU 上，各版本的相对性能大致为：

| 版本 | 工作量 | 共享内存使用 | Bank 冲突 | 相对速度 |
|------|--------|-------------|-----------|----------|
| V1 Hillis-Steele | $O(N\log N)$ | 高 | 有 | 1x (基准) |
| V2 Blelloch | $O(N)$ | 中 | 有 | 2-3x |
| V3 Bank-Free | $O(N)$ | 中 | 无 | 3-4x |
| V4 Warp Primitives | $O(N)$ | 低 | N/A | 5-10x |

## 优化建议

1. **学习路径**: 建议按 V2 → V3 → V4 顺序学习，逐步理解优化技巧
2. **生产环境**: 直接使用 **NVIDIA CUB 库** (`cub::DeviceScan::ExclusiveSum`)
3. **大规模数据**: 需要实现 Decoupled Look-back 算法处理多 Block 情况
4. **数据类型**: 可使用向量化加载 (`float4`) 进一步提升带宽利用率

## 注意事项

1. 当前实现假设数据长度是 2 的幂次
2. V1-V3 限制在单个 Block 内（最大 1024 元素）
3. V4 支持多 Block，但简化了 Block 间前缀和传播
4. 完整的多 Block 实现应参考 CUB 库的 Decoupled Look-back 算法

## 参考资源

- [NVIDIA CUB Library](https://github.com/NVIDIA/cub)
- [CUDA Reduction Tutorial](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)
- [Parallel Prefix Sum with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

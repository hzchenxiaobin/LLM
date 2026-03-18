# RTX 5090 GEMM 优化硬件约束详解

本文档详细分析 NVIDIA GeForce RTX 5090 的硬件架构约束，以及如何针对这些约束优化 GEMM kernel。

---

## 目录
1. [RTX 5090 硬件规格概览](#1-rtx-5090-硬件规格概览)
2. [寄存器限制与优化](#2-寄存器限制与优化)
3. [共享内存容量与限制](#3-共享内存容量与限制)
4. [Bank Conflict 详解](#4-bank-conflict-详解)
5. [理论性能峰值分析](#5-理论性能峰值分析)
6. [GEMM 优化策略总结](#6-gemm-优化策略总结)

---

## 1. RTX 5090 硬件规格概览

### 1.1 核心硬件参数

| 参数 | RTX 5090 数值 | 说明 |
|------|--------------|------|
| **架构** | Blackwell (GB202) | NVIDIA 最新消费级架构 |
| **制程** | 5nm | 台积电定制工艺 |
| **CUDA Cores** | 21,760 | FP32 计算单元 |
| **Tensor Cores** | 680 (第 5 代) | 支持 FP16/BF16/FP8/FP4 |
| **RT Cores** | 170 | 光追单元 |
| **显存** | 32 GB GDDR7 | 28 Gbps 等效频率 |
| **显存位宽** | 512-bit | - |
| **显存带宽** | **1,792 GB/s** | - |
| **峰值算力 (FP32)** | **104.88 TFLOPS** | CUDA Core |
| **峰值算力 (FP16 Tensor)** | **835.3 TFLOPS** | Tensor Core with Accumulation |
| **TGP** | 575W | 总功耗 |

### 1.2 SM (Streaming Multiprocessor) 架构

| SM 参数 | 数值 | 说明 |
|---------|------|------|
| **SM 数量** | 170 | - |
| **每 SM CUDA Core** | 128 | 相比 Ada 架构无变化 |
| **每 SM Tensor Core** | 4 | 第 5 代 Tensor Core |
| **每 SM 寄存器文件** | **256 KB** | 约 65,536 个 32-bit 寄存器 |
| **每 SM 共享内存** | **164 KB** (可配置) | 128 KB 默认 + 扩展 |
| **每 SM L1 缓存** | 与共享内存共享 | 可动态配置 |
| **Warp 调度器** | 4 个 | 每周期可发射 4 条 warp 指令 |

### 1.3 计算能力 (Compute Capability)

```
RTX 5090: Compute Capability 10.0 (Blackwell)
```

**关键特性**：
- 支持 FP8 和 FP4 Tensor Core 运算
- 增强的异步拷贝指令 (`cp.async`)
- 改进的 Warp Group 同步原语
- 更大的寄存器文件和共享内存

---

## 2. 寄存器限制与优化

### 2.1 寄存器文件架构

#### 寄存器文件大小
```
每 SM: 256 KB 寄存器文件
      = 256 × 1024 × 8 bits
      = 65,536 个 32-bit 寄存器
```

#### 寄存器分配粒度
```
每 Warp (32 线程): 可分配最多 255 个寄存器/线程
每线程最大: 255 个 32-bit 寄存器
```

**关键公式**：
```
每 SM 最大并发 Warp 数 = 寄存器文件大小 / (每线程寄存器数 × 32线程 × 32位)

示例：
- 如果每线程使用 128 个寄存器
- 每 Warp 需要: 128 × 32 × 4 bytes = 16 KB
- 每 SM 可同时运行: 256 KB / 16 KB = 16 个 Warp
```

### 2.2 Occupancy 与寄存器使用

| 每线程寄存器数 | 每 Warp 占用 | 每 SM 最大 Warp | Occupancy | 备注 |
|--------------|------------|----------------|----------|------|
| **32** | 4 KB | 64 | 100% | 理论最大 |
| **64** | 8 KB | 32 | 50% | 通常足够 |
| **96** | 12 KB | 21 | 33% | 中等 |
| **128** | 16 KB | 16 | 25% | 常用选择 |
| **160** | 20 KB | 12 | 19% | 占用较高 |
| **192** | 24 KB | 10 | 16% | 接近限制 |
| **255** | ~32 KB | 8 | 12.5% | 最小值 |

### 2.3 GEMM 中的寄存器使用分析

#### sgemm_register.cu 的寄存器使用

```cuda
float accum[TM][TN];    // accum[8][8]  = 64 个寄存器
float frag_a[TM];        // frag_a[8]    = 8 个寄存器
float frag_b[TN];        // frag_b[8]    = 8 个寄存器
// 其他临时变量:         ~10 个寄存器
// 总计:                 ~90 个寄存器/线程
```

**计算验证**：
```
每线程: 90 个寄存器
每 Warp: 90 × 32 = 2,880 个寄存器 = ~11.25 KB
每 Block (256 线程): 90 × 256 = 23,040 个寄存器
每 Block Warp 数: 256 / 32 = 8 Warps
每 SM 可同时运行: 256 KB / 11.25 KB ≈ 22 Warps
考虑 Block 限制: 22 / 8 = 2.75 个 Block
```

**结论**：
- 当前实现每线程使用约 90 个寄存器
- Occupancy 约 33%，仍有优化空间
- 可以尝试增加到 128 寄存器以获得更好的 ILP

### 2.4 寄存器优化策略

#### 策略 1: 调整 TM/TN 大小
```cuda
// 当前配置
#define TM 8
#define TN 8
// accum[8][8] = 64 个寄存器

// 增加计算强度
#define TM 16
#define TN 8
// accum[16][8] = 128 个寄存器
// 每线程计算量翻倍，但占用率下降
```

#### 策略 2: 利用寄存器缓存更多数据
```cuda
// 原始: 每 k 迭代加载一次
float frag_a[TM];
#pragma unroll
for (int i = 0; i < TM; ++i) 
    frag_a[i] = sA[...][k];

// 优化: 预加载多个 k 值到寄存器
float frag_a[2][TM];
// 使用双缓冲重叠计算和加载
```

#### 策略 3: 避免寄存器 Spilling
```cuda
// 坏: 数组太大会溢出到本地内存
float large_array[256];  // 可能触发 spilling

// 好: 控制数组大小在寄存器限制内
float small_array[128];  // 通常安全
```

**寄存器 Spilling 的危害**：
- 寄存器数据溢出到 L1/L2 缓存甚至全局内存
- 性能下降 10-100 倍
- 可通过 `nvcc -Xptxas -v` 查看 spilling 情况

---

## 3. 共享内存容量与限制

### 3.1 共享内存架构

#### 共享内存大小
```
RTX 5090 每 SM: 164 KB 共享内存 (可配置)

分配方式：
- 静态分配: __shared__ float arr[N];
- 动态分配: 通过 kernel 启动参数 <<<grid, block, smem_size>>>
```

#### 共享内存分区
```
164 KB 共享内存可按以下方式配置：
- 128 KB Shared Memory + 36 KB L1
- 164 KB Shared Memory + 0 KB L1 (最大化共享内存)
- 96 KB Shared Memory + 68 KB L1 (平衡配置)
```

### 3.2 Bank 架构

#### Bank 数量
```
共享内存组织为 32 个 Bank
每个 Bank 宽度: 4 bytes (32-bit)
总带宽: 32 banks × 4 bytes × clock = 极高带宽 (~10+ TB/s)
```

**地址映射**：
```
Bank ID = (地址 / 4) % 32

示例：
地址 0x00: bank 0
地址 0x04: bank 1
...
地址 0x7C: bank 31
地址 0x80: bank 0 (循环)
```

### 3.3 sgemm_register.cu 的共享内存使用

```cuda
__shared__ float sA[BM][BK];  // sA[128][8]  = 4 KB
__shared__ float sB[BK][BN];  // sB[8][128]  = 4 KB
// 总计: 8 KB / Block
```

**分析**：
```
每 Block 使用 8 KB 共享内存
每 SM 最大 Block 数: 164 KB / 8 KB ≈ 20 个 Block
受限于 Warp/线程限制，实际并行度更低
```

### 3.4 共享内存容量瓶颈

| Block 配置 | 共享内存使用 | 每 SM 最大 Block | 实际限制因素 |
|-----------|-------------|----------------|-------------|
| **当前 (8 KB)** | sA[128][8] + sB[8][128] | 20 | 线程数 |
| **增大 BK=16** | 16 KB | 10 | 共享内存 |
| **增大 BK=32** | 32 KB | 5 | 共享内存 |
| **BM=256, BN=256** | sA[256][8] = 8KB, sB[8][256] = 8KB = 16 KB | 10 | 共享内存 |

### 3.5 共享内存优化策略

#### 策略 1: 调整 BK 平衡计算和内存
```cuda
// 当前: BK=8, 计算/内存比 = 4.0
// 可以尝试: BK=16
#define BK 16
// sA[128][16] = 8 KB, sB[16][128] = 8 KB, 总计 16 KB
// 每迭代计算 16×16 = 256 FMA
// 共享内存读取: 32 次
// 计算/内存比: 256/32 = 8.0 (提升 2×)
```

#### 策略 2: 动态共享内存分配
```cuda
// 使用动态分配适应不同配置
extern __shared__ float smem[];
float* sA = smem;
float* sB = &smem[BM * BK];

// 启动时指定大小
sgemm_kernel<<<grid, block, (BM*BK + BK*BN)*sizeof(float)>>>(...);
```

#### 策略 3: 共享内存数据重用
```cuda
// 原始: 每个线程单独加载
sA[ty][tx] = A[...];

// 优化: 协作加载，每个线程加载多个元素
// 256 线程加载 1024 个元素，每线程 4 个
```

---

## 4. Bank Conflict 详解

### 4.1 什么是 Bank Conflict？

**定义**：当多个线程同时访问同一个 Bank 的不同地址时，访问会被串行化，导致性能下降。

**理想情况**（无 Conflict）：
```
32 个线程访问 32 个不同 Banks
→ 单次内存事务完成
→ 带宽 = 32 × 4 bytes × clock
```

**Conflict 情况**：
```
32 个线程访问 8 个 Banks（每个 Bank 4 个线程）
→ 需要 4 次内存事务
→ 带宽下降为 1/4
```

### 4.2 Bank Conflict 模式分析

#### sgemm_shared.cu 的 Conflict 分析

```cuda
__shared__ float sA[32][32];  // 行优先存储
__shared__ float sB[32][32];  // 行优先存储

// 访问模式:
// sA[ty][k] - 行访问，连续地址，无 Conflict ✓
// sB[k][tx] - 列访问，stride-32，32× Conflict ✗
```

**详细分析**：
```
假设 warp 内线程 (tx=0..31, ty=0) 同时执行:

sB[k][tx] 的地址:
- 线程 0: &sB[k][0] = base + k*32 + 0
- 线程 1: &sB[k][1] = base + k*32 + 1
- ...
- 线程 31: &sB[k][31] = base + k*32 + 31

Bank 计算: (地址/4) % 32
- 线程 0: (k*8 + 0) % 32
- 线程 1: (k*8 + 1) % 32
- 所有线程访问不同 Banks，看起来无 Conflict?

实际 Conflict:
当 sB 是 float sB[32][32] 时，实际存储是连续的:
&sB[k][tx] = base + (k*32 + tx)*4

对于 warp 中的 32 线程:
- 线程 tx 访问地址 offset = (k*32 + tx) * 4
- Bank = (k*32 + tx) % 32 = tx % 32

看起来是每个线程访问不同 Bank，但实际编译器可能优化为 vectorized load，
导致某些情况下的 Conflict。
```

#### sgemm_register.cu 的 Conflict 分析

```cuda
__shared__ float sA[128][8];   // 行优先
__shared__ float sB[8][128];   // 行优先

// 访问模式:
// sA[ty*TM + i][k] - 行访问，无 Conflict ✓
// sB[k][tx*TN + j] - 行访问，无 Conflict ✓
```

**优势**：
```
因为 BK=8 很小，sB 的形状是 [8][128]，
访问 sB[k][tx*8 + j] 时，
对于不同的 k，地址间隔是 128*4 = 512 bytes = 128 banks
对于不同的 tx，地址间隔是 8*4 = 32 bytes = 8 banks

因为 warp 内线程的 tx 连续，
所以 sB[k][tx*8 + j] 访问的是连续的 8-bank 范围，
不会与相邻线程产生 Conflict。
```

### 4.3 Bank Conflict 量化

#### Shared Kernel 的 Conflict 代价
```
sB[k][tx] 访问模式:
- 理论带宽: ~10 TB/s
- 实际带宽: ~2.5 TB/s (估计，4-way conflict)
- 性能损失: ~75%
```

#### Register Kernel 的改进
```
sB[k][tx*8 + j] 访问模式:
- 理论带宽: ~10 TB/s
- 实际带宽: ~8-10 TB/s (无 Conflict)
- 性能提升: 3-4×
```

### 4.4 消除 Bank Conflict 的策略

#### 策略 1: Padding（填充）
```cuda
// 原始: 有 Conflict
__shared__ float sB[BK][BN];  // sB[8][128]

// 优化: Padding 一列
__shared__ float sB[BK][BN + 1];  // sB[8][129]
// 现在 sB[k][x] 和 sB[k+1][x] 不在同一 bank
```

#### 策略 2: 转置存储
```cuda
// 原始: sB 行优先，访问列有 Conflict
__shared__ float sB[BK][BN];  // 行优先

// 优化: sB 列优先存储
__shared__ float sB_t[BN][BK];  // 转置存储
// 访问时: sB_t[tx*8+j][k]
```

#### 策略 3: Swizzling（地址混排）
```cuda
// 高级技巧：自定义地址映射
// 例如 NVIDIA CUTLASS 中的 swizzle 模式
// 将 sB[i][j] 实际存储在 bank (i ^ j) % 32
// 确保相邻线程访问不同 banks
```

#### 策略 4: 向量化加载
```cuda
// 使用 float4 加载 4 个 float
float4 data = reinterpret_cast<float4*>(sB[k])[tx];
// 一个线程加载 16 bytes，天然对齐，减少 conflict
```

---

## 5. 理论性能峰值分析

### 5.1 CUDA Core 峰值

#### FP32 峰值计算
```
公式: Peak FLOPS = CUDA Core 数 × 2 FLOPs/clock × Boost Clock

RTX 5090:
- CUDA Cores: 21,760
- Boost Clock: ~2.41 GHz
- FP32 FMA: 2 FLOPs/操作 (乘 + 加)

Peak FP32 = 21,760 × 2 × 2.41e9
          = 104.88 TFLOPS
```

#### 实际可达到的 CUDA Core 性能
```
理论限制:
- 内存带宽限制: 1,792 GB/s
- Ridge Point: 104.88 / 1.792 = 58.5 FLOPs/byte

对于 GEMM (AI > 58.5):
- 可以达到接近峰值算力
- 实际限制因素: 指令发射率、Warp 调度效率

实际可达到: 30-50 TFLOPS (使用 CUDA Core)
```

### 5.2 Tensor Core 峰值

#### 第 5 代 Tensor Core 特性
```
RTX 5090 Tensor Core 支持:
- FP16: 835.3 TFLOPS (with Accumulation)
- BF16: 835.3 TFLOPS
- FP8: 1,670.6 TFLOPS
- FP4: 3,341.2 TFLOPS

相比 CUDA Core FP32 (104.9 TFLOPS):
- FP16: 8× 算力
- FP8: 16× 算力
```

#### Tensor Core 编程模型
```cuda
// WMMA (Warp Matrix Multiply Accumulate)
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, half> c_frag;

wmma::load_matrix_sync(a_frag, A, K);
wmma::load_matrix_sync(b_frag, B, N);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
```

**使用 Tensor Core 的 GEMM 可以达到**: 200+ TFLOPS (FP16)

### 5.3 内存带宽限制

#### 全局内存带宽
```
理论带宽: 1,792 GB/s (GDDR7)
实际可达到: ~70-80% = 1,250-1,430 GB/s
```

#### 共享内存带宽
```
每 SM 共享内存带宽: ~10-15 TB/s
全芯片: 170 SM × 10 TB/s = 1,700 TB/s (理论)

实际限制:
- Bank conflicts
- 访问模式
- 实际可用: ~5-10 TB/s per SM
```

### 5.4 Ridge Point 与优化策略

#### Ridge Point 计算
```
Ridge Point = Peak FLOPS / Memory Bandwidth
            = 104.88e12 / 1.792e12
            = 58.5 FLOPs/byte

含义:
- AI < 58.5: 受内存带宽限制
- AI > 58.5: 受计算能力限制
```

#### 不同 Kernel 的理论位置

| Kernel | Arithmetic Intensity | 瓶颈 | 理论利用率 |
|--------|---------------------|------|-----------|
| Naive | 0.5 | 内存带宽 | 0.85% |
| Shared | 683 | 计算 | 8.7% (实际被共享内存限制) |
| Register | 683 | 计算 | 30-50% |
| Tensor Core | ~100+ | 计算 | 60-80% |

### 5.5 为什么达不到 100% 峰值？

#### 主要限制因素

1. **Occupancy 限制** (25-50%)
   - 寄存器使用过多
   - 共享内存使用过多
   - Block 大小选择不当

2. **指令发射瓶颈** (60-80%)
   - Warp 调度器每周期只能发射有限指令
   - 内存指令和计算指令竞争发射槽
   - 分支发散降低效率

3. **内存延迟隐藏不足** (70-85%)
   - 需要足够的 Warp 来隐藏全局内存延迟 (~500 cycles)
   - 建议每 SM 至少 16-32 个 Warp

4. **Bank Conflicts** (75-90%)
   - 共享内存访问冲突
   - 降低有效带宽

5. **同步开销** (85-95%)
   - `__syncthreads()` 导致 Warp 空闲等待
   - 特别是在小矩阵或大量迭代时

#### 理论最大可达到
```
使用 CUDA Core:     30-50 TFLOPS (30-50%)
使用 Tensor Core:   200+ TFLOPS (FP16, 60-80%)
                    400+ TFLOPS (FP8, 60-80%)
```

---

## 6. GEMM 优化策略总结

### 6.1 针对 RTX 5090 的优化建议

#### Level 1: 基础优化 (达到 10-15 TFLOPS)
```
✓ 使用共享内存 Tiling
✓ Block 大小 16×16 或 32×32
✓ 合理的 BK 选择 (8-16)
✓ 循环展开 (#pragma unroll)
```

#### Level 2: 中级优化 (达到 20-40 TFLOPS)
```
✓ 寄存器分块 (TM×TN = 8×8 或更大)
✓ 消除 Bank Conflicts (padding/swizzling)
✓ 向量化加载 (float4)
✓ 优化 Occupancy (寄存器 ≤ 128)
✓ 双缓冲 (隐藏加载延迟)
```

#### Level 3: 高级优化 (达到 40-60 TFLOPS)
```
✓ Warp-level Tiling
✓ 使用异步拷贝 (cp.async)
✓ 精细的流水线调度
✓ 动态共享内存配置
```

#### Level 4: 极致优化 (达到 100+ TFLOPS)
```
✓ 使用 Tensor Core (WMMA/MMA)
✓ FP16/BF16 混合精度
✓ 专用汇编优化
✓ CUTLASS 类似实现
```

### 6.2 推荐配置

| 场景 | 推荐配置 | 预期性能 |
|------|---------|---------|
| **学习/通用** | TM=8, TN=8, BK=8, Block=16×16 | 20-30 TFLOPS |
| **FP32 峰值** | TM=16, TN=8, BK=16, Block=16×16 | 40-50 TFLOPS |
| **混合精度** | 使用 Tensor Core FP16 | 200+ TFLOPS |
| **极限性能** | CUTLASS/cuBLAS | 60-80% 峰值 |

### 6.3 关键检查清单

```
□ 寄存器使用 ≤ 128/线程
□ 共享内存 ≤ 48 KB/Block (留余量)
□ Bank Conflict 已消除
□ 循环已展开 (#pragma unroll)
□ Occupancy ≥ 25%
□ 异步拷贝 (如可用)
□ Tensor Core (需要极致性能时)
```

---

## 7. 调试与 Profiling 工具

### 7.1 编译时检查
```bash
# 查看寄存器使用和 spilling
nvcc -Xptxas -v -O3 kernel.cu

# 输出示例:
# ptxas info    : Compiling entry function '_Z13sgemm_register...'
# ptxas info    : Used 88 registers, 8192 bytes smem, 392 bytes cmem[0]
```

### 7.2 运行时 Profiling
```bash
# 使用 ncu (NVIDIA Compute Profiler)
ncu -o profile_report.ncu-rep ./benchmark_gemm

# 关键指标:
# - sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio
# - sm__sass_average_data_bytes_per_sector_mem_shared_op_ld.ratio
# - sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict
```

### 7.3 常用 ncu 指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Achieved Occupancy** | 实际占用率 | > 25% |
| **Shared Memory Bank Conflicts** | Bank 冲突次数 | 0 |
| **Warp Stall Long Scoreboard** | 等待内存的 Warp | 最小化 |
| **FMA Utilization** | FMA 单元利用率 | > 50% |
| **Memory Throughput** | 内存带宽利用率 | > 70% |

---

## 总结

RTX 5090 的硬件约束与优化要点：

| 约束项 | 限制 | 优化策略 |
|--------|------|---------|
| **寄存器** | 255/线程, 256KB/SM | 控制 90-128 个，避免 spilling |
| **共享内存** | 164KB/SM | 使用 ≤ 48KB/Block，避免容量瓶颈 |
| **Bank Conflict** | 32 Banks | Padding 或 Swizzling 消除 |
| **理论峰值** | 104.9 TFLOPS (FP32) | 实际可达 30-50% |
| **Tensor Core** | 835 TFLOPS (FP16) | 可达 60-80% 峰值 |

通过理解这些硬件约束并针对性优化，可以将 SGEMM 性能从 9 TFLOPS (Shared) 提升到 30-50 TFLOPS (Register Optimized)。

---

*文档生成时间: 2026年3月17日*
*适用硬件: NVIDIA GeForce RTX 5090 (Blackwell Architecture)*

# 第四部分：性能分析与 Roofline 模型

> **学习目标**：掌握 Roofline 模型和 Arithmetic Intensity 分析，科学定位性能瓶颈

---

## 目录

1. [Arithmetic Intensity 基础](#1-arithmetic-intensity-基础)
2. [Roofline 模型](#2-roofline-模型)
3. [GEMM 各版本分析](#3-gemm-各版本分析)
4. [瓶颈识别与优化](#4-瓶颈识别与优化)

---

## 1. Arithmetic Intensity 基础

### 1.1 定义与公式

**Arithmetic Intensity (AI)**：计算密度，衡量每字节内存访问能执行多少次浮点运算

$$AI = \frac{\text{FLOPs}}{\text{Bytes}} \quad \text{[单位: FLOPs/byte]}$$

### 1.2 GEMM 计算分析

```
矩阵乘法 C = A × B 的计算量：

FLOPs = 2 × M × N × K
（每次乘加 = 2 FLOPS：1 次乘法 + 1 次加法）

示例 (M=N=K=4096):
FLOPs = 2 × 4096³ ≈ 137.4 GFLOPs
```

### 1.3 内存访问分析

| Kernel 类型 | 内存访问模式 | 总字节数 | AI |
|:---|:---|:---:|:---:|
| **Naive** | 每次计算都读全局内存 | ~2×M×K×N×4 | ~0.5 |
| **Shared** | 数据只读 1 次到 Shared | ~(M×K+K×N+M×N)×4 | ~M/6 |
| **Register** | 同 Shared | ~(M×K+K×N+M×N)×4 | ~M/6 |

```
Shared/Register Kernel AI 推导：

内存访问 = (M×K + K×N + M×N) × 4 bytes
当 M=N=K 时:
内存访问 = (M² + M² + M²) × 4 = 12M² bytes

AI = 2M³ / 12M² = M / 6

当 M=4096 时:
AI = 4096 / 6 ≈ 682.7 FLOPs/byte
```

---

## 2. Roofline 模型

### 2.1 模型定义

```
Roofline 模型公式：

Performance = min(
    Peak FLOPS,           ← 计算屋顶（水平线）
    Memory Bandwidth × AI  ← 内存屋顶（斜线）
)

图示：

Performance (TFLOPS)
    │
105 ┤────────────────────────┐ ← Peak FLOPS (RTX 5090)
    │                        │
 70 ┤                   ┌────┘
    │              ┌────┘
 35 ┤         ┌────┘
    │    ┌────┘ ← Ridge Point
 10 ┤────┘──────────────────────────────
    │   /
    │  /  Memory Bandwidth 斜线
    │ /
    ┼/──────────────────────────────────
     1   10   58.5  100  1000  → AI
              ↑
         Ridge Point

Ridge Point = Peak FLOPS / Memory Bandwidth
            = 104.9 / 1.792
            ≈ 58.5 FLOPs/byte

关键洞察：
- AI < 58.5: 内存受限（斜线区域）
- AI > 58.5: 计算受限（平顶区域）
```

### 2.2 RTX 5090 参数

| 参数 | 数值 | 说明 |
|:---|:---:|:---|
| **FP32 Peak** | 104.9 TFLOPS | CUDA Core 峰值 |
| **Memory BW** | 1,792 GB/s | GDDR7 显存带宽 |
| **Ridge Point** | 58.5 FLOPs/byte | 性能转折点 |
| **Tensor FP16** | 835.3 TFLOPS | Tensor Core 峰值 |
| **Tensor Ridge** | 466.5 FLOPs/byte | Tensor Core 转折点 |

---

## 3. GEMM 各版本分析

### 3.1 Roofline 图上的位置

```
RTX 5090 Roofline 图与 GEMM Kernel 位置：

Performance (TFLOPS)
    │
835 ┤◌ Tensor Core Target              ┐
    │                              │
105 ┤───────────────────────────┐  │
    │                           │  │
 50 ┤                      ◌ Reg│  │  (~50 TFLOPS)
    │                 ┌────────┘  │
 30 ┤            ┌────┘           │
    │       ┌────┘                │
 10 ┤──────┘◌ Shared            │
    │      / (~9 TFLOPS)         │
    │     /                      │
    │    /                       │
    │   /  ◌ Naive (~1 TFLOPS)   │
    │  /                         │
    ┼/───────────────────────────────
     0.5  1   10   58.5  100  683  → AI
              ↑                  ↑
         Ridge Point         Shared/Reg AI

关键观察：
1. Naive: AI=0.5 << 58.5，严重内存受限，~1 TFLOPS
2. Shared: AI=683 >> 58.5，进入计算受限区，可达 ~9 TFLOPS
3. Register: 同 AI=683，但更高效，~50 TFLOPS
4. Tensor: 新屋顶线 835 TFLOPS，可达 ~200+ TFLOPS
```

### 3.2 性能差距分析

| Kernel | AI | 理论上限 | 实际性能 | 利用率 |
|:---:|:---:|:---:|:---:|:---:|
| **Naive** | 0.5 | 0.9 TFLOPS | ~7 TFLOPS | ~7% |
| **Shared** | 683 | 104.9 TFLOPS | ~9 TFLOPS | ~9% |
| **Register** | 683 | 104.9 TFLOPS | ~50 TFLOPS | ~48% |
| **Tensor Core** | 100+ | 835 TFLOPS | ~300 TFLOPS | ~36% |

**为什么 AI 很高，但利用率不高？**

```
主要原因：

1. 指令发射瓶颈 (60-80% 峰值)
   - Warp 调度器每周期只能发射有限指令
   - 内存指令和计算指令竞争发射槽
   
2. 内存延迟隐藏不足 (70-85% 峰值)
   - 需要足够的 Warp 来隐藏全局内存延迟 (~500 cycles)
   - 建议每 SM 至少 16-32 个 Warp

3. Bank Conflicts (75-90% 峰值)
   - 共享内存访问冲突降低有效带宽

4. 同步开销 (85-95% 峰值)
   - __syncthreads() 导致 Warp 空闲等待

5. 数据依赖 (90-95% 峰值)
   - 计算指令之间的数据依赖限制并行
```

---

## 4. 瓶颈识别与优化

### 4.1 瓶颈诊断流程

```
性能瓶颈诊断树：

开始性能分析
    │
    ▼
检查 AI 与 Ridge Point 关系
    │
    ├── AI < Ridge Point (内存受限)
    │   │
    │   ├── 全局内存带宽未饱和 → 检查 Memory Coalescing
    │   │
    │   ├── 全局内存带宽已饱和 → 使用 Tiling 提升 AI
    │   │
    │   └── L2 Cache Miss 高 → 改进数据局部性
    │
    └── AI > Ridge Point (计算受限)
        │
        ├── Occupancy 低 → 减少寄存器/共享内存使用
        │
        ├── Bank Conflict 高 → 添加 Padding
        │
        ├── 指令发射率低 → 增加 ILP，减少 Divergence
        │
        └── 未使用 Tensor Core → 切换到 WMMA
```

### 4.2 优化策略矩阵

| 瓶颈类型 | 诊断指标 | 优化策略 | 预期提升 |
|:---|:---|:---|:---:|
| **内存受限** | AI < 58.5 | Shared Memory Tiling | 100×+ |
| **Bank Conflict** | Nsight: Bank Conflict > 0 | Padding +1 | 20-30% |
| **低 Occupancy** | Occupancy < 25% | 减少寄存器使用 | 10-30% |
| **指令瓶颈** | Issue Slot Util < 80% | 增加 ILP，展开循环 | 10-20% |
| **未用 Tensor** | 使用 CUDA Core | 切换到 WMMA | 4-8× |

### 4.3 实用检查清单

```
GEMM Kernel 优化检查清单：

□ Arithmetic Intensity > 58.5 (RTX 5090 Ridge Point)
□ Memory Coalescing > 80%
□ Shared Memory Bank Conflict = 0
□ Occupancy 25-50%（GEMM 最佳范围）
□ 指令发射率 > 70%
□ 使用 Tensor Core (FP16 可达 8× 性能)
□ 循环展开 (#pragma unroll)
□ 向量化加载 (float4)
```

---

## 5. 实际案例分析

### 5.1 案例：优化 Shared Memory Kernel

```
问题：Shared Kernel 只有 ~9 TFLOPS，远低于理论 104 TFLOPS

分析：
1. AI = 683 > 58.5 → 计算受限（正确方向）
2. Occupancy = 100% → 可能线程太多，每个线程工作量太少
3. 每个线程只计算 1 个元素 → 共享内存访问 64 次/计算 32 次

优化方案：
→ 使用 Register Tiling
   - 每个线程计算 8×8 = 64 个元素
   - 寄存器缓存减少共享内存访问
   - 计算/访存比从 0.5 → 4.0

结果：性能从 ~9 TFLOPS → ~50 TFLOPS（5.5× 提升）
```

### 5.2 案例：优化 Bank Conflict

```
问题：Register Kernel 只有 ~35 TFLOPS，低于预期

分析（Nsight Compute）：
- sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict > 0
- 存在 Shared Memory Bank Conflict

诊断：
sA[128][8] 导致行 0 和行 8 映射到相同 Bank

优化方案：
→ 添加 Padding: sA[128][9]
   - 行间距从 32 bytes → 36 bytes
   - 打破 Bank 对齐，消除 Conflict

结果：性能从 ~35 TFLOPS → ~45 TFLOPS（28% 提升）
```

---

## 6. 总结

### 6.1 核心要点

| 概念 | 要点 |
|:---|:---|
| **AI** | FLOPs / Bytes，衡量计算密度 |
| **Ridge Point** | Peak FLOPS / BW，性能转折点 |
| **Memory Bound** | AI < Ridge Point，优化目标是提升 AI |
| **Compute Bound** | AI > Ridge Point，优化目标是提高利用率 |
| **GEMM 优化路径** | Naive → Shared → Register → Tensor |

### 6.2 性能优化路线图

```
RTX 5090 GEMM 优化路线图：

阶段 0: Naive (AI=0.5, Memory Bound)
    │
    ▼ 应用 Shared Memory Tiling
阶段 1: Shared (AI=683, Compute Bound)
    │  性能: ~9 TFLOPS (8%)
    ▼ 应用 Register Tiling
阶段 2: Register (AI=683, Compute Bound)
    │  性能: ~50 TFLOPS (48%)
    ▼ 应用向量化 + Padding
阶段 3: Optimized (AI=683, Compute Bound)
    │  性能: ~45 TFLOPS (43%)
    ▼ 应用 Tensor Core
阶段 4: Tensor Core (AI=100+, Compute Bound)
       性能: 200-300 TFLOPS (FP16, 24-36%)
```

---

*下一步学习：[09_profiling_tools.md](09_profiling_tools.md) - 使用 Nsight Compute 进行实际性能分析*

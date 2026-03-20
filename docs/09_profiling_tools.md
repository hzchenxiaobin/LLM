# 第四部分：Nsight Compute 性能分析实战

> **学习目标**：掌握 Nsight Compute 工具使用，实际诊断性能瓶颈

---

## 目录

1. [Nsight Compute 基础](#1-nsight-compute-基础)
2. [关键性能指标](#2-关键性能指标)
3. [GEMM 性能分析实战](#3-gemm-性能分析实战)
4. [常见问题诊断](#4-常见问题诊断)

---

## 1. Nsight Compute 基础

### 1.1 工具简介

Nsight Compute (ncu) 是 NVIDIA 提供的 GPU Kernel 性能分析工具，可以：
- 收集详细的硬件性能计数器
- 分析内存访问模式
- 识别计算和内存瓶颈
- 对比不同版本的性能差异

### 1.2 基本使用方法

```bash
# 1. 基本性能分析
ncu ./benchmark_gemm

# 2. 保存结果到文件
ncu -o report.ncu-rep ./benchmark_gemm

# 3. 指定特定 kernel 分析
ncu --kernel-name sgemm_register_kernel ./benchmark_gemm

# 4. 收集特定指标
ncu --metrics sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio \
    ./benchmark_gemm

# 5. 交互式 GUI 查看
ncu-ui report.ncu-rep
```

---

## 2. 关键性能指标

### 2.1 核心指标速查表

| 指标 | 命令行名称 | 目标值 | 说明 |
|:---|:---|:---:|:---|
| **Achieved Occupancy** | `sm__warps_active.avg.pct_of_peak_sustained_active` | 25-50% | 实际占用率 |
| **Memory Coalescing** | `sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | > 80% | 全局内存合并访问率 |
| **Bank Conflicts** | `sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict` | 0 | 共享内存 Bank 冲突 |
| **FMA Utilization** | `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | > 50% | FMA 单元利用率 |
| **Memory Throughput** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | - | 显存带宽利用率 |
| **L2 Cache Hit** | `lts__t_sector_hit_rate.pct` | > 80% | L2 缓存命中率 |

### 2.2 详细指标解释

#### Occupancy（占用率）

```
指标：sm__warps_active.avg.pct_of_peak_sustained_active

含义：
- 实际运行的 Warp 数 / SM 最大 Warp 数
- 反映 SM 资源利用情况

目标值：
- GEMM: 25-50%（允许更多寄存器用于 ILP）
- 内存密集型: 70-100%（需要更多 Warp 隐藏延迟）

诊断：
- < 25%: 检查寄存器使用是否过多
- > 75%: 可能有提升空间（增加寄存器使用）
```

#### Memory Coalescing（内存合并）

```
指标：sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio

含义：
- 平均每个 sector 传输的有效字节数 / 32 bytes
- 32 bytes = 100%（完美合并）

目标值：
- > 80%: 良好
- 50-80%: 一般
- < 50%: 差，需要优化

示例：
- 连续访问: 32 bytes / 32 bytes = 100%
- 跨步访问: 4 bytes / 32 bytes = 12.5%
```

#### Bank Conflict

```
指标：sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict

含义：
- 共享内存访问冲突次数
- 理想值为 0

诊断：
- > 0: 存在 Bank Conflict，添加 Padding
- 每轮 k 迭代都有 Conflict: 严重的布局问题
```

---

## 3. GEMM 性能分析实战

### 3.1 Naive Kernel 分析

```bash
# 运行分析
ncu --metrics sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio,\
              sm__warps_active.avg.pct_of_peak_sustained_active,\
              sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed \
    ./benchmark_gemm --kernel sgemm_naive_kernel
```

**预期输出分析**：

```
预期指标值：
┌──────────────────────────────────────┬────────┬──────────┐
│ 指标                                  │ 预期值  │ 含义     │
├──────────────────────────────────────┼────────┼──────────┤
│ Memory Coalescing (A 访问)            │ ~100%  │ 良好     │
│ Memory Coalescing (B 访问)            │ ~12%   │ 极差     │
│ Achieved Occupancy                    │ ~100%  │ 正常     │
│ FMA Utilization                       │ ~5%    │ 极低     │
│ Memory Throughput                     │ ~90%   │ 瓶颈     │
└──────────────────────────────────────┴────────┴──────────┘

诊断：
- B 矩阵访问合并率低 → 跨步访问问题
- FMA 利用率极低 → 计算单元饥饿
- 内存吞吐量高 → 确认是内存瓶颈

优化方向：
→ 使用 Shared Memory Tiling 提升 AI
→ 解决 B 矩阵的内存访问模式
```

### 3.2 Shared Kernel 分析

```bash
ncu --metrics sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict,\
              sm__sass_average_data_bytes_per_sector_mem_shared_op_ld.ratio,\
              sm__warps_active.avg.pct_of_peak_sustained_active,\
              sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed \
    ./benchmark_gemm --kernel sgemm_shared_kernel
```

**预期输出分析**：

```
预期指标值：
┌──────────────────────────────────────┬────────┬──────────┐
│ 指标                                  │ 预期值  │ 含义     │
├──────────────────────────────────────┼────────┼──────────┤
│ Shared Memory Bank Conflicts          │ > 0    │ 有冲突   │
│ Achieved Occupancy                    │ ~100%  │ 过高     │
│ FMA Utilization                       │ ~10%   │ 偏低     │
│ Global Memory Coalescing              │ ~90%   │ 良好     │
└──────────────────────────────────────┴────────┴──────────┘

诊断：
- Bank Conflict 存在 → sB 访问模式问题
- Occupancy 100% → 每线程工作量太少
- FMA 利用率 10% → 共享内存带宽瓶颈

优化方向：
→ 使用 Register Tiling 减少共享内存访问
→ 增加每线程计算量，降低 Occupancy
```

### 3.3 Register Kernel 分析

```bash
ncu --metrics sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict,\
              sm__warps_active.avg.pct_of_peak_sustained_active,\
              sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
              sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio \
    ./benchmark_gemm --kernel sgemm_register_kernel
```

**预期输出分析**：

```
预期指标值：
┌──────────────────────────────────────┬────────┬──────────┐
│ 指标                                  │ 预期值  │ 含义     │
├──────────────────────────────────────┼────────┼──────────┤
│ Bank Conflicts                        │ 0      │ 理想     │
│ Achieved Occupancy                    │ ~33%   │ 良好     │
│ FMA Utilization                       │ ~50%   │ 较好     │
│ Global Memory Coalescing              │ ~90%   │ 良好     │
│ Issue Slot Utilization                │ ~60%   │ 有提升空间│
└──────────────────────────────────────┴────────┴──────────┘

诊断：
- Bank Conflict = 0 → Padding 有效
- Occupancy 33% → 寄存器使用合理
- FMA 50% → 仍有提升空间
- Issue Slot 60% → 指令发射瓶颈

优化方向：
→ 使用 Tensor Core 突破瓶颈
→ 进一步优化指令调度
```

---

## 4. 常见问题诊断

### 4.1 问题诊断矩阵

| 症状 | 检查指标 | 诊断方法 | 解决方案 |
|:---|:---|:---|:---|
| **性能远低于预期** | FMA Util, Occupancy | `ncu --metrics sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 检查 AI，是否内存受限 |
| **共享内存慢** | Bank Conflicts | `sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict` | 添加 Padding |
| **全局内存慢** | Coalescing | `sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 检查访问模式 |
| **Occupancy 低** | Register Usage | `launch__registers_per_thread` | 减少寄存器使用 |
| **Tensor Core 未生效** | Tensor Util | `sm__inst_executed_pipe_tensor.sum` | 检查 WMMA 配置 |

### 4.2 诊断脚本示例

```bash
#!/bin/bash
# GEMM 性能诊断脚本

KERNEL=$1

echo "===== Analyzing $KERNEL ====="

# 1. 基础指标
echo "--- Basic Metrics ---"
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
              launch__registers_per_thread,\
              launch__shared_mem_per_block_static \
    ./benchmark_gemm --kernel $KERNEL

# 2. 内存指标
echo "--- Memory Metrics ---"
ncu --metrics sm__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio,\
              sm__sass_l1tex_data_pipe_lsu_mem_shared_op_ld.banks_conflict,\
              dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./benchmark_gemm --kernel $KERNEL

# 3. 计算指标
echo "--- Compute Metrics ---"
ncu --metrics sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
              sm__inst_executed_pipe_tensor.sum,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./benchmark_gemm --kernel $KERNEL

echo "===== Analysis Complete ====="
```

### 4.3 编译时检查

```bash
# 查看寄存器使用和共享内存
nvcc -Xptxas -v -O3 sgemm_register.cu

# 输出示例：
# ptxas info    : Compiling entry function '_Z13sgemm_register...'
# ptxas info    : Used 88 registers, 8192 bytes smem, 392 bytes cmem[0]
#                ↑ 寄存器使用    ↑ 共享内存使用
```

---

## 5. 性能优化工作流

```
性能优化完整工作流：

1. 基准测试
   │
   ▼
2. Nsight Compute 分析
   ├── 检查 Occupancy
   ├── 检查 Memory Coalescing
   ├── 检查 Bank Conflict
   └── 检查 FMA Utilization
   │
   ▼
3. 识别瓶颈
   ├── AI < Ridge Point → 内存瓶颈
   │   └── 使用 Tiling 提升 AI
   ├── Bank Conflict > 0 → 共享内存问题
   │   └── 添加 Padding
   ├── Occupancy < 25% → 资源限制
   │   └── 调整寄存器/共享内存使用
   └── FMA Util < 50% → 计算效率低
       └── 增加 ILP，使用 Tensor Core
   │
   ▼
4. 实施优化
   │
   ▼
5. 验证性能提升
   └── 重复步骤 1-4 直到满意
```

---

## 6. 总结

### 6.1 关键指标速查

| 指标 | 良好值 | 警告值 | 危险值 |
|:---|:---:|:---:|:---:|
| **Occupancy** | 25-50% | < 25% | > 75% (GEMM) |
| **Memory Coalescing** | > 80% | 50-80% | < 50% |
| **Bank Conflict** | 0 | < 100 | > 1000 |
| **FMA Utilization** | > 50% | 30-50% | < 30% |
| **L2 Hit Rate** | > 80% | 60-80% | < 60% |

### 6.2 工具使用要点

1. **定期 Profiling**：每次优化后都用 ncu 验证
2. **关注关键指标**：Occupancy、Coalescing、Bank Conflict
3. **对比分析**：保存基准结果，对比优化效果
4. **结合 Roofline**：用 AI 指导优化方向

---

*本教程完。继续学习 CUTLASS 框架请参考官方文档。*

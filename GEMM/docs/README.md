# CUDA GEMM 优化文档索引

本目录包含 GEMM (General Matrix Multiply) 优化的完整学习文档，从基础概念到高级技术。

---

## 文档导航

### 入门必读

| 文档 | 内容 | 适合人群 |
|-----|------|---------|
| **[01_fundamentals.md](01_fundamentals.md)** | CUDA 线程层级、SM 架构、内存层次、GEMM 基础概念 | 初学者 |
| **[sgemm_shared_kernel_explained.md](sgemm_shared_kernel_explained.md)** | Shared Memory Tiling 详解（代码逐行解读） | 初学者 |

### 进阶优化

| 文档 | 内容 | 适合人群 |
|-----|------|---------|
| **[02_optimization_guide.md](02_optimization_guide.md)** | Naive → Shared → Register → Vectorized 完整优化路径 | 中级开发者 |
| **[03_advanced_techniques.md](03_advanced_techniques.md)** | Bank Conflict 消除、Roofline 模型、双缓冲、Tensor Core | 高级开发者 |

---

## 学习路径建议

### 路径 1：快速入门（1-2 天）
1. 阅读 [01_fundamentals.md](01_fundamentals.md) 理解基础概念
2. 阅读 [sgemm_shared_kernel_explained.md](sgemm_shared_kernel_explained.md) 掌握 Shared Memory Tiling
3. 运行 `make && ./benchmark_gemm` 观察性能差异

### 路径 2：系统学习（1-2 周）
1. **第 1-2 天**：01_fundamentals.md + sgemm_shared_kernel_explained.md
2. **第 3-5 天**：02_optimization_guide.md，实现 Register Tiling
3. **第 6-8 天**：03_advanced_techniques.md，实现 Vectorized + Padding
4. **第 9-10 天**：Tensor Core (WMMA) 实践

### 路径 3：面试准备
- 快速浏览 01_fundamentals.md 的核心概念
- 重点阅读 02_optimization_guide.md 的优化对比表格
- 掌握 03_advanced_techniques.md 的 Roofline 和 Bank Conflict

---

## 关键概念速查

| 概念 | 一句话解释 | 所在文档 |
|-----|-----------|---------|
| **Warp** | 32 线程同时执行，最小调度单位 | 01_fundamentals.md |
| **Occupancy** | 实际运行 Warp 数 / 最大 Warp 数，25-50% 是 GEMM 最佳 | 01_fundamentals.md |
| **Arithmetic Intensity** | 计算强度 = FLOPs / 内存访问，GEMM 目标是 >58.5 | 01_fundamentals.md |
| **Bank Conflict** | 同一 Warp 多线程访问同一 Bank，用 Padding 消除 | 03_advanced_techniques.md |
| **Ridge Point** | AI = Peak FLOPS / Memory BW，RTX 5090 是 58.5 | 03_advanced_techniques.md |

---

## 代码与文档对应关系

| 代码文件 | 优化技术 | 参考文档 |
|---------|---------|---------|
| `sgemm_naive.cu` | 基础实现 | 02_optimization_guide.md |
| `sgemm_shared.cu` | Shared Memory Tiling | sgemm_shared_kernel_explained.md |
| `sgemm_register.cu` | Register Tiling | 02_optimization_guide.md |
| `sgemm_register_v2.cu` | Vectorized + Padding | 02_optimization_guide.md |
| `sgemm_register_bank_conflict.cu` | Bank Conflict 优化 | 03_advanced_techniques.md |
| `sgemm_register_v3.cu` | Double Buffering | 03_advanced_techniques.md |
| `sgemm_wmma.cu` | Tensor Core (WMMA) | 03_advanced_techniques.md |

---

## 性能参考（RTX 5090，4096×4096×4096）

| Kernel | TFLOPS | 峰值利用率 | 学习优先级 |
|--------|--------|-----------|-----------|
| Naive | ~0.5 | 0.5% | 了解 |
| Shared | ~9 | 9% | ⭐⭐⭐ |
| Register | ~35 | 33% | ⭐⭐⭐⭐ |
| Register V2 | ~50 | 48% | ⭐⭐⭐⭐ |
| Tensor Core | ~300+ | 70%+ | ⭐⭐⭐⭐⭐ |
| cuBLAS | ~105 | 100% | 目标 |

---

## 更多资源

- **面试题**：[../exercises/](../exercises/) 目录下有完整面试题集
- **代码参考**：[../src/](../src/) 目录包含所有实现
- **硬件规格**：参见 01_fundamentals.md 的 SM 架构章节

---

*文档更新时间：2026年3月19日*

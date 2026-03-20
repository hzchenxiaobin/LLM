# 批量矩阵乘法 (Batched GEMM) 优化教程

> **注意**：本目录下的详细文档已整合到新的体系化教程中。

---

## 新版教程位置

批量 GEMM 相关内容已整合到：
- [10_batched_gemm.md](/docs/10_batched_gemm.md)

---

## 原教程存档

原教程 `CUDA_Batched_GEMM_Optimization_Tutorial.md` 内容已整合到新体系。

主要内容包括：
1. V0: 朴素 Batched GEMM
2. V1: Shared Memory 分块
3. V2: 寄存器分块
4. V3: 向量化访存
5. V4: Tensor Core 优化

---

## 学习建议

请先完成基础 GEMM 教程，再学习批量 GEMM：
1. [CUDA 基础](/docs/01_cuda_fundamentals.md)
2. [GEMM 优化阶梯](/docs/03_gemm_naive.md) ~ [05_register_tiling.md)
3. [批量 GEMM 专题](/docs/10_batched_gemm.md)

---

*更新日期：2026年3月*

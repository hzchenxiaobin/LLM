# SGEMM Register Kernel 代码逐行解读

## 文件：`sgemm_register.cu`

本文档对 `sgemm_register.cu` 的每一行代码进行详细解读，解释其设计原理和优化思路。

---

## 目录
1. [头文件与分块参数定义](#1-头文件与分块参数定义)
2. [Kernel 函数入口与索引计算](#2-kernel-函数入口与索引计算)
3. [内存申请：共享内存与寄存器](#3-内存申请共享内存与寄存器)
4. [数据加载：协作加载到共享内存](#4-数据加载协作加载到共享内存)
5. [核心计算：寄存器分块外积](#5-核心计算寄存器分块外积)
6. [结果写回全局内存](#6-结果写回全局内存)
7. [启动配置](#7-启动配置)

---

## 1. 头文件与分块参数定义

```cuda
#include "common.h"
#include "gemm_kernels.h"
```

**解读**：
- `common.h`：包含 CUDA 错误检查宏和统一接口定义
- `gemm_kernels.h`：包含所有 SGEMM kernel 的函数声明

---

```cuda
// 定义分块大小
#define BM 128  // Block在M维度的负责大小
#define BN 128  // Block在N维度的负责大小
#define BK 8    // Block在K维度的步长
#define TM 8    // Thread在M维度的负责大小
#define TN 8    // Thread在N维度的负责大小
```

**解读**：
这是 Register Tiling 的核心参数设计，采用**双层分块策略**：

| 参数 | 值 | 含义 | 计算验证 |
|------|-----|------|---------|
| **BM** | 128 | 每个 Block 负责计算 128 行 | - |
| **BN** | 128 | 每个 Block 负责计算 128 列 | - |
| **BK** | 8 | K维度每次处理 8 个元素 | - |
| **TM** | 8 | 每个线程负责 8 行 | - |
| **TN** | 8 | 每个线程负责 8 列 | 8×8=64 元素 |

**关键设计决策**：

1. **BK = 8 而非 32**：
   - 让 A 的 8 个元素和 B 的 8 个元素能完全放入寄存器
   - 减少共享内存访问次数（从 64 次降到 16 次）
   - 代价：需要更多 K 迭代（512 次 vs 128 次）

2. **Block 大小 16×16 = 256 线程**：
   - 相比 Shared Kernel 的 1024 线程，每个线程工作量更大
   - 256 线程分摊 1024 个元素加载（每线程 4 个）

3. **验证计算**：
   ```
   Block 负责区域 = BM × BN = 128 × 128 = 16,384 元素
   每线程计算量 = TM × TN = 8 × 8 = 64 元素
   总线程数 = 16 × 16 = 256
   验证：256 × 64 = 16,384 ✓
   ```

---

## 2. Kernel 函数入口与索引计算

```cuda
__global__ void sgemm_register_kernel(int M, int N, int K, float alpha, 
                                      const float *A, const float *B, 
                                      float beta, float *C) {
```

**解读**：
标准 SGEMM 接口：`C = alpha × A × B + beta × C`
- 输入：M, N, K 是矩阵维度；alpha, beta 是缩放系数
- A：M×K 矩阵，B：K×N 矩阵，C：M×N 矩阵

---

```cuda
    // Block 索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
```

**解读**：
- `blockIdx.x`：Block 在 N 维度（列）的位置
- `blockIdx.y`：Block 在 M 维度（行）的位置
- 注意：CUDA 中 x 通常对应列（N），y 对应行（M）

---

```cuda
    // Thread 索引 (16x16 = 256 个线程 / Block)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
```

**解读**：
- `threadIdx.x`：线程在 Block 内的列索引（0~15）
- `threadIdx.y`：线程在 Block 内的行索引（0~15）
- 每个 Block 共 16×16 = 256 个线程

---

```cuda
    // 线程的全局一维 ID (Block内)
    int tid = ty * blockDim.x + tx;
```

**解读**：
**关键设计**：将二维线程索引转换为一维 ID（0~255）

**用途**：
- 用于协作加载数据时计算每个线程负责的位置
- `blockDim.x` = 16（线程块的 x 维度大小）

**示例**：
```
Thread (tx=3, ty=5) → tid = 5 × 16 + 3 = 83
```

---

## 3. 内存申请：共享内存与寄存器

```cuda
    // 1. 申请共享内存
    __shared__ float sA[BM][BK];  // sA[128][8]
    __shared__ float sB[BK][BN];  // sB[8][128]
```

**解读**：

**共享内存大小计算**：
```
sA: 128 × 8 × 4 bytes = 4,096 bytes = 4 KB
sB: 8 × 128 × 4 bytes = 4,096 bytes = 4 KB
总计: 8 KB / Block
```

**设计对比（vs Shared Kernel）**：

| 特性 | Shared Kernel | Register Kernel | 差异 |
|------|--------------|-----------------|------|
| sA 大小 | [32][32] = 1K 元素 | [128][8] = 1K 元素 | 相同元素数 |
| sB 大小 | [32][32] = 1K 元素 | [8][128] = 1K 元素 | 相同元素数 |
| K维度 | 32 | 8 | 更小，更多数据进寄存器 |
| M/N维度 | 32 | 128 | 更大，Block 负责更多 |

**关键洞察**：
- 虽然元素数相同（1024），但形状不同
- BK=8 使得 sA 的列和 sB 的行很小，可以放入寄存器缓存

---

```cuda
    // 2. 申请线程私有的寄存器数组，用于保存 C 的中间结果
    // 每个线程负责计算 8x8 = 64 个元素
    float accum[TM][TN] = {0.0f};  // accum[8][8]
```

**解读**：
**这是 Register Tiling 最核心的优化！**

**寄存器数组分析**：
- `accum[8][8]` = 64 个 float = 256 bytes
- 全部存储在寄存器中（最快的内存层级）
- 相比 Shared Kernel 的 `float tmp`（4 bytes），增加了 64 倍！

**为什么需要这么多寄存器？**
```
每个线程负责计算 8×8 = 64 个 C 元素
需要 64 个累加器分别存储这 64 个元素的部分和
```

**硬件限制检查**：
```
现代 GPU（如 RTX 4090/5090）每个线程最多可使用 255 个寄存器
accum[8][8] 使用 64 个寄存器，远低于限制
加上 frag_a[8], frag_b[8] 等其他变量，总共约 80 个寄存器
```

---

```cuda
    // 计算当前线程负责的 C 矩阵元素的全局起始坐标
    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;
```

**解读**：
**计算该线程负责的 8×8 C 子矩阵的起始位置**

**分解**：
```
row_start = by × 128 + ty × 8
          ↑ Block偏移    ↑ Thread在Block内的偏移

col_start = bx × 128 + tx × 8
          ↑ Block偏移    ↑ Thread在Block内的偏移
```

**示例**：
```
假设 Block(1, 2) 中的 Thread(3, 5):
- by=2, bx=1
- ty=5, tx=3

row_start = 2 × 128 + 5 × 8 = 256 + 40 = 296
col_start = 1 × 128 + 3 × 8 = 128 + 24 = 152

该线程负责计算 C[296:304][152:160]（8×8 子矩阵）
```

---

## 4. 数据加载：协作加载到共享内存

```cuda
    // 预计算从全局内存搬运数据到共享内存时的坐标 (共 256 个线程协作)
    // 搬运 A: 需要加载 128x8 = 1024 个元素，256个线程每个加载 4 个
    int load_a_row = tid / BK;     // tid / 8
    int load_a_col = tid % BK;     // tid % 8
```

**解读**：
**协作加载策略设计**

**计算**：
```
总元素数: BM × BK = 128 × 8 = 1,024 个 float
线程数: 256
每线程加载: 1024 / 256 = 4 个元素
```

**坐标计算**：
```
load_a_row = tid / 8  // 行坐标 (0~127)
load_a_col = tid % 8  // 列坐标 (0~7)

示例：
tid = 83 → row = 10, col = 3
表示该线程负责加载 sA[10][3], sA[42][3], sA[74][3], sA[106][3]
（间隔 32 行，共 4 个元素）
```

**为什么这样分配？**
- 256 线程各自负责 4 个元素，覆盖全部 1024 个元素
- 通过 `i * 32` 跳跃（见下文循环），实现均匀分配

---

```cuda
    // 搬运 B: 需要加载 8x128 = 1024 个元素，256个线程每个加载 4 个
    int load_b_row = tid / BN;     // tid / 128
    int load_b_col = tid % BN;     // tid % 128
```

**解读**：
同理，为 B 矩阵设计加载坐标：
```
总元素数: BK × BN = 8 × 128 = 1,024 个 float
每线程加载: 1024 / 256 = 4 个元素

load_b_row = tid / 128  // 行坐标 (0~7)
load_b_col = tid % 128  // 列坐标 (0~127)
```

---

```cuda
    // 3. 沿 K 维度分块滑动
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;
```

**解读**：
**K 维度循环**

**循环次数计算**：
```
(K + BK - 1) / BK = ceil(K / 8)

当 K = 4096 时：
迭代次数 = (4096 + 7) / 8 = 512 次

对比 Shared Kernel（BK=32）：
迭代次数 = 4096 / 32 = 128 次
```

**代价与收益**：
- **代价**：4× 迭代次数，4× 同步次数
- **收益**：每次迭代计算量更大（64 FMA vs 32 FMA），且寄存器缓存更好

---

```cuda
        // --- 步骤 A: 协作加载 A 块到 sA ---
        #pragma unroll
        for (int i = 0; i < BM * BK / 256; ++i) {
            int a_row_idx = load_a_row + i * 32; // 256/8 = 32
            int a_col_idx = load_a_col;
            int global_a_row = by * BM + a_row_idx;
            int global_a_col = k_offset + a_col_idx;
            if (global_a_row < M && global_a_col < K) {
                sA[a_row_idx][a_col_idx] = A[global_a_row * K + global_a_col];
            } else {
                sA[a_row_idx][a_col_idx] = 0.0f;
            }
        }
```

**解读**：
**协作加载 A 矩阵到共享内存**

**循环展开**：
```
#pragma unroll  // 编译器展开循环
i 从 0 到 BM*BK/256 - 1 = 1024/256 - 1 = 3
即 i = 0, 1, 2, 3（共 4 次迭代）
```

**地址计算**：
```
a_row_idx = load_a_row + i * 32
           = (tid/8) + i * 32

示例：tid = 10
i=0: row = 1, col = 2 → 加载 sA[1][2]
i=1: row = 33, col = 2 → 加载 sA[33][2]
i=2: row = 65, col = 2 → 加载 sA[65][2]
i=3: row = 97, col = 2 → 加载 sA[97][2]
```

**全局内存地址**：
```
A[global_a_row * K + global_a_col]
  ↑ 行主序：跳过 K 列
```

**边界检查**：
```cuda
if (global_a_row < M && global_a_col < K)
```
处理矩阵维度非 128 整数倍的情况，越界填充 0

---

```cuda
        // --- 步骤 B: 协作加载 B 块到 sB ---
        #pragma unroll
        for (int i = 0; i < BK * BN / 256; ++i) {
            int b_row_idx = load_b_row + i * 2; // 256/128 = 2
            int b_col_idx = load_b_col;
            int global_b_row = k_offset + b_row_idx;
            int global_b_col = bx * BN + b_col_idx;
            if (global_b_row < K && global_b_col < N) {
                sB[b_row_idx][b_col_idx] = B[global_b_row * N + global_b_col];
            } else {
                sB[b_row_idx][b_col_idx] = 0.0f;
            }
        }
```

**解读**：
同理加载 B 矩阵，关键区别：
```
i * 2 而非 i * 32，因为 BN=128（大于 BK=8）

示例：tid = 10
i=0: row = 0, col = 10 → 加载 sB[0][10]
i=1: row = 2, col = 10 → 加载 sB[2][10]
i=2: row = 4, col = 10 → 加载 sB[4][10]
i=3: row = 6, col = 10 → 加载 sB[6][10]
```

---

```cuda
        __syncthreads(); // 等待数据全部加载到共享内存
```

**解读**：
**同步点 1**：确保所有线程完成加载，才能开始计算

---

## 5. 核心计算：寄存器分块外积

```cuda
        // --- 步骤 C: 寄存器分块计算 ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
```

**解读**：
**K 维度内层循环**，k 从 0 到 7（共 8 次迭代）

`#pragma unroll` 让编译器展开这 8 次循环，消除分支开销

---

```cuda
            // 将 A 和 B 的数据从共享内存拉取到寄存器中
            float frag_a[TM];  // frag_a[8]
            float frag_b[TN];  // frag_b[8]
```

**解读**：
**关键优化：寄存器缓存！**

**设计原理**：
- `frag_a[8]`：缓存 sA 的 8 个元素（当前线程需要的 8 行）
- `frag_b[8]`：缓存 sB 的 8 个元素（当前线程需要的 8 列）
- 这 16 个 float 全部存储在寄存器中

**对比 Shared Kernel**：
```cuda
// Shared Kernel：每次从共享内存读取
tmp += sA[ty][k] * sB[k][tx];  // 2 次共享内存访问

// Register Kernel：从寄存器读取
accum[i][j] += frag_a[i] * frag_b[j];  // 0 次共享内存访问
```

---

```cuda
            #pragma unroll
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];
```

**解读**：
**批量加载到寄存器**

**加载模式**：
```
对于 Thread(ty, tx)：

frag_a[i] = sA[ty * 8 + i][k]  // i=0~7
表示加载该线程负责的 8 行中第 i 行、第 k 列的元素

frag_b[j] = sB[k][tx * 8 + j]  // j=0~7
表示加载该线程负责的 8 列中第 j 列、第 k 行的元素
```

**示例**：
```
Thread(3, 5), k=2:
frag_a[0] = sA[40][2]  // ty*8 + 0 = 40
frag_a[1] = sA[41][2]
...
frag_a[7] = sA[47][2]

frag_b[0] = sB[2][24]  // tx*8 + 0 = 24
frag_b[1] = sB[2][25]
...
frag_b[7] = sB[2][31]
```

**关键点**：
- 只需 16 次共享内存读取（8+8）
- 后续 64 次计算全部从寄存器读取

---

```cuda
            // 在寄存器中完成 8x8 的外积 (FFMA)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }
```

**解读**：
**外积计算（核心！）**

**数学原理**：
```
accum[i][j] += frag_a[i] * frag_b[j]

表示：
C[row_start + i][col_start + j] += A[row][k] * B[k][col]

其中：
- frag_a[i] 对应 A 矩阵的一个元素
- frag_b[j] 对应 B 矩阵的一个元素
- frag_a[i] * frag_b[j] 是它们的外积贡献到 C 的一个元素
```

**计算统计**：
```
8 × 8 = 64 次 FMA（乘加）
每次 k 迭代执行 64 次 FMA
共 8 次 k 迭代
总计：512 次 FMA 每 k_step
```

**指令级并行 (ILP)**：
- 64 个 FMA 指令相互独立
- GPU 可以同时发射多个 FMA
- 计算单元可以流水线执行

---

```cuda
        __syncthreads(); // 等待当前块计算完成，再进入下一个 K 步
    }
```

**解读**：
**同步点 2**：防止快线程覆盖慢线程还在使用的共享内存数据

---

## 6. 结果写回全局内存

```cuda
    // 4. 将寄存器中的累加结果写回全局内存
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_row = row_start + i;
            int global_col = col_start + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = alpha * accum[i][j] + beta * C[global_row * N + global_col];
            }
        }
    }
}
```

**解读**：
**将 64 个累加器结果写回全局内存**

**循环展开**：
`#pragma unroll` 展开 8×8 = 64 次循环，消除分支

**地址计算**：
```
global_row = row_start + i = by*128 + ty*8 + i
global_col = col_start + j = bx*128 + tx*8 + j

C[global_row * N + global_col]：行主序寻址
```

**GEMM 完整公式**：
```cuda
C[row][col] = alpha * accum[i][j] + beta * C[row][col]
```
- `alpha`：缩放矩阵乘法结果
- `beta`：缩放原有 C 矩阵值（常用于累加）

**边界检查**：
```cuda
if (global_row < M && global_col < N)
```
确保不写入越界区域

---

## 7. 启动配置

```cuda
// 包装函数
void run_sgemm_register(int M, int N, int K, float alpha, const float *A, 
                        const float *B, float beta, float *C) {
    // 使用 16x16 的线程块，每个线程计算 8x8，因此一个 Block 负责计算 128x128 的 C 矩阵块
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
```

**解读**：
**Kernel 启动配置**

### 7.1 Block 配置
```cuda
dim3 block(16, 16);  // 256 线程 / Block
```
- 16×16 = 256 线程
- 每个线程计算 8×8 = 64 元素
- 每 Block 计算 128×128 = 16,384 元素

### 7.2 Grid 配置
```cuda
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
```

**向上取整计算**：
```
grid.x = ceil(N / 128)
grid.y = ceil(M / 128)

示例：M=N=4096
grid.x = (4096 + 127) / 128 = 32
grid.y = (4096 + 127) / 128 = 32
总 Block 数 = 32 × 32 = 1024
```

### 7.3 总线程数计算
```
总线程数 = grid.x × grid.y × block.x × block.y
        = 32 × 32 × 16 × 16
        = 262,144 线程

总计算量 = 262,144 × 64 = 16,777,216 元素
验证：4096 × 4096 = 16,777,216 ✓
```

---

## 8. 优化总结

### 8.1 相比 Shared Kernel 的改进

| 优化点 | 具体实现 | 效果 |
|--------|---------|------|
| **双层分块** | Block 128×128 + Thread 8×8 | 每线程工作量 64× |
| **寄存器缓存** | frag_a[8], frag_b[8], accum[8][8] | 共享内存访问 4×↓ |
| **小 BK 策略** | BK=8（vs 32） | 更多数据进寄存器 |
| **外积计算** | 8×8=64 FMA/迭代 | 计算密度 8×↑ |
| **向量化加载** | 256 线程协作加载 1024 元素 | 带宽利用率↑ |
| **循环展开** | #pragma unroll 所有内层循环 | 消除分支开销 |

### 8.2 关键代码片段总结

```cuda
// 1. 分块参数（双层）
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// 2. 寄存器数组（核心优化）
float accum[TM][TN] = {0.0f};  // 64 个累加器

// 3. 协作加载（256 线程分工）
int tid = ty * blockDim.x + tx;
int load_a_row = tid / BK;
int load_a_col = tid % BK;

// 4. 寄存器缓存
float frag_a[TM];
float frag_b[TN];
#pragma unroll
for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
#pragma unroll
for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];

// 5. 外积计算（核心）
#pragma unroll
for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
        accum[i][j] += frag_a[i] * frag_b[j];
```

---

## 9. 完整代码（带详细注释）

```cuda
// ==========================================
// 算子 3: 二维寄存器分块 GEMM (Register Tiling)
// ==========================================

// 分块参数定义
#define BM 128  // Block 在 M 维度负责 128 行
#define BN 128  // Block 在 N 维度负责 128 列
#define BK 8    // K 维度步长为 8（关键：小步长让更多数据进寄存器）
#define TM 8    // 每个线程负责 8 行
#define TN 8    // 每个线程负责 8 列（共 64 元素）

__global__ void sgemm_register_kernel(int M, int N, int K, float alpha, 
                                      const float *A, const float *B, 
                                      float beta, float *C) {
    // ========== 索引计算 ==========
    int bx = blockIdx.x;           // Block 在 N 维度的索引
    int by = blockIdx.y;           // Block 在 M 维度的索引
    int tx = threadIdx.x;          // Thread 在 Block 内的列索引（0~15）
    int ty = threadIdx.y;          // Thread 在 Block 内的行索引（0~15）
    int tid = ty * 16 + tx;        // Thread 的一维 ID（0~255）
    
    // ========== 内存申请 ==========
    // 共享内存：sA[128][8], sB[8][128]，共 8KB
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    
    // 寄存器数组：64 个累加器 + 16 个缓存，共 80 个 float
    float accum[TM][TN] = {0.0f};  // 64 个累加器，初始化为 0
    
    // 当前线程负责的 C 子矩阵起始坐标
    int row_start = by * BM + ty * TM;  // 全局行坐标
    int col_start = bx * BN + tx * TN;  // 全局列坐标
    
    // 协作加载坐标（每线程负责 4 个元素）
    int load_a_row = tid / BK;         // A 的行坐标
    int load_a_col = tid % BK;         // A 的列坐标
    int load_b_row = tid / BN;         // B 的行坐标
    int load_b_col = tid % BN;         // B 的列坐标
    
    // ========== 主循环：沿 K 维度滑动 ==========
    for (int k_step = 0; k_step < (K + BK - 1) / BK; ++k_step) {
        int k_offset = k_step * BK;    // 当前 K 块的起始位置
        
        // --- 协作加载 A 到共享内存（256 线程 × 4 元素 = 1024 元素）---
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = load_a_row + i * 32;
            int col = load_a_col;
            int global_row = by * BM + row;
            int global_col = k_offset + col;
            if (global_row < M && global_col < K)
                sA[row][col] = A[global_row * K + global_col];
            else
                sA[row][col] = 0.0f;
        }
        
        // --- 协作加载 B 到共享内存（256 线程 × 4 元素 = 1024 元素）---
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = load_b_row + i * 2;
            int col = load_b_col;
            int global_row = k_offset + row;
            int global_col = bx * BN + col;
            if (global_row < K && global_col < N)
                sB[row][col] = B[global_row * N + global_col];
            else
                sB[row][col] = 0.0f;
        }
        
        __syncthreads();  // 同步：等待所有线程完成加载
        
        // --- 寄存器分块计算（核心优化）---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 1. 从共享内存加载到寄存器（16 次读取）
            float frag_a[TM];
            float frag_b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) frag_a[i] = sA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = sB[k][tx * TN + j];
            
            // 2. 外积计算（64 次 FMA，全部从寄存器读取）
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    accum[i][j] += frag_a[i] * frag_b[j];
        }
        
        __syncthreads();  // 同步：等待所有线程完成计算
    }
    
    // ========== 写回结果 ==========
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = row_start + i;
            int col = col_start + j;
            if (row < M && col < N)
                C[row * N + col] = alpha * accum[i][j] + beta * C[row * N + col];
        }
}

// 启动函数
void run_sgemm_register(int M, int N, int K, float alpha, const float *A, 
                        const float *B, float beta, float *C) {
    dim3 block(16, 16);                           // 256 线程 / Block
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // 向上取整
    sgemm_register_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
```

---

*文档生成时间：2026年3月17日*

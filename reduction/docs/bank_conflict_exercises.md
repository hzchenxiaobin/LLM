# CUDA Bank Conflict 练习题

## 说明

本练习题旨在帮助你深入理解 Bank Conflict 的概念、产生机制和解决方案。建议先阅读 `bank_conflict_tutorial.md` 后再完成练习。

**难度标识**：
- ⭐ 基础题
- ⭐⭐ 进阶题
- ⭐⭐⭐ 挑战题

---

## 第一部分：判断题（True/False）

### 题目 1 ⭐
**Bank Conflict 只会在写入共享内存时发生，读取时不会发生。**

<details>
<summary>点击显示答案</summary>

**False**

Bank Conflict 在读取和写入时都会发生。只要同一个 Warp 内的多个线程同时访问同一个 Bank 的不同地址，无论是读还是写，都会产生冲突。

</details>

---

### 题目 2 ⭐
**如果 Warp 内的所有线程都访问共享内存的同一个地址，会产生 Bank Conflict。**

<details>
<summary>点击显示答案</summary>

**False**

这是广播（Broadcast）模式，不会产生冲突。GPU 硬件支持将一个 Bank 的数据广播给多个线程，这是无冲突的。

```
情况 1（广播，无冲突）:
tid 0,1,2,3 都访问 sdata[0] (Bank 0)
→ 广播，1 个时钟周期完成

情况 2（冲突）:
tid 0 访问 sdata[0] (Bank 0)
tid 1 访问 sdata[32] (Bank 0)  ← 冲突！
→ 2-way conflict，2 个时钟周期
```

</details>

---

### 题目 3 ⭐
**在计算能力 7.0+ 的 GPU 上，共享内存被划分为 64 个 Bank。**

<details>
<summary>点击显示答案</summary>

**False**

从 Fermi 到 Hopper 架构，默认情况下共享内存都是 32 个 Bank（每个 Bank 4 字节）。虽然可以通过 `cudaDeviceSetSharedMemConfig` 配置为 16 个 Bank（8 字节模式），但默认是 32 个 Bank。

</details>

---

### 题目 4 ⭐⭐
**以下代码不会产生 Bank Conflict：**
```cuda
__shared__ float sdata[64];
int tid = threadIdx.x;
float val = sdata[tid * 2];
```

<details>
<summary>点击显示答案</summary>

**True**

分析：
- tid 0: 访问 sdata[0] → Bank 0
- tid 1: 访问 sdata[2] → Bank 2
- tid 2: 访问 sdata[4] → Bank 4
- ...
- tid 15: 访问 sdata[30] → Bank 30

每个线程访问不同的 Bank（间隔为 2），所以没有冲突。

</details>

---

### 题目 5 ⭐⭐
**Sequential Addressing（连续线程访问连续地址）一定不会有 Bank Conflict。**

<details>
<summary>点击显示答案</summary>

**True**

连续线程访问连续地址时，线程 tid 访问 address tid，Bank ID 为 `tid % 32`。Warp 内的 32 个线程分别访问 Bank 0-31，每个 Bank 只被一个线程访问，因此无冲突。

```
tid 0 → address 0 → Bank 0
tid 1 → address 1 → Bank 1
...
tid 31 → address 31 → Bank 31
```

</details>

---

## 第二部分：计算分析题

### 题目 6 ⭐
**计算以下地址对应的 Bank ID（假设 32 个 Bank，每个 Bank 4 字节）：**
- 地址 0
- 地址 16
- 地址 128
- 地址 132

<details>
<summary>点击显示答案</summary>

公式：`Bank ID = (address / 4) % 32`

| 地址 | 计算 | Bank ID |
|------|------|---------|
| 0 | (0/4) % 32 = 0 | 0 |
| 16 | (16/4) % 32 = 4 | 4 |
| 128 | (128/4) % 32 = 0 | 0 |
| 132 | (132/4) % 32 = 1 | 1 |

**注意**：地址 0 和 128 都在 Bank 0，如果同时访问会产生冲突！

</details>

---

### 题目 7 ⭐⭐
**分析以下代码会产生几 way 的 Bank Conflict？**

```cuda
__shared__ float sdata[256];
int tid = threadIdx.x;  // tid 范围 0-31
float val = sdata[tid * 32];
```

<details>
<summary>点击显示答案</summary>

**32-way Bank Conflict！**

分析：
- tid 0: address = 0 * 4 = 0 → Bank 0
- tid 1: address = 32 * 4 = 128 → Bank 0 (128/4=32, 32%32=0)
- tid 2: address = 64 * 4 = 256 → Bank 0 (256/4=64, 64%32=0)
- ...
- tid 31: address = 992 * 4 = 3968 → Bank 0

所有 32 个线程都访问 Bank 0 的不同地址，产生 32-way conflict，需要 32 个时钟周期串行完成！

</details>

---

### 题目 8 ⭐⭐
**分析以下结构体访问模式，判断是否有 Bank Conflict：**

```cuda
struct Point {
    float x;
    float y;
};

__shared__ Point points[32];
int tid = threadIdx.x;
float val = points[tid].x;  // 访问 x 坐标
```

<details>
<summary>点击显示答案</summary>

**无冲突！**

分析结构体内存布局：
```
points[0]: x at 0, y at 4
points[1]: x at 8, y at 12
points[2]: x at 16, y at 20
...

points[tid].x 的地址 = tid * 8
Bank = (tid * 8 / 4) % 32 = tid * 2 % 32

结果：
tid 0 → Bank 0
tid 1 → Bank 2
tid 2 → Bank 4
...
tid 15 → Bank 30
tid 16 → Bank 0 (32%32=0) ← 和 tid 0 冲突！
```

Wait！实际上 Warp 只有 32 个线程，tid 范围 0-31：
- tid 0 → Bank 0
- tid 1 → Bank 2
- ...
- tid 15 → Bank 30
- tid 16 → Bank 0 (32%32=0) ← 与 tid 0 都是 Bank 0

所以实际上有 2-way conflict！

**正确答案**：有 2-way Bank Conflict（线程 0 和 16 都访问 Bank 0，线程 1 和 17 都访问 Bank 2，等等）

</details>

---

### 题目 9 ⭐⭐⭐
**矩阵转置中的 Bank Conflict 分析：**

```cuda
#define TILE_DIM 32
__shared__ float tile[TILE_DIM][TILE_DIM];

// 写入列（转置操作）
int col = threadIdx.x;
for (int row = 0; row < TILE_DIM; row++) {
    tile[row][col] = data[row];  // 同一列的元素
}
```

**问题**：
1. 这段代码是否有 Bank Conflict？
2. 如果有，是几 way 的冲突？
3. 如何修复？

<details>
<summary>点击显示答案</summary>

**1. 有 Bank Conflict！**

**2. 分析冲突程度：**
```
tile[row][col] 的地址 = (row * 32 + col) * 4
Bank = (row * 32 + col) % 32 = col % 32

对于固定的 col：
- row 0: tile[0][col] → Bank col
- row 1: tile[1][col] → Bank col (32+col)%32 = col
- row 2: tile[2][col] → Bank col (64+col)%32 = col
...
- row 31: tile[31][col] → Bank col

32 个 row 循环，每次 Warp 的 32 个线程都访问同一个 Bank！
→ 32-way conflict！
```

**3. 修复方案（Padding）：**
```cuda
__shared__ float tile[TILE_DIM][TILE_DIM + 1];  // 添加 1 列 padding
```

修复后：
```
地址 = (row * 33 + col) * 4
Bank = (row * 33 + col) % 32

row 0: Bank col
row 1: Bank (33 + col) % 32 = (1 + col) % 32  ← 不同！
row 2: Bank (66 + col) % 32 = (2 + col) % 32  ← 不同！
...
现在每个 row 访问不同的 Bank，无冲突！
```

</details>

---

## 第三部分：代码分析题

### 题目 10 ⭐⭐
**分析以下两种归约代码的 Bank Conflict 情况：**

**版本 A：**
```cuda
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**版本 B：**
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**问题**：
1. 哪个版本有 Bank Conflict？
2. 冲突发生在哪些轮次？
3. 冲突程度如何？

<details>
<summary>点击显示答案</summary>

**版本 A（Interleaved Addressing）分析：**

```
s=1: tid 0,2,4,6... 访问 sdata[tid] 和 sdata[tid+1]
  - 读取 sdata[0](Bank0), sdata[1](Bank1)
  - 读取 sdata[2](Bank2), sdata[3](Bank3)
  → 无冲突

s=2: tid 0,4,8... 访问 sdata[tid] 和 sdata[tid+2]
  - tid 0: sdata[0](Bank0) + sdata[2](Bank2)
  - tid 4: sdata[8](Bank8) + sdata[10](Bank10)
  → 无冲突
  
实际上这个版本在归约的前几轮没有严重冲突
冲突主要发生在后期，当活跃线程减少时
```

**版本 B（Sequential Addressing）分析：**

```
s=16 (假设 blockSize=32):
  tid 0: sdata[0](Bank0) + sdata[16](Bank16) - 不同 ✓
  tid 1: sdata[1](Bank1) + sdata[17](Bank17) - 不同 ✓
  ...
  → 无冲突

s=8:
  tid 0: sdata[0](Bank0) + sdata[8](Bank8) - 不同 ✓
  ...
  → 无冲突

只要 s < 32 且不是 32 的约数导致重复，就无冲突
实际上对于标准的 2 的幂次方的 blockSize，Sequential Addressing 基本无冲突
```

**结论**：
- 版本 A 可能有轻微冲突（取决于具体访问模式）
- 版本 B 在大多数轮次无冲突，是更好的选择

</details>

---

### 题目 11 ⭐⭐⭐
**以下代码尝试实现向量化加载以避免 Bank Conflict，但有 bug，请找出并修复：**

```cuda
__global__ void buggyKernel(float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // 尝试向量化加载 4 个 float
    float4 *g_idata_f4 = reinterpret_cast<float4*>(g_idata);
    float4 vec = g_idata_f4[tid];  // 每个线程加载 1 个 float4
    
    sdata[tid * 4 + 0] = vec.x;
    sdata[tid * 4 + 1] = vec.y;
    sdata[tid * 4 + 2] = vec.z;
    sdata[tid * 4 + 3] = vec.w;
    
    __syncthreads();
    // ... 后续处理
}
```

**问题**：
1. 这段代码有什么 Bank Conflict 问题？
2. 如何修复？

<details>
<summary>点击显示答案</summary>

**问题分析：**

问题在于写入共享内存的顺序：
```
线程 tid 写入：
- sdata[tid * 4 + 0] (Bank: (tid*4)%32)
- sdata[tid * 4 + 1] (Bank: (tid*4+1)%32)
- sdata[tid * 4 + 2] (Bank: (tid*4+2)%32)
- sdata[tid * 4 + 3] (Bank: (tid*4+3)%32)

假设 Warp 有 32 个线程：
tid 0: Bank 0, 1, 2, 3
tid 1: Bank 4, 5, 6, 7
tid 2: Bank 8, 9, 10, 11
...
tid 7: Bank 28, 29, 30, 31
tid 8: Bank 0, 1, 2, 3  ← 与 tid 0 冲突！

tid 0 和 tid 8 都写入 Bank 0-3，产生 2-way conflict
```

**修复方案：**

方案 1：改变存储顺序，使用 Sequential 模式
```cuda
// 将 4 个 float 分散到连续的共享内存位置
// 使用合并存储
sdata[tid] = vec.x;                    // tid 0-31
sdata[tid + blockDim.x] = vec.y;       // tid 32-63
sdata[tid + 2*blockDim.x] = vec.z;     // tid 64-95
sdata[tid + 3*blockDim.x] = vec.w;     // tid 96-127
```

方案 2：使用 float4 直接存储到共享内存
```cuda
// 如果共享内存按 float4 对齐
float4 *sdata_f4 = reinterpret_cast<float4*>(sdata);
sdata_f4[tid] = vec;
```

</details>

---

## 第四部分：编程实践题

### 题目 12 ⭐⭐
**实现一个无 Bank Conflict 的矩阵转置核函数。**

**要求**：
1. 使用共享内存缓存
2. 避免 Bank Conflict
3. 处理任意大小的矩阵（非 32 的倍数）

**提示**：
- 使用 TILE 分块策略
- 使用 Padding 避免冲突
- 处理边界条件

<details>
<summary>点击显示参考答案</summary>

```cuda
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeNoBankConfict(float *g_idata, float *g_odata, 
                                        int width, int height) {
    // 使用 padding 避免 Bank Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 是关键！
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 读取行（无冲突，因为连续线程访问连续地址）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = 
                g_idata[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // 计算转置后的坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 写入列（因为 padding，无冲突）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            g_odata[(y + j) * height + x] = 
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

**关键点**：
1. `tile[TILE_DIM][TILE_DIM + 1]` 的 +1 padding 消除了列访问冲突
2. 读取行时连续访问，无冲突
3. 写入列时因为有 padding，每个线程访问不同 Bank

</details>

---

### 题目 13 ⭐⭐⭐
**实现一个 Bank Conflict Free 的并行前缀和（Scan）算法。**

**要求**：
1. 使用共享内存
2. 完全无 Bank Conflict
3. 处理 block 级别的扫描（无需处理跨 block）

**提示**：
- 使用 Sequential Addressing
- 使用 Kogge-Stone 或 Brent-Kung 算法
- 注意树形归约/扫描的模式

<details>
<summary>点击显示参考答案</summary>

```cuda
// Kogge-Stone 扫描算法（无 Bank Conflict 版本）
__global__ void scanKoggeStone(float *g_idata, float *g_odata, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    
    // 加载到共享内存（Sequential，无冲突）
    temp[tid] = (tid < n) ? g_idata[tid] : 0;
    __syncthreads();
    
    // Kogge-Stone 扫描（使用 offset 为 2^k 的模式）
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0;
        if (tid >= offset) {
            // 读取 tid 和 tid-offset（Sequential 模式）
            val = temp[tid] + temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp[tid] = val;
        }
        __syncthreads();
    }
    
    // 写回结果
    if (tid < n) {
        g_odata[tid] = temp[tid];
    }
}

// 更优的版本：使用 Brent-Kung 算法减少同步次数
__global__ void scanBrentKung(float *g_idata, float *g_odata, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    
    temp[tid] = (tid < n) ? g_idata[tid] : 0;
    __syncthreads();
    
    // 向上归约阶段（Reduction）
    for (int d = 1; d < blockDim.x; d *= 2) {
        int offset = tid * 2 * d;
        if (offset + d < blockDim.x) {
            temp[offset + 2*d - 1] += temp[offset + d - 1];
        }
        __syncthreads();
    }
    
    // 清除最后一个元素（Exclusive Scan）
    if (tid == 0) temp[blockDim.x - 1] = 0;
    __syncthreads();
    
    // 向下分发阶段（Distribution）
    for (int d = blockDim.x / 2; d >= 1; d /= 2) {
        int offset = tid * 2 * d;
        if (offset + d < blockDim.x) {
            float t = temp[offset + d - 1];
            temp[offset + d - 1] = temp[offset + 2*d - 1];
            temp[offset + 2*d - 1] += t;
        }
        __syncthreads();
    }
    
    if (tid < n) {
        g_odata[tid] = temp[tid];
    }
}
```

**注意**：Brent-Kung 算法虽然同步次数少，但要注意其访问模式可能有轻微冲突。Kogge-Stone 更清晰但同步多。实际选择需要权衡。

</details>

---

## 第五部分：思考题

### 题目 14 ⭐⭐⭐
**在现代 GPU（如 A100/H100）上，Bank Conflict 的影响是否越来越小？为什么？**

<details>
<summary>点击显示答案</summary>

**是的，Bank Conflict 的影响确实在减小，但并未消失。**

**原因：**

1. **共享内存容量增加**
   - V100: 96 KB/SM
   - A100: 164 KB/SM
   - H100: 228 KB/SM
   - 更大的容量允许更激进的 padding 和算法优化

2. **更好的硬件仲裁**
   - 新架构有更智能的 Bank 访问仲裁机制
   - 可以更好地处理轻度冲突

3. **L1 缓存效率提升**
   - 共享内存访问可能通过 L1 缓存
   - 缓存可以平滑 Bank 访问压力

**但是，Bank Conflict 仍然重要：**

1. **共享内存仍然是关键资源**
   - 延迟仍然远低于全局内存
   - 高效使用共享内存是性能优化的基础

2. **极端冲突仍然严重影响性能**
   - 32-way conflict 仍然会导致 32 倍延迟
   - 在计算密集型 kernel 中，这可能成为瓶颈

3. **算法设计原则不变**
   - Sequential Addressing 仍然是最优模式
   - 理解 Bank 机制有助于设计更高效的算法

**结论**：虽然硬件进步减轻了 Bank Conflict 的惩罚，但作为 CUDA 程序员，理解并避免 Bank Conflict 仍然是优化技能的重要组成部分。

</details>

---

### 题目 15 ⭐⭐⭐
**设计一个测试程序，自动检测任意共享内存访问模式的 Bank Conflict 程度。**

**要求**：
1. 输入：访问模式函数（如 `index = tid * stride`）
2. 输出：最大 conflict degree（1-way 到 32-way）
3. 可以测试不同的 blockSize 和访问步长

**提示**：
- 使用 Nsight Compute metrics
- 或者使用软件模拟计算 Bank 分布

<details>
<summary>点击显示参考答案</summary>

```cuda
#include <stdio.h>
#include <algorithm>

// 软件模拟计算 Bank Conflict
template<int BLOCK_SIZE>
__host__ int calculateBankConflict(int (*accessFunc)(int)) {
    int bankCount[32] = {0};
    int maxConflict = 1;
    
    // 模拟 Warp 内所有线程的访问
    for (int tid = 0; tid < 32 && tid < BLOCK_SIZE; tid++) {
        int index = accessFunc(tid);
        int address = index * sizeof(float);
        int bank = (address / 4) % 32;
        bankCount[bank]++;
        maxConflict = std::max(maxConflict, bankCount[bank]);
    }
    
    printf("Bank distribution (BLOCK_SIZE=%d):\n", BLOCK_SIZE);
    for (int i = 0; i < 32; i++) {
        if (bankCount[i] > 0) {
            printf("  Bank %2d: %d threads\n", i, bankCount[i]);
        }
    }
    printf("Max conflict: %d-way\n\n", maxConflict);
    
    return maxConflict;
}

// 测试不同的访问模式
int access_linear(int tid) { return tid; }           // Sequential
int access_stride2(int tid) { return tid * 2; }      // Stride 2
int access_stride32(int tid) { return tid * 32; }    // Stride 32 (冲突！)
int access_modulo(int tid) { return tid % 4; }       // 广播模式

int main() {
    printf("=== Bank Conflict Analyzer ===\n\n");
    
    printf("1. Sequential Access (tid):\n");
    calculateBankConflict<32>(access_linear);
    // 预期: 1-way (无冲突)
    
    printf("2. Stride 2 (tid * 2):\n");
    calculateBankConflict<32>(access_stride2);
    // 预期: 1-way (无冲突)
    
    printf("3. Stride 32 (tid * 32):\n");
    calculateBankConflict<32>(access_stride32);
    // 预期: 32-way (严重冲突！)
    
    printf("4. Modulo 4 (tid %% 4):\n");
    calculateBankConflict<32>(access_modulo);
    // 预期: 8-way (广播模式，tid 0-3 都访问 Bank 0-3，然后重复)
    
    // 使用 Nsight Compute 验证（实际 GPU 运行）
    printf("\n=== GPU Verification with Nsight Compute ===\n");
    printf("Run: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./program\n");
    
    return 0;
}
```

**使用方法**：

1. **软件模拟**：编译运行 CPU 版本，快速验证访问模式
2. **GPU 验证**：使用 Nsight Compute 实际测量

```bash
# 编译
nvcc bank_conflict_analyzer.cu -o analyzer

# 运行 CPU 分析
./analyzer

# GPU 验证（示例）
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./analyzer
```

</details>

---

## 附录：参考答案速查表

| 题号 | 难度 | 核心知识点 | 答案概要 |
|------|------|-----------|---------|
| 1 | ⭐ | 读写冲突 | False，读写都会产生冲突 |
| 2 | ⭐ | 广播模式 | False，广播无冲突 |
| 3 | ⭐ | Bank 数量 | False，默认 32 个 Bank |
| 4 | ⭐ | 间隔访问 | True，间隔为 2，无冲突 |
| 5 | ⭐ | Sequential | True，连续访问无冲突 |
| 6 | ⭐ | Bank 计算 | 0, 4, 0, 1 |
| 7 | ⭐⭐ | 严重冲突 | 32-way conflict |
| 8 | ⭐⭐ | 结构体布局 | 2-way conflict |
| 9 | ⭐⭐⭐ | 矩阵转置 | 32-way，用 padding 修复 |
| 10 | ⭐⭐ | 归约模式 | 版本 B 更优 |
| 11 | ⭐⭐⭐ | 向量化 | tid 0 和 8 冲突，用 strided 存储修复 |
| 12 | ⭐⭐ | 编程题 | TILE + padding |
| 13 | ⭐⭐⭐ | 编程题 | Kogge-Stone 或 Brent-Kung |
| 14 | ⭐⭐⭐ | 架构演进 | 影响减小但仍重要 |
| 15 | ⭐⭐⭐ | 编程题 | 软件模拟 + Nsight Compute |

---

## 结语

完成这些练习后，你应该能够：

1. ✅ 快速判断任意访问模式是否会产生 Bank Conflict
2. ✅ 计算 Conflict 的严重程度（N-way）
3. ✅ 设计避免 Bank Conflict 的数据结构和算法
4. ✅ 使用工具检测和分析 Bank Conflict
5. ✅ 在实际 CUDA 编程中应用这些知识

**建议下一步**：
- 使用 Nsight Compute 分析实际 CUDA 程序中的 Bank Conflict
- 尝试优化你现有的 CUDA kernel，消除 Bank Conflict
- 阅读更多关于共享内存优化的 CUDA 示例

---

## 参考文档

- `bank_conflict_tutorial.md` - Bank Conflict 深度解析
- `reduce_v2_strided_explanation.md` - V2 版本分析
- `reduce_v3_sequential_explanation.md` - V3 版本分析
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

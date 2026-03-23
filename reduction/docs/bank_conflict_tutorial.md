# CUDA Bank Conflict 深度解析教程

## 目录
1. [什么是 Bank Conflict](#1-什么是-bank-conflict)
2. [GPU 共享内存架构](#2-gpu-共享内存架构)
3. [Bank Conflict 的产生机制](#3-bank-conflict-的产生机制)
4. [Reduction 中的 Bank Conflict 案例](#4-reduction-中的-bank-conflict-案例)
5. [如何检测 Bank Conflict](#5-如何检测-bank-conflict)
6. [避免 Bank Conflict 的技巧](#6-避免-bank-conflict-的技巧)
7. [不同架构的 Bank 配置](#7-不同架构的-bank-配置)
8. [总结](#8-总结)

---

## 1. 什么是 Bank Conflict

### 通俗理解

**Bank Conflict（存储体冲突）** 是 CUDA 编程中影响共享内存性能的关键问题。简单来说：

> 当同一个 Warp 中的多个线程**同时访问同一个 Bank** 的不同地址时，就会产生 Bank Conflict，导致内存访问被串行化，降低性能。

```
类比：超市收银台

┌─────────────────────────────────────────────────────────────┐
│                    超市收银场景                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  理想情况（无冲突）:                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                          │
│  │ 台1 │ │ 台2 │ │ 台3 │ │ 台4 │  ← 4 个顾客去不同收银台    │
│  │  A  │ │  B  │ │  C  │ │  D  │  → 同时结账，无等待       │
│  └─────┘ └─────┘ └─────┘ └─────┘                          │
│                                                             │
│  Bank Conflict（冲突）:                                      │
│  ┌─────┐                                                   │
│  │ 台1 │  ← 4 个顾客都挤到同一个收银台                     │
│  │ A,B │                                                    │
│  │ C,D │  → 必须排队，串行结账                              │
│  └─────┘  → 性能下降！                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 技术定义

```
共享内存被划分为 32 个 Bank（计算能力 3.0+）
每个 Bank 每个时钟周期只能服务一个访问请求
如果 Warp 内的多个线程同时访问同一个 Bank，就会产生冲突
冲突的访问会被串行处理，增加访问延迟
```

---

## 2. GPU 共享内存架构

### 2.1 共享内存的分层结构

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 内存架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   全局内存 (Global Memory)            │   │
│  │            (慢，但容量大，所有线程可访问)            │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↕ (数据流向)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              L2 缓存 (所有 SM 共享)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↕                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              SM (Streaming Multiprocessor)   │   │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │   │
│  │  │  │  Warp 0 │ │  Warp 1 │ │  Warp 2 │ ...    │   │   │
│  │  │  └────┬────┘ └────┬────┘ └────┬────┘        │   │   │
│  │  │       └───────────┴───────────┘              │   │   │
│  │  │                   ↓                          │   │   │
│  │  │  ┌─────────────────────────────────────────┐ │   │   │
│  │  │  │           共享内存 (Shared Memory)       │ │   │   │
│  │  │  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐  │ │   │   │
│  │  │  │  │Bank0│Bank1│Bank2│ ... │Bank30│Bank31│ │ │   │   │
│  │  │  │  │  0  │  1  │  2  │     │  30  │  31  │ │ │   │   │
│  │  │  │  │  32 │  33 │  34 │     │  62  │  63  │ │ │   │   │
│  │  │  │  │  64 │  65 │  66 │     │  94  │  95  │ │ │   │   │
│  │  │  │  └─────┴─────┴─────┴─────┴─────┴─────┘  │ │   │   │
│  │  │  └─────────────────────────────────────────┘ │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  • 每个 SM 有独立的共享内存                                 │
│  • 共享内存划分为 32 个 Bank                                │
│  • Bank 宽度：4 字节（32-bit）                              │
│  • 总容量：通常 48KB-164KB（依架构而定）                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Bank 地址映射

```
共享内存地址到 Bank 的映射公式：
Bank ID = (address / 4) % 32

示例（假设 float = 4 字节）:
地址 0:  Bank (0/4)%32 = 0  → Bank 0
地址 4:  Bank (4/4)%32 = 1  → Bank 1
地址 8:  Bank (8/4)%32 = 2  → Bank 2
...
地址 124: Bank (124/4)%32 = 31 → Bank 31
地址 128: Bank (128/4)%32 = 0  → Bank 0 (循环)

可视化：
┌─────────────────────────────────────────────────────────────┐
│  Bank  0  │  1  │  2  │ ... │ 30 │ 31 │  0  │  1  │  2  │  │
│  ─────────┼─────┼─────┼─────┼────┼────┼─────┼─────┼─────┼───│
│  float[0] │ [1] │ [2] │     │[30]│[31]│[32] │[33] │[34] │  │
│  地址 0   │  4  │  8  │     │120 │124 │ 128 │ 132 │ 136 │  │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 访问模式分类

```
┌─────────────────────────────────────────────────────────────┐
│                    Bank 访问模式分类                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式 1: 广播（Broadcast）- 无冲突                            │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │Bank0│Bank1│Bank2│Bank3│                                 │
│  └─────┴──┬──┴─────┴─────┘                                 │
│           │                                                 │
│      ┌────┴────┐                                            │
│      ↓         ↓                                            │
│    tid0      tid1                                          │
│    tid2      tid3                                          │
│    tid4      tid5                                          │
│    ...                                                      │
│  多个线程同时读取同一地址 → 广播，无冲突                      │
│                                                             │
│  【详细解释】为什么广播无冲突？                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Bank Conflict 的本质是：                                 │ │
│  │  "同一 Bank 的同一时钟周期内收到多个不同地址的访问请求"   │ │
│  │                                                         │ │
│  │  广播模式的情况是：                                       │ │
│  │  "同一 Bank 的同一时钟周期内收到同一地址的多次读取请求"   │ │
│  │                                                         │ │
│  │  GPU 硬件支持广播机制：                                   │ │
│  │  1. Bank 0 只需读取一次地址 0 的数据                     │ │
│  │  2. 然后将这个数据同时分发给 Warp 内的所有线程            │ │
│  │  3. 这只需要 1 个时钟周期，没有冲突！                     │ │
│  │                                                         │ │
│  │  类比：                                                   │ │
│  │  • 冲突情况：32 个人分别要买 32 种不同的饮料              │ │
│  │    → 店员需要一个个拿，需要 32 次操作                     │ │
│  │                                                         │ │
│  │  • 广播情况：32 个人都要同一种饮料                       │ │
│  │    → 店员拿一瓶，然后大家分，只需 1 次操作                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  模式 2: 跨 Bank 访问 - 无冲突 ✓                            │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │Bank0│Bank1│Bank2│Bank3│                                 │
│  └──┬──┴──┬──┴──┬──┴──┬──┘                                 │
│     ↓     ↓     ↓     ↓                                     │
│   tid0  tid1  tid2  tid3                                    │
│  每个线程访问不同 Bank → 并行，无冲突                        │
│                                                             │
│  模式 3: Bank Conflict - 有冲突 ✗                          │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │Bank0│Bank1│Bank2│Bank3│                                 │
│  └─────┴─────┴─────┴──┬──┘                                 │
│                       │                                     │
│                  ┌────┴────┐                                │
│                  ↓    ↓    ↓                                │
│                tid0 tid4 tid8                               │
│  多个线程访问同一 Bank 的不同地址 → 串行化                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Bank Conflict 的产生机制

### 3.1 冲突的严重程度

```
N-way Bank Conflict: 表示同一 Bank 被 N 个线程同时访问

1-way (无冲突):   1 个时钟周期完成
2-way:            2 个时钟周期（2 倍延迟）
4-way:            4 个时钟周期（4 倍延迟）
...
32-way:           32 个时钟周期（32 倍延迟！）
```

### 3.2 典型冲突场景分析

#### 场景 1：直接冲突（Direct Conflict）

```cuda
// 场景：线程 tid 访问 sdata[tid * 2]
// 假设 blockSize = 32

__shared__ float sdata[64];
int tid = threadIdx.x;

float val = sdata[tid * 2];  // 每个线程访问间隔为 2
```

```
地址计算：
tid 0: address = 0 * 2 * 4 = 0   → Bank 0
      
tid 1: address = 1 * 2 * 4 = 8   → Bank 2
      
tid 2: address = 2 * 2 * 4 = 16  → Bank 4
      
... (每隔一个 Bank)

这个例子实际没有冲突！因为间隔是 2，Bank ID 差也是 2

但如果是 tid * 32：
tid 0: address = 0  → Bank 0
tid 1: address = 128 → Bank 0 (128/4=32, 32%32=0)  ← 冲突！
tid 2: address = 256 → Bank 0 (256/4=64, 64%32=0)  ← 冲突！
... 32 个线程全部访问 Bank 0 → 32-way conflict！
```

#### 场景 2：结构体导致的冲突

```cuda
// 结构体导致 Bank Conflict 的经典案例

struct MyData {
    float x;  // 4 bytes
    float y;  // 4 bytes
    float z;  // 4 bytes
};  // 总共 12 bytes

__shared__ MyData sdata[32];

// 线程 tid 访问 sdata[tid].x
float val = sdata[tid].x;
```

内存布局分析：

```
┌─────────────────────────────────────────────────────────────┐
│  MyData[0]   MyData[1]   MyData[2]   MyData[3]   ...       │
│  ┌────┐      ┌────┐      ┌────┐      ┌────┐                │
│  │x:0 │      │x:12│      │x:24│      │x:36│                │
│  ├────┤      ├────┤      ├────┤      ├────┤                │
│  │y:4 │      │y:16│      │y:28│      │y:40│                │
│  ├────┤      ├────┤      ├────┤      ├────┤                │
│  │z:8 │      │z:20│      │z:32│      │z:44│                │
│  └────┘      └────┘      └────┘      └────┘                │
│                                                             │
│  Bank 分析（每个 float 占一个 Bank）:                        │
│  sdata[0].x: addr 0   → Bank 0                              │
│  sdata[1].x: addr 12  → Bank 3  (12/4=3)                    │
│  sdata[2].x: addr 24  → Bank 6  (24/4=6)                    │
│  sdata[3].x: addr 36  → Bank 9  (36/4=9)                    │
│                                                             │
│  → 每个线程访问不同 Bank，无冲突 ✓                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

但如果访问 .y：

```
  sdata[0].y: addr 4   → Bank 1   (4/4=1)
  sdata[1].y: addr 16  → Bank 4   (16/4=4, 4%32=4)
  sdata[2].y: addr 28  → Bank 7   (28/4=7)
  ...
  
访问模式更复杂，需要具体分析
```

#### 场景 3：矩阵转置中的冲突

```cuda
// 矩阵转置：读取行，写入列

#define N 32
__shared__ float tile[N][N];

// 读取：行优先，无冲突
for (int j = 0; j < N; j += 4) {
    float4 val = reinterpret_cast<float4*>(&tile[row][j])[0];
}

// 写入：列优先，有冲突！
for (int i = 0; i < N; i++) {
    tile[i][col] = data[i];  // 同一列的元素间隔 N 个 float
}
// 地址 = (i * N + col) * 4
// i=0: col * 4
// i=1: (N + col) * 4 = col*4 + N*4
// 如果 N=32, 则 (N*4)/4 = 32, 32%32 = 0
// → 所有行同一列的元素都在同一个 Bank！
```

---

## 4. Reduction 中的 Bank Conflict 案例

### 4.1 V2 Strided 的 Bank Conflict

```cuda
// ============== V2: Strided Addressing ==============
// 问题版本：存在 Bank Conflict

for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
        sdata[index] += sdata[index + s];  // 冲突发生在这里
    }
    __syncthreads();
}
```

**冲突分析**（blockSize = 32，假设 warp 大小为 32）：

Bank 计算公式：`Bank = (address / 4) % 32 = index % 32`

```
第 1 轮 (s=1):
index = 2 * 1 * tid = 2 * tid

线程访问模式（所有 32 个线程同时执行）：
tid 0: 读取 sdata[0] (Bank 0) 和 sdata[1] (Bank 1)
tid 1: 读取 sdata[2] (Bank 2) 和 sdata[3] (Bank 3)
tid 2: 读取 sdata[4] (Bank 4) 和 sdata[5] (Bank 5)
...
tid 15: 读取 sdata[30] (Bank 30) 和 sdata[31] (Bank 31)
tid 16-31: 被 if (index < 32) 过滤掉，不参与

→ 第 1 轮：16 个线程各自访问不同 Bank，无冲突！

第 2 轮 (s=2):
index = 2 * 2 * tid = 4 * tid

线程访问模式（16 个活跃线程）：
tid 0: 读取 sdata[0] (Bank 0) 和 sdata[2] (Bank 2)
tid 1: 读取 sdata[4] (Bank 4) 和 sdata[6] (Bank 6)
tid 2: 读取 sdata[8] (Bank 8) 和 sdata[10] (Bank 10)
tid 3: 读取 sdata[12] (Bank 12) 和 sdata[14] (Bank 14)
tid 4: 读取 sdata[16] (Bank 16) 和 sdata[18] (Bank 18)
tid 5: 读取 sdata[20] (Bank 20) 和 sdata[22] (Bank 22)
tid 6: 读取 sdata[24] (Bank 24) 和 sdata[26] (Bank 26)
tid 7: 读取 sdata[28] (Bank 28) 和 sdata[30] (Bank 30)
tid 8-15: 被过滤

→ 第 2 轮：8 个线程各自访问不同 Bank，无冲突！

第 3 轮 (s=4):
index = 8 * tid

tid 0: sdata[0](Bank 0) + sdata[4](Bank 4)
tid 1: sdata[8](Bank 8) + sdata[12](Bank 12)
tid 2: sdata[16](Bank 16) + sdata[20](Bank 20)
tid 3: sdata[24](Bank 24) + sdata[28](Bank 28)

→ 第 3 轮：4 个线程，无冲突！

第 4 轮 (s=8):
index = 16 * tid

tid 0: sdata[0](Bank 0) + sdata[8](Bank 8)
tid 1: sdata[16](Bank 16) + sdata[24](Bank 24)

→ 第 4 轮：2 个线程，无冲突！

第 5 轮 (s=16):
index = 32 * tid

tid 0: sdata[0](Bank 0) + sdata[16](Bank 16)

→ 第 5 轮：只有 tid 0 活跃，无冲突！

**结论：当 blockSize = 32 时，V2 实际上没有 Bank Conflict！**
```

**但是，当 blockSize = 64 时，情况完全不同：**

```
第 1 轮 (s=1):
index = 2 * tid

活跃线程：tid 0-31 (32 个线程同时执行)

tid 0: sdata[0](Bank 0)  + sdata[1](Bank 1)
tid 1: sdata[2](Bank 2)  + sdata[3](Bank 3)
...
tid 15: sdata[30](Bank 30) + sdata[31](Bank 31)
tid 16: sdata[32](Bank 0)  + sdata[33](Bank 1)  ← Bank 0 冲突！
tid 17: sdata[34](Bank 2)  + sdata[35](Bank 3)  ← Bank 2 冲突！
...
tid 31: sdata[62](Bank 30) + sdata[63](Bank 31) ← Bank 30 冲突！

→ tid 0 和 tid 16 同时访问 Bank 0
→ tid 1 和 tid 17 同时访问 Bank 2
→ ...
→ 这是 2-way Bank Conflict！
```

**关键发现：**
- `sdata[index]` 的 Bank = `index % 32 = (2*s*tid) % 32`
- 当 `2*s*tid` 跨越 32 边界时，不同 tid 可能映射到相同 Bank
- **V2 的 Bank Conflict 程度取决于 blockSize，当 blockSize > 32 时会出现 2-way conflict**

### 4.2 V3 Sequential 如何解决

```cuda
// ============== V3: Sequential Addressing ==============
// 解决方案：连续线程访问连续地址

for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        // tid 和 tid+s 访问的 Bank 相差 s
        // 只要 s 不是 32 的倍数，就不会冲突
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**分析（blockSize = 32）：**

```
第 1 轮 (s=16):
tid 0: sdata[0](Bank0) + sdata[16](Bank16) - 不同 Bank ✓
tid 1: sdata[1](Bank1) + sdata[17](Bank17) - 不同 Bank ✓
...
tid 15: sdata[15](Bank15) + sdata[31](Bank31) - 不同 Bank ✓

第 2 轮 (s=8):
tid 0: sdata[0](Bank0) + sdata[8](Bank8) - 不同 Bank ✓
tid 1: sdata[1](Bank1) + sdata[9](Bank9) - 不同 Bank ✓
...

只要 s < 32 且 s 不整除 32，就不会有冲突
因为 tid 和 tid+s 的 Bank ID 差为 s，不会相同
```

**但是，当 blockSize = 64 时，情况需要仔细分析：**

```
第 1 轮 (s=32):
index = tid, 读取 sdata[tid] 和 sdata[tid + 32]

活跃线程：tid 0-31 (32 个线程同时在一个 warp 中执行)

tid 0: sdata[0](Bank 0)  + sdata[32](Bank 0)  ← 冲突！
tid 1: sdata[1](Bank 1)  + sdata[33](Bank 1)  ← 冲突！
tid 2: sdata[2](Bank 2)  + sdata[34](Bank 2)  ← 冲突！
...
tid 31: sdata[31](Bank 31) + sdata[63](Bank 31) ← 冲突！

→ 每个线程同时访问同一个 Bank 的两个不同地址！
→ 这是 2-way Bank Conflict！

第 2 轮 (s=16):
index = tid, 读取 sdata[tid] 和 sdata[tid + 16]

tid 0: sdata[0](Bank 0) + sdata[16](Bank 16) - 不同 Bank ✓
tid 1: sdata[1](Bank 1) + sdata[17](Bank 17) - 不同 Bank ✓
...
tid 15: sdata[15](Bank 15) + sdata[31](Bank 31) - 不同 Bank ✓

→ 16 个活跃线程，Bank 相差 16，无冲突！

第 3 轮 (s=8):
tid 0: sdata[0](Bank 0) + sdata[8](Bank 8) - 不同 Bank ✓
...
→ 无冲突！

第 4 轮 (s=4): 无冲突 ✓
第 5 轮 (s=2): 无冲突 ✓
第 6 轮 (s=1): 无冲突 ✓
```

**关键发现：**
- V3 只有在 `s >= 32` 且 `s` 是 32 的倍数时才会出现 Bank Conflict
- 当 blockSize = 64 时，只有第 1 轮 (s=32) 会发生 2-way conflict
- 相比之下，V2 Strided 在多轮中都会发生冲突（当 blockSize > 32 时）
- **V3 显著减少了 Bank Conflict 的发生次数！**

### 4.3 可视化对比

```
V2 Strided 访问模式（跳跃式）:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Bank0│Bank1│Bank2│Bank3│Bank4│ ... │Bank30│Bank31│
└──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
   │     │     │     │     │     │     │     │
  tid0  tid2  tid4  tid6  tid8  ... （跳跃访问）
  
→ 可能多个线程映射到同一 Bank

V3 Sequential 访问模式（连续式）:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Bank0│Bank1│Bank2│Bank3│Bank4│ ... │Bank30│Bank31│
└──┬──┴──┬──┴──┬──┴──┬──┴──┴───┴───┴───┴───┘
   │     │     │     │
  tid0  tid1  tid2  tid3  （连续访问）
  
→ 连续线程访问不同 Bank，无冲突
```

---

## 5. 如何检测 Bank Conflict

### 5.1 使用 Nsight Compute

```bash
# 使用 ncu 命令行工具检测 Bank Conflict
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./your_program

# 输出示例
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  : 1024
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  : 512
# 数值越大表示冲突越严重
```

### 5.2 使用 CUDA Profiler

```bash
# 使用 nvprof（旧版本）
nvprof --metrics shared_load_transactions,shared_store_transactions \
    ./your_program

# 理想情况：transactions = threads / 32（warp数）
# 如果有冲突：transactions > threads / 32
```

### 5.3 代码层面的检查方法

```cuda
// 手动检查 Bank 分配
__device__ void checkBankAccess(int tid, int index) {
    int bank = (index * sizeof(float) / 4) % 32;
    printf("tid %d accessing index %d -> Bank %d\n", tid, index, bank);
}

// 在 kernel 中调用检查
__global__ void myKernel() {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // 检查访问模式
    checkBankAccess(tid, tid * 2);  // 或其他访问模式
}
```

---

## 6. 避免 Bank Conflict 的技巧

### 技巧 1：Sequential Addressing（连续寻址）

```cuda
// 好：连续线程访问连续地址
for (int i = tid; i < N; i += blockDim.x) {
    float val = sdata[i];  // 无冲突
}
```

### 技巧 2：Padding（填充）

```cuda
// 问题：32x32 float 数组，列访问导致冲突
__shared__ float tile[32][32];
// tile[i][0] 和 tile[i+1][0] 都在 Bank 0

// 解决：添加 padding 列
__shared__ float tile[32][33];  // 33 而不是 32
// 现在同一列的元素间隔 33 个 float
// Bank = (i * 33 + col) % 32，不会全部相同
```

```
原始布局（33x32）:
地址 = row * 32 + col
row 0 col 0: 0  → Bank 0
row 1 col 0: 32 → Bank 0 (32%32=0)  ← 冲突！

Padding 后（33x33）:
地址 = row * 33 + col
row 0 col 0: 0  → Bank 0
row 1 col 0: 33 → Bank 1 (33%32=1)  ← 无冲突！
row 2 col 0: 66 → Bank 2 (66%32=2)  ← 无冲突！
```

### 技巧 3：Swizzle（地址重排）

```cuda
// 使用 swizzle 模式优化矩阵访问
#define SWIZZLE(addr) (((addr) / 32) ^ ((addr) % 32)) % 32

__device__ int getSwizzledIndex(int row, int col) {
    return SWIZZLE(row) * 32 + col;
}
```

### 技巧 4：Vectorized Access（向量化访问）

```cuda
// 使用 float4 读取 4 个连续 float
// 4 个 float 必然在不同的 Bank（间隔 4 bytes）
float4 val = reinterpret_cast<float4*>(&sdata[index])[0];
// Bank: index%32, (index+1)%32, (index+2)%32, (index+3)%32
// 只要 index%32 < 29，就不会有 Bank 重复
```

### 技巧 5：转置时的对角线访问

```cuda
// 矩阵转置优化：使用对角线起始坐标
int row = (blockIdx.y * TILE_DIM + threadIdx.y);
int col = (blockIdx.x * TILE_DIM + threadIdx.x);

// 对角线起始
int bankOffset = (row + col) % 32;
int index = row * TILE_DIM + threadIdx.x + bankOffset;
```

---

## 7. 不同架构的 Bank 配置

### GPU 架构演进

| 架构 | 计算能力 | Bank 数量 | Bank 宽度 | 共享内存/SM |
|------|---------|----------|----------|------------|
| Fermi | 2.x | 32 | 4 bytes | 48 KB |
| Kepler | 3.x | 32 | 4 bytes | 48 KB / 16 KB 可选 |
| Maxwell | 5.x | 32 | 4 bytes | 96 KB |
| Pascal | 6.x | 32 | 4 bytes | 64 KB |
| Volta | 7.x | 32 | 4 bytes | 96 KB |
| Turing | 7.5 | 32 | 4 bytes | 64 KB |
| Ampere | 8.x | 32 | 4 bytes | 164 KB |
| Hopper | 9.x | 32 | 4 bytes | 228 KB |

**注意**：
- 现代 GPU（Volta+）支持配置为 16 或 32 个 Bank
- 默认通常是 32 个 Bank
- 可以使用 `cudaDeviceSetSharedMemConfig()` 修改

### 配置 Bank 模式

```cuda
// 查看当前配置
cudaSharedMemConfig config;
cudaDeviceGetSharedMemConfig(&config);
// cudaSharedMemBankSizeFourByte (默认，32 banks)
// cudaSharedMemBankSizeEightByte (16 banks，用于 double)

// 设置为 8-byte 模式（16 banks）
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
```

---

## 8. 总结

### Bank Conflict 核心要点

```
┌─────────────────────────────────────────────────────────────┐
│                    Bank Conflict 速查表                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Bank 映射公式:                                          │
│     Bank ID = (address / 4) % 32                            │
│                                                             │
│  2. 冲突条件:                                               │
│     同一 Warp 内的多个线程同时访问同一 Bank 的不同地址       │
│                                                             │
│  3. 无冲突模式:                                             │
│     • 连续线程访问连续地址 (tid → address)                   │
│     • 所有线程访问同一地址（广播）                           │
│     • 间隔为 32 的倍数时小心（可能冲突）                     │
│                                                             │
│  4. 解决方案:                                               │
│     • Sequential Addressing                                  │
│     • Padding                                                │
│     • Swizzle                                                │
│     • Vectorized Access                                      │
│                                                             │
│  5. 检测工具:                                               │
│     • Nsight Compute (l1tex__data_bank_conflicts_*)         │
│     • nvprof (shared_load_transactions)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键公式

```
Bank ID = (byte_address / 4) % 32

冲突检查：
对于 Warp 中的所有线程：
    bank_ids = {(addr[i] / 4) % 32 for i in warp}
    max_occurrence = max(count(bank_id) for bank_id in bank_ids)
    conflict_degree = max_occurrence  # N-way conflict
```

### 最佳实践

1. **优先使用 Sequential Addressing**：连续线程访问连续地址几乎总是最优的
2. **小心结构体布局**：结构体可能导致意外的 Bank 分布
3. **矩阵操作使用 Padding**：`[N][N]` → `[N][N+1]` 或其他对齐方式
4. **使用向量化加载**：`float4` 可以一次加载 4 个不同 Bank 的数据
5. **使用工具验证**：Nsight Compute 是检测 Bank Conflict 的金标准

---

## 参考

- [CUDA Parallel Reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [CUDA C Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA Best Practices Guide - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [Nsight Compute Documentation - Memory Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-analysis)
- "Programming Massively Parallel Processors" - David Kirk, Wen-mei Hwu

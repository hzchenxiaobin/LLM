# V2: Blelloch 工作高效扫描 - 详细解析

## 概述

Blelloch 算法是一种**工作高效的并行扫描算法**（Work-Efficient Scan），由 Guy Blelloch 在 1990 年提出。与 Hillis-Steele 的 O(N log N) 工作量相比，Blelloch 算法将工作量优化到 **O(N)**，同时保持 O(log N) 的跨度。

---

## 算法原理

Blelloch 算法分为两个阶段：
1. **Up-Sweep（归约阶段）**：自底向上构建部分和树
2. **Down-Sweep（分发阶段）**：自顶向下计算前缀和

### 核心思想

利用**二叉树结构**来组织计算：
- 叶子节点是输入数据
- 内部节点存储部分和
- 通过两次遍历（上归约 + 下传播）得到前缀和

---

## 完整示例：N=6 (填充到 8)

以输入 `[3, 1, 7, 0, 4, 1]` 为例，展示完整流程：

```
                         UP-SWEEP 阶段: 自底向上归约 (N=6)


输入:        3       1       7       0       4       1
             |       |       |       |       |       |
             v       v       v       v       v       v

                         +------------------+
                         |  temp[7] = 16    |  <- 根节点 (总和)
                         |  [0..5]累加      |
                         +--------+---------+
                                  |
              +-------------------+-------------------+
              |                                       |
    +---------+---------+                   +---------+---------+
    | temp[3] = 11      |  <- Level 2       | temp[7] = 5       |
    | [0..3]累加        |                   | [4..5]累加+填充   |
    +---------+---------+                   +---------+---------+
              |                                       |
    +---------+---------+                   +---------+
    |                   |                   |
+---+-----+     +-----+-----+         +-----+-----+
|temp[1]=4|     |temp[3]=7  |         |temp[5]=5  |
|[0..1]   |<-L1 |[2..3]     |<-L1     |[4..5]     |<-L1
+---+-----+     +-----+-----+         +-----+-----+
    |                 |                     |
    |       +---------+                     |
    |       |                               |
+---+-----+-----+   +-----+-----+   +-----+-----+   +-----+-----+   +-----+-----+   +-----+-----+
|             |   |         |   |         |   |         |   |         |   |         |
| temp[0]=3   |   |temp[1]=1 |   |temp[2]=7|   |temp[3]=0|   |temp[4]=4|   |temp[5]=1|
|  a[0]       |   |  a[1]    |   |  a[2]   |   |  a[3]   |   |  a[4]   |   |  a[5]   |
|  叶子       |   |  叶子    |   |  叶子   |   |  叶子   |   |  叶子   |   |  叶子   |
+-------------+   +---------+   +---------+   +---------+   +---------+   +---------+

                                                             +---------+   +---------+
                                                             |temp[6]=0|   |temp[7]=0|
                                                             |  (填充) |   |  (根)   |
                                                             +---------+   +---------+


计算步骤 (N=6, 实际数组大小=8):
  Level 1 (stride=1):  temp[1]+=temp[0]  temp[3]+=temp[2]  temp[5]+=temp[4]  temp[7]+=temp[6]
                      (1+3=4)         (0+7=7)         (1+4=5)         (0+0=0)

  Level 2 (stride=2):  temp[3]+=temp[1]              temp[7]+=temp[5]
                      (7+4=11)                      (0+5=5)

  Level 3 (stride=4): temp[7]+=temp[3]
                      (5+11=16)  <- 总和

  清零: temp[7] = 0 (准备 Down-Sweep)
```

树节点值变化:

Level 3 (d=4, offset=1):
  temp[0]=3, temp[1]=4(3+1), temp[2]=7, temp[3]=7(7+0), 
  temp[4]=4, temp[5]=5(4+1), temp[6]=6, temp[7]=9(6+3)

Level 2 (d=2, offset=2):
  temp[3]=11(4+7), temp[7]=14(5+9)

Level 1 (d=1, offset=4):
  temp[7]=25(11+14) ← 总和

根节点清零:
  temp[7] = 0  (关键！准备排他型扫描)

╔══════════════════════════════════════════════════════════════════════╗
║                     2. DOWN-SWEEP 阶段 (分发)                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  交换与累加操作示意图:                                                ║
║                                                                       ║
║  d=1 (offset=2, 从根开始):                                            ║
║                                                                       ║
║       temp[3] ← temp[7] = 0           temp[7] += temp[3] (old)        ║
║       temp[7] ← 0 + 11 = 11                                           ║
║                                                                       ║
║  d=2 (offset=1):                                                      ║
║                                                                       ║
║    temp[1] ← temp[3] = 0              temp[3] += temp[1] (old=4)      ║
║    temp[3] ← 0 + 4 = 4                                                ║
║                                                                       ║
║    temp[5] ← temp[7] = 11            temp[7] += temp[5] (old=5)       ║
║    temp[7] ← 11 + 5 = 16                                              ║
║                                                                       ║
║  d=4 (offset=0.5→取1, 叶子层):                                        ║
║                                                                       ║
║  temp[0] ← temp[1] = 0     temp[2] ← temp[3] = 4     ...              ║
║  temp[1] ← 0 + 4 = 4       temp[3] ← 4 + 7 = 11      ...              ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝

最终结果: [0, 3, 4, 11, 11, 15, 16, 22] ✓
```

---

## 树形索引计算详解

### Up-Sweep 索引计算

```
对于 N=8，线程数 = N/2 = 4，thid = 0,1,2,3

迭代 d=4 (offset=1):
  thid=0: ai = 1*(2*0+1)-1 = 0,  bi = 1*(2*0+2)-1 = 1
          temp[1] += temp[0]  →  temp[1] = 1+3 = 4
  thid=1: ai = 1*(2*1+1)-1 = 2,  bi = 1*(2*1+2)-1 = 3
          temp[3] += temp[2]  →  temp[3] = 0+7 = 7
  thid=2: ai = 1*(2*2+1)-1 = 4,  bi = 1*(2*2+2)-1 = 5
          temp[5] += temp[4]  →  temp[5] = 1+4 = 5
  thid=3: ai = 1*(2*3+1)-1 = 6,  bi = 1*(2*3+2)-1 = 7
          temp[7] += temp[6]  →  temp[7] = 3+6 = 9

迭代 d=2 (offset=2):
  thid=0: ai = 2*(2*0+1)-1 = 1,  bi = 2*(2*0+2)-1 = 3
          temp[3] += temp[1]  →  temp[3] = 7+4 = 11
  thid=1: ai = 2*(2*1+1)-1 = 5,  bi = 2*(2*1+2)-1 = 7
          temp[7] += temp[5]  →  temp[7] = 9+5 = 14? 
          等等，让我修正：temp[7] 应该是 9+5=14，但之前是9
          实际: temp[7]=11 (上一轮temp[5]=5)

迭代 d=1 (offset=4):
  thid=0: ai = 4*(2*0+1)-1 = 3,  bi = 4*(2*0+2)-1 = 7
          temp[7] += temp[3]  →  temp[7] = 11+11 = 22
```

### 索引可视化

```
N=8 的树结构索引布局:

Shared Memory 索引: [0, 1, 2, 3, 4, 5, 6, 7]

                    7 (根节点)
                   / \
                  /   \
                 3     7 ← 注意：这是同一个位置！
                / \   / \
               1   3 5   7
              / \ / \ / \ / \
             0  1 2 3 4 5 6 7  ← 叶子节点

Up-Sweep 配对 (d=4, offset=1):
  (0→1), (2→3), (4→5), (6→7)
  
Up-Sweep 配对 (d=2, offset=2):
  (1→3), (5→7)
  
Up-Sweep 配对 (d=1, offset=4):
  (3→7)
```

---

## Down-Sweep 交换操作详解

```
Down-Sweep 的核心操作 (三行代码):

  T t = temp[ai_local];           // 保存 ai 的旧值
  temp[ai_local] = temp[bi_local]; // ai 获得 bi 的值（父节点的和）
  temp[bi_local] += t;             // bi 累加 ai 的旧值（传播前缀和）

视觉化理解 (单个节点对):

  操作前:              操作后:
  ┌─────┐ ┌─────┐     ┌─────┐ ┌─────┐
  │ ai  │ │ bi  │     │ ai  │ │ bi  │
  │  4  │ │  7  │ ──→ │  7  │ │ 11  │
  └─────┘ └─────┘     └─────┘ └─────┘
  
  t = 4
  temp[ai] = temp[bi] = 7    ← ai 获得"从开头到父节点"的和
  temp[bi] = 7 + t = 11      ← bi 累加"左兄弟子树的和"

这样 ai 变成了排他型前缀和（不包含自己），bi 变成了包含型前缀和。
```

---

## 代码逐行解析

### 内核函数

```cpp
template <typename T>
__global__ void blelloch_scan_kernel(T* g_odata, const T* g_idata, int n) {
    extern __shared__ T temp[];     // 动态分配共享内存，大小 = n * sizeof(T)
    int thid = threadIdx.x;          // 线程在 block 内的索引
    int offset = 1;                  // 初始偏移量，每轮翻倍

    // 每个线程加载两个相邻元素
    // ai = 偶数索引, bi = 奇数索引
    int ai = 2 * thid;               // 线程 thid 负责的第一个元素索引
    int bi = 2 * thid + 1;           // 线程 thid 负责的第二个元素索引

    // 从 Global Memory 加载到 Shared Memory
    // 边界检查处理 N 不是 2 的幂次的情况
    temp[ai] = (ai < n) ? g_idata[ai] : 0;
    temp[bi] = (bi < n) ? g_idata[bi] : 0;

    // ========== 1. Up-Sweep 阶段 (归约) ==========
    // d 从 n/2 开始，每次减半，直到 d=1
    // 这对应于从树的叶子层向上到根节点
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();             // 确保所有线程完成上一轮写入
        
        // 当前轮只有前 d 个线程工作
        // d 表示当前层的"节点对"数量
        if (thid < d) {
            // 计算当前线程需要处理的两个位置
            // 公式推导基于二叉树结构
            int ai_local = offset * (2 * thid + 1) - 1;  // 左子树和的位置
            int bi_local = offset * (2 * thid + 2) - 1;  // 右子树和的位置
            
            // 关键归约操作：右子树累加左子树的和
            temp[bi_local] += temp[ai_local];
        }
        offset *= 2;                  // 偏移量翻倍，向上一层
    }

    // 将根节点设为 0
    // 这是排他型扫描的关键：根节点代表"所有前面元素的和"
    // 设为 0 表示"没有前面的元素"
    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // ========== 2. Down-Sweep 阶段 (分发) ==========
    // d 从 1 开始，每次翻倍，直到 d=n
    // 这对应于从根节点向下到叶子层
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;                 // 偏移量减半，向下一层
        __syncthreads();
        
        if (thid < d) {
            int ai_local = offset * (2 * thid + 1) - 1;
            int bi_local = offset * (2 * thid + 2) - 1;

            // 交换并累加：
            // - ai 获得 bi 的旧值（父节点的前缀和）
            // - bi 累加 ai 的旧值（左兄弟子树的和）
            T t = temp[ai_local];           // 保存左子树的和
            temp[ai_local] = temp[bi_local]; // 左位置获得父节点的和
            temp[bi_local] += t;             // 右位置累加左子树的和
        }
    }
    __syncthreads();

    // 将结果写回 Global Memory
    if (ai < n) g_odata[ai] = temp[ai];
    if (bi < n) g_odata[bi] = temp[bi];
}
```

### 主机端包装函数

```cpp
void blelloch_scan(const float* d_input, float* d_output, int n) {
    // Blelloch 算法要求每个线程处理 2 个元素
    // 所以线程数 = n/2
    int threads = n / 2;
    
    // 限制检查：不能超过 CUDA 最大线程数 (1024)
    // 实际生产环境需要实现分块处理
    if (threads > 1024) {
        threads = 1024;
    }

    // 共享内存大小：只需要 n 个元素（单缓冲）
    // 比 Hillis-Steele 的 2n 节省一半空间
    int smem_size = n * sizeof(float);

    // 启动内核：1 个 block，threads 个线程
    blelloch_scan_kernel<<<1, threads, smem_size>>>(d_output, d_input, n);

    cudaDeviceSynchronize();
}
```

---

## 复杂度分析

### 工作量 (Work)

```
Up-Sweep:
  第 1 层: N/2 次加法
  第 2 层: N/4 次加法
  第 3 层: N/8 次加法
  ...
  总计: N/2 + N/4 + N/8 + ... + 1 = N - 1 ≈ N

Down-Sweep:
  与 Up-Sweep 对称，也是 N - 1 次操作

总工作量: 2N = O(N) ✓
```

### 跨度 (Span)

```
最长依赖链 = 树的高度 = log₂N

Up-Sweep: log₂N 步
Down-Sweep: log₂N 步
总计: 2log₂N = O(log N)
```

### 空间复杂度

```
共享内存: N × sizeof(T) (单缓冲)

比 Hillis-Steele 节省 50% 共享内存！
```

---

## 与 Hillis-Steele 对比

| 特性 | Hillis-Steele | Blelloch |
|------|---------------|----------|
| **工作量** | O(N log N) | **O(N)** ✓ |
| **跨度** | O(log N) | O(log N) |
| **共享内存** | 2N | **N** ✓ |
| **同步次数** | log N | 2log N |
| **代码复杂度** | 简单 | 中等 |
| **Bank Conflict** | 有 | 有 |

### 工作量对比实例 (N=1024)

```
Hillis-Steele:
  1024 × log₂(1024) = 1024 × 10 = 10,240 次操作

Blelloch:
  2 × 1024 = 2,048 次操作

节省: (10240 - 2048) / 10240 = 80% 的操作！
```

---

## Bank Conflict 问题

Blelloch 算法在 Up-Sweep 和 Down-Sweep 中，访问步长会翻倍（2, 4, 8...），这会导致 Shared Memory Bank 冲突。

```
Bank 布局 (32 个 Bank):

地址: 0  1  2  3  ... 31 32 33 ...
Bank: 0  1  2  3  ... 31  0  1  ...

冲突示例 (offset=2):
  线程 0 访问地址 2 (Bank 2)
  线程 1 访问地址 4 (Bank 4)  
  线程 2 访问地址 6 (Bank 6)
  ...
  这不会冲突，因为每个线程访问不同 Bank

冲突示例 (offset=32):
  线程 0 访问地址 32 (Bank 0)
  线程 1 访问地址 64 (Bank 0)
  线程 2 访问地址 96 (Bank 0)
  ...
  所有线程访问 Bank 0！严重冲突！

解决方案: V3 将使用 Padding 技术
```

---

## 线程负载均衡

```
N=8, 线程数=4 (每个线程处理 2 个元素)

线程分配:
  Thread 0: 处理索引 0, 1
  Thread 1: 处理索引 2, 3
  Thread 2: 处理索引 4, 5
  Thread 3: 处理索引 6, 7

Up-Sweep 各轮活跃线程:
  d=4: 线程 0,1,2,3 工作 (全部)
  d=2: 线程 0,1 工作 (一半)
  d=1: 线程 0 工作 (只有一个)

这就是负载不均衡问题：越往上层，工作的线程越少
但注意：这不会浪费太多效率，因为工作量本身在减少
```

---

## 使用建议

1. **学习路径**: 在理解 Hillis-Steele 后，Blelloch 是理解工作高效算法的入门
2. **实际应用**: 比 Hillis-Steele 快 2-3 倍，但仍不如 V3/V4
3. **下一步**: 学习 V3 消除 Bank Conflict，V4 使用 Warp Primitives
4. **生产环境**: 使用 NVIDIA CUB 库 (`cub::DeviceScan::ExclusiveSum`)

---

## 参考

- Blelloch, G. E. (1990). Prefix sums and their applications.
- GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA
- Harris, M. (2007). Optimizing parallel reduction in CUDA

# V2: Blelloch 工作高效扫描 - 详细解析

## 概述

Blelloch 扫描算法是一种**工作高效（Work-Efficient）的并行扫描算法**，由 Guy Blelloch 于 1990 年提出。与 Hillis-Steele 算法的 O(N log N) 工作量不同，Blelloch 算法达到了理论最优的 **O(N) 工作量**，同时保持 O(log N) 的跨度。

---

## 核心思想

Blelloch 算法分为两个主要阶段：

1. **Up-Sweep 阶段（归约/Reduce）**：自底向上构建部分和树，将相邻元素配对归约
2. **Down-Sweep 阶段（分发/Distribute）**：自顶向下传播前缀和，计算最终结果

**关键洞察**：通过巧妙的索引安排和交换操作，Blelloch 算法实现了排他型扫描（Exclusive Scan）而无需额外的右移操作。

---

## 算法流程图解

### 整体流程概览

```mermaid
flowchart TB
    subgraph Input["输入数组 (N=8)"]
        IN["[3, 1, 7, 0, 4, 1, 6, 3]"]
    end

    subgraph UpSweep["阶段 1: Up-Sweep 归约"]
        direction TB
        US1["Step 1: offset=1<br/>相邻配对相加"]
        US2["Step 2: offset=2<br/>间隔 2 相加"]
        US3["Step 3: offset=4<br/>间隔 4 相加"]
        USR["根节点 = 总和"]
    end

    subgraph RootClear["根节点清零"]
        RC["temp[N-1] = 0<br/>关键步骤！"]
    end

    subgraph DownSweep["阶段 2: Down-Sweep 分发"]
        direction TB
        DS1["Step 1: offset=4<br/>传播到子节点"]
        DS2["Step 2: offset=2<br/>继续传播"]
        DS3["Step 3: offset=1<br/>最终位置"]
    end

    subgraph Output["输出: 排他型前缀和"]
        OUT["[0, 3, 4, 11, 11, 15, 16, 22]"]
    end

    IN --> US1 --> US2 --> US3 --> USR
    USR --> RC
    RC --> DS1 --> DS2 --> DS3 --> OUT
```

---

## Up-Sweep 阶段详解

### 概念说明

Up-Sweep 阶段构建一棵**归约树（Reduction Tree）**，自底向上逐层计算部分和。

```mermaid
flowchart TB
    subgraph UpSweepTree["Up-Sweep 归约树 (二叉树结构)"]
        direction TB

        L3["Level 0 (叶子):<br/>[3, 1, 7, 0, 4, 1, 6, 3]"]

        L2["Level 1 (offset=1):<br/>[4, 7, 5, 9]"]

        L1["Level 2 (offset=2):<br/>[11, 14]"]

        L0["Level 3 (offset=4):<br/>[25] ← 总和"]

        L3 -->|"3+1=4, 7+0=7<br/>4+1=5, 6+3=9"| L2
        L2 -->|"4+7=11, 5+9=14"| L1
        L1 -->|"11+14=25"| L0
    end

    style L0 fill:#ff9999,stroke:#333,stroke-width:2px
    style L3 fill:#99ccff,stroke:#333,stroke-width:1px
```

### Up-Sweep 详细步骤

**初始数据加载到 Shared Memory：**

```mermaid
flowchart LR
    subgraph LoadData["数据加载 (每个线程加载 2 个元素)"]
        direction LR

        TID0["Thread 0"]
        TID1["Thread 1"]
        TID2["Thread 2"]
        TID3["Thread 3"]

        IDX0["ai=0:<br/>temp[0]=3"]
        IDX1["bi=1:<br/>temp[1]=1"]
        IDX2["ai=2:<br/>temp[2]=7"]
        IDX3["bi=3:<br/>temp[3]=0"]
        IDX4["ai=4:<br/>temp[4]=4"]
        IDX5["bi=5:<br/>temp[5]=1"]
        IDX6["ai=6:<br/>temp[6]=6"]
        IDX7["bi=7:<br/>temp[7]=3"]

        TID0 --> IDX0
        TID0 --> IDX1
        TID1 --> IDX2
        TID1 --> IDX3
        TID2 --> IDX4
        TID2 --> IDX5
        TID3 --> IDX6
        TID3 --> IDX7
    end
```

**Step 1: offset=1 (d=4, n/2)**

```mermaid
flowchart TB
    subgraph Step1["Step 1: offset=1 (d=4 线程工作)"]
        direction TB

        Before1["Before:<br/>[3, 1, 7, 0, 4, 1, 6, 3]"]

        T0["thid=0:<br/>ai=0, bi=1<br/>temp[1] += temp[0]<br/>1 += 3 → 4"]
        T1["thid=1:<br/>ai=2, bi=3<br/>temp[3] += temp[2]<br/>0 += 7 → 7"]
        T2["thid=2:<br/>ai=4, bi=5<br/>temp[5] += temp[4]<br/>1 += 4 → 5"]
        T3["thid=3:<br/>ai=6, bi=7<br/>temp[7] += temp[6]<br/>3 += 6 → 9"]

        After1["After:<br/>[3, 4, 7, 7, 4, 5, 6, 9]"]

        Before1 --> T0 & T1 & T2 & T3 --> After1
    end
```

索引计算公式：
- `ai = offset * (2*thid + 1) - 1 = 1*(2*thid+1) - 1 = 2*thid`
- `bi = offset * (2*thid + 2) - 1 = 1*(2*thid+2) - 1 = 2*thid + 1`

**Step 2: offset=2 (d=2, 2 线程工作)**

```mermaid
flowchart TB
    subgraph Step2["Step 2: offset=2 (d=2 线程工作)"]
        direction TB

        Before2["Before:<br/>[3, 4, 7, 7, 4, 5, 6, 9]"]

        T0_2["thid=0:<br/>ai=1, bi=3<br/>temp[3] += temp[1]<br/>7 += 4 → 11"]
        T1_2["thid=1:<br/>ai=5, bi=7<br/>temp[7] += temp[5]<br/>9 += 5 → 14"]

        After2["After:<br/>[3, 4, 7, 11, 4, 5, 6, 14]"]

        Before2 --> T0_2 & T1_2 --> After2
    end
```

索引计算公式：
- `ai = offset * (2*thid + 1) - 1 = 2*(2*thid+1) - 1 = 4*thid + 1`
- `bi = offset * (2*thid + 2) - 1 = 2*(2*thid+2) - 1 = 4*thid + 3`

**Step 3: offset=4 (d=1, 1 线程工作)**

```mermaid
flowchart TB
    subgraph Step3["Step 3: offset=4 (d=1 线程工作)"]
        direction TB

        Before3["Before:<br/>[3, 4, 7, 11, 4, 5, 6, 14]"]

        T0_3["thid=0:<br/>ai=3, bi=7<br/>temp[7] += temp[3]<br/>14 += 11 → 25"]

        After3["After:<br/>[3, 4, 7, 11, 4, 5, 6, 25]"]

        Before3 --> T0_3 --> After3
    end

    style After3 fill:#ffcccc,stroke:#333,stroke-width:2px
```

**根节点清零（排他型扫描的关键）：**

```mermaid
flowchart LR
    subgraph RootZero["根节点设为 0"]
        BeforeR["Up-Sweep 结束:<br/>[3, 4, 7, 11, 4, 5, 6, 25]"]
        ZeroOp["thid==0:<br/>temp[N-1] = 0<br/>temp[7] = 0"]
        AfterR["清零后:<br/>[3, 4, 7, 11, 4, 5, 6, 0]"]

        BeforeR --> ZeroOp --> AfterR
    end

    style AfterR fill:#ffffcc,stroke:#333,stroke-width:2px
```

---

## Down-Sweep 阶段详解

### 概念说明

Down-Sweep 阶段自顶向下传播前缀和，通过交换和累加操作将部分和分发到正确位置。

```mermaid
flowchart TB
    subgraph DownSweepTree["Down-Sweep 分发树"]
        direction TB

        Root["根节点: 0<br/>(原本存储总和 25)"]

        L1["Level 2:<br/>[11, 0]"]

        L2["Level 1:<br/>[4, 11, 5, 11]"]

        L3["Level 0 (结果):<br/>[0, 3, 4, 11, 11, 15, 16, 22]"]

        Root -->|"左子=父, 右子=父+原左子"| L1
        L1 -->|"继续传播"| L2
        L2 -->|"最终前缀和"| L3
    end

    style L3 fill:#99ff99,stroke:#333,stroke-width:2px
```

### Down-Sweep 详细步骤

**核心操作（交换与累加）：**
```
t = temp[ai]                    // 保存左节点值
temp[ai] = temp[bi]           // 左节点获得右节点值（父节点传来的前缀和）
temp[bi] += t                 // 右节点累加左节点原值（成为新的前缀和）
```

**Step 1: offset=4 (d=1)**

```mermaid
flowchart TB
    subgraph DS1["Down-Sweep Step 1: offset=4"]
        direction TB

        BeforeDS1["Before:<br/>[3, 4, 7, 11, 4, 5, 6, 0]"]

        Op1["thid=0, offset=4:<br/>ai=3, bi=7<br/>t = temp[3] = 11<br/>temp[3] = temp[7] = 0<br/>temp[7] += t → 0+11=11"]

        AfterDS1["After:<br/>[3, 4, 7, 0, 4, 5, 6, 11]"]

        BeforeDS1 --> Op1 --> AfterDS1
    end

    note["注意: temp[3]获得父节点前缀和(0)<br/>temp[7]获得子树前缀和(11)"]
    AfterDS1 -.-> note
```

**Step 2: offset=2 (d=2)**

```mermaid
flowchart TB
    subgraph DS2["Down-Sweep Step 2: offset=2"]
        direction TB

        BeforeDS2["Before:<br/>[3, 4, 7, 0, 4, 5, 6, 11]"]

        Op2_1["thid=0, offset=2:<br/>ai=1, bi=3<br/>t = temp[1] = 4<br/>temp[1] = temp[3] = 0<br/>temp[3] += t → 0+4=4"]

        Op2_2["thid=1, offset=2:<br/>ai=5, bi=7<br/>t = temp[5] = 5<br/>temp[5] = temp[7] = 11<br/>temp[7] += t → 11+5=16"]

        AfterDS2["After:<br/>[3, 0, 7, 4, 4, 11, 6, 16]"]

        BeforeDS2 --> Op2_1 & Op2_2 --> AfterDS2
    end
```

**Step 3: offset=1 (d=4)**

```mermaid
flowchart TB
    subgraph DS3["Down-Sweep Step 3: offset=1"]
        direction TB

        BeforeDS3["Before:<br/>[3, 0, 7, 4, 4, 11, 6, 16]"]

        Op3_0["thid=0:<br/>ai=0, bi=1<br/>t=3, temp[0]=0, temp[1]=3"]
        Op3_1["thid=1:<br/>ai=2, bi=3<br/>t=7, temp[2]=4, temp[3]=11"]
        Op3_2["thid=2:<br/>ai=4, bi=5<br/>t=4, temp[4]=11, temp[5]=15"]
        Op3_3["thid=3:<br/>ai=6, bi=7<br/>t=6, temp[6]=16, temp[7]=22"]

        AfterDS3["最终结果:<br/>[0, 3, 4, 11, 11, 15, 16, 22]"]

        BeforeDS3 --> Op3_0 & Op3_1 & Op3_2 & Op3_3 --> AfterDS3
    end

    style AfterDS3 fill:#ccffcc,stroke:#333,stroke-width:3px
```

---

## 完整算法数据流图

```mermaid
flowchart TB
    subgraph DataFlow["完整数据流 (N=8 示例: [3,1,7,0,4,1,6,3])"]
        direction TB

        Input["① 输入加载<br/>[3, 1, 7, 0, 4, 1, 6, 3]"]

        US1["② Up-Sweep offset=1<br/>[3, 4, 7, 7, 4, 5, 6, 9]"]
        US2["③ Up-Sweep offset=2<br/>[3, 4, 7, 11, 4, 5, 6, 14]"]
        US3["④ Up-Sweep offset=4<br/>[3, 4, 7, 11, 4, 5, 6, 25]"]

        Zero["⑤ 根节点清零<br/>[3, 4, 7, 11, 4, 5, 6, 0]"]

        DS1["⑥ Down-Sweep offset=4<br/>[3, 4, 7, 0, 4, 5, 6, 11]"]
        DS2["⑦ Down-Sweep offset=2<br/>[3, 0, 7, 4, 4, 11, 6, 16]"]
        DS3["⑧ Down-Sweep offset=1<br/>[0, 3, 4, 11, 11, 15, 16, 22]"]

        Output["⑨ 输出结果<br/>排他型前缀和"]

        Input --> US1 --> US2 --> US3 --> Zero --> DS1 --> DS2 --> DS3 --> Output
    end

    style Input fill:#99ccff,stroke:#333,stroke-width:1px
    style Output fill:#99ff99,stroke:#333,stroke-width:2px
```

---

## 代码逐行解析

### 内核函数

```cpp
template <typename T>
__global__ void blelloch_scan_kernel(T* g_odata, const T* g_idata, int n) {
    extern __shared__ T temp[];    // 动态分配共享内存，只需 N 个元素
    int thid = threadIdx.x;        // 线程在 block 内的索引
    int offset = 1;                // 用于计算索引的偏移量
```

**关键点**：与 Hillis-Steele 的双缓冲（2N 内存）不同，Blelloch 只需 N 个元素的共享内存。

### 数据加载

```cpp
    // 每个线程加载两个元素到 Shared Memory
    // ai = 偶数索引, bi = 奇数索引
    int ai = 2 * thid;             // 偶数位置索引
    int bi = 2 * thid + 1;         // 奇数位置索引

    temp[ai] = (ai < n) ? g_idata[ai] : 0;   // 加载偶数位置
    temp[bi] = (bi < n) ? g_idata[bi] : 0;   // 加载奇数位置
```

**线程与数据映射**：
- Thread 0: 处理 temp[0] 和 temp[1]
- Thread 1: 处理 temp[2] 和 temp[3]
- ...以此类推

### Up-Sweep 阶段代码

```cpp
    // ========== 1. Up-Sweep 阶段 (归约) ==========
    // 从叶子节点向上归约，构建部分和树
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();           // 确保上一轮写入完成

        if (thid < d) {            // 只有前 d 个线程工作
            // 计算要操作的索引
            int ai_local = offset * (2 * thid + 1) - 1;   // 左节点索引
            int bi_local = offset * (2 * thid + 2) - 1;   // 右节点索引

            // 右节点累加左节点的值
            temp[bi_local] += temp[ai_local];
        }
        offset *= 2;               // 偏移量翻倍
    }
```

**循环分析**：
- 初始 `d = n/2`，每轮减半
- 线程数逐轮减少：n/2 → n/4 → n/8 → ... → 1
- `offset` 记录当前层级的跨度

### 根节点清零

```cpp
    // 将根节点设为 0 (这是排他型扫描的关键)
    if (thid == 0) {
        temp[n - 1] = 0;           // 最后一个元素清零
    }
```

**为什么清零能实现排他型扫描？**
- Up-Sweep 后，根节点存储总和（包含型）
- 清零后，Down-Sweep 会将 0 作为初始前缀和传播下去
- 最终每个位置存储的是它之前所有元素的和（不包含自己）

### Down-Sweep 阶段代码

```cpp
    // ========== 2. Down-Sweep 阶段 (分发) ==========
    // 自顶向下计算前缀和
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;              // 偏移量减半（反向）
        __syncthreads();           // 确保上一轮完成

        if (thid < d) {            // 逐轮增加工作线程
            int ai_local = offset * (2 * thid + 1) - 1;
            int bi_local = offset * (2 * thid + 2) - 1;

            // 交换并累加
            T t = temp[ai_local];              // 保存左节点值
            temp[ai_local] = temp[bi_local];   // 左节点获得右节点值
            temp[bi_local] += t;                // 右节点累加左节点原值
        }
    }
    __syncthreads();
```

**交换累加的含义**：
- `ai` 位置获得父节点传来的前缀和（原来的 `bi` 值）
- `bi` 位置成为新的前缀和（父节点前缀和 + 本子树和）

### 结果写回

```cpp
    // 将结果写回 Global Memory
    if (ai < n) g_odata[ai] = temp[ai];
    if (bi < n) g_odata[bi] = temp[bi];
}
```

### 主机端函数

```cpp
void blelloch_scan(const float* d_input, float* d_output, int n) {
    // Blelloch 算法需要 n 是 2 的幂次
    int threads = n / 2;           // 每个线程处理 2 个元素

    if (threads > 1024) {          // CUDA 最大线程数限制
        threads = 1024;
    }

    int smem_size = n * sizeof(float);   // 共享内存大小：N 个元素

    blelloch_scan_kernel<<<1, threads, smem_size>>>(d_output, d_input, n);
    cudaDeviceSynchronize();
}
```

---

## 线程活动可视化

### Up-Sweep 线程活动

```mermaid
flowchart TB
    subgraph UpSweepActivity["Up-Sweep 阶段线程活动"]
        direction TB

        Step1US["Step 1 (d=4, offset=1)<br/>┌─────┬─────┬─────┬─────┐<br/>│ T0  │ T1  │ T2  │ T3  │<br/>│ 活  │ 活  │ 活  │ 活  │<br/>└─────┴─────┴─────┴─────┘"]

        Step2US["Step 2 (d=2, offset=2)<br/>┌─────┬─────┬─────┬─────┐<br/>│ T0  │ T1  │ -   │ -   │<br/>│ 活  │ 活  │ 闲  │ 闲  │<br/>└─────┴─────┴─────┴─────┘"]

        Step3US["Step 3 (d=1, offset=4)<br/>┌─────┬─────┬─────┬─────┐<br/>│ T0  │ -   │ -   │ -   │<br/>│ 活  │ 闲  │ 闲  │ 闲  │<br/>└─────┴─────┴─────┴─────┘"]

        Step1US --> Step2US --> Step3US
    end
```

### Down-Sweep 线程活动

```mermaid
flowchart TB
    subgraph DownSweepActivity["Down-Sweep 阶段线程活动"]
        direction TB

        Step1DS["Step 1 (d=1, offset=4)<br/>┌─────┬─────┬─────┬─────┐<br/>│ T0  │ -   │ -   │ -   │<br/>│ 活  │ 闲  │ 闲  │ 闲  │<br/>└─────┴─────┴─────┴─────┘"]

        Step2DS["Step 2 (d=2, offset=2)<br/>┌─────┬─────┬─────┬─────┐<br/>│ T0  │ T1  │ -   │ -   │<br/>│ 活  │ 活  │ 闲  │ 闲  │<br/>└─────┴─────┴─────┴─────┘"]

        Step3DS["Step 3 (d=4, offset=1)<br/>┌─────┬─────┬─────┬─────┐<br/>│ T0  │ T1  │ T2  │ T3  │<br/>│ 活  │ 活  │ 活  │ 活  │<br/>└─────┴─────┴─────┴─────┘"]

        Step1DS --> Step2DS --> Step3DS
    end
```

---

## 复杂度分析

### 工作量 (Work)

```mermaid
pie showData
    title Up-Sweep 工作量分布
    "Level 1 (N/2 线程)" : 50
    "Level 2 (N/4 线程)" : 25
    "Level 3 (N/8 线程)" : 12.5
    "..." : 12.5
```

**总工作量计算**：

```
Up-Sweep:
  N/2 + N/4 + N/8 + ... + 1 = N - 1 次加法

Down-Sweep:
  1 + 2 + 4 + ... + N/2 = N - 1 次交换/加法

总工作量 = 2(N-1) = O(N)
```

对比：
- **Hillis-Steele**: O(N log N) = 10N (N=1024)
- **Blelloch**: O(N) = 2N
- **提升**：约 5 倍效率提升

### 跨度 (Span/Critical Path)

```
跨度 = 2 × log₂N

Up-Sweep: log₂N
Down-Sweep: log₂N
总计: 2log₂N = O(log N)
```

### 空间复杂度

```
共享内存使用 = N × sizeof(T)

对比 Hillis-Steele 的 2N，Blelloch 节省 50% 共享内存
```

---

## 与 Hillis-Steele 对比

| 特性 | Hillis-Steele | Blelloch |
|------|---------------|----------|
| **工作量** | O(N log N) | **O(N)** ✓ |
| **跨度** | O(log N) | O(log N) |
| **共享内存** | 2N | **N** ✓ |
| **同步次数** | log N | 2 log N |
| **线程利用率** | 高 (每轮全工作) | 低 (部分线程空闲) |
| **实现复杂度** | 简单 | 中等 |
| **结果类型** | 包含型/排他型 | 直接排他型 ✓ |

```mermaid
flowchart LR
    subgraph Compare["算法对比"]
        direction TB

        HS["Hillis-Steele<br/>O(N log N)<br/>简单直观<br/>适合教学"]
        BL["Blelloch<br/>O(N)<br/>工作高效<br/>适合生产"]

        HS <-->|"效率差距<br/>log N 倍"| BL
    end
```

---

## 优缺点总结

### 优点

1. **工作高效**：O(N) 工作量，理论最优
2. **内存高效**：仅需 N 的共享内存
3. **直接排他型**：无需额外转换步骤
4. **适合大规模数据**：N 越大，优势越明显

### 缺点

1. **线程利用率低**：Up-Sweep 和 Down-Sweep 初期很多线程空闲
2. **需要 2 的幂次**：数据长度必须是 2 的幂次（可 padding 解决）
3. **实现稍复杂**：索引计算比 Hillis-Steele 复杂

---

## 使用建议

1. **生产环境首选**：Blelloch 是 CUDA 扫描的标准算法
2. **大规模数据**：N > 512 时，Blelloch 优势明显
3. **内存受限**：共享内存紧张时，Blelloch 更优
4. **学习曲线**：建议先学 Hillis-Steele，再学 Blelloch

---

## 参考

- Blelloch, G. E. (1990). Prefix sums and their applications.
- GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA
- Harris, M. (2007). Parallel Prefix Sum with CUDA

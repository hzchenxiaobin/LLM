# V1 算法可视化详解

## 1. 数据流动图

```mermaid
flowchart TB
    subgraph GlobalMemory[HBM - 全局内存]
        Q[Q: N×d]
        K[K: N×d]
        V[V: N×d]
        O[O: N×d]
    end

    subgraph ThreadExecution[单个线程的执行流程]
        direction TB

        subgraph Step1[步骤1: 加载Q]
            LoadQ["从HBM加载Q的一行到寄存器 q_vec[d]"]
        end

        subgraph Step2[步骤2: 循环遍历KV]
            direction TB
            LoopStart[for k_idx=0 to N-1]

            subgraph InnerStep1[计算注意力分数]
                LoadK["加载K行到寄存器"]
                Dot["计算 dot(q_vec, K行)"]
                Scale["乘以 scale=1/√d"]
                ResultQK["qk值"]
            end

            subgraph InnerStep2[Online Softmax更新]
                UpdateM["m = max(m, qk)"]
                CalcExp["计算 exp_factor"]
                UpdateL["l = l × exp_prev + exp_curr"]
            end

            subgraph InnerStep3[更新输出累加器]
                LoadV["加载V行到寄存器"]
                UpdateO["o_acc = o_acc × exp_prev + exp_curr × V行"]
            end

            LoopEnd{继续循环?}
        end

        subgraph Step3[步骤3: 归一化输出]
            Normalize["O = o_acc / l"]
            WriteO["写回HBM"]
        end
    end

    Q --> LoadQ
    K --> LoadK
    V --> LoadV
    WriteO --> O
```

---

## 2. Online Softmax 动态演示

### 场景：序列长度 N=4，处理 4 个 KV 值

```mermaid
timeline
    title Online Softmax 更新过程
    section 初始化
        状态 : m = -∞
             : l = 0
             : o_acc = [0, 0, ...]

    section 处理 x0=2.0
        更新 : m = max(-∞, 2.0) = 2.0
             : exp_prev = exp(-∞ - 2.0) ≈ 0
             : exp_curr = exp(2.0 - 2.0) = 1.0
             : l = 0 × 0 + 1.0 = 1.0
             : o_acc += 1.0 × V[0]
        权重 : w0 = 1.0 / 1.0 = 1.0 (暂时)

    section 处理 x1=3.0
        更新 : m_prev = 2.0
             : m = max(2.0, 3.0) = 3.0
             : exp_prev = exp(2.0 - 3.0) = 0.368
             : exp_curr = exp(3.0 - 3.0) = 1.0
             : l = 1.0 × 0.368 + 1.0 = 1.368
             : o_acc = o_acc × 0.368 + 1.0 × V[1]
        权重 : w0 = exp(2-3)/1.368 ≈ 0.269
             : w1 = 1.0/1.368 ≈ 0.731

    section 处理 x2=1.0
        更新 : m_prev = 3.0
             : m = max(3.0, 1.0) = 3.0 (不变!)
             : exp_prev = exp(3.0 - 3.0) = 1.0
             : exp_curr = exp(1.0 - 3.0) = 0.135
             : l = 1.368 × 1.0 + 0.135 = 1.503
             : o_acc = o_acc × 1.0 + 0.135 × V[2]
        权重 : w0 = 0.368/1.503 ≈ 0.245
             : w1 = 1.0/1.503 ≈ 0.665
             : w2 = 0.135/1.503 ≈ 0.090

    section 处理 x3=4.0
        更新 : m_prev = 3.0
             : m = max(3.0, 4.0) = 4.0
             : exp_prev = exp(3.0 - 4.0) = 0.368
             : exp_curr = exp(4.0 - 4.0) = 1.0
             : l = 1.503 × 0.368 + 1.0 = 1.553
             : o_acc = o_acc × 0.368 + 1.0 × V[3]
        权重 : 所有旧权重 × 0.368
             : w3 = 1.0/1.553 ≈ 0.644

    section 最终
        结果 : 输出 = o_acc / l
             : 权重和 = 1.0 ✓
```

---

## 3. GPU 线程并行视图

### Grid-Block-Thread 层次

```mermaid
flowchart TB
    subgraph Grid["Grid (num_blocks = ceil(N/64))"]
        direction TB

        subgraph Block0["Block 0 (64 threads)"]
            direction LR
            T00["Thread 0: Q[0]"]
            T01["Thread 1: Q[1]"]
            T02["..."]
            T063["Thread 63: Q[63]"]
        end

        subgraph Block1["Block 1 (64 threads)"]
            direction LR
            T10["Thread 0: Q[64]"]
            T11["Thread 1: Q[65]"]
            T12["..."]
            T163["Thread 63: Q[127]"]
        end

        subgraph BlockN["..."]
            BNE["..."]
        end

        subgraph BlockLast["Block num_blocks-1"]
            direction LR
            TL0["Thread 0"]
            TL1["..."]
            TLX["Thread x (可能小于63)"]
        end
    end

    QData["Q[0:N-1]"]

    QData --> Block0
    QData --> Block1
    QData --> BlockN
    QData --> BlockLast
```

### 单个线程的内存访问模式

```mermaid
sequenceDiagram
    participant T as Thread tid
    participant HBM as Global Memory
    participant REG as "Registers (q_vec, o_acc, m, l)"

    Note over T,HBM: 初始化阶段
    T->>HBM: 读取 Q[q_row×d + 0:d-1]
    HBM->>REG: 存入 q_vec[0:d-1]
    T->>REG: o_acc = [0,...], m=-∞, l=0

    Note over T,HBM: 循环 k_idx = 0 to N-1
    loop 每个KV位置
        T->>HBM: 读取 K[k_idx×d + 0:d-1]
        HBM->>REG: 临时寄存器
        T->>REG: 计算 qk = dot(q_vec, K[k_idx])
        T->>REG: qk *= scale

        T->>REG: m_prev = m
        T->>REG: m = max(m, qk)
        T->>REG: exp_prev = exp(m_prev - m)
        T->>REG: exp_curr = exp(qk - m)
        T->>REG: l = l × exp_prev + exp_curr

        T->>HBM: 读取 V[k_idx×d + 0:d-1]
        HBM->>REG: 临时寄存器
        T->>REG: o_acc = o_acc × exp_prev + exp_curr × V[k_idx]
    end

    Note over T,HBM: 输出阶段
    T->>REG: 计算 O = o_acc / l
    REG->>HBM: 写回 O[q_row×d + 0:d-1]
```

---

## 4. 矩阵运算可视化

### Attention 计算分解

```mermaid
flowchart TB
    subgraph Input[输入矩阵]
        Q["Q: N×d (每行是一个query向量)"]
        K["K: N×d (每行是一个key向量)"]
        V["V: N×d (每行是一个value向量)"]
    end

    subgraph Computation[计算过程]
        direction TB

        subgraph Step1[Q @ K^T]
            S["S = Q×K^T: N×N - 注意力分数矩阵 (V1不显式存储)"]
        end

        subgraph Step2[Softmax]
            P["P = softmax(S): N×N - 注意力权重 (V1用online softmax增量计算)"]
        end

        subgraph Step3[× V]
            O["O = P×V: N×d<br/>输出"]
        end
    end

    Q --> Step1
    K --> Step1
    Step1 --> Step2
    Step2 --> Step3
    V --> Step3

    style S fill:#faa,stroke:#333
    style P fill:#faa,stroke:#333
```

**V1的特殊之处**：S 和 P 矩阵**不需要完全存储**，而是通过 online softmax 流式处理！

---

## 5. 内存访问热点图

### 全局内存访问频率（颜色越深访问越频繁）

```
Q矩阵 (每个元素被读取1次):
┌─────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░ │  ░ = 1次
│ ░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────┘

K矩阵 (每个元素被读取Br次=64次):
┌─────────────────────────┐
│ ███████████████████████ │  █ = 64次
│ ███████████████████████ │
│ ███████████████████████ │
└─────────────────────────┘

V矩阵 (每个元素被读取Br次=64次):
┌─────────────────────────┐
│ ███████████████████████ │  █ = 64次
│ ███████████████████████ │
│ ███████████████████████ │
└─────────────────────────┘
```

**问题**：K和V被过度重复加载！

---

## 6. Online Softmax 公式推导

### 为什么公式是正确的？

**标准 Softmax**:
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**数值稳定版本**:
```
softmax(x_i) = exp(x_i - m) / Σ exp(x_j - m)
其中 m = max(x_j)
```

**Online 版本推导**:

假设已经处理了前 k 个值，有：
```
m_k = max(x_0, ..., x_{k-1})
l_k = Σ_{i=0}^{k-1} exp(x_i - m_k)
```

新来一个值 x_k，新的最大值：
```
m_{k+1} = max(m_k, x_k)
```

需要重新计算 l_k（因为 m 变了）：
```
l_k' = Σ_{i=0}^{k-1} exp(x_i - m_{k+1})
     = Σ_{i=0}^{k-1} exp(x_i - m_k + m_k - m_{k+1})
     = Σ_{i=0}^{k-1} exp(x_i - m_k) × exp(m_k - m_{k+1})
     = l_k × exp(m_k - m_{k+1})
```

所以：
```
l_{k+1} = l_k × exp(m_k - m_{k+1}) + exp(x_k - m_{k+1})
```

这就是代码中的更新公式！

---

## 7. 执行时间线

### 单个线程的执行时间线

```mermaid
gantt
    title 单线程执行时间线 (N=1024, d=64)
    dateFormat X
    axisFormat %s

    section 加载Q
    加载Q行(64 floats)           :0, 10

    section 循环1024次
    加载K[0] + 计算qk           :10, 30
    Online Softmax更新          :30, 35
    加载V[0] + 更新o_acc        :35, 55

    加载K[1] + 计算qk           :55, 75
    Online Softmax更新          :75, 80
    加载V[1] + 更新o_acc        :80, 100

    ... (重复1022次)            :100, 20000

    section 写回
    归一化 + 写回O              :20000, 20010
```

**关键观察**：
- 每个 KV 迭代需要约 20 个时间单位
- 总共 1024 × 20 = ~20480 时间单位
- 其中大部分时间花在全局内存加载！

---

## 8. 对比：标准 Attention vs FlashAttention V1

### 内存使用对比

```mermaid
flowchart LR
    subgraph StdAttention[标准 Attention]
        direction TB
        Q1[Q: N×d]
        K1[K: N×d]
        V1[V: N×d]
        S1["S: N×N<br/>(HBM存储)"]
        P1["P: N×N<br/>(HBM存储)"]
        O1[O: N×d]

        Q1 --> S1
        K1 --> S1
        S1 --> P1
        P1 --> O1
        V1 --> O1
    end

    subgraph FlashV1[FlashAttention V1]
        direction TB
        Q2[Q: N×d]
        K2[K: N×d]
        V2[V: N×d]
        REG["寄存器: q_vec[d], o_acc[d], m, l"]
        O2[O: N×d]

        Q2 --> REG
        K2 -.流式.-> REG
        V2 -.流式.-> REG
        REG --> O2
    end

    style S1 fill:#faa
    style P1 fill:#faa
    style REG fill:#afa
```

**内存复杂度**：
- 标准 Attention: O(N² + N×d)
- FlashAttention V1: O(N×d) ✓

---

## 9. 关键代码片段索引

| 行号范围 | 内容 | 重要性 |
|---------|------|--------|
| 47-51 | 配置常量 | ⭐⭐ |
| 53-57 | Kernel签名 | ⭐⭐⭐ |
| 59-67 | 线程索引计算 | ⭐⭐⭐⭐ |
| 74-84 | 寄存器分配 | ⭐⭐⭐ |
| 87-91 | Q行加载 | ⭐⭐ |
| 95-134 | 核心KV循环 | ⭐⭐⭐⭐⭐ |
| 98-106 | qk计算 | ⭐⭐⭐⭐ |
| 110-122 | Online Softmax | ⭐⭐⭐⭐⭐ |
| 127-133 | 输出累加 | ⭐⭐⭐⭐ |
| 137-141 | 最终归一化 | ⭐⭐⭐ |
| 144-169 | Host wrapper | ⭐⭐ |

---

*此文档配合 `V1_NAIVE_EXPLAINED.md` 使用，提供视觉化理解*

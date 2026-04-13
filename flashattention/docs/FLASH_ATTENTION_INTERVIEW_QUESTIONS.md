# Flash Attention AI Infra 面试题集

> 基于 FlashAttention-4 (FA4) 仓库代码实现整理
> 适用于 AI Infra / GPU Kernel 优化 / 深度学习框架开发岗位

---

## 面试题难度分级

- **初级 (L3)**: 基础概念理解
- **中级 (L4)**: 代码实现与优化
- **高级 (L5+)**: 架构设计与深度优化

---

## 一、基础概念题 (初级)

### Q1: 什么是 Flash Attention？它解决了传统 Attention 的什么问题？

**答案要点：**

**传统 Attention 的问题：**

```python
# 标准 Attention 内存访问模式
S = Q @ K.T        # O(N²) HBM 写入
P = softmax(S)     # O(N²) HBM 读写
O = P @ V          # O(N²) HBM 读写
# 总 HBM 访问: O(N²) 次读写
```

**Flash Attention 核心思想：**

1. **Tiling (分块计算)**: 将 Q、K、V 分成小块在 SRAM 中计算
2. **Online Softmax**: 不存储完整 S 矩阵，只保存统计量 (row_max, row_sum)
3. **Recomputation**: 反向传播时重计算而非存储中间结果

**复杂度对比：**

| 指标 | 标准 Attention | Flash Attention |
|------|---------------|-----------------|
| 内存复杂度 | O(N²) | O(N) |
| HBM 访问 | 3N² + N·d | ~2N² (最优) |
| 计算复杂度 | O(N²·d) | O(N²·d) (不变) |

---

### Q2: 解释 Online Softmax 算法及其数值稳定性处理

**答案要点：**

**算法原理** (来自 `softmax.py` 52-116行):

```python
@cute.jit
def online_softmax(self, acc_S, is_first=False):
    """
    分块 softmax 计算
    - acc_S: 当前块的 S = QK^T 矩阵 (shape: tile_m × tile_n)
    - row_max: 每行的最大值 (O(1) 存储)
    - row_sum: 每行的指数和 (O(1) 存储)
    """
    for r in range(num_rows):
        acc_S_row = acc_S[r, :]  # 当前行
        
        # 1. 更新当前块的最大值
        row_max_cur = max(acc_S_row)
        row_max_prev = row_max[r]
        row_max[r] = max(row_max_prev, row_max_cur)
        
        # 2. 计算缩放因子 (数值稳定性关键)
        row_scale = exp2((row_max_prev - row_max_cur) * scale_log2)
        
        # 3. 更新指数和
        if is_first:
            row_sum[r] = sum(exp2((acc_S_row - row_max_cur) * scale_log2))
            row_scale = 1.0
        else:
            row_sum[r] = row_sum[r] * row_scale + \
                         sum(exp2((acc_S_row - row_max_cur) * scale_log2))
        
        # 4. 存储归一化后的值
        acc_S[r, :] = exp2((acc_S_row - row_max_cur) * scale_log2)
    
    return row_scale  # 用于缩放输出 O
```

**数值稳定性处理：**

1. **减去最大值**: 计算 `exp(x - max)` 而非直接 `exp(x)`，避免溢出
2. **使用 exp2**: 利用硬件指令 `exp2`，比 `exp` 更快
3. **累加缩放**: 通过 `row_scale` 累积之前的指数和

**数学公式：**

对于分块计算的 softmax，假设当前块前已有统计量 $(m_{old}, s_{old})$：

$$
m_{new} = \max(m_{old}, m_{cur})
$$

$$
s_{new} = s_{old} \cdot e^{m_{old} - m_{new}} + \sum_j e^{x_j - m_{new}}
$$

---

### Q3: Flash Attention 支持哪些 Attention 变体？

**答案要点：**

**支持的 Attention 类型** (来自 `interface.py`):

| 类型 | 说明 | 参数 |
|------|------|------|
| **MHA** | Multi-Head Attention | num_heads_q = num_heads_k = num_heads_v |
| **GQA** | Grouped Query Attention | num_heads_q > num_heads_k = num_heads_v |
| **MQA** | Multi-Query Attention | num_heads_q > 1, num_heads_k = num_heads_v = 1 |

**关键参数：**

```python
def flash_attn_func(q, k, v, ...):
    """
    Tensor shape: (batch, seqlen, num_heads, head_dim)
    - q: (batch, seqlen_q, nheads_q, head_dim)
    - k: (batch, seqlen_k, nheads_kv, head_dim)
    - v: (batch, seqlen_k, nheads_kv, head_dim_v)
    """
```

**GQA 的优化 - Pack GQA:**

```python
# 将多个 Q head 打包到序列维度
# (seqlen_q, headdim, nheads_q, batch) → ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch)
# 例如: 8 Q heads, 2 KV heads → qhead_per_kvhead = 4, packed_seqlen = 4 * 1024 = 4096
```

---

## 二、算法实现题 (中级)

### Q4: 解释 Flash Attention 中的 Tiling 策略和 BlockInfo 的作用

**答案要点：**

**BlockInfo 类** (来自 `block_info.py` 12-140行):

```python
@dataclass(frozen=True)
class BlockInfo:
    tile_m: int   # Q tile 大小 (通常为 128 或 192)
    tile_n: int   # KV tile 大小 (通常为 128)
    is_causal: bool
    is_local: bool  # 滑动窗口
    window_size_left/right: Optional[int]
```

**核心方法 - 计算 KV block 范围：**

```python
@cute.jit
def get_n_block_min_max(self, seqlen_info, m_block, split_idx=0, num_splits=1):
    """
    对于给定的 Q tile (m_block)，确定需要计算的 KV tile 范围
    考虑 causal mask 和 local window
    """
    n_block_max = ceil_div(seqlen_info.seqlen_k, self.tile_n)
    
    if self.is_causal or self.is_local:
        # Causal: 只需要计算 n <= m 的位置
        m_idx_max = (m_block + 1) * self.tile_m
        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_block_max = min(n_block_max, ceil_div(n_idx, self.tile_n))
    
    if self.is_local and self.window_size_left is not None:
        # Local: 考虑左窗口边界
        m_idx_min = m_block * self.tile_m
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_left = n_idx - self.window_size_left
        n_block_min = max(0, n_idx_left // self.tile_n)
    
    if self.is_split_kv:
        # SplitKV: 分割 KV 序列到不同 block
        num_n_blocks_per_split = (n_block_max - n_block_min + num_splits - 1) // num_splits
        n_block_min = n_block_min + split_idx * num_n_blocks_per_split
        n_block_max = min(n_block_min + num_n_blocks_per_split, n_block_max)
    
    return n_block_min, n_block_max
```

**Tiling 执行流程：**

```
Q tile (m_block)     KV tiles
┌─────────┐          ┌───┬───┬───┐
│   m=0   │    ×     │0  │1  │2  │  →  只计算 n <= m 的块 (causal)
├─────────┤          ├───┼───┼───┤
│   m=1   │    ×     │0  │1  │2  │  →  n=0,1 计算，n=2 跳过
├─────────┤          ├───┼───┼───┤
│   m=2   │    ×     │0  │1  │2  │  →  全部计算
└─────────┘          └───┴───┴───┘
```

---

### Q5: Pipeline 机制在 Flash Attention 中的作用是什么？

**答案要点：**

**PipelineStateSimple 类** (来自 `pipeline.py` 38-84行):

```python
class PipelineStateSimple:
    """
    单一 Int32 存储 index 和 phase 的 pipeline 状态
    - index: 环形缓冲区位置 (0 到 stages-1)
    - phase: 阶段标记 (0 或 1，用于同步)
    """
    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index
    
    @property
    def index(self) -> Int32:
        if self._stages == 1:
            return Int32(0)
        else:
            return self._phase_index % self._stages  # 环形位置
    
    @property
    def phase(self) -> Int32:
        if self._stages == 1:
            return self._phase_index
        else:
            return self._phase_index // self._stages  # 阶段标记
    
    def advance(self):
        if self._stages == 1:
            self._phase_index ^= 1  # 单stage时翻转phase
        else:
            self._phase_index += 1  # 多stage时递增
```

**Pipeline 类型：**

| Pipeline | 用途 | 特点 |
|----------|------|------|
| `PipelineAsync` | 通用异步拷贝 | 基础双缓冲 |
| `PipelineCpAsync` | CP 异步拷贝 | cp.async 指令 |
| `PipelineTmaAsync` | TMA 异步加载 | Hopper+ TMA |
| `PipelineTmaUmma` | TMA + UMMA | Blackwell 专用 |

**执行流程 (双缓冲):**

```
时间 →
阶段0:  加载 K[0], V[0] ──────────────────────────
阶段1:  同步 ── 计算 Q×K[0]^T ── 加载 K[1], V[1] ──
阶段2:  ────────────────── 同步 ── 计算 Q×K[1]^T ── 加载 K[2], V[2]
```

**优势：**
1. **隐藏内存延迟**: 计算与加载并行
2. **提高 SRAM 利用率**: 数据保持在快速内存中
3. **减少 HBM 访问**: 批量加载/存储

---

### Q6: 解释 SplitKV 技术的原理和适用场景

**答案要点：**

**SplitKV 原理：**

当 KV 序列很长时，单个 thread block 无法处理完整 KV，将 KV 分割到多个 block 并行处理。

```
        Q (固定)              KV 序列分割
          │              ┌──────┬──────┬──────┐
          │              │Split0│Split1│Split2│
          └──────────────┼──────┼──────┼──────┤
                         ↓      ↓      ↓
                    ┌─────────┐
                    │ block 0 │ → 计算 partial O_0, LSE_0
                    │ block 1 │ → 计算 partial O_1, LSE_1
                    │ block 2 │ → 计算 partial O_2, LSE_2
                    └────┬────┘
                         ↓
                    ┌─────────────┐
                    │ flash_fwd_combine │
                    │ 合并 partial 结果  │
                    │ 计算最终 O         │
                    └─────────────┘
```

**启发式选择** (来自 `interface.py`):

```python
def determine_num_splits(seqlen_q, seqlen_k, head_dim, num_heads, num_sms):
    """
    基于序列长度、head维度、可用SM数量决定 splits
    """
    if seqlen_k > 4096 and head_dim <= 128:
        # 长序列且 head_dim 较小，启用 split
        num_splits = min(4, num_sms // (batch_size * num_heads))
    else:
        num_splits = 1
    return num_splits
```

**合并公式** (来自 `flash_fwd_combine.py`):

```python
# 合并多个 partial 结果
global_max = max(LSE_0, LSE_1, ..., LSE_n)

# 加权平均
O_final = Σ O_i * exp2(LSE_i - global_max) / Σ exp2(LSE_i - global_max)
```

---

## 三、GPU 架构题 (中级)

### Q7: Hopper (SM90) 和 Blackwell (SM100) 架构在 FlashAttention 中的主要区别？

**答案要点：**

**架构对比：**

| 特性 | Hopper (SM90) | Blackwell (SM100) |
|------|---------------|-------------------|
| MMA 指令 | WGMMA | UMMA (Unified MMA) |
| 协作计算 | Warp Group | 2CTA (2 Cooperative Thread Arrays) |
| 共享内存 | TMA | TMA + TMEM |
| 特殊优化 | TMA 异步 | Exp2 模拟、Warp 分工 |
| Tile 大小 | 192×128, 128×128 | 128×128 (2CTA) |

**Hopper (SM90) 实现：**

```python
# flash_fwd_sm90.py
class FlashAttentionForwardSm90:
    def __call__(self, Q, K, V):
        # 使用 WGMMA (Warp Group MMA) 指令
        # Tile sizes: hdim≤64 用 192×128, hdim≤128 用 128×128
        
        # Pipeline: TMA 异步加载
        pipeline = PipelineTmaAsync(stages=2)
        
        # 循环加载 KV tiles
        for n_block in range(n_block_min, n_block_max):
            pipeline.producer_acquire(state)
            cute.tma_load(tma_k, K_tile)  # 异步加载 K
            cute.tma_load(tma_v, V_tile)  # 异步加载 V
            pipeline.producer_commit(state)
            
            pipeline.consumer_wait(state)
            acc_s = cute.wgmma(Q_tile, K_tile)  # Warp Group MMA
            pipeline.consumer_release(state)
```

**Blackwell (SM100) 核心创新：**

```python
# blackwell_helpers.py (163-207行)
# UMMA 指令使用 PTX 内联汇编
def umma_mma(acc_reg, a_desc, b_desc, c_desc):
    asm volatile(
        "tcgen05.mma.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3}, %4, %5, %6;\n"
        : "=r"(acc[0]), "=r"(acc[1]), "=r"(acc[2]), "=r"(acc[3])
        : "r"(a_desc), "r"(b_desc), "r"(c_desc)
    )

# 2CTA 启用条件 (interface.py 518-531行)
use_2cta = (
    arch in [10, 11] and  # Blackwell
    not causal and not local and  # 非因果、非局部
    not is_split_kv and
    head_dim in [128, 192]  # 特定维度才启用
)

# Warp 分工 (flash_fwd_sm100.py 182-221行)
softmax0_warp_ids = (0, 1, 2, 3)      # 4 warps 做 softmax
softmax1_warp_ids = (4, 5, 6, 7)      # 4 warps 做 softmax
correction_warp_ids = (8, 9, 10, 11)  # 修正
mma_warp_id = 12                      # MMA 计算
epilogue_warp_ids = (13,)             # 输出存储
load_warp_ids = (14,)                 # 数据加载
```

---

### Q8: 解释 Flash Attention 中的 Tile Scheduler 及其调度策略

**答案要点：**

**TileScheduler 类型：**

| Scheduler | 用途 | 特点 |
|-----------|------|------|
| `SingleTileScheduler` | 基础调度 | 每个 block 处理一个 tile |
| `StaticPersistentTileScheduler` | 持久化 kernel | block 处理多个 tiles |
| `SingleTileLPTScheduler` | 变长序列优化 | LPT (Longest Processing Time) 策略 |

**LPT 调度策略** (来自 `tile_scheduler.py` 257-381行):

```python
class SingleTileLPTScheduler:
    """L2 感知的 LPT 调度器，优化变长序列的 load balancing"""
    
    def __init__(self, seqlens_k, num_heads, batch_size):
        # 1. 计算每个 head 的 KV 块大小
        kv_sizes = [ceil(seqlen_k / tile_n) for seqlen_k in seqlens_k]
        
        # 2. 按 L2 缓存容量 (50MB) 确定 swizzle 大小
        l2_capacity = 50 * 1024 * 1024
        self.swizzle_size = self.compute_swizzle(l2_capacity, tile_n, head_dim)
        
        # 3. 重排序: 大的 KV 块优先 (LPT 策略)
        self.execution_order = sorted(
            range(len(kv_sizes)),
            key=lambda i: kv_sizes[i],
            reverse=True  # 降序，大的先处理
        )
```

**LPT 策略原理：**

```
标准调度:  序列A(长) → 序列B(短) → 序列C(中) → 序列D(长)
           (顺序执行，尾部可能空闲)

LPT调度:   序列A(长) → 序列D(长) → 序列C(中) → 序列B(短)
           (优先处理长的，减少尾部空闲)
```

**优势：**
1. **更好的 load balancing**: 长序列先开始，避免尾部只有一个长序列拖慢整体
2. **L2 缓存优化**: 相同大小的序列一起处理，提高缓存命中率
3. **适合变长**: 根据实际 KV 大小动态排序

---

## 四、高级特性题 (高级)

### Q9: Block Sparse Attention 的实现原理是什么？

**答案要点：**

**数据结构** (来自 `block_sparsity.py`):

```python
@dataclass
class BlockSparseTensors:
    mask_block_cnt: Tensor  # 每个 Q 块需要计算的 KV 块数
    mask_block_idx: Tensor  # 需要计算的 KV 块索引
    full_block_cnt: Tensor  # 完全计算块数 (跳过 mask_mod)
    full_block_idx: Tensor  # 完全计算块索引
```

**执行逻辑：**

```python
# flash_fwd_sm90.py / flash_fwd_sm100.py
for each Q tile (m_block):
    # 1. 获取该 Q tile 需要计算的 KV 块列表
    n_blocks = get_sparse_blocks(m_block)
    
    for n_block in n_blocks:
        if n_block in full_blocks:
            # FULL block: 直接计算，不应用 mask_mod
            acc_s = gemm(Q, K[n_block])
            acc_s = apply_softmax(acc_s)
        else:
            # PARTIAL block: 需要应用 mask_mod
            acc_s = gemm(Q, K[n_block])
            mask = mask_mod(q_idx, k_idx)
            acc_s = apply_mask(acc_s, mask)
            acc_s = apply_softmax(acc_s)
        
        O += acc_s @ V[n_block]
```

**与 FlexAttention 集成：**

```python
# 自定义 mask_mod 函数
@cute.jit
def sliding_window_mask(q_idx, k_idx, window_size=512):
    """滑动窗口 mask"""
    return abs(q_idx - k_idx) <= window_size

@cute.jit
def prefix_lm_mask(q_idx, k_idx, prefix_len=1024):
    """Prefix LM mask: prefix 部分双向，其余 causal"""
    if k_idx < prefix_len:
        return True  # prefix 部分全可见
    else:
        return k_idx <= q_idx  # 其余 causal
```

**优势：**
- 长序列场景下减少无效计算 (如 MoE、滑动窗口)
- 支持结构化稀疏模式

---

### Q10: Paged KV Cache 是如何实现的？有什么优势？

**答案要点：**

**Paged KV Cache 实现** (来自 `paged_kv.py`):

```python
class PagedKVManager:
    def __init__(self, page_table, page_size=128):
        self.page_table = page_table  # 页表映射: 逻辑页 → 物理页
        self.page_size = page_size
    
    def compute_K_ptr(self, batch_idx, head_idx, seq_idx):
        """通过页表查找物理地址"""
        page_idx = seq_idx // self.page_size
        page_offset = seq_idx % self.page_size
        physical_page = self.page_table[batch_idx, page_idx]
        
        return base_ptr + physical_page * page_size * head_dim + \
               page_offset * head_dim
    
    def load_kv_tile(self, tma_k, tma_v, n_block):
        """TMA 加载一页 KV"""
        # SM90: V 布局为 (page_size, dv, num_pages)
        # SM100: V 布局为 (dv, page_size, num_pages) - 转置优化
        k_ptr = self.compute_K_ptr(batch_idx, head_idx, n_block * tile_n)
        v_ptr = self.compute_V_ptr(batch_idx, head_idx, n_block * tile_n)
        
        cute.tma_load(tma_k, k_ptr)
        cute.tma_load(tma_v, v_ptr)
```

**布局差异：**

| 特性 | SM90 | SM100 |
|------|------|-------|
| V 布局 | (page_size, dv, num_pages) | (dv, page_size, num_pages) - 转置 |
| TMA 支持 | 是 | 是 |
| 加载优化 | 标准 | 优化内存访问模式 |

**优势：**
1. **减少内存碎片**: 固定页大小避免变长分配
2. **支持大 batch**: 动态页分配适应不同序列长度
3. **推理优化**: 与 vLLM 等推理框架兼容
4. **内存复用**: 释放的页可以回收再利用

---

### Q11: 解释 Flash Attention 后向传播 (Backward) 的梯度计算流程

**答案要点：**

**梯度计算公式：**

```
已知: dO (输出梯度), O (输出), LSE (log-sum-exp)
求: dQ, dK, dV

1. 重计算 S = QK^T
2. P = softmax(S)  (用保存的 LSE)
3. dV = P^T @ dO
4. dP = dO @ V^T  
5. dS = P * (dP - sum(dO * O, dim=-1))  # softmax 导数
6. dQ = dS @ K
7. dK = dS^T @ Q
```

**文件结构：**

| 文件 | 功能 |
|------|------|
| `flash_bwd_sm90.py` | Hopper 后向实现 |
| `flash_bwd_sm100.py` | Blackwell 后向 (支持块稀疏) |
| `flash_bwd_preprocess.py` | 预处理: 计算 dO * O |
| `flash_bwd_postprocess.py` | 后处理: 缩放和存储梯度 |

**实现流程** (来自 `flash_bwd_sm90.py`):

```python
class FlashAttentionBackwardSm90:
    def __call__(self, dO, Q, K, V, O, LSE):
        # 预处理: 计算 dO * O
        delta = sum(dO * O, dim=-1)  # flash_bwd_preprocess.py
        
        # KV 迭代 Q 的 tiling 策略 (与正向相反)
        for n_block in range(num_n_blocks):
            # 加载 K[n_block], V[n_block]
            
            for m_block in range(num_m_blocks):
                # 计算 S = Q[m_block] @ K[n_block]^T
                # 应用 softmax (用保存的 LSE)
                # 计算 dV_accum += P^T @ dO
                # 计算 dS
                # 累加 dQ[m_block], dK[n_block]
        
        # 后处理: 缩放和存储
        dQ, dK, dV = postprocess(dQ_accum, dK_accum, dV_accum)
```

**优化策略：**

1. **重计算而非存储**: 反向时重新计算 S 矩阵，节省 HBM 带宽
2. **反向 tiling**: KV 迭代 Q，获得更好并行度
3. **块稀疏支持** (SM100): 只对需要的块计算梯度

---

## 五、性能调优与工程实践题 (高级)

### Q12: Flash Attention 的 JIT 编译缓存系统是如何工作的？

**答案要点：**

**缓存系统组成** (来自 `cache_utils.py`):

```python
def get_jit_cache(cache_dir=None):
    """
    两级缓存: 内存 LRU + 磁盘缓存
    缓存路径: /tmp/${USER}/flash_attention_cute_dsl_cache/
    """
    return JITCache(memory_lru_size=100, disk_cache_dir=cache_dir)
```

**缓存键构成** (来自 `interface.py` 592-625行):

```python
compile_key = (
    dtype,              # 数据类型 (fp16/bf16/fp8)
    head_dim,           # 头维度 (64/96/128/192/256)
    head_dim_v,         # V 的头维度
    qhead_per_kvhead,   # GQA 打包数
    causal,             # 是否 causal
    score_mod_hash,     # score 修改函数 hash
    mask_mod_hash,      # mask 修改函数 hash
    use_block_sparsity, # 是否块稀疏
    tile_m, tile_n,     # tile 大小
    num_threads,        # 线程数
    arch,               # 架构 (90/100/110/120)
    use_2cta_instrs,    # 是否用 2CTA
    mma_pv_is_rs,       # MMA 类型
    intra_wg_overlap,   # warp group 内重叠
)
```

**快速测试流程：**

```bash
# Pass 1: 并行编译 (无 GPU 内存分配)
FLASH_ATTENTION_FAKE_TENSOR=1 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest -n 64 tests/cute/test_flash_attn.py

# Pass 2: 运行测试 (使用缓存)
FLASH_ATTENTION_FAKE_TENSOR=0 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest tests/cute/test_flash_attn.py
```

**环境变量：**

| 变量 | 作用 |
|------|------|
| `FLASH_ATTENTION_FAKE_TENSOR=1` | 使用 FakeTensorMode 编译，不分配 GPU 内存 |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` | 启用磁盘缓存 |
| `CUTE_CUBIN_PATH` | 导出 CUBIN/SASS 路径 |
| `CUTE_DSL_KEEP_PTX=1` | 保留 PTX 文件 |

---

### Q13: 如何调试 Flash Attention GPU kernel 的性能问题？

**答案要点：**

**1. 编译与调试工具：**

```bash
# 导出 PTX 和 SASS
CUTE_DSL_KEEP_PTX=1 \
CUTE_DSL_LINEINFO=1 \
python test_script.py

# 使用 compute-sanitizer (注意 TMA 有假阳性)
compute-sanitizer --tool=racecheck python test.py

# Nsight Compute 性能分析
ncu -k flash_attn --metrics sm__throughput python test.py
```

**2. Kernel 内打印调试：**

```python
# cute.printf 配合线程过滤
cute.printf("tidx=%d, value=%f\n", 
            cute.arch.thread_idx(), 
            value,
            cond=cute.arch.thread_idx() % 32 == 0)  # 只打印每 32 个线程

# 使用 elect_one() 只让一个线程打印
if cute.arch.elect_one():
    cute.printf("Checkpoint A reached\n")
```

**3. 2CTA 调试：**

```python
# 检查 cluster 同步
cute.arch.mbarrier_init(...)
cute.arch.cluster_sync()  # 同步点

# 检查 tx_count 设置 (2CTA 需要 ×2)
tx_count = base_tx_count * cta_group_size  # cta_group_size=2 for 2CTA
```

**4. 常见问题排查：**

| 问题 | 排查方法 | 解决 |
|------|---------|------|
| OOM | `nvidia-smi` | 减少 batch_size / tile_size |
| Hang | printf bisection | 检查 barrier 同步 |
| 结果不对 | FakeTensor 对比 | 检查 index 计算 |
| 性能差 | Nsight Compute | 调整 tile 大小 |

---

### Q14: 解释 SM100 中 Softmax 的特殊优化 - Exp2 模拟

**答案要点：**

**背景**：SM100 (Blackwell) 架构上 `exp2` 指令吞吐量较低，需要特殊优化。

**Exp2 模拟** (来自 `softmax.py` 238-330行):

```python
class SoftmaxSm100(Softmax):
    def apply_exp2_convert(self, acc_S_row, acc_S_row_converted,
                          ex2_emu_freq=0, ex2_emu_res=4):
        """
        ex2_emu_freq: 每隔多少元素使用模拟
        ex2_emu_res: 模拟的精度级别
        
        ex2_emu_freq=0: 全部使用标准 exp2
        ex2_emu_freq>0: 部分使用模拟，平衡精度和性能
        """
        frg_tile = 32
        frg_cnt = len(acc_S_row) // frg_tile
        
        for j in range(frg_cnt):
            for k in range(0, frg_tile, 2):
                if ex2_emu_freq == 0:
                    # 标准 exp2
                    acc_S_row[k, j] = cute.math.exp2(acc_S_row[k, j], fastmath=True)
                    acc_S_row[k+1, j] = cute.math.exp2(acc_S_row[k+1, j], fastmath=True)
                else:
                    # 混合策略: 大部分用标准 exp2，少量用模拟
                    if (k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res or 
                        j >= frg_cnt - 1 or j < ex2_emu_start_frg):
                        acc_S_row[k, j] = cute.math.exp2(acc_S_row[k, j], fastmath=True)
                        acc_S_row[k+1, j] = cute.math.exp2(acc_S_row[k+1, j], fastmath=True)
                    else:
                        # 使用 exp2 模拟 (多项式逼近)
                        acc_S_row[k, j], acc_S_row[k+1, j] = \
                            utils.ex2_emulation_2(acc_S_row[k, j], acc_S_row[k+1, j])
```

**多项式系数** (来自 `fast_math.py`):

```python
# exp2 多项式逼近系数
EXP2_POLY_COEF = [
    1.0,           # 常数项
    0.693147,      # x^1
    0.240227,      # x^2
    0.055504,      # x^3
    0.009618,      # x^4
]

def ex2_emulation(x):
    """exp2 的多项式逼近"""
    result = 0.0
    for i, coef in enumerate(EXP2_POLY_COEF):
        result += coef * (x ** i)
    return result
```

---

### Q15: 如何自定义 score_mod 和 mask_mod 函数？

**答案要点：**

**score_mod 示例** (来自 `tests/cute/score_mod_definitions.py`):

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def alibi_score_mod(score, batch_idx, head_idx, q_idx, kv_idx, **kwargs):
    """Alibi 位置编码: 基于 q 和 k 的距离增加惩罚"""
    distance = abs(q_idx - kv_idx)
    # 每个 head 不同 slope
    slope = 2.0 ** (-(8 + head_idx) / 8.0)
    return score - distance * slope

@cute.jit  
def softcap_score_mod(score, batch_idx, head_idx, q_idx, kv_idx, **kwargs):
    """Softcap: 限制 score 最大值"""
    softcap = 50.0
    return softcap * cute.math.tanh(score / softcap)

# 使用
flash_attn_func(q, k, v, score_mod=alibi_score_mod)
```

**mask_mod 示例** (来自 `tests/cute/mask_mod_definitions.py`):

```python
@cute.jit
def sliding_window_mask(q_idx, kv_idx, window_size=512):
    """滑动窗口 mask: 只关注附近 window_size 个 token"""
    return abs(q_idx - kv_idx) <= window_size

@cute.jit
def prefix_lm_mask(q_idx, kv_idx, prefix_len=1024):
    """Prefix LM mask: prefix 部分双向，其余 causal"""
    if kv_idx < prefix_len:
        return True  # prefix 部分全可见
    else:
        return kv_idx <= q_idx  # 其余 causal

@cute.jit
def document_mask(q_idx, kv_idx, seqlen_info, **kwargs):
    """文档边界 mask: 不同文档间不可见"""
    # 使用 seqlen_info 获取文档边界
    return seqlen_info.get_document_id(q_idx) == seqlen_info.get_document_id(kv_idx)

# 使用
flash_attn_func(q, k, v, mask_mod=sliding_window_mask)
```

**实现原理** (来自 `softmax.py` 343-471行):

```python
@cute.jit
def apply_score_mod_inner(score_tensor, index_tensor, score_mod, 
                          batch_idx, head_idx, softmax_scale,
                          vec_size=4,  # 向量化处理
                          ...):
    """向量化应用 score_mod，一次处理 4 个元素"""
    n_vals = len(score_tensor)
    score_vec = cute.make_rmem_tensor(vec_size, Float32)
    
    for i in range(0, n_vals, vec_size):
        # 加载 vec_size 个元素
        for j in range(vec_size):
            score_vec[j] = score_tensor[i + j] * softmax_scale
        
        # 获取索引
        q_idx_vec = index_tensor[i:i+vec_size, 0]
        kv_idx_vec = index_tensor[i:i+vec_size, 1]
        
        # 调用 score_mod (向量化)
        modified = score_mod(
            score_vec.load(),
            batch_idx.broadcast(vec_size),
            head_idx.broadcast(vec_size),
            q_idx_vec,
            kv_idx_vec,
            ...
        )
        
        # 写回
        score_vec.store(modified)
        for j in range(vec_size):
            score_tensor[i + j] = score_vec[j]
```

---

## 六、代码阅读题 (高级)

### Q16: 阅读以下代码并解释其作用

```python
# 来自 mask.py (15-66行)
def r2p_bitmask_below(limit, s):
    """Register-to-Predicate 位掩码"""
    m = max((s + 1) * 32 - limit, 0)
    return shr_u32(0xFFFFFFFF, m)
```

**答案要点：**

**R2P (Register-to-Predicate)** 是高效的 mask 实现方式，使用 32-bit 寄存器批量表示 32 个 boolean 值。

**原理：**

```
对于 limit=10，需要表示第 0-9 位为 1 (有效)，第 10-31 位为 0 (无效)

m = max((s + 1) * 32 - limit, 0)
  = max(32 - 10, 0) = 22

shr_u32(0xFFFFFFFF, 22) = 0x000003FF
二进制: 0000 0000 0000 0000 0000 0011 1111 1111
                              ↑       ↑
                             位10    位0-9全为1
```

**应用场景：**
- SM90 warp-level mask 操作
- 批量处理 32 个元素的 mask
- 避免逐个元素的条件分支

---

### Q17: 阅读以下代码并解释 Warp Group 同步机制

```python
# 来自 hopper_helpers.py / blackwell_helpers.py
def wgmma_fence():
    """WGMMA 同步栅栏"""
    cute.arch.wgmma_fence_sync()

def wgmma_commit_group():
    """提交 WGMMA 操作组"""
    cute.arch.wgmma_commit_group()

def wgmma_wait_group(num_pending=0):
    """等待 WGMMA 操作完成"""
    cute.arch.wgmma_wait_group(num_pending)
```

**答案要点：**

**WGMMA (Warp Group Matrix Multiply Accumulate) 同步：**

```
执行流程:
1. wgmma_fence()        - 确保之前内存操作完成
2. wgmma_commit_group() - 提交一组 WGMMA 指令
3. ... 其他计算 ...    - 重叠执行
4. wgmma_wait_group(0)  - 等待所有 WGMMA 完成
```

**Pipeline 中的使用：**

```python
# 典型的 WGMMA pipeline
pipeline.consumer_wait(state)
wgmma_fence()
acc_s = cute.wgmma(Q_tile, K_tile)  # 提交 WGMMA
wgmma_commit_group()
pipeline.consumer_release(state)

# 下一次迭代前等待
wgmma_wait_group(0)  # 等待完成
```

---

## 七、系统设计题 (高级)

### Q18: 设计一个支持动态序列长度的 Flash Attention 优化方案

**答案要点：**

**问题分析：**
- 实际场景中序列长度变化大 (128 ~ 128K)
- 固定 tile 大小无法兼顾短序列和长序列性能
- 需要平衡计算并行度和内存利用率

**设计方案：**

```python
# 1. 自适应 Tile 大小选择
def select_tile_size(seqlen, head_dim, is_causal):
    """基于序列长度选择最优 tile 大小"""
    if seqlen <= 1024:
        # 短序列: 大 tile，提高数据复用
        return FwdConfig(192, 192, mma_pv_is_rs=False)
    elif seqlen <= 4096:
        # 中等序列: 平衡
        return FwdConfig(192, 128, mma_pv_is_rs=True)
    else:
        # 长序列: 小 tile，提高并行度
        return FwdConfig(128, 128, mma_pv_is_rs=True)

# 2. 混合调度策略
class AdaptiveScheduler:
    def __init__(self, seqlens):
        # 分类序列
        short_seqs = [s for s in seqlens if s <= 1024]
        medium_seqs = [s for s in seqlens if 1024 < s <= 4096]
        long_seqs = [s for s in seqlens if s > 4096]
        
        # 不同长度使用不同调度器
        self.schedulers = {
            'short': SingleTileScheduler(short_seqs),
            'medium': StaticPersistentTileScheduler(medium_seqs),
            'long': SingleTileLPTScheduler(long_seqs, use_splitkv=True)
        }

# 3. 变长序列 Pack 策略
def pack_variable_length(sequences, max_pack_size=4):
    """将不同长度的序列打包到同一 batch，用 padding 对齐"""
    # 类似 vLLM 的 iteration-level scheduling
    # 但粒度更细: tile-level packing
```

**性能优化点：**
1. **LPT 调度**: 长序列优先处理，减少尾部空闲
2. **SplitKV**: 超长序列分割到多个 SM
3. **Dynamic Tile**: 根据长度动态选择 tile 大小
4. **Persistent Kernel**: 短序列合并处理，减少 kernel 启动开销

---

### Q19: 如何为新的 GPU 架构 (如 SM120) 添加 Flash Attention 支持？

**答案要点：**

**实现步骤：**

```python
# 1. 创建新的 forward kernel (flash_fwd_sm120.py)
class FlashAttentionForwardSm120:
    def __init__(self, ...):
        # 初始化架构特定参数
        self.use_2cta = ...  # 是否支持 2CTA
        self.mma_inst = ...  # MMA 指令选择
    
    def __call__(self, Q, K, V):
        # 实现 SM120 特定的 kernel 逻辑
        # 参考 SM100 实现，调整:
        # - MMA 指令
        # - Pipeline 类型
        # - Tile 大小
        pass

# 2. 创建新的 backward kernel (flash_bwd_sm120.py)
class FlashAttentionBackwardSm120:
    # 类似 forward，实现反向传播
    pass

# 3. 在 interface.py 中注册
_ARCH_TO_FWD_KERNEL = {
    80: FlashAttentionForwardSm80,
    90: FlashAttentionForwardSm90,
    100: FlashAttentionForwardSm100,
    110: FlashAttentionForwardSm100,  # SM110 复用 SM100
    120: FlashAttentionForwardSm120,  # 新增
}

# 4. 添加 tile size 配置
def _tile_size_fwd_sm120(head_dim, head_dim_v, is_causal):
    """SM120 最优 tile 配置"""
    # 基于硬件特性调整
    if head_dim <= 64:
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    else:
        return FwdConfig(128, 96, True, True)

# 5. 添加测试
# tests/cute/test_flash_attn.py 会自动覆盖
```

**关键考虑点：**

1. **架构特性分析**：
   - 新的 MMA 指令格式
   - 共享内存大小变化
   - Warp 组织方式

2. **Tile 大小调优**：
   - 使用 grid search 寻找最优配置
   - 考虑不同 head_dim 和 seqlen

3. **Pipeline 选择**：
   - 是否支持新的异步加载指令
   - Barrier 同步机制变化

---

## 附录: 面试准备建议

### 重点掌握

1. **核心算法**：Online Softmax 数学推导、Tiling 策略
2. **GPU 架构**：SM90 vs SM100 差异、WGMMA/UMMA/TMA
3. **优化技巧**：Pack GQA、SplitKV、Block Sparse、LPT 调度
4. **工程实践**：JIT 缓存、调试技巧、性能分析

### 推荐阅读

- FlashAttention-1/2/3 原始论文
- NVIDIA CUTLASS/CuTe 文档
- CUDA Programming Guide (Hopper/Blackwell 章节)
- 本仓库 `CLAUDE.md` 开发指南

### 实践建议

```bash
# 1. 阅读核心实现
vim flash_attn/cute/softmax.py      # Online softmax
vim flash_attn/cute/block_info.py   # Tiling 逻辑
vim flash_attn/cute/interface.py    # API 入口
vim flash_attn/cute/flash_fwd_sm90.py   # Hopper 实现
vim flash_attn/cute/flash_fwd_sm100.py  # Blackwell 实现

# 2. 运行测试理解行为
pytest tests/cute/test_flash_attn.py::test_flash_attn_output -xvs
pytest tests/cute/test_flash_attn_varlen.py -xvs

# 3. 调试观察
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 python your_test.py

# 4. 性能分析
ncu -k flash_attn --metrics sm__throughput python benchmark.py
```

---

**祝你面试顺利！**

*文档生成日期: 2026-04-13*  
*基于 FlashAttention-4 仓库*

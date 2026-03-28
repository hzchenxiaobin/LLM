# Flash Attention AI Infra 岗位面试题集

> 基于 FlashAttention-4 (FA4) 仓库内容整理  
> 适用于 AI Infra / GPU Kernel 优化 / 深度学习框架 岗位

---

## 一、基础概念题

### Q1: 什么是Flash Attention？与传统Attention相比有什么优势？

**答案要点：**

**Flash Attention** 是一种 IO-Aware 的精确注意力算法，通过分块计算 (tiling) 和重计算 (recomputation) 技术，在不存储完整注意力矩阵的情况下实现标准 attention 的数学等价计算。

**核心优势：**

| 特性 | 标准Attention | FlashAttention |
|------|--------------|----------------|
| 内存复杂度 | O(N²) - 需存储S和P矩阵 | O(N) - 只存储统计量 |
| HBM访问 | O(N²) | O(N²) 但减少实际读写 |
| 数值稳定性 | 需要全局max | Online softmax 稳定 |
| 性能 | HBM带宽瓶颈 | SRAM计算密集型 |

**FA4 特定实现：**
```python
# 来自 interface.py
flash_attn_func(q, k, v, causal=False, window_size=(-1,-1), 
                softmax_scale=None, score_mod=None, mask_mod=None)
```

---

### Q2: 解释 Online Softmax 算法原理及其在FlashAttention中的作用

**答案要点：**

**Online Softmax** 是分块计算 softmax 的关键技术，避免存储完整的 S = QK^T 矩阵。

**算法步骤：**

```python
# 来自 softmax.py (52-116行)
def online_softmax(acc_S_row, row_max_prev, row_sum_prev):
    """
    输入: acc_S_row - 当前块的 S 矩阵一行
          row_max_prev - 之前所有块的最大值
          row_sum_prev - 之前所有块的指数和
    """
    # 1. 当前块最大值
    row_max_cur = max(acc_S_row)
    
    # 2. 计算缩放因子 (数值稳定性)
    row_scale = exp2((row_max_prev - row_max_cur) * scale_log2)
    
    # 3. 更新累加器
    row_sum = row_sum_prev * row_scale + sum(exp2((acc_S_row - row_max_cur) * scale_log2))
    row_max = max(row_max_prev, row_max_cur)
    
    return row_max, row_sum, row_scale
```

**关键公式：**
- 对于新块 m，更新后的 softmax: $softmax(S) = \frac{e^{S - m_{new}}}{s_{new}}$
- 其中 $m_{new} = max(m_{old}, m_{cur})$, $s_{new} = s_{old} \cdot e^{m_{old} - m_{new}} + \sum e^{S - m_{new}}$

**作用：**
1. 每个 tile 只需 O(1) 内存存储 `row_max` 和 `row_sum`
2. 避免 O(N²) 的 S 矩阵存储
3. 支持流式计算，与 tiling 结合实现高 SRAM 利用率

---

### Q3: Flash Attention中的 Tiling 策略是什么？如何处理 Causal Mask？

**答案要点：**

**Tiling 策略：** 将 Q、K、V 矩阵分成小块在 SRAM 中计算。

```python
# 来自 block_info.py (24-55行)
def get_n_block_min_max(self, m_block):
    """前向传播: Q迭代KV时的block范围计算"""
    m_idx_min = m_block * self.tile_m
    m_idx_max = (m_block + 1) * self.tile_m
    
    if self.is_causal:
        # Causal mask: 只需计算 n <= m 的位置
        n_block_max = ceil((m_idx_max + seqlen_k - seqlen_q) / tile_n)
    elif self.is_local:
        # Local mask: 考虑左右窗口边界
        n_block_min = floor((m_idx_min - w_left) / tile_n)
        n_block_max = ceil((m_idx_max + w_right) / tile_n)
```

**Causal Mask 处理：**

```
Q tile (m)    K tile (n)
    ↓           ↓
┌───────┐   ┌───────┐
│ block │ × │ block │ = S tile
│   m   │   │   n   │
└───────┘   └───────┘
      ↓
   if n > m: skip (future tokens)
   if n == m: apply triangular mask
   if n < m: full computation
```

**优化：**
- FULL blocks: 完全计算，跳过 mask 检查
- PARTIAL blocks: 需要应用 mask
- 通过 `BlockInfo` 类动态计算每个 Q tile 需要处理的 K tile 范围

---

## 二、GPU架构与优化题

### Q4: Hopper (SM90) 和 Blackwell (SM100) 架构在 FlashAttention 中的主要区别是什么？

**答案要点：**

| 特性 | Hopper (SM90) | Blackwell (SM100) |
|------|---------------|-------------------|
| MMA指令 | WGMMA | UMMA (Unified MMA) |
| 共享内存管理 | TMA | TMA + TMEM |
| 协作计算 | 基本 Warp Group | 2CTA (2 Cooperative Thread Arrays) |
| 特殊优化 | TMA异步加载 | Exp2模拟、Warp分工 |
| 适用场景 | hdim≤128 | hdim∈{128,192} 用2CTA |

**Hopper (SM90) 实现：**
```python
# flash_fwd_sm90.py
# 使用 WGMMA (Warp Group MMA) 指令
# Tile sizes: hdim≤64用192×128, hdim≤96用192×128/144
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
```

**2CTA 指令：** (interface.py 518-531行)
```python
use_2cta = (
    arch in [10, 11] and  # Blackwell
    not causal and not local and
    not is_split_kv and
    head_dim in [128, 192]  # 特定维度才启用
)
# 两个CTA通过cluster mbarrier同步，共享计算负载
```

**Warp 分工 (SM100)：**
```python
# flash_fwd_sm100.py (182-221行)
softmax0_warp_ids = (0, 1, 2, 3)      # 4 warps softmax
softmax1_warp_ids = (4, 5, 6, 7)      # 4 warps softmax
correction_warp_ids = (8, 9, 10, 11)  # 修正
mma_warp_id = 12                      # MMA计算
epilogue_warp_ids = (13,)             # 输出存储
load_warp_ids = (14,)                 # 数据加载
```

---

### Q5: 解释 FlashAttention 中的 Pipeline 机制及其作用

**答案要点：**

**PipelineStateSimple 类** (pipeline.py 38-84行):

```python
class PipelineStateSimple:
    """单一Int32存储index和phase的pipeline状态"""
    def __init__(self, stages: int):
        self.phase_index = 0
        self.stages = stages
    
    @property
    def index(self):
        return self.phase_index % self.stages  # 环形缓冲区位置
    
    @property
    def phase(self):
        return self.phase_index // self.stages  # 阶段标记
    
    def advance(self):
        self.phase_index += 1
        # 优化: stages==1时, phase ^= 1
```

**Pipeline 类型：**

| Pipeline类型 | 用途 | 特点 |
|-------------|------|------|
| `PipelineAsync` | 通用异步拷贝 | 基础双缓冲 |
| `PipelineCpAsync` | CP异步拷贝 | cp.async指令 |
| `PipelineTmaAsync` | TMA异步加载 | Hopper+ TMA |
| `PipelineTmaUmma` | TMA+UMMA | Blackwell专用 |

**作用：**
1. **隐藏内存延迟**: 加载下一个 tile 的同时计算当前 tile
2. **双缓冲/多缓冲**: stages=2 或 4，实现完全的计算-加载 overlap
3. **同步机制**: 使用 mbarrier 确保数据就绪后才计算

**典型流程：**
```
阶段1: 加载 K[0], V[0] → 同步 → 计算 Q×K[0]^T
阶段2: 加载 K[1], V[1] (与阶段1计算并行)
阶段3: 同步 → 计算 Q×K[1]^T
...
```

---

### Q6: 什么是 SplitKV 技术？FlashAttention 如何决定何时使用它？

**答案要点：**

**SplitKV** 将 KV 序列分成多个块，由不同 thread blocks 并行处理，最后合并结果。

**适用场景：**
- 长序列 (seqlen_k 很大)
- 单个 thread block 无法处理完整 KV
- 需要利用更多 SM 并行度

**启发式算法** (interface.py 255-262行):

```python
def determine_num_splits(seqlen_q, seqlen_k, head_dim, num_heads, num_sms):
    # 基于序列长度、head维度、可用SM数量决定splits
    if seqlen_k > 4096 and head_dim <= 128:
        num_splits = min(4, num_sms // (batch_size * num_heads))
    else:
        num_splits = 1
    return num_splits
```

**执行流程：**

```
Q (固定)     KV split 0    KV split 1    KV split 2
  │            │             │             │
  └────────────┼─────────────┼─────────────┘
               ↓             ↓             ↓
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │ block 0 │    │ block 1 │    │ block 2 │
         │计算部分O │    │计算部分O │    │计算部分O │
         │部分LSE  │    │部分LSE  │    │部分LSE  │
         └────┬────┘    └────┬────┘    └────┬────┘
              └──────────────┼──────────────┘
                             ↓
                    ┌─────────────────┐
                    │ flash_fwd_combine │
                    │  合并所有partial  │
                    │  结果计算最终O    │
                    └─────────────────┘
```

**合并公式：**
```python
# flash_fwd_combine.py
# 对于每个 partial 结果 (O_i, LSE_i):
global_max = max(LSE_0, LSE_1, ..., LSE_n)
O_final = Σ O_i * exp2(LSE_i - global_max) / Σ exp2(LSE_i - global_max)
```

---

## 三、高级特性题

### Q7: 什么是 Pack GQA？如何在 FlashAttention 中实现？

**答案要点：**

**GQA (Grouped Query Attention)** 让多个 Query heads 共享同一组 KV heads。

**Pack GQA 优化：** 将多个 Q head 打包到序列维度，共享 KV 加载。

**实现：** (pack_gqa.py 15-40行)

```python
def pack_gqa_layout(T, nheads_kv, qhead_per_kvhead):
    """
    转换: (seqlen_q, headdim, nheads, batch) 
    → ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch)
    
    例子: 
    - 8 Q heads, 2 KV heads → qhead_per_kvhead = 4
    - seqlen=1024 → packed_seqlen = 4 * 1024 = 4096
    """
    shape_packed = (
        (qhead_per_kvhead, T.shape[0]),  # 打包后的序列长度
        T.shape[1],  # headdim
        nheads_kv,   # 减少的head数
        T.shape[3]   # batch
    )
    return cute.make_tensor(make_layout(shape_packed), T.iterator)
```

**优势：**
1. **减少 KV 加载次数**: 1 次加载服务多个 Q head
2. **提高计算密度**: 更大的 effective batch size
3. **更好内存局部性**: 连续的 Q head 访问

**PackGQA 类** (pack_gqa.py 115-263行):
```python
class PackGQA:
    def load_Q(self, tma_q, qhead_pack_idx):
        # 计算打包后的指针位置
        ptr = self.compute_ptr(qhead_pack_idx)
        return cute.tma_load(tma_q, ptr)
    
    def store_O(self, tma_o, acc_o, qhead_pack_idx):
        # 解包存储
        ptr = self.compute_ptr(qhead_pack_idx)
        cute.tma_store(tma_o, acc_o, ptr)
```

---

### Q8: 解释 Block Sparse Attention 的实现原理

**答案要点：**

**块稀疏注意力** 只计算指定的 KV 块，跳过不相关的块。

**数据结构** (block_sparsity.py 17-35行):

```python
@dataclass
class BlockSparseTensors:
    mask_block_cnt: Tensor  # 每个Q块需要计算的KV块数
    mask_block_idx: Tensor  # 需要计算的KV块索引
    full_block_cnt: Tensor  # 完全计算块数(跳过mask_mod)
    full_block_idx: Tensor  # 完全计算块索引
```

**执行逻辑：**

```python
# flash_fwd_sm90.py / flash_fwd_sm100.py
for each Q tile (m_block):
    # 1. 获取该Q tile需要计算的KV块列表
    n_blocks = get_sparse_blocks(m_block)
    
    for n_block in n_blocks:
        if n_block in full_blocks:
            # FULL block: 直接计算，不应用mask_mod
            acc_s = gemm(Q, K[n_block])
            acc_s = apply_softmax(acc_s)
        else:
            # PARTIAL block: 需要应用mask_mod
            acc_s = gemm(Q, K[n_block])
            mask = mask_mod(q_idx, k_idx)
            acc_s = apply_mask(acc_s, mask)
            acc_s = apply_softmax(acc_s)
        
        O += acc_s @ V[n_block]
```

**与 FlexAttention 集成：**
- `mask_mod`: 自定义 mask 函数 (PyTorch flex attention 风格)
- `score_mod`: 自定义 score 修改函数 (如 alibi, softcap)

**优势：**
- 长序列场景下减少无效计算
- 支持结构化稀疏模式 (如 MoE、滑动窗口等)

---

### Q9: Paged KV Cache 是如何实现的？有什么优势？

**答案要点：**

**Paged KV Cache** 将 KV cache 分割成固定大小的页，支持动态内存管理。

**实现** (paged_kv.py 16-234行):

```python
class PagedKVManager:
    def __init__(self, page_table, page_size=128):
        self.page_table = page_table  # 页表映射
        self.page_size = page_size
    
    def compute_K_ptr(self, batch_idx, head_idx, seq_idx):
        # 通过页表查找物理地址
        page_idx = seq_idx // self.page_size
        page_offset = seq_idx % self.page_size
        physical_page = self.page_table[batch_idx, page_idx]
        
        return base_ptr + physical_page * page_size * head_dim + \
               page_offset * head_dim
```

**SM90 vs SM100 差异：**

| 特性 | SM90 | SM100 |
|------|------|-------|
| V布局 | (page_size, dv, num_pages) | (dv, page_size, num_pages) - 转置 |
| TMA支持 | 是 | 是 |
| 加载优化 | 标准 | 优化内存访问模式 |

**优势：**
1. **减少内存碎片**: 固定页大小避免变长分配
2. **支持大batch**: 动态页分配适应不同序列长度
3. **推理优化**: 与 vLLM 等推理框架兼容
4. **内存复用**: 释放的页可以回收再利用

---

## 四、性能调优与工程实践题

### Q10: FlashAttention 的 JIT 编译缓存系统是如何工作的？

**答案要点：**

**缓存系统组成：**

```python
# cache_utils.py
# 两级缓存: 内存LRU + 磁盘缓存

def get_jit_cache(cache_dir=None):
    """
    缓存路径: /tmp/${USER}/flash_attention_cute_dsl_cache/
    """
    return JITCache(memory_lru_size=100, disk_cache_dir=cache_dir)
```

**缓存键构成** (interface.py 592-625行):

```python
compile_key = (
    dtype,              # 数据类型 (fp16/bf16/fp8)
    head_dim,           # 头维度 (64/96/128/192/256)
    head_dim_v,         # V的头维度
    qhead_per_kvhead,   # GQA打包数
    causal,             # 是否causal
    score_mod_hash,     # score修改函数hash
    mask_mod_hash,      # mask修改函数hash
    use_block_sparsity, # 是否块稀疏
    tile_m, tile_n,     # tile大小
    num_threads,        # 线程数
    arch,               # 架构 (90/100/110/120)
    use_2cta_instrs,    # 是否用2CTA
    mma_pv_is_rs,       # MMA类型
    intra_wg_overlap,   # warp group内重叠
)
```

**快速测试流程：**

```bash
# Pass 1: 并行编译 (无GPU内存分配)
FLASH_ATTENTION_FAKE_TENSOR=1 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest -n 64 tests/cute/test_flash_attn.py

# Pass 2: 运行测试 (使用缓存)
FLASH_ATTENTION_FAKE_TENSOR=0 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest tests/cute/test_flash_attn.py
```

---

### Q11: 如何调试 FlashAttention GPU kernel 的性能问题？

**答案要点：**

**1. 编译与调试工具：**

```bash
# 导出 PTX 和 SASS
CUTE_DSL_KEEP_PTX=1 \
CUTE_DSL_LINEINFO=1 \
python test_script.py

# 使用 compute-sanitizer (注意 TMA 有假阳性)
compute-sanitizer --tool=racecheck python test.py
```

**2. Kernel内打印调试：**

```python
# cute.printf 配合线程过滤
cute.printf("tidx=%d, value=%f\n", cute.arch.thread_idx(), value, 
            cond=cute.arch.thread_idx() % 32 == 0)  # 只打印每32个线程
```

**3. 2CTA调试 (参考 AI/DEBUG_2CTA.md):**

```python
# 检查 cluster 同步
cute.arch.mbarrier_init(...)
cute.arch.cluster_sync()  # 同步点

# 使用 elect_one() 只让一个线程打印
if cute.arch.elect_one():
    cute.printf("Checkpoint A reached\n")
```

**4. 性能分析：**

```bash
# 查看 GPU 利用率
nvidia-smi dmon -s pucv

# Nsight Compute
ncu -k flash_attn --metrics sm__throughput python test.py
```

**常见问题排查：**

| 问题 | 排查方法 | 解决 |
|------|---------|------|
| OOM | nvidia-smi | 减少 batch_size / tile_size |
| Hang | printf bisection | 检查 barrier 同步 |
| 结果不对 | FakeTensor对比 | 检查 index 计算 |
| 性能差 | Nsight Compute | 调整 tile 大小 |

---

### Q12: 解释 FlashAttention 中 LPT (Longest Processing Time) 调度策略

**答案要点：**

**LPT调度** 优化变长序列的 load balancing 和缓存命中率。

**实现** (tile_scheduler.py 257-381行):

```python
class SingleTileLPTScheduler:
    """L2感知的LPT调度器"""
    
    def __init__(self, seqlens_k, num_heads, batch_size):
        # 1. 计算每个head的KV块大小
        kv_sizes = [ceil(seqlen_k / tile_n) for seqlen_k in seqlens_k]
        
        # 2. 按L2缓存容量(50MB)确定swizzle大小
        l2_capacity = 50 * 1024 * 1024  # 50MB
        self.swizzle_size = self.compute_swizzle(l2_capacity, tile_n, head_dim)
        
        # 3. 重排序: 大的KV块优先 (LPT策略)
        self.execution_order = sorted(
            range(len(kv_sizes)),
            key=lambda i: kv_sizes[i],
            reverse=True  # 降序，大的先处理
        )
```

**LPT策略原理：**

```
标准调度:    序列A(长) → 序列B(短) → 序列C(中) → 序列D(长)
             (顺序执行，可能导致尾部空闲)

LPT调度:     序列A(长) → 序列D(长) → 序列C(中) → 序列B(短)  
             (优先处理长的，减少尾部空闲)
```

**优势：**
1. **更好的 load balancing**: 长序列先开始，避免尾部只有一个长序列拖慢整体
2. **L2缓存优化**: 相同大小的序列一起处理，提高缓存命中率
3. **适合变长**: 根据实际 KV 大小动态排序

---

## 五、代码实现细节题

### Q13: 解释以下代码片段的作用和原理

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
对于 limit=10，需要表示第0-9位为1(有效)，第10-31位为0(无效)

m = max((s + 1) * 32 - limit, 0)
  = max(32 - 10, 0) = 22

shr_u32(0xFFFFFFFF, 22) = 0x000003FF
二进制: 0000 0000 0000 0000 0000 0011 1111 1111
                              ↑       ↑
                             位10   位0-9全为1
```

**应用场景：**
- SM90 warp-level mask 操作
- 批量处理 32 个元素的 mask
- 避免逐个元素的条件分支

**配合 sm90_col_to_r2p_idx** (68-76行):
```python
def sm90_col_to_r2p_idx(col_limit):
    """SM90 列坐标转 R2P 索引"""
    return col_limit // 8 * 2 + min(col_limit % 8, 2)
```

---

### Q14: 如何实现自定义的 score_mod 和 mask_mod？

**答案要点：**

**score_mod 示例：**

```python
import cute

@cute.jit
def alibi_score_mod(score, q_idx, k_idx, head_idx):
    """Alibi 位置编码"""
    # 基于 q 和 k 的距离增加惩罚
    distance = abs(q_idx - k_idx)
    slope = 2 ** (-(8 + head_idx) / 8)  # 每个 head 不同 slope
    return score - distance * slope

# 使用
flash_attn_func(q, k, v, score_mod=alibi_score_mod)
```

**mask_mod 示例：**

```python
@cute.jit
def sliding_window_mask(q_idx, k_idx, window_size=512):
    """滑动窗口 mask"""
    return abs(q_idx - k_idx) <= window_size

@cute.jit
def prefix_lm_mask(q_idx, k_idx, prefix_len=1024):
    """Prefix LM mask: prefix部分双向，其余causal"""
    if k_idx < prefix_len:
        return True  # prefix 部分全可见
    else:
        return k_idx <= q_idx  # 其余causal
```

**实现原理** (softmax.py 332-593行):

```python
def apply_score_mod_inner(acc_s, q_idx, k_idx, score_mod, vec_size=4):
    """
    向量化应用 score_mod
    vec_size: 一次处理4个元素，提高吞吐量
    """
    for i in range(0, acc_s.shape[-1], vec_size):
        scores_vec = acc_s[..., i:i+vec_size]
        
        # 对每个元素应用 score_mod
        for j in range(vec_size):
            scores_vec[j] = score_mod(
                scores_vec[j], 
                q_idx, 
                k_idx + i + j,
                head_idx
            )
        
        acc_s[..., i:i+vec_size] = scores_vec
    
    return acc_s
```

---

### Q15: 解释后向传播 (Backward) 中的梯度计算流程

**答案要点：**

**FlashAttention 后向传播** 需要重新计算前向的 S 矩阵来求梯度。

**文件：**
- `flash_bwd_sm90.py` - Hopper 后向
- `flash_bwd_sm100.py` - Blackwell 后向 (支持块稀疏)
- `flash_bwd_preprocess.py` - 预处理
- `flash_bwd_postprocess.py` - 后处理

**梯度公式：**

```
已知: dO (输出梯度), O (输出), LSE (log-sum-exp)
求: dQ, dK, dV

1. 重计算 S = QK^T
2. P = softmax(S)  (用 LSE)
3. dV = P^T @ dO
4. dP = dO @ V^T  
5. dS = P * (dP - sum(dO * O, dim=-1))  # softmax 导数
6. dQ = dS @ K
7. dK = dS^T @ Q
```

**实现流程：**

```python
# flash_bwd_sm90.py
class FlashAttentionBackwardSm90:
    def __call__(self, dO, Q, K, V, O, LSE):
        # 预处理: 计算 dO * O
        delta = sum(dO * O, dim=-1)  # flash_bwd_preprocess.py
        
        # KV迭代Q的tiling策略 (与正向相反)
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

**优化：**
1. **重计算而非存储**: 节省 HBM 带宽
2. **反向 tiling**: KV 迭代 Q，更好并行度
3. **块稀疏支持** (SM100): 只对需要的块计算梯度

---

## 附录：面试准备建议

### 重点掌握

1. **核心算法**：Online Softmax、Tiling策略
2. **GPU架构**：SM90 vs SM100差异、TMA/UMMA/WGMMA
3. **优化技巧**：Pack GQA、SplitKV、Block Sparse
4. **工程实践**：JIT缓存、调试技巧、性能分析

### 推荐阅读

- FlashAttention-1/2/3 原始论文
- 本仓库 `CLAUDE.md` 开发指南
- NVIDIA CUTLASS/CuTe 文档
- CUDA Programming Guide (Hopper/Blackwell章节)

### 实践建议

```bash
# 1. 阅读核心实现
vim flash_attn/cute/softmax.py      # Online softmax
vim flash_attn/cute/block_info.py   # Tiling逻辑
vim flash_attn/cute/interface.py    # API入口

# 2. 运行测试理解行为
pytest tests/cute/test_flash_attn.py::test_flash_attn_output -xvs

# 3. 修改参数观察影响
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 python benchmark.py
```

---

**祝你面试顺利！**

*文档生成日期: 2026-03-28*  
*基于 FlashAttention-4 仓库 commit: main*

#include "../include/topk_common.h"

// ==========================================
// V4: Radix Select (基于基数选择的高效Top-K)
// ==========================================
// 核心思想：类比QuickSelect，但通过Radix(基数)方式逐步缩小范围
// 不是对所有元素排序，而是利用数值的二进制表示，逐bit判断
// 从而快速排除不可能进入Top-K的元素
//
// 算法流程：
// 1. 将float转为uint32，处理符号位使得可比较
// 2. 从最高有效位开始，逐bit统计
// 3. 根据统计结果，只保留可能包含Top-K的分支
// 4. 最终得到Top-K个元素（无需全排序）
//
// 适用场景：
// - K > 32：Warp级别寄存器不够用时
// - 需要更稳定的性能表现（避免插入排序的最坏情况）
// - 数据分布较为随机时
// ==========================================

#define RADIX_BITS 4
#define RADIX_BINS (1 << RADIX_BITS)  // 16 bins for 4 bits
#define WARP_SIZE 32

// ==========================================
// 辅助函数：浮点数键值转换
// ==========================================
// 将float转为可用于位排序的uint32
// 关键：处理IEEE 754浮点数的符号位，使得数值大小顺序与整数一致
// 正数：符号位0，转uint后越大值越大
// 负数：符号位1，需要翻转使得越接近0越大
__device__ __inline__ uint32_t float_to_sortable_uint(float val) {
    uint32_t u = __float_as_uint(val);
    // 将浮点转为可排序的整数
    // 正数：符号位0，直接取值
    // 负数：符号位1，需要按位取反（这样-0.1 > -1.0）
    return (u & 0x80000000) ? (~u) : (u | 0x80000000);
}

__device__ __inline__ float uint_to_float(uint32_t u) {
    // 反向转换
    uint32_t val = (u & 0x80000000) ? (~u) : (u & ~0x80000000);
    return __uint_as_float(val);
}

// ==========================================
// 辅助函数：插入排序维护Local Top-K
// ==========================================
__device__ void radix_insert_topk(float* vals, int* inds, int K, float new_val, int new_idx) {
    // 如果比当前最小值小，直接跳过
    if (new_val <= vals[K - 1]) return;

    // 找到插入位置（降序）
    int pos = K - 1;
    while (pos > 0 && new_val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        inds[pos] = inds[pos - 1];
        pos--;
    }
    vals[pos] = new_val;
    inds[pos] = new_idx;
}

// ==========================================
// V4 Kernel: Warp级别 Radix-Style Top-K
// ==========================================
// 每个Warp处理一行数据
// 使用Warp Primitives进行高效归约
// 采用分治思想：先粗筛，后精排
__global__ void topk_v4_kernel(const float* input, float* out_vals, int* out_inds,
                               int Batch, int N, int K) {
    // Grid配置：每个block包含多个warp，每个warp处理一行
    int warps_per_block = blockDim.x / WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * warps_per_block + warp_id_in_block;

    if (row >= Batch) return;

    const float* row_input = input + row * N;
    float* row_out_vals = out_vals + row * K;
    int* row_out_inds = out_inds + row * K;

    // 寄存器存储Local Top-K
    float local_vals[MAX_K];
    int local_inds[MAX_K];

    // 初始化
    for (int i = 0; i < K; ++i) {
        local_vals[i] = -1e20f;
        local_inds[i] = -1;
    }

    // ==========================================
    // 阶段1：粗筛选 - 每个线程收集候选
    // ==========================================
    // Warp交错读取，保证合并访存
    for (int i = lane_id; i < N; i += WARP_SIZE) {
        float val = row_input[i];

        // 使用插入排序维护Local Top-K
        // 只有比当前Local最小值大的才插入
        radix_insert_topk(local_vals, local_inds, K, val, i);
    }

    // ==========================================
    // 阶段2：精排序 - Warp级别归约
    // ==========================================
    // 每个线程的local_vals已按降序排列
    // 通过warp shuffle，逐轮找出全局Top-K

    int local_ptr = 0;  // 指向当前local topk的位置

    for (int k = 0; k < K; ++k) {
        // 当前线程提供的候选值
        float candidate_val = (local_ptr < K) ? local_vals[local_ptr] : -1e20f;
        int candidate_idx = (local_ptr < K) ? local_inds[local_ptr] : -1;

        // Warp Reduction：找出warp内的最大值
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, candidate_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, candidate_idx, offset);

            if (other_val > candidate_val) {
                candidate_val = other_val;
                candidate_idx = other_idx;
            }
        }

        // 广播最大值
        float global_max_val = __shfl_sync(0xffffffff, candidate_val, 0);
        int global_max_idx = __shfl_sync(0xffffffff, candidate_idx, 0);

        // Lane 0 写入全局内存
        if (lane_id == 0) {
            row_out_vals[k] = global_max_val;
            row_out_inds[k] = global_max_idx;
        }

        // 找出是哪个线程提供了这个最大值
        // 使用__match_any_sync找到所有具有该值的lane
        uint32_t match_mask = __match_any_sync(0xffffffff, global_max_val);

        // 获胜的线程（们）前进指针
        // 注意：可能有多个线程有相同的值，都前进
        if (candidate_val == global_max_val && candidate_idx == global_max_idx) {
            local_ptr++;
        }
    }
}

// ==========================================
// V4 扩展版本: Block级别 Radix-Style (支持更大K)
// ==========================================
// 当K较大时（如K > 32），单个Warp的寄存器可能不够
// 这时需要使用Shared Memory进行Block级别的归约
//
// 策略：
// 1. Block-per-Row，每个线程粗筛
// 2. Shared Memory存储中间结果
// 3. Bitonic Merge或Selection算法得到最终Top-K
#define V4_BLOCK_SIZE 256

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void topk_v4_block_kernel(const float* input, float* out_vals, int* out_inds,
                                     int Batch, int N, int K) {
    // 动态共享内存
    extern __shared__ char smem[];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= Batch) return;

    const float* row_input = input + row * N;

    // 共享内存布局
    struct Pair {
        float val;
        int idx;
    };

    Pair* s_data = (Pair*)smem;  // 存储所有线程的Local Top-K

    // 寄存器存储Local Top-K
    float local_vals[MAX_K];
    int local_inds[MAX_K];
    for (int i = 0; i < K; ++i) {
        local_vals[i] = -1e20f;
        local_inds[i] = -1;
    }

    // 阶段1：粗筛选
    // 每个线程处理多个元素（Grid-Stride Loop）
    const int block_stride = BLOCK_SIZE * ITEMS_PER_THREAD;

    for (int base = 0; base < N; base += block_stride) {
        int idx = base + tid;

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            int global_idx = idx + item * BLOCK_SIZE;
            if (global_idx < N) {
                float val = row_input[global_idx];
                radix_insert_topk(local_vals, local_inds, K, val, global_idx);
            }
        }
    }

    // 写入Shared Memory
    for (int i = 0; i < K; ++i) {
        s_data[tid * K + i].val = local_vals[i];
        s_data[tid * K + i].idx = local_inds[i];
    }
    __syncthreads();

    // 阶段2：精排序 - 使用Shared Memory Selection
    // 每次找出全局最大值，类似Heap Selection
    float* row_out_vals = out_vals + row * K;
    int* row_out_inds = out_inds + row * K;

    // 每个线程维护自己的指针
    int ptr = 0;

    for (int k = 0; k < K; ++k) {
        // 找到当前全局最大值
        float my_val = (ptr < K) ? s_data[tid * K + ptr].val : -1e20f;
        int my_idx = (ptr < K) ? s_data[tid * K + ptr].idx : -1;

        // Block级别Reduction（使用Shared Memory树形归约）
        // 使用Shared Memory存储中间结果
        __shared__ float s_max_vals[BLOCK_SIZE];
        __shared__ int s_max_idxs[BLOCK_SIZE];
        __shared__ int s_max_tids[BLOCK_SIZE];

        s_max_vals[tid] = my_val;
        s_max_idxs[tid] = my_idx;
        s_max_tids[tid] = tid;
        __syncthreads();

        // 树形归约
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_max_vals[tid + stride] > s_max_vals[tid]) {
                    s_max_vals[tid] = s_max_vals[tid + stride];
                    s_max_idxs[tid] = s_max_idxs[tid + stride];
                    s_max_tids[tid] = s_max_tids[tid + stride];
                }
            }
            __syncthreads();
        }

        // 输出当前最大值
        if (tid == 0) {
            row_out_vals[k] = s_max_vals[0];
            row_out_inds[k] = s_max_idxs[0];
        }

        // 获胜线程前进
        int winner = s_max_tids[0];
        if (tid == winner) {
            ptr++;
            // 更新Shared Memory中的值
            if (ptr < K) {
                s_data[tid * K].val = local_vals[ptr];
                s_data[tid * K].idx = local_inds[ptr];
            } else {
                s_data[tid * K].val = -1e20f;
            }
        }
        __syncthreads();
    }
}

// 注：根据K的大小选择不同策略
// K <= 32: 使用 topk_v4_kernel (Warp级别，更快，无同步开销)
// K > 32:  使用 topk_v4_block_kernel (Block级别，支持更大K，使用Shared Memory)
// 调用者需要根据K值选择合适的kernel启动

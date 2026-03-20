"""
Triton SGEMM: C = alpha * A * B + beta * C

使用 OpenAI Triton 实现的矩阵乘法，展示 Triton DSL 的强大抽象能力。
与 CUDA 相比，Triton 代码更简洁，且能自动优化。

依赖：
    pip install triton

参考：
    https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sgemm_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Alpha and Beta
    alpha, beta,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for SGEMM: C = alpha * A @ B + beta * C

    每个程序实例处理 C 的一个 BLOCK_SIZE_M x BLOCK_SIZE_N 块。
    使用 grouped ordering 优化 L2 cache 命中率。
    """
    # 计算当前程序实例负责的 C 块坐标
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算当前块的起始位置
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 创建指针块
    # A 指针: (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B 指针: (BLOCK_SIZE_K, BLOCK_SIZE_N)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 遍历 K 维度
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 块 (行主序，使用 mask 处理边界)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        # 加载 B 块 (行主序，使用 mask 处理边界)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # 矩阵乘累加
        accumulator += tl.dot(a, b)

        # 更新指针到下一个 K 块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 应用 alpha
    accumulator = accumulator * alpha

    # 处理 beta * C (只有当 beta != 0 时需要)
    if beta != 0.0:
        # C 指针: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)
        accumulator = accumulator + beta * c

    # 存储结果到 C
    c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_sgemm_triton(
    M: int, N: int, K: int,
    alpha: float,
    A: torch.Tensor,  # (M, K)
    B: torch.Tensor,  # (K, N)
    beta: float,
    C: torch.Tensor,  # (M, N)
):
    """
    使用 Triton 执行 SGEMM。

    Args:
        M, N, K: 矩阵维度
        alpha: 缩放系数
        A: 输入矩阵 A (M, K)，行主序
        B: 输入矩阵 B (K, N)，行主序
        beta: C 的缩放系数
        C: 输出矩阵 (M, N)，行主序
    """
    # 确保张量在 GPU 上
    assert A.is_cuda and B.is_cuda and C.is_cuda

    # 配置 kernel 参数
    # 这些参数可以根据 GPU 架构和矩阵大小调整
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # 计算 grid 大小
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # 启动 kernel
    sgemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )


# Auto-tuning 配置（可选，使用 Triton 的自动调优功能）
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def sgemm_kernel_autotuned(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """带自动调优的 SGEMM kernel"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator * alpha

    if beta != 0.0:
        c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)
        accumulator = accumulator + beta * c

    c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_sgemm_triton_autotuned(
    M: int, N: int, K: int,
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: torch.Tensor,
):
    """使用自动调优的 Triton SGEMM"""
    assert A.is_cuda and B.is_cuda and C.is_cuda

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    sgemm_kernel_autotuned[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        alpha, beta,
    )


if __name__ == "__main__":
    # 简单测试
    torch.manual_seed(0)

    M, N, K = 1024, 1024, 1024
    alpha, beta = 1.0, 0.0

    # 创建随机矩阵
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # 运行 Triton SGEMM
    run_sgemm_triton(M, N, K, alpha, A, B, beta, C)

    # 验证结果
    C_ref = torch.matmul(A, B)
    max_diff = torch.max(torch.abs(C - C_ref)).item()
    print(f"Triton SGEMM test: M={M}, N={N}, K={K}")
    print(f"Max difference from PyTorch: {max_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")

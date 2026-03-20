"""
Triton SGEMM 性能基准测试

与 PyTorch (cuBLAS) 和自定义 CUDA kernel 进行性能对比。

使用方法：
    cd GEMM/triton_kernels
    python benchmark_triton.py

依赖：
    pip install triton torch numpy matplotlib
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from sgemm_triton import run_sgemm_triton, run_sgemm_triton_autotuned


def benchmark_gemm(M, N, K, num_repeats=20, warmup=3):
    """
    对比 Triton、PyTorch (cuBLAS) 和 torch.matmul 的性能。

    Args:
        M, N, K: 矩阵维度
        num_repeats: 测量重复次数
        warmup: 预热次数
    """
    alpha, beta = 1.0, 0.0

    # 创建随机矩阵
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    C_triton = torch.empty((M, N), device='cuda', dtype=torch.float32)
    C_torch = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # 预热 GPU
    for _ in range(warmup):
        run_sgemm_triton(M, N, K, alpha, A, B, beta, C_triton)
        torch.matmul(A, B, out=C_torch)
    torch.cuda.synchronize()

    # 验证正确性
    run_sgemm_triton(M, N, K, alpha, A, B, beta, C_triton)
    C_ref = torch.matmul(A, B)
    max_diff = torch.max(torch.abs(C_triton - C_ref)).item()
    passed = max_diff < 1e-2

    # Benchmark Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_repeats):
        run_sgemm_triton(M, N, K, alpha, A, B, beta, C_triton)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_repeats  # ms

    # Benchmark PyTorch (cuBLAS)
    start.record()
    for _ in range(num_repeats):
        torch.matmul(A, B, out=C_torch)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_repeats  # ms

    # 计算 TFLOPS
    flops = 2.0 * M * N * K  # 乘加操作数
    triton_tflops = (flops / triton_time) / 1e9  # TFLOPS
    torch_tflops = (flops / torch_time) / 1e9  # TFLOPS

    return {
        'M': M, 'N': N, 'K': K,
        'triton_ms': triton_time,
        'torch_ms': torch_time,
        'triton_tflops': triton_tflops,
        'torch_tflops': torch_tflops,
        'speedup': torch_time / triton_time,
        'passed': passed,
        'max_diff': max_diff,
    }


def print_results(results):
    """打印基准测试结果"""
    print("\n" + "="*80)
    print("Triton SGEMM 性能基准测试")
    print("="*80)
    print(f"{'M':>6} {'N':>6} {'K':>6} | "
          f"{'Triton(ms)':>12} {'cuBLAS(ms)':>12} | "
          f"{'Triton(TF)':>10} {'cuBLAS(TF)':>10} | "
          f"{'Speedup':>8} {'Pass':>6}")
    print("-"*80)

    for r in results:
        status = "✓" if r['passed'] else "✗"
        print(f"{r['M']:>6} {r['N']:>6} {r['K']:>6} | "
              f"{r['triton_ms']:>12.3f} {r['torch_ms']:>12.3f} | "
              f"{r['triton_tflops']:>10.2f} {r['torch_tflops']:>10.2f} | "
              f"{r['speedup']:>8.2f} {status:>6}")

    print("-"*80)

    # 统计
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"\n平均加速比: {avg_speedup:.2f}x")

    # 找到 Triton 最接近 cuBLAS 的 case
    best_idx = np.argmin([abs(r['speedup'] - 1.0) for r in results])
    best = results[best_idx]
    print(f"最接近的性能: M={best['M']}, N={best['N']}, K={best['K']} "
          f"(Triton: {best['triton_tflops']:.1f} TF, cuBLAS: {best['torch_tflops']:.1f} TF)")


def run_benchmark_suite():
    """运行一系列矩阵尺寸的基准测试"""
    test_cases = [
        # 方形矩阵
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # 非方形矩阵
        (1024, 2048, 512),
        (2048, 1024, 512),
        (4096, 1024, 4096),
        (1024, 4096, 4096),
        # 大矩阵
        (8192, 8192, 1024),
        (1024, 8192, 8192),
    ]

    results = []
    print("开始基准测试...")

    for M, N, K in test_cases:
        print(f"Testing M={M}, N={N}, K={K}...", end=" ")
        try:
            result = benchmark_gemm(M, N, K, num_repeats=10, warmup=2)
            results.append(result)
            status = "✓" if result['passed'] else "✗"
            print(f"{status} ({result['triton_tflops']:.1f} TFLOPS)")
        except Exception as e:
            print(f"Failed: {e}")

    print_results(results)
    return results


def profile_kernel(M=2048, N=2048, K=2048):
    """
    使用 PyTorch Profiler 分析 kernel 性能
    """
    print(f"\nProfiling kernel: M={M}, N={N}, K={K}")

    alpha, beta = 1.0, 0.0
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # 预热
    for _ in range(5):
        run_sgemm_triton(M, N, K, alpha, A, B, beta, C)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            run_sgemm_triton(M, N, K, alpha, A, B, beta, C)

    print("\nTop 10 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    # 检查 GPU
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__ if hasattr(triton, '__version__') else 'unknown'}")

    # 运行基准测试
    results = run_benchmark_suite()

    # 可选：进行 profiling
    # profile_kernel(2048, 2048, 2048)

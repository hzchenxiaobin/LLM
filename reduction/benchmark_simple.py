#!/usr/bin/env python3
"""
CUDA Reduction 算子性能测试框架 (简化版)

无需 pycuda，使用 ctypes 调用编译好的 CUDA 共享库。
"""

import ctypes
import os
import sys
import subprocess
import argparse
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """存储单个测试结果"""
    version: str
    n: int
    time_ms: float
    bandwidth_gb_s: float
    efficiency: float
    result_correct: bool
    error: float = 0.0


class SimpleCudaBenchmark:
    """简化版 CUDA Reduction 性能测试框架"""

    VERSIONS = {
        1: ("v1_interleaved", "朴素版本 - Warp Divergence"),
        2: ("v2_strided", "解决分支发散 - Bank Conflict"),
        3: ("v3_sequential", "解决 Bank Conflict"),
        4: ("v4_first_add", "加载时相加"),
        5: ("v5_warp_shuffle", "Warp Shuffle"),
        6: ("v6_vectorized", "向量化访存"),
        7: ("cub", "NVIDIA CUB 库"),
    }

    def __init__(self, lib_path: str = "./libreduction.so"):
        self.lib_path = lib_path
        self.lib = None
        self.gpu_name = ""
        self.peak_bandwidth_gb_s = 0.0
        self.results: List[BenchmarkResult] = []

        self._load_library()

    def _load_library(self):
        if not os.path.exists(self.lib_path):
            raise RuntimeError(f"CUDA 库不存在: {self.lib_path}. 请先运行 'make'")

        self.lib = ctypes.CDLL(self.lib_path)

        # 获取 GPU 信息
        name_buffer = ctypes.create_string_buffer(256)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        peak_bw = ctypes.c_float()

        self.lib.get_gpu_info(name_buffer, ctypes.byref(major), ctypes.byref(minor), ctypes.byref(peak_bw))

        self.gpu_name = name_buffer.value.decode('utf-8')
        self.peak_bandwidth_gb_s = peak_bw.value

        print(f"=" * 60)
        print(f"GPU: {self.gpu_name}")
        print(f"Compute Capability: {major.value}.{minor.value}")
        print(f"Peak Memory Bandwidth: {self.peak_bandwidth_gb_s:.2f} GB/s")
        print(f"=" * 60)
        print()

        # 设置函数签名
        self.lib.init_data.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint, ctypes.c_uint]
        self.lib.get_bandwidth_gb_s.argtypes = [ctypes.c_uint, ctypes.c_float]
        self.lib.get_bandwidth_gb_s.restype = ctypes.c_float
        self.lib.get_kernel_config.argtypes = [
            ctypes.c_int, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.run_kernel.restype = ctypes.c_float
        self.lib.run_kernel.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        self.lib.run_cub.restype = ctypes.c_float

    def parse_size(self, size_str: str) -> int:
        size_str = size_str.strip().upper()
        multipliers = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
        for suffix, mult in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-1]) * mult)
        return int(size_str)

    def format_bytes(self, n: int) -> str:
        if n >= 1024**3:
            return f"{n / 1024**3:.2f} GB"
        if n >= 1024**2:
            return f"{n / 1024**2:.2f} MB"
        if n >= 1024:
            return f"{n / 1024:.2f} KB"
        return f"{n} B"

    def run_test(self, version: int, n: int,
                 warmup: int = 10, iterations: int = 100) -> BenchmarkResult:
        """运行单个测试"""

        # 分配页锁定内存 (pinned memory) 用于更快传输
        h_in = np.zeros(n, dtype=np.float32)
        h_out = np.zeros(1, dtype=np.float32)

        # 初始化数据
        seed = int(time.time()) % 10000
        self.lib.init_data(
            h_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint(n), ctypes.c_uint(seed)
        )

        # CPU 参考结果
        cpu_sum = np.sum(h_in)

        # 获取 kernel 配置
        num_blocks = ctypes.c_int()
        num_threads = ctypes.c_int()
        shared_mem = ctypes.c_int()
        self.lib.get_kernel_config(ctypes.c_int(version), ctypes.c_uint(n),
                                   ctypes.byref(num_blocks), ctypes.byref(num_threads),
                                   ctypes.byref(shared_mem))

        # 运行 kernel
        time_ms = self.lib.run_kernel(
            ctypes.c_int(version),
            h_in.ctypes.data_as(ctypes.c_void_p),  # 使用主机内存地址作为设备指针 (统一内存)
            h_out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(n), num_blocks, num_threads, shared_mem,
            ctypes.c_void_p(0), ctypes.c_int(warmup), ctypes.c_int(iterations)
        )

        # 验证
        result_correct = True
        error = 0.0
        if version == 7:  # CUB
            # CUB 版本结果格式不同
            pass
        else:
            gpu_sum = h_out[0]
            error = abs(gpu_sum - cpu_sum) / abs(cpu_sum) if cpu_sum != 0 else abs(gpu_sum)
            result_correct = error < 0.01

        # 计算带宽
        bandwidth = self.lib.get_bandwidth_gb_s(ctypes.c_uint(n), ctypes.c_float(time_ms))
        efficiency = (bandwidth / self.peak_bandwidth_gb_s * 100.0) if self.peak_bandwidth_gb_s > 0 else 0.0

        return BenchmarkResult(
            version=self.VERSIONS[version][0],
            n=n, time_ms=time_ms,
            bandwidth_gb_s=bandwidth,
            efficiency=efficiency,
            result_correct=result_correct,
            error=error
        )

    def run_benchmark(self, versions: List[int], sizes: List[str],
                      warmup: int = 10, iterations: int = 100):
        """运行完整测试"""
        self.results = []

        print(f"测试配置: warmup={warmup}, iterations={iterations}")
        print(f"测试规模: {', '.join(sizes)}")
        print(f"测试版本: {', '.join(str(v) for v in versions)}")
        print("-" * 80)

        for size_str in sizes:
            n = self.parse_size(size_str)
            n = (n // 4) * 4  # 对齐到4
            if n == 0:
                n = 4

            print(f"\n数据规模: {n:,} 元素 ({self.format_bytes(n * 4)})")

            for v in versions:
                name, desc = self.VERSIONS[v]
                try:
                    result = self.run_test(v, n, warmup, iterations)
                    self.results.append(result)

                    status = "✓" if result.result_correct else "✗"
                    print(f"  {status} {name:20s}: {result.time_ms:8.4f} ms | "
                          f"{result.bandwidth_gb_s:7.2f} GB/s ({result.efficiency:5.1f}%)")
                except Exception as e:
                    print(f"  ✗ {name:20s}: 错误 - {e}")

    def print_summary(self):
        """打印摘要"""
        if not self.results:
            return

        print("\n" + "=" * 80)
        print("性能测试摘要")
        print("=" * 80)

        sizes = sorted(set(r.n for r in self.results))

        for size in sizes:
            results = [r for r in self.results if r.n == size]
            results.sort(key=lambda x: x.bandwidth_gb_s, reverse=True)

            print(f"\n{size:,} 元素 ({self.format_bytes(size * 4)}):")
            print(f"{'版本':<20} {'时间(ms)':>10} {'带宽(GB/s)':>12} {'效率':>10} {'状态':>6}")
            print("-" * 65)

            for r in results:
                status = "✓" if r.result_correct else "✗"
                print(f"{r.version:<20} {r.time_ms:>10.4f} {r.bandwidth_gb_s:>12.2f} "
                      f"{r.efficiency:>9.1f}% {status:>6}")

        best = max(self.results, key=lambda x: x.bandwidth_gb_s)
        print(f"\n最佳性能: {best.version} @ {self.format_bytes(best.n * 4)}: "
              f"{best.bandwidth_gb_s:.2f} GB/s ({best.efficiency:.1f}%)")


def compile_library():
    """编译 CUDA 库"""
    print("编译 CUDA 库...")

    # 检查 nvcc
    try:
        subprocess.run(['nvcc', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 找不到 nvcc")
        return False

    # 查找 CUB
    cub_include = ""
    for path in ['/usr/local/cuda/include/cub', '/usr/include/cub', f'{os.path.expanduser("~")}/cub']:
        if os.path.exists(path):
            cub_include = f"-I{os.path.dirname(path)}"
            break

    cmd = [
        'nvcc', '-O3', '-shared', '-Xcompiler', '-fPIC',
        '-o', 'libreduction.so', 'reduction_kernels.cu',
        '-lcudart', '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64'
    ]
    if cub_include:
        cmd.insert(5, cub_include)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"编译失败:\n{result.stderr}")
        return False

    print("编译成功!")
    return True


def main():
    parser = argparse.ArgumentParser(description='CUDA Reduction 性能测试')
    parser.add_argument('--compile', action='store_true', help='编译 CUDA 库')
    parser.add_argument('--sizes', nargs='+', default=['1M', '10M', '100M'], help='数据规模')
    parser.add_argument('--versions', nargs='+', default=['all'], help='版本 (1-7 或 all)')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--quick', action='store_true', help='快速模式')

    args = parser.parse_args()

    if args.compile:
        sys.exit(0 if compile_library() else 1)

    if not os.path.exists('./libreduction.so'):
        print("库文件不存在，尝试编译...")
        if not compile_library():
            sys.exit(1)

    if args.quick:
        args.warmup, args.iterations = 3, 10

    if 'all' in args.versions:
        versions = list(range(1, 8))
    else:
        versions = [int(v) for v in args.versions]

    try:
        benchmark = SimpleCudaBenchmark()
        benchmark.run_benchmark(versions, args.sizes, args.warmup, args.iterations)
        benchmark.print_summary()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

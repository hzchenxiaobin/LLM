#!/usr/bin/env python3
"""
CUDA Reduction 算子性能测试框架

支持测试 6 个版本的 reduction 实现：
- v1: 朴素版本 (Interleaved Addressing) - 有 Warp Divergence
- v2: 解决分支发散 (Strided Index) - 有 Bank Conflict
- v3: 解决 Bank Conflict (Sequential Addressing)
- v4: 提高指令吞吐 (First Add During Load)
- v5: Warp Shuffle (终结 Shared Memory)
- v6: 向量化访存 (Vectorized Memory Access)
- cub: NVIDIA CUB 库 (基准)

使用方法:
    python benchmark.py --sizes 1M 10M 100M 1G --versions all
    python benchmark.py --compare --plot
"""

import ctypes
import os
import sys
import subprocess
import argparse
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from tabulate import tabulate


@dataclass
class BenchmarkResult:
    """存储单个测试结果"""
    version: str
    n: int
    time_ms: float
    bandwidth_gb_s: float
    efficiency: float  # 相对于峰值带宽的效率
    result_correct: bool
    error: float = 0.0


class CudaReductionBenchmark:
    """CUDA Reduction 性能测试框架"""

    # 版本名称和描述
    VERSIONS = {
        1: ("v1_interleaved", "朴素版本 (Interleaved Addressing) - Warp Divergence"),
        2: ("v2_strided", "解决分支发散 (Strided Index) - Bank Conflict"),
        3: ("v3_sequential", "解决 Bank Conflict (Sequential Addressing)"),
        4: ("v4_first_add", "提高指令吞吐 (First Add During Load)"),
        5: ("v5_warp_shuffle", "Warp Shuffle (终结 Shared Memory)"),
        6: ("v6_vectorized", "向量化访存 (Vectorized Memory Access)"),
        7: ("cub", "NVIDIA CUB 库 (基准)"),
    }

    def __init__(self, lib_path: str = "./libreduction.so"):
        """初始化，加载 CUDA 库"""
        self.lib_path = lib_path
        self.lib = None
        self.gpu_name = ""
        self.peak_bandwidth_gb_s = 0.0
        self.results: List[BenchmarkResult] = []
        self.d_temp_storage = None
        self.temp_storage_bytes = ctypes.c_size_t(0)

        self._load_library()

    def _load_library(self):
        """加载编译好的 CUDA 共享库"""
        if not os.path.exists(self.lib_path):
            raise RuntimeError(f"CUDA 库不存在: {self.lib_path}. 请先运行 'make' 编译")

        self.lib = ctypes.CDLL(self.lib_path)

        # 获取 GPU 信息
        name_buffer = ctypes.create_string_buffer(256)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        peak_bw = ctypes.c_float()

        self.lib.get_gpu_info(name_buffer, ctypes.byref(major), ctypes.byref(minor), ctypes.byref(peak_bw))

        self.gpu_name = name_buffer.value.decode('utf-8')
        self.peak_bandwidth_gb_s = peak_bw.value

        print(f"GPU: {self.gpu_name}")
        print(f"Compute Capability: {major.value}.{minor.value}")
        print(f"Peak Memory Bandwidth: {self.peak_bandwidth_gb_s:.2f} GB/s")
        print()

        # 设置函数签名
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """设置 C 函数签名"""
        lib = self.lib

        # void init_data(float *data, unsigned int n, unsigned int seed)
        lib.init_data.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint, ctypes.c_uint]
        lib.init_data.restype = None

        # float get_bandwidth_gb_s(unsigned int n, float time_ms)
        lib.get_bandwidth_gb_s.argtypes = [ctypes.c_uint, ctypes.c_float]
        lib.get_bandwidth_gb_s.restype = ctypes.c_float

        # float run_kernel(int version, float *d_in, float *d_out, unsigned int n,
        #                int num_blocks, int num_threads, int shared_mem_bytes,
        #                cudaStream_t stream, int warmup_iters, int test_iters)
        lib.run_kernel.argtypes = [
            ctypes.c_int,                           # version
            ctypes.c_void_p,                        # d_in
            ctypes.c_void_p,                        # d_out
            ctypes.c_uint,                          # n
            ctypes.c_int,                           # num_blocks
            ctypes.c_int,                           # num_threads
            ctypes.c_int,                           # shared_mem_bytes
            ctypes.c_void_p,                        # stream
            ctypes.c_int,                           # warmup_iters
            ctypes.c_int,                           # test_iters
        ]
        lib.run_kernel.restype = ctypes.c_float

        # float run_cub(...)
        lib.run_cub.argtypes = [
            ctypes.c_void_p,                        # d_in
            ctypes.c_void_p,                        # d_out
            ctypes.c_uint,                          # n
            ctypes.POINTER(ctypes.c_void_p),        # d_temp_storage
            ctypes.POINTER(ctypes.c_size_t),        # temp_storage_bytes
            ctypes.c_void_p,                        # stream
            ctypes.c_int,                           # warmup_iters
            ctypes.c_int,                           # test_iters
        ]
        lib.run_cub.restype = ctypes.c_float

        # void get_kernel_config(int version, unsigned int n, int *num_blocks, int *num_threads, int *shared_mem_bytes)
        lib.get_kernel_config.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.get_kernel_config.restype = None

    def parse_size(self, size_str: str) -> int:
        """解析大小字符串，如 '1M', '10K', '1G'"""
        size_str = size_str.strip().upper()
        multipliers = {
            'K': 1024,
            'M': 1024 * 1024,
            'G': 1024 * 1024 * 1024,
        }

        for suffix, mult in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-1]) * mult)

        return int(size_str)

    def run_single_test(self, version: int, n: int,
                        warmup_iters: int = 10, test_iters: int = 100,
                        verify: bool = True) -> BenchmarkResult:
        """运行单个版本的测试"""

        # 分配主机内存
        h_in = np.empty(n, dtype=np.float32)
        h_out = np.empty(1, dtype=np.float32)

        # 初始化数据
        seed = int(time.time()) % 10000
        self.lib.init_data(
            h_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint(n),
            ctypes.c_uint(seed)
        )

        # CPU 参考结果
        cpu_sum = np.sum(h_in)

        # 分配设备内存
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        d_in = cuda.mem_alloc(h_in.nbytes)
        d_out = cuda.mem_alloc(h_out.nbytes)

        cuda.memcpy_htod(d_in, h_in)

        # 获取 kernel 配置
        num_blocks = ctypes.c_int()
        num_threads = ctypes.c_int()
        shared_mem_bytes = ctypes.c_int()

        self.lib.get_kernel_config(
            ctypes.c_int(version),
            ctypes.c_uint(n),
            ctypes.byref(num_blocks),
            ctypes.byref(num_threads),
            ctypes.byref(shared_mem_bytes)
        )

        # 运行测试
        if version == 7:  # CUB 版本
            time_ms = self.lib.run_cub(
                d_in, d_out, ctypes.c_uint(n),
                ctypes.byref(self.d_temp_storage),
                ctypes.byref(self.temp_storage_bytes),
                ctypes.c_void_p(0),  # default stream
                ctypes.c_int(warmup_iters),
                ctypes.c_int(test_iters)
            )
        else:
            time_ms = self.lib.run_kernel(
                ctypes.c_int(version),
                d_in, d_out, ctypes.c_uint(n),
                num_blocks, num_threads, shared_mem_bytes,
                ctypes.c_void_p(0),
                ctypes.c_int(warmup_iters),
                ctypes.c_int(test_iters)
            )

        # 验证结果
        result_correct = True
        error = 0.0
        if verify:
            cuda.memcpy_dtoh(h_out, d_out)
            gpu_sum = h_out[0]
            error = abs(gpu_sum - cpu_sum) / abs(cpu_sum) if cpu_sum != 0 else abs(gpu_sum)
            result_correct = error < 0.01  # 1% 误差容忍

        # 清理
        d_in.free()
        d_out.free()

        # 计算带宽
        bandwidth = self.lib.get_bandwidth_gb_s(ctypes.c_uint(n), ctypes.c_float(time_ms))
        efficiency = (bandwidth / self.peak_bandwidth_gb_s) * 100.0 if self.peak_bandwidth_gb_s > 0 else 0.0

        version_name = self.VERSIONS.get(version, ("unknown", ""))[0]

        return BenchmarkResult(
            version=version_name,
            n=n,
            time_ms=time_ms,
            bandwidth_gb_s=bandwidth,
            efficiency=efficiency,
            result_correct=result_correct,
            error=error
        )

    def run_benchmark(self, versions: List[int], sizes: List[int],
                      warmup_iters: int = 10, test_iters: int = 100,
                      verify: bool = True) -> List[BenchmarkResult]:
        """运行完整的基准测试"""
        self.results = []

        print(f"开始测试 {len(versions)} 个版本，{len(sizes)} 个数据规模")
        print(f"Warmup iterations: {warmup_iters}, Test iterations: {test_iters}")
        print("-" * 80)

        for size in sizes:
            n = self.parse_size(size) if isinstance(size, str) else size

            # 对齐到 4 的倍数 (v6 需要)
            n = (n // 4) * 4
            if n == 0:
                n = 4

            print(f"\n数据规模: {n:,} 元素 ({n * 4 / (1024**2):.2f} MB)")

            for version in versions:
                name, desc = self.VERSIONS.get(version, ("unknown", ""))

                try:
                    result = self.run_single_test(version, n, warmup_iters, test_iters, verify)
                    self.results.append(result)

                    status = "✓" if result.result_correct else "✗"
                    print(f"  {status} {name:20s}: {result.time_ms:8.3f} ms | "
                          f"{result.bandwidth_gb_s:7.2f} GB/s ({result.efficiency:5.1f}%)")

                except Exception as e:
                    print(f"  ✗ {name:20s}: 错误 - {e}")

        return self.results

    def print_summary(self):
        """打印测试摘要"""
        if not self.results:
            print("没有测试结果")
            return

        print("\n" + "=" * 80)
        print("性能测试摘要")
        print("=" * 80)

        # 按数据规模分组
        sizes = sorted(set(r.n for r in self.results))

        for size in sizes:
            print(f"\n数据规模: {size:,} 元素 ({size * 4 / (1024**2):.2f} MB)")

            results_for_size = [r for r in self.results if r.n == size]

            # 排序：按带宽降序
            results_for_size.sort(key=lambda x: x.bandwidth_gb_s, reverse=True)

            table_data = []
            for r in results_for_size:
                table_data.append([
                    r.version,
                    f"{r.time_ms:.3f}",
                    f"{r.bandwidth_gb_s:.2f}",
                    f"{r.efficiency:.1f}%",
                    "✓" if r.result_correct else "✗"
                ])

            headers = ["版本", "时间 (ms)", "带宽 (GB/s)", "效率", "正确"]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # 整体最佳
        print("\n整体最佳性能:")
        best = max(self.results, key=lambda x: x.bandwidth_gb_s)
        print(f"  {best.version} @ {best.n:,} 元素: {best.bandwidth_gb_s:.2f} GB/s ({best.efficiency:.1f}% 峰值)")

    def plot_results(self, output_file: str = "reduction_performance.png"):
        """绘制性能图表"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("警告: matplotlib 未安装，无法绘制图表")
            return

        if not self.results:
            print("没有数据可绘制")
            return

        # 准备数据
        versions = sorted(set(r.version for r in self.results))
        sizes = sorted(set(r.n for r in self.results))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图 1: 带宽 vs 数据规模
        ax1 = axes[0]
        for version in versions:
            data = [(r.n, r.bandwidth_gb_s) for r in self.results if r.version == version]
            data.sort()
            xs = [d[0] / (1024**2) for d in data]  # MB
            ys = [d[1] for d in data]
            ax1.plot(xs, ys, marker='o', label=version, linewidth=2)

        ax1.axhline(y=self.peak_bandwidth_gb_s, color='r', linestyle='--',
                    label=f'Peak ({self.peak_bandwidth_gb_s:.1f} GB/s)')
        ax1.set_xlabel('Data Size (MB)')
        ax1.set_ylabel('Bandwidth (GB/s)')
        ax1.set_title('Reduction Kernel Performance')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 图 2: 效率对比（取最大数据规模的测试结果）
        ax2 = axes[1]
        max_size = max(sizes)
        eff_data = [(r.version, r.efficiency) for r in self.results if r.n == max_size]
        eff_data.sort(key=lambda x: x[1], reverse=True)

        names = [d[0] for d in eff_data]
        effs = [d[1] for d in eff_data]

        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax2.barh(names, effs, color=colors)
        ax2.set_xlabel('Efficiency (% of Peak Bandwidth)')
        ax2.set_title(f'Efficiency Comparison @ {max_size / (1024**2):.1f} MB')
        ax2.set_xlim(0, 100)

        # 添加数值标签
        for bar, eff in zip(bars, effs):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{eff:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存到: {output_file}")


def compile_library():
    """编译 CUDA 库"""
    print("编译 CUDA 库...")

    # 检查 nvcc 是否可用
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("错误: 找不到 nvcc，请确保 CUDA 工具链已安装")
            return False
    except FileNotFoundError:
        print("错误: 找不到 nvcc，请确保 CUDA 工具链已安装并添加到 PATH")
        return False

    # 编译
    cmd = [
        'nvcc', '-O3', '-shared', '-Xcompiler', '-fPIC',
        '-o', 'libreduction.so',
        'reduction_kernels.cu',
        '-lcudart',
        '-I/usr/local/cuda/include',
        '-L/usr/local/cuda/lib64',
    ]

    # 添加 CUB 支持（如果可用）
    cub_paths = [
        '/usr/local/cuda/include/cub',
        '/usr/include/cub',
        os.path.expanduser('~/cub'),
    ]
    for cub_path in cub_paths:
        if os.path.exists(cub_path):
            cmd.insert(-3, f'-I{os.path.dirname(cub_path)}')
            break

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"编译失败:\n{result.stderr}")
        return False

    print("编译成功!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='CUDA Reduction 算子性能测试框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 编译 CUDA 库
    %(prog)s --compile

    # 测试所有版本
    %(prog)s --sizes 1M 10M 100M --versions all

    # 只测试特定版本
    %(prog)s --sizes 100M --versions 3 4 5 6

    # 快速测试（较少迭代次数）
    %(prog)s --sizes 10M --quick

    # 测试并生成图表
    %(prog)s --sizes 1M 10M 100M 1G --plot
        """
    )

    parser.add_argument('--compile', action='store_true',
                        help='编译 CUDA 库')
    parser.add_argument('--sizes', nargs='+', default=['1M', '10M', '100M'],
                        help='测试数据规模 (如: 1M 10M 100M 1G)')
    parser.add_argument('--versions', nargs='+', default=['all'],
                        help='要测试的版本 (1-7，或 all)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热迭代次数')
    parser.add_argument('--iterations', type=int, default=100,
                        help='测试迭代次数')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式 (warmup=3, iterations=10)')
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过结果验证')
    parser.add_argument('--plot', action='store_true',
                        help='生成性能图表')
    parser.add_argument('--output', default='reduction_performance.png',
                        help='图表输出文件名')
    parser.add_argument('--lib', default='./libreduction.so',
                        help='CUDA 库路径')

    args = parser.parse_args()

    # 编译模式
    if args.compile:
        success = compile_library()
        sys.exit(0 if success else 1)

    # 快速模式
    if args.quick:
        args.warmup = 3
        args.iterations = 10

    # 检查库文件
    if not os.path.exists(args.lib):
        print(f"CUDA 库不存在: {args.lib}")
        print("尝试自动编译...")
        if not compile_library():
            print("自动编译失败，请手动运行: make")
            sys.exit(1)

    # 解析版本
    if 'all' in args.versions:
        versions = list(range(1, 8))  # 1-7
    else:
        versions = [int(v) for v in args.versions]

    try:
        # 运行测试
        benchmark = CudaReductionBenchmark(args.lib)
        benchmark.run_benchmark(
            versions=versions,
            sizes=args.sizes,
            warmup_iters=args.warmup,
            test_iters=args.iterations,
            verify=not args.no_verify
        )

        benchmark.print_summary()

        if args.plot:
            benchmark.plot_results(args.output)

    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install pycuda numpy tabulate matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

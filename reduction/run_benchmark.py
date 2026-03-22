#!/usr/bin/env python3
"""
CUDA Reduction 性能测试 - Python 包装器

调用编译好的 benchmark 可执行文件，并提供结果可视化。
"""

import subprocess
import re
import sys
import argparse
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    version: str
    n: int
    time_ms: float
    bandwidth_gb_s: float
    efficiency: float
    correct: bool


def compile_benchmark() -> bool:
    """编译 benchmark 程序"""
    print("编译 benchmark...")
    result = subprocess.run(['make'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"编译失败:\n{result.stderr}")
        return False
    return True


def parse_output(output: str) -> List[TestResult]:
    """解析 benchmark 输出"""
    results = []

    # 匹配行:   ✓ v1_interleaved      :  12.3456 ms |  123.45 GB/s ( 45.6%)
    pattern = re.compile(
        r'\s+([✓✗])\s+(\S+)\s*:\s*([\d.]+)\s+ms\s*\|\s*([\d.]+)\s+GB/s\s*\(([\d.]+)%\)'
    )

    current_n = 0

    for line in output.split('\n'):
        # 查找数据规模
        if '数据规模:' in line:
            match = re.search(r'(\d+) 元素', line)
            if match:
                current_n = int(match.group(1))

        # 匹配结果行
        match = pattern.match(line)
        if match:
            correct = match.group(1) == '✓'
            version = match.group(2)
            time_ms = float(match.group(3))
            bandwidth = float(match.group(4))
            efficiency = float(match.group(5))

            results.append(TestResult(
                version=version,
                n=current_n,
                time_ms=time_ms,
                bandwidth_gb_s=bandwidth,
                efficiency=efficiency,
                correct=correct
            ))

    return results


def print_table(results: List[TestResult]):
    """打印结果表格"""
    if not results:
        print("没有结果")
        return

    # 按数据规模分组
    sizes = sorted(set(r.n for r in results))

    for size in sizes:
        size_results = [r for r in results if r.n == size]
        size_results.sort(key=lambda x: x.bandwidth_gb_s, reverse=True)

        print(f"\n{size:,} 元素 ({size * 4 / (1024**2):.2f} MB):")
        print(f"{'版本':<20} {'时间(ms)':>10} {'带宽(GB/s)':>12} {'效率':>10} {'正确':>6}")
        print("-" * 65)

        for r in size_results:
            status = "✓" if r.correct else "✗"
            print(f"{r.version:<20} {r.time_ms:>10.4f} {r.bandwidth_gb_s:>12.2f} "
                  f"{r.efficiency:>9.1f}% {status:>6}")


def plot_results(results: List[TestResult], output_file: str = "benchmark_results.png"):
    """绘制结果图表"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("警告: matplotlib 未安装，跳过绘图")
        print("安装: pip install matplotlib")
        return

    if not results:
        return

    versions = sorted(set(r.version for r in results))
    sizes = sorted(set(r.n for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 带宽 vs 数据规模
    ax1 = axes[0]
    for version in versions:
        data = [(r.n, r.bandwidth_gb_s) for r in results if r.version == version]
        data.sort()
        xs = [d[0] / (1024**2) for d in data]  # MB
        ys = [d[1] for d in data]
        ax1.plot(xs, ys, marker='o', label=version, linewidth=2, markersize=6)

    ax1.set_xlabel('Data Size (MB)')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Reduction Kernel Bandwidth')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 图2: 效率对比（最大数据规模）
    ax2 = axes[1]
    max_size = max(sizes)
    eff_data = [(r.version, r.efficiency) for r in results if r.n == max_size]
    eff_data.sort(key=lambda x: x[1], reverse=True)

    if eff_data:
        names = [d[0] for d in eff_data]
        effs = [d[1] for d in eff_data]

        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax2.barh(names, effs, color=colors)
        ax2.set_xlabel('Efficiency (% of Peak Bandwidth)')
        ax2.set_title(f'Efficiency @ {max_size / (1024**2):.1f} MB')
        ax2.set_xlim(0, 100)

        # 添加数值标签
        for bar, eff in zip(bars, effs):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{eff:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='CUDA Reduction 性能测试')
    parser.add_argument('--compile', action='store_true', help='仅编译')
    parser.add_argument('--sizes', nargs='+', default=['1M', '10M', '100M'],
                        help='数据规模 (如: 1M 10M 100M)')
    parser.add_argument('--versions', nargs='+', default=['all'],
                        help='版本 (1-7 或 all)')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--quick', action='store_true', help='快速模式')
    parser.add_argument('--plot', action='store_true', help='生成图表')
    parser.add_argument('--output', default='benchmark_results.png',
                        help='图表输出文件名')

    args = parser.parse_args()

    # 检查/编译 benchmark
    if not os.path.exists('./benchmark'):
        if not compile_benchmark():
            sys.exit(1)

    if args.compile:
        sys.exit(0)

    # 快速模式
    if args.quick:
        args.warmup = 3
        args.iterations = 10

    # 构建命令
    cmd = ['./benchmark',
           '--warmup', str(args.warmup),
           '--iterations', str(args.iterations),
           '--sizes'] + args.sizes + ['--versions'] + args.versions

    print("运行测试...")
    print(f"命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 打印原始输出
    print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)

    # 解析结果
    if result.returncode == 0:
        results = parse_output(result.stdout)

        if results and args.plot:
            plot_results(results, args.output)

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()

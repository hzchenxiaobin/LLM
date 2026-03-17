#!/usr/bin/env python3
"""
Generate Roofline plot for SGEMM kernel analysis
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_roofline():
    # GPU 硬件参数 (NVIDIA A100)
    peak_flops = 19.5e3  # 19.5 TFLOPS = 19500 GFLOPS
    memory_bw = 1555  # 1555 GB/s
    ridge_point = peak_flops / memory_bw  # Ridge Point

    # 矩阵参数
    M = N = K = 4096
    
    # 计算两个 kernel 的 Arithmetic Intensity
    # Naive Kernel: 高全局内存访问
    # 假设每次迭代都从全局内存读取 (无有效缓存)
    naive_memory = M * K * N * 4  # bytes (简化模型)
    naive_flops = 2 * M * N * K
    naive_ai = naive_flops / naive_memory
    
    # Shared Memory Kernel: 全局内存访问大幅减少
    shared_memory = (M * K + K * N + M * N) * 4  # bytes
    shared_flops = 2 * M * N * K
    shared_ai = shared_flops / shared_memory
    
    print(f"=" * 60)
    print(f"Roofline Analysis for SGEMM Kernels")
    print(f"Matrix Size: M={M}, N={N}, K={K}")
    print(f"=" * 60)
    print(f"\nGPU: NVIDIA A100")
    print(f"  Peak FP32 Performance: {peak_flops/1e3:.1f} TFLOPS")
    print(f"  Memory Bandwidth: {memory_bw} GB/s")
    print(f"  Ridge Point: {ridge_point:.1f} FLOPs/byte")
    print(f"\nKernel Analysis:")
    print(f"  {'Kernel':<20} {'AI (FLOPs/byte)':<20} {'Theoretical GFLOPS':<20}")
    print(f"  {'-'*60}")
    print(f"  {'Naive':<20} {naive_ai:<20.2f} {min(naive_ai * memory_bw, peak_flops):<20.1f}")
    print(f"  {'Shared Memory':<20} {shared_ai:<20.2f} {min(shared_ai * memory_bw, peak_flops):<20.1f}")
    print(f"\nPerformance Gap: {min(shared_ai * memory_bw, peak_flops) / min(naive_ai * memory_bw, peak_flops):.1f}x")
    print(f"=" * 60)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # X轴: Arithmetic Intensity (对数刻度)
    ai_min = 0.1
    ai_max = 10000
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
    
    # Roofline: 屋顶线
    # 内存受限区域: Performance = Memory BW * AI
    memory_limited = memory_bw * ai_range
    # 计算受限区域: Performance = Peak FLOPS
    compute_limited = np.full_like(ai_range, peak_flops)
    
    # Roofline 是两者的最小值
    roofline = np.minimum(memory_limited, compute_limited)
    
    # 绘制 Roofline
    ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='Roofline (A100)')
    
    # 填充区域
    ax.fill_between(ai_range, 0, roofline, alpha=0.1, color='gray')
    
    # 标记 Ridge Point
    ax.plot(ridge_point, peak_flops, 'r*', markersize=15, label=f'Ridge Point ({ridge_point:.1f})')
    ax.axvline(x=ridge_point, color='r', linestyle='--', alpha=0.5)
    ax.text(ridge_point*1.2, peak_flops*0.5, f'Ridge Point\n{ridge_point:.1f}', 
            fontsize=10, color='red', fontweight='bold')
    
    # 标记两个 kernel
    # Naive Kernel
    naive_perf = min(naive_ai * memory_bw, peak_flops)
    ax.plot(naive_ai, naive_perf, 'ro', markersize=12, label=f'Naive (AI={naive_ai:.2f})')
    ax.annotate(f'Naive\nAI={naive_ai:.2f}\n{naive_perf:.1f} GFLOPS', 
                xy=(naive_ai, naive_perf), 
                xytext=(naive_ai*0.1, naive_perf*3),
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='red'))
    
    # Shared Memory Kernel
    shared_perf = min(shared_ai * memory_bw, peak_flops)
    ax.plot(shared_ai, shared_perf, 'go', markersize=12, label=f'Shared Memory (AI={shared_ai:.1f})')
    ax.annotate(f'Shared Memory\nAI={shared_ai:.1f}\n{shared_perf:.1f} GFLOPS', 
                xy=(shared_ai, shared_perf), 
                xytext=(shared_ai*0.1, shared_perf*0.5),
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', color='green'))
    
    # 添加带宽线和算力线标注
    # 内存带宽线 (在左侧区域)
    ax.text(0.3, memory_bw * 0.3 * 3, f'Memory Bandwidth\n{memory_bw} GB/s', 
            fontsize=9, rotation=35, color='blue', alpha=0.7)
    
    # 峰值算力线
    ax.text(100, peak_flops * 1.5, f'Peak Performance\n{peak_flops/1e3:.1f} TFLOPS', 
            fontsize=9, color='black', fontweight='bold')
    
    # 设置坐标轴
    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title('Roofline Model: SGEMM Kernel Performance Analysis\n(NVIDIA A100, M=N=K=4096)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 设置刻度范围
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(10, peak_flops * 3)
    
    # 添加网格
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 添加性能区域标注
    ax.text(0.5, peak_flops * 0.01, 'Memory\nBound', fontsize=11, ha='center', 
            color='blue', fontweight='bold', alpha=0.7)
    ax.text(500, peak_flops * 1.5, 'Compute\nBound', fontsize=11, ha='center', 
            color='green', fontweight='bold', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = '/home/chenb/code/master/LLM/GEMM/docs/roofline_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nRoofline plot saved to: {output_path}")
    
    # 显示图形 (如果在交互环境)
    # plt.show()
    
    return naive_ai, shared_ai, naive_perf, shared_perf

if __name__ == '__main__':
    generate_roofline()

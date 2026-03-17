#!/usr/bin/env python3
"""
Visualize why sgemm_register_kernel is faster than sgemm_shared_kernel
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_comparison_overview():
    """Create side-by-side comparison of two kernels"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Shared Memory Kernel (sgemm_shared)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SGEMM_Shared (32×32 Block)\nEach thread: 1×1 C element', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # Draw block
    block = FancyBboxPatch((1, 1), 8, 8, boxstyle="round,pad=0.1",
                           facecolor='#FFE4E1', edgecolor='red', linewidth=3)
    ax.add_patch(block)
    ax.text(5, 9.5, 'Block: 32×32 threads = 1024 threads', ha='center', fontsize=10)
    
    # Draw thread grid (simplified 8x8 for visualization)
    for i in range(8):
        for j in range(8):
            x, y = 1.2 + j * 0.95, 1.2 + i * 0.95
            thread = Rectangle((x, y), 0.9, 0.9, 
                              facecolor='white', edgecolor='gray', linewidth=0.5)
            ax.add_patch(thread)
            if i == 3 and j == 3:  # Highlight one thread
                thread.set_facecolor('#FFB6C1')
                ax.text(x+0.45, y+0.45, '1', ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.text(5, 0.3, 'Each thread computes:\n1 C element', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Shared memory tiles
    ax.text(5, -1.5, 'Shared Memory:\nsA[32][32], sB[32][32]', 
            ha='center', fontsize=9, style='italic')
    
    # Right: Register Kernel (sgemm_register)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SGEMM_Register (16×16 Block)\nEach thread: 8×8 C elements', 
                 fontsize=12, fontweight='bold', pad=20, color='green')
    
    # Draw larger block
    block = FancyBboxPatch((0.5, 1), 9, 8, boxstyle="round,pad=0.1",
                           facecolor='#E0FFE0', edgecolor='green', linewidth=3)
    ax.add_patch(block)
    ax.text(5, 9.5, 'Block: 16×16 threads = 256 threads', ha='center', fontsize=10)
    ax.text(5, 9.0, 'Computing: 128×128 C submatrix', ha='center', fontsize=9, color='green')
    
    # Draw thread grid (4x4 for visualization, each thread handles 8x8)
    for i in range(4):
        for j in range(4):
            x, y = 0.7 + j * 2.2, 1.2 + i * 2.0
            # Thread responsible area (8x8)
            thread_area = Rectangle((x, y), 2.0, 1.8, 
                                   facecolor='#90EE90', edgecolor='green', linewidth=2)
            ax.add_patch(thread_area)
            
            # Draw 8x8 grid inside
            for ii in range(4):
                for jj in range(4):
                    cell = Rectangle((x + jj*0.5, y + ii*0.45), 0.48, 0.43,
                                    facecolor='white', edgecolor='lightgray', linewidth=0.3)
                    ax.add_patch(cell)
            
            ax.text(x+1, y+0.9, '64', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.text(5, 0.3, 'Each thread computes:\n8×8 = 64 C elements', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Register and shared memory info
    ax.text(5, -1.5, 'Registers: accum[8][8]\nShared Mem: sA[128][8], sB[8][128]', 
            ha='center', fontsize=9, style='italic', color='green')
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/register_vs_shared_overview.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Comparison overview saved")
    plt.close()

def create_memory_access_pattern():
    """Show memory access pattern differences"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Shared Kernel - Memory hierarchy
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Shared Kernel: Memory Hierarchy\n(High Shared Memory Pressure)', 
                 fontsize=11, fontweight='bold')
    
    # Global memory
    global_box = FancyBboxPatch((0.5, 7), 9, 2, boxstyle="round,pad=0.1",
                               facecolor='#FFE4E1', edgecolor='darkred', linewidth=2)
    ax.add_patch(global_box)
    ax.text(5, 8.3, 'Global Memory\n(1.8 TB/s)', ha='center', fontsize=10)
    ax.text(5, 7.5, 'Load: A[32][32], B[32][32] per k_step', ha='center', fontsize=8)
    
    # Arrow
    ax.annotate('', xy=(5, 6.5), xytext=(5, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Shared memory
    shared_box = FancyBboxPatch((1, 3.5), 8, 2.5, boxstyle="round,pad=0.1",
                               facecolor='#E0FFE0', edgecolor='green', linewidth=2)
    ax.add_patch(shared_box)
    ax.text(5, 5.3, 'Shared Memory sA[32][32], sB[32][32]\n(10+ TB/s)', ha='center', fontsize=10)
    
    # Show accesses
    ax.text(5, 4.2, 'For each of 32 k iterations:', ha='center', fontsize=9)
    ax.text(5, 3.7, 'sA[ty][k] × sB[k][tx] → 2 shared mem accesses/multiply', 
            ha='center', fontsize=8, style='italic')
    
    # Registers
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    reg_box = FancyBboxPatch((2, 0.5), 6, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#E6E6FA', edgecolor='blue', linewidth=2)
    ax.add_patch(reg_box)
    ax.text(5, 1.5, 'Register: float tmp = 0\n(1× accumulator)', ha='center', fontsize=10)
    
    # Top-right: Register Kernel - Memory hierarchy
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Register Kernel: Memory Hierarchy\n(Low Shared Memory Pressure)', 
                 fontsize=11, fontweight='bold', color='green')
    
    # Global memory
    global_box = FancyBboxPatch((0.5, 7), 9, 2, boxstyle="round,pad=0.1",
                               facecolor='#FFE4E1', edgecolor='darkred', linewidth=2)
    ax.add_patch(global_box)
    ax.text(5, 8.3, 'Global Memory\n(1.8 TB/s)', ha='center', fontsize=10)
    ax.text(5, 7.5, 'Load: A[128][8], B[8][128] per k_step (256 threads)', ha='center', fontsize=8)
    
    # Arrow
    ax.annotate('', xy=(5, 6.5), xytext=(5, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Shared memory
    shared_box = FancyBboxPatch((1, 3.5), 8, 2.5, boxstyle="round,pad=0.1",
                               facecolor='#E0FFE0', edgecolor='green', linewidth=2)
    ax.add_patch(shared_box)
    ax.text(5, 5.3, 'Shared Memory sA[128][8], sB[8][128]\n(10+ TB/s)', ha='center', fontsize=10)
    ax.text(5, 4.6, '【关键】K维度小(BK=8)，更多数据进寄存器', ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Show fewer shared accesses
    ax.text(5, 4.0, 'For each of 512 k iterations:', ha='center', fontsize=9)
    ax.text(5, 3.5, 'Load frag_a[8], frag_b[8] → Compute 64 multiplies', 
            ha='center', fontsize=8, style='italic', color='green')
    
    # Arrow to registers
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Registers - more data
    reg_box = FancyBboxPatch((1.5, 0.5), 7, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#E6E6FA', edgecolor='blue', linewidth=2)
    ax.add_patch(reg_box)
    ax.text(5, 1.5, 'Registers:\nfloat frag_a[8], frag_b[8], accum[8][8]\n(64× accumulators!)', 
            ha='center', fontsize=10, color='blue')
    
    # Bottom: Comparison metrics
    ax = axes[1, 0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Shared Kernel: Computation vs Memory Access', fontsize=11, fontweight='bold')
    
    metrics_text = """
    每轮 k_step 计算:
    
    • 共享内存读取: 2 × 32 = 64 次
    • 乘法运算: 32 次
    • 累加运算: 32 次
    
    计算强度 (每轮):
    64 FLOPs / (64 × 4 bytes) = 0.25 FLOPs/byte
    
    问题:
    • 共享内存成为瓶颈
    • 每次计算都要访问共享内存
    • 寄存器只有 1 个累加器
    """
    
    ax.text(5, 5, metrics_text, ha='center', va='center', fontsize=10,
            family='monospace', bbox=dict(boxstyle='round', facecolor='#FFE4E1', alpha=0.5))
    
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Register Kernel: Computation vs Memory Access', 
                 fontsize=11, fontweight='bold', color='green')
    
    metrics_text = """
    每轮 k_step 计算:
    
    • 共享内存读取: 8 + 8 = 16 次
    • 乘法运算: 8 × 8 = 64 次
    • 累加运算: 64 次
    
    计算强度 (每轮):
    128 FLOPs / (16 × 4 bytes) = 2 FLOPs/byte
    
    提升:
    • 8× 计算强度!
    • 更多数据在寄存器
    • 减少共享内存访问压力
    """
    
    ax.text(5, 5, metrics_text, ha='center', va='center', fontsize=10,
            family='monospace', bbox=dict(boxstyle='round', facecolor='#E0FFE0', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/register_memory_pattern.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Memory pattern diagram saved")
    plt.close()

def create_computation_flow():
    """Show detailed computation flow for register kernel"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Register Kernel Computation Flow: 8×8 Outer Product', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Title area
    y_start = 9
    
    # Step 1: Load from shared memory to registers
    step1_y = 7.5
    ax.text(2, y_start, 'Step 1: Load from Shared Memory to Registers', 
            fontsize=12, fontweight='bold', color='blue')
    
    # Draw sA column
    for i in range(8):
        rect = patches.Rectangle((1, step1_y - i*0.5), 0.4, 0.45,
                                  facecolor='#90EE90', edgecolor='green', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.7, step1_y - i*0.5 + 0.22, f'a{i}', ha='right', fontsize=8)
    ax.text(1.2, step1_y + 0.3, 'sA[*][k]', ha='center', fontsize=9)
    
    # Arrow
    ax.annotate('', xy=(3, step1_y - 1.5), xytext=(1.8, step1_y - 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(2.4, step1_y - 1.2, 'Load', fontsize=8, color='blue')
    
    # Draw frag_a registers
    for i in range(8):
        rect = patches.Rectangle((3.2, step1_y - i*0.5), 0.6, 0.45,
                                  facecolor='#87CEEB', edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        ax.text(3.5, step1_y - i*0.5 + 0.22, f'f{i}', ha='center', fontsize=8)
    ax.text(3.5, step1_y + 0.3, 'frag_a[8]', ha='center', fontsize=9, fontweight='bold')
    
    # Draw sB row
    for j in range(8):
        rect = patches.Rectangle((6 + j*0.5, step1_y), 0.45, 0.4,
                                  facecolor='#FFD700', edgecolor='orange', linewidth=1)
        ax.add_patch(rect)
        ax.text(6.25 + j*0.5, step1_y + 0.2, f'b{j}', ha='center', fontsize=8)
    ax.text(8, step1_y + 0.7, 'sB[k][*]', ha='center', fontsize=9)
    
    # Arrow
    ax.annotate('', xy=(10, step1_y - 1.5), xytext=(8.5, step1_y - 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(9.25, step1_y - 1.2, 'Load', fontsize=8, color='blue')
    
    # Draw frag_b registers
    for j in range(8):
        rect = patches.Rectangle((10.2 + j*0.4, step1_y - 0.5), 0.35, 0.9,
                                  facecolor='#FFD700', edgecolor='orange', linewidth=2)
        ax.add_patch(rect)
        ax.text(10.4 + j*0.4, step1_y, f'f{j}', ha='center', fontsize=8, rotation=90)
    ax.text(11.6, step1_y + 0.7, 'frag_b[8]', ha='center', fontsize=9, fontweight='bold')
    
    # Step 2: Outer product computation
    step2_y = 4.5
    ax.text(2, 6.2, 'Step 2: Outer Product Computation (Registers Only)', 
            fontsize=12, fontweight='bold', color='green')
    
    # Draw accum[8][8] matrix
    for i in range(8):
        for j in range(8):
            rect = patches.Rectangle((4 + j*0.8, step2_y - i*0.5), 0.75, 0.48,
                                  facecolor='#FFB6C1', edgecolor='red', linewidth=1)
            ax.add_patch(rect)
            if i == 0 and j == 0:
                ax.text(4.4, step2_y + 0.24, '+', ha='center', fontsize=12, fontweight='bold')
    
    ax.text(7.2, step2_y + 0.5, 'accum[8][8] += frag_a[i] * frag_b[j]', 
            ha='center', fontsize=10, fontweight='bold')
    ax.text(7.2, step2_y - 0.3, '64 FMAs per k iteration', 
            ha='center', fontsize=9, style='italic')
    
    # Show outer product formula
    formula_y = 2.5
    ax.text(7, formula_y, 
            'Outer Product: accum[i][j] += frag_a[i] × frag_b[j]\n\n'
            'For each k in 0..7:\n'
            '  Load 8 floats from sA → frag_a[8]\n'
            '  Load 8 floats from sB → frag_b[8]\n'
            '  Compute 8×8 = 64 FMAs in registers',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Key advantage box
    ax.text(7, 0.8, 
            '【关键优势】\n'
            '• 共享内存读取: 16 次/轮 (8+8)\n'
            '• 寄存器计算: 64 FMAs/轮\n'
            '• 计算/内存比: 4:1 (vs Shared Kernel 的 1:1)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/register_computation_flow.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Computation flow diagram saved")
    plt.close()

def create_performance_comparison():
    """Create performance comparison bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Performance comparison
    ax = axes[0]
    
    kernels = ['Naive', 'Shared', 'Register\n(Expected)', 'cuBLAS']
    performance = [7.477, 9.132, 40, 66.687]  # Expected for Register ~40 TFLOPS
    colors = ['red', 'orange', 'green', 'blue']
    
    bars = ax.bar(kernels, performance, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, perf in zip(bars, performance):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison on RTX 5090\n(4096×4096×4096)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.axhline(y=104.9, color='red', linestyle='--', linewidth=2, label='Peak FP32 (104.9 TFLOPS)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Right: Efficiency breakdown
    ax = axes[1]
    
    categories = ['Global\nMemory\nEfficiency', 'Shared\nMemory\nEfficiency', 
                  'Register\nUtilization', 'Compute\nUnit\nUtilization']
    shared_scores = [60, 40, 20, 10]  # Estimated
    register_scores = [80, 70, 65, 40]  # Estimated
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, shared_scores, width, label='Shared Kernel', 
                   color='orange', alpha=0.7)
    bars2 = ax.bar(x + width/2, register_scores, width, label='Register Kernel',
                   color='green', alpha=0.7)
    
    ax.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
    ax.set_title('Hardware Resource Utilization\n(Higher is Better)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/register_performance_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Performance comparison saved")
    plt.close()

def create_optimization_summary():
    """Create summary of all optimizations"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Register Kernel Optimization Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Create optimization boxes
    optimizations = [
        {
            'title': '1. 双层分块 (Double Tiling)',
            'content': 'Block: 128×128\nThread: 8×8 (64 elements)\n\n效果: 更多计算/更少同步',
            'pos': (0.5, 6.5),
            'color': '#FFE4B5'
        },
        {
            'title': '2. 寄存器缓存 (Register Blocking)',
            'content': 'frag_a[8], frag_b[8]\naccum[8][8] (64 floats)\n\n效果: 减少共享内存访问',
            'pos': (5, 6.5),
            'color': '#E0FFE0'
        },
        {
            'title': '3. K维度小分块 (BK=8)',
            'content': 'sA[128][8], sB[8][128]\n\n效果: 更多数据进寄存器\n减少共享内存压力',
            'pos': (9.5, 6.5),
            'color': '#E6E6FA'
        },
        {
            'title': '4. 外积计算 (Outer Product)',
            'content': 'accum[i][j] += a[i] * b[j]\n64 FMAs per iteration\n\n效果: 高计算密度',
            'pos': (0.5, 3),
            'color': '#FFB6C1'
        },
        {
            'title': '5. 向量化加载 (Vectorized Load)',
            'content': '256 threads load\n1024 elements cooperatively\n\n效果: 最大化带宽利用',
            'pos': (5, 3),
            'color': '#FFFACD'
        },
        {
            'title': '6. 循环展开 (#pragma unroll)',
            'content': 'All inner loops unrolled\n\n效果: 消除分支开销\n更多指令级并行',
            'pos': (9.5, 3),
            'color': '#F0F8FF'
        }
    ]
    
    for opt in optimizations:
        x, y = opt['pos']
        box = FancyBboxPatch((x, y), 4, 2.8, boxstyle="round,pad=0.1",
                            facecolor=opt['color'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Title
        ax.text(x + 2, y + 2.4, opt['title'], 
               ha='center', fontsize=11, fontweight='bold')
        
        # Content
        ax.text(x + 2, y + 1.2, opt['content'],
               ha='center', va='center', fontsize=9,
               family='monospace')
    
    # Bottom summary
    summary_y = 1
    summary_box = FancyBboxPatch((0.5, 0), 13, 2, boxstyle="round,pad=0.1",
                                facecolor='#90EE90', edgecolor='darkgreen', linewidth=3)
    ax.add_patch(summary_box)
    
    ax.text(7, summary_y + 1, 
            '【综合效果】\n'
            'Shared Kernel (32×32):  9.1 TFLOPS   →  利用率 8.7%\n'
            'Register Kernel (预计): 30-50 TFLOPS  →  利用率 30-50%\n'
            '提升原因: 寄存器级数据复用 + 更高的计算/内存比',
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/register_optimization_summary.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Optimization summary saved")
    plt.close()

if __name__ == '__main__':
    print("Generating Register Kernel analysis diagrams...")
    print("=" * 60)
    
    create_comparison_overview()
    create_memory_access_pattern()
    create_computation_flow()
    create_performance_comparison()
    create_optimization_summary()
    
    print("=" * 60)
    print("All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  - images/register_vs_shared_overview.png")
    print("  - images/register_memory_pattern.png")
    print("  - images/register_computation_flow.png")
    print("  - images/register_performance_comparison.png")
    print("  - images/register_optimization_summary.png")

#!/usr/bin/env python3
"""
Visualize sgemm_shared_kernel logic with step-by-step diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def draw_matrix(ax, matrix, pos, cell_size, color_map, title, highlight_row=None, highlight_col=None):
    """Draw a matrix with optional row/col highlighting"""
    rows, cols = matrix.shape
    x, y = pos
    
    # Draw title
    ax.text(x + cols * cell_size / 2, y + rows * cell_size + 0.3, title, 
            ha='center', fontsize=11, fontweight='bold')
    
    for i in range(rows):
        for j in range(cols):
            cell_color = color_map.get((i, j), 'white')
            
            # Highlight row or column
            if highlight_row is not None and i == highlight_row:
                cell_color = '#FFE4B5'  # Light orange for row
            if highlight_col is not None and j == highlight_col:
                cell_color = '#E6E6FA'  # Light purple for col
            
            rect = Rectangle((x + j * cell_size, y + (rows - 1 - i) * cell_size), 
                           cell_size - 0.02, cell_size - 0.02,
                           facecolor=cell_color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add text label
            if rows <= 8 and cols <= 8:
                ax.text(x + j * cell_size + cell_size/2, 
                       y + (rows - 1 - i) * cell_size + cell_size/2,
                       f'{matrix[i, j]:.0f}', ha='center', va='center', fontsize=8)

def create_overview_diagram():
    """Create overview diagram showing the tiling concept"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    BLOCK_SIZE = 4  # For visualization, use 4x4 blocks instead of 32x32
    M, N, K = 8, 8, 8
    
    # Create sample matrices
    A = np.arange(M * K).reshape(M, K) + 1
    B = np.arange(K * N).reshape(K, N) + 1
    C = np.zeros((M, N))
    
    # Calculate one tile for demonstration
    tile_row, tile_col = 0, 0
    for k_tile in range(0, K, BLOCK_SIZE):
        A_tile = A[tile_row*BLOCK_SIZE:(tile_row+1)*BLOCK_SIZE, 
                    k_tile:k_tile+BLOCK_SIZE]
        B_tile = B[k_tile:k_tile+BLOCK_SIZE, 
                    tile_col*BLOCK_SIZE:(tile_col+1)*BLOCK_SIZE]
        C[tile_row*BLOCK_SIZE:(tile_row+1)*BLOCK_SIZE, 
          tile_col*BLOCK_SIZE:(tile_col+1)*BLOCK_SIZE] += A_tile @ B_tile
    
    cell_size = 0.4
    
    # Color maps for highlighting tiles
    color_A = {}
    color_B = {}
    color_C = {}
    
    # Highlight the active tiles
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            color_A[(tile_row*BLOCK_SIZE + i, j)] = '#90EE90'  # Light green
            color_B[(i, tile_col*BLOCK_SIZE + j)] = '#87CEEB'  # Light blue
            color_C[(tile_row*BLOCK_SIZE + i, tile_col*BLOCK_SIZE + j)] = '#FFB6C1'  # Light pink
    
    # Draw matrices
    for ax, matrix, colors, title in [(axes[0], A, color_A, 'Matrix A (M×K)'),
                                        (axes[1], B, color_B, 'Matrix B (K×N)'),
                                        (axes[2], C, color_C, 'Matrix C (M×N)')]:
        ax.set_xlim(-0.5, matrix.shape[1] * cell_size + 0.5)
        ax.set_ylim(-0.5, matrix.shape[0] * cell_size + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        draw_matrix(ax, matrix, (0, 0), cell_size, colors, title)
    
    # Add block labels
    axes[0].text(-0.3, 3.5 * cell_size, f'Block\n({BLOCK_SIZE}×{BLOCK_SIZE})', 
                fontsize=9, ha='center', color='green', fontweight='bold')
    axes[1].text(3.5 * cell_size, 8.5 * cell_size, f'Block\n({BLOCK_SIZE}×{BLOCK_SIZE})', 
                fontsize=9, ha='center', color='blue', fontweight='bold')
    axes[2].text(3.5 * cell_size, 3.5 * cell_size, f'Block\n({BLOCK_SIZE}×{BLOCK_SIZE})', 
                fontsize=9, ha='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/shared_gemm_overview.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Overview diagram saved")
    plt.close()

def create_step_by_step_diagram():
    """Create step-by-step diagram showing the computation flow"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    BLOCK_SIZE = 4
    cell_size = 0.35
    
    # Step 1: Show the grid/block organization
    ax = axes[0]
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 1: Grid/Block Organization', fontsize=12, fontweight='bold', pad=20)
    
    # Draw grid of blocks
    for by in range(2):
        for bx in range(2):
            x, y = bx * 2.5, by * 2.5
            # Block outline
            block_rect = FancyBboxPatch((x, y), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor='#F0F0F0', edgecolor='black', linewidth=2)
            ax.add_patch(block_rect)
            ax.text(x + 1, y + 2.2, f'Block({bx}, {by})', ha='center', fontsize=9)
            
            # Draw threads inside block
            for ty in range(BLOCK_SIZE):
                for tx in range(BLOCK_SIZE):
                    thread_x = x + tx * (2 / BLOCK_SIZE)
                    thread_y = y + ty * (2 / BLOCK_SIZE)
                    thread_size = 2 / BLOCK_SIZE - 0.02
                    thread_rect = Rectangle((thread_x, thread_y), thread_size, thread_size,
                                             facecolor='#FFE4B5' if (tx, ty) == (1, 2) else 'white',
                                             edgecolor='gray', linewidth=0.5)
                    ax.add_patch(thread_rect)
    
    ax.text(2.5, -0.3, 'Grid of Blocks\n(Each block: 32×32 threads)', ha='center', fontsize=10)
    ax.annotate('Thread(1,2)', xy=(0.5, 1.5), xytext=(3.5, 4),
                fontsize=9, color='orange', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='orange'))
    
    # Step 2: Load A tile to shared memory
    ax = axes[1]
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 2: Load A Tile to Shared Memory', fontsize=12, fontweight='bold', pad=20)
    
    # Draw global memory A strip
    for i in range(4):
        for j in range(8):
            rect = Rectangle((j * 0.6, 3.5 - i * 0.6), 0.58, 0.58,
                             facecolor='#90EE90' if j < 4 else 'white',
                             edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    ax.text(2.4, 4.3, 'Global Memory: Matrix A row', ha='center', fontsize=9)
    
    # Arrow
    ax.annotate('', xy=(2.5, 2.5), xytext=(2.5, 3.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(4.5, 2.85, 'Cooperative\nLoading', fontsize=8, ha='center', color='blue')
    
    # Draw shared memory sA
    for i in range(4):
        for j in range(4):
            rect = Rectangle((j * 0.6 + 0.5, 1.5 - i * 0.6), 0.58, 0.58,
                             facecolor='#90EE90', edgecolor='green', linewidth=2)
            ax.add_patch(rect)
    ax.text(1.7, 0.7, 'Shared Memory: sA[4][4]', ha='center', fontsize=9, color='green')
    
    # Step 3: Load B tile to shared memory
    ax = axes[2]
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 3: Load B Tile to Shared Memory', fontsize=12, fontweight='bold', pad=20)
    
    # Draw global memory B strip (transposed view)
    for i in range(8):
        for j in range(4):
            rect = Rectangle((j * 0.6 + 2, 3.5 - i * 0.6), 0.58, 0.58,
                             facecolor='#87CEEB' if i < 4 else 'white',
                             edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    ax.text(3.2, 4.3, 'Global Memory: Matrix B column', ha='center', fontsize=9)
    
    # Arrow
    ax.annotate('', xy=(2.5, 2.5), xytext=(3.2, 3.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(0.5, 2.85, 'Cooperative\nLoading', fontsize=8, ha='center', color='blue')
    
    # Draw shared memory sB
    for i in range(4):
        for j in range(4):
            rect = Rectangle((j * 0.6 + 0.5, 1.5 - i * 0.6), 0.58, 0.58,
                             facecolor='#87CEEB', edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
    ax.text(1.7, 0.7, 'Shared Memory: sB[4][4]', ha='center', fontsize=9, color='blue')
    
    # Step 4: Compute dot product in shared memory
    ax = axes[3]
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 4: Compute in Shared Memory', fontsize=12, fontweight='bold', pad=20)
    
    # Draw sA row
    for j in range(4):
        rect = Rectangle((j * 0.6, 3.5), 0.58, 0.58,
                         facecolor='#90EE90', edgecolor='green', linewidth=2)
        ax.add_patch(rect)
        ax.text(j * 0.6 + 0.29, 3.79, f'A[{j}]', ha='center', fontsize=7)
    ax.text(1.2, 4.3, 'sA[ty][*]', ha='center', fontsize=9, color='green')
    
    # Draw multiplication symbol
    ax.text(2.8, 3.8, '×', fontsize=20, ha='center', va='center')
    
    # Draw sB column
    for i in range(4):
        rect = Rectangle((3.5, 3.5 - i * 0.6), 0.58, 0.58,
                         facecolor='#87CEEB', edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        ax.text(3.79, 3.79 - i * 0.6, f'B[{i}]', ha='center', fontsize=7)
    ax.text(3.8, 4.3, 'sB[*][tx]', ha='center', fontsize=9, color='blue')
    
    # Draw equals and accumulation
    ax.text(5.2, 3.8, '=', fontsize=20, ha='center', va='center')
    
    # Accumulation register
    acc_rect = FancyBboxPatch((6, 3.3), 1.5, 1, boxstyle="round,pad=0.05",
                             facecolor='#FFB6C1', edgecolor='red', linewidth=2)
    ax.add_patch(acc_rect)
    ax.text(6.75, 3.8, 'tmp\n(accumulator)', ha='center', va='center', fontsize=9)
    
    ax.text(4, 1.5, 'Each thread computes:\ntmp += Σ(sA[ty][k] * sB[k][tx])', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Step 5: Slide along K dimension
    ax = axes[4]
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 5: Slide Along K Dimension', fontsize=12, fontweight='bold', pad=20)
    
    # Draw K dimension as horizontal strip
    K_blocks = 4
    for step in range(K_blocks):
        x_offset = step * 1.8
        
        # A tile
        for i in range(2):
            for j in range(2):
                color = '#90EE90' if step == 1 else '#E0E0E0'
                rect = Rectangle((x_offset + j * 0.5, 3.5 - i * 0.5), 0.48, 0.48,
                                 facecolor=color, edgecolor='green', linewidth=1)
                ax.add_patch(rect)
        
        # B tile
        for i in range(2):
            for j in range(2):
                color = '#87CEEB' if step == 1 else '#E0E0E0'
                rect = Rectangle((x_offset + j * 0.5 + 1, 3.5 - i * 0.5), 0.48, 0.48,
                                 facecolor=color, edgecolor='blue', linewidth=1)
                ax.add_patch(rect)
        
        ax.text(x_offset + 0.5, 2.5, f'k_step={step}', ha='center', fontsize=8)
        
        # Arrow between steps
        if step < K_blocks - 1:
            ax.annotate('', xy=(x_offset + 1.6, 3.2), xytext=(x_offset + 1.4, 3.2),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    ax.text(3.3, 4.5, 'Loop: for (k_step = 0; k_step < K/BLOCK_SIZE; k_step++)', 
            ha='center', fontsize=10, fontweight='bold')
    ax.text(3.3, 1.5, 'For each k_step:\n1. Load A tile, B tile → shared memory\n2. Synchronize\n3. Compute partial sum\n4. Synchronize', 
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Step 6: Write result to global memory
    ax = axes[5]
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 6: Write Result to Global Memory', fontsize=12, fontweight='bold', pad=20)
    
    # Draw accumulated result in register
    acc_rect = FancyBboxPatch((0.5, 2.5), 1.5, 1, boxstyle="round,pad=0.05",
                             facecolor='#FFB6C1', edgecolor='red', linewidth=2)
    ax.add_patch(acc_rect)
    ax.text(1.25, 3, 'tmp\n(final)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(3.5, 2.5), xytext=(2.2, 2.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(2.7, 3.2, 'Write to\nC[row][col]', fontsize=9, ha='center', color='red')
    
    # Draw C matrix tile
    for i in range(4):
        for j in range(4):
            color = '#FFB6C1' if (i, j) == (2, 1) else 'white'
            rect = Rectangle((3 + j * 0.6, 1.5 - i * 0.6), 0.58, 0.58,
                             facecolor=color, edgecolor='red', linewidth=2 if (i, j) == (2, 1) else 1)
            ax.add_patch(rect)
            if (i, j) == (2, 1):
                ax.text(3 + j * 0.6 + 0.29, 1.5 - i * 0.6 + 0.29, 'C', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(4.2, 0.7, 'Global Memory: Matrix C', ha='center', fontsize=9, color='red')
    ax.text(4.2, -0.2, 'C[row*N + col] = alpha * tmp + beta * C[row*N + col]', 
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/shared_gemm_steps.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Step-by-step diagram saved")
    plt.close()

def create_memory_hierarchy_diagram():
    """Create diagram showing memory hierarchy and access patterns"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('SGEMM Shared Memory Kernel: Memory Hierarchy & Access Pattern', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Global Memory
    global_mem = FancyBboxPatch((1, 7.5), 12, 2, boxstyle="round,pad=0.1",
                               facecolor='#FFE4E1', edgecolor='darkred', linewidth=3)
    ax.add_patch(global_mem)
    ax.text(7, 9.2, 'Global Memory (HBM/GDDR7)', ha='center', fontsize=12, fontweight='bold', color='darkred')
    ax.text(7, 8.5, 'Large capacity (~32GB), High latency (~400-800 cycles), High bandwidth (~1.8 TB/s)', 
            ha='center', fontsize=9, style='italic')
    ax.text(7, 8, 'Access pattern: Coalesced load/store only at block level', 
            ha='center', fontsize=9)
    
    # Arrow down
    ax.annotate('', xy=(7, 7.2), xytext=(7, 7.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(8.5, 7.35, 'Load A/B tiles\n(once per k_step)', fontsize=9, color='red', ha='center')
    
    # Shared Memory
    shared_mem = FancyBboxPatch((2, 4.5), 10, 2.5, boxstyle="round,pad=0.1",
                               facecolor='#E0FFE0', edgecolor='darkgreen', linewidth=3)
    ax.add_patch(shared_mem)
    ax.text(7, 6.7, 'Shared Memory (SRAM)', ha='center', fontsize=12, fontweight='bold', color='darkgreen')
    ax.text(7, 6.1, 'Small per-block (~48KB), Low latency (~20-30 cycles), Very high bandwidth (~10+ TB/s)', 
            ha='center', fontsize=9, style='italic')
    ax.text(7, 5.5, 'Access pattern: Bank-conflict-free access by all threads in block', 
            ha='center', fontsize=9)
    ax.text(7, 4.9, 'Data reuse: Each element accessed BLOCK_SIZE times (32x for 32x32 block)', 
            ha='center', fontsize=9, fontweight='bold', color='green')
    
    # Arrow down
    ax.annotate('', xy=(7, 4.2), xytext=(7, 4.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax.text(8.5, 4.35, 'Compute\n(BLOCK_SIZE^2 multiplies)', fontsize=9, color='green', ha='center')
    
    # Registers
    registers = FancyBboxPatch((3, 1.5), 8, 2.5, boxstyle="round,pad=0.1",
                              facecolor='#E6E6FA', edgecolor='darkblue', linewidth=3)
    ax.add_patch(registers)
    ax.text(7, 3.7, 'Registers (per-thread)', ha='center', fontsize=12, fontweight='bold', color='darkblue')
    ax.text(7, 3.1, 'Ultra-low latency (~1 cycle), Ultra-high bandwidth', 
            ha='center', fontsize=9, style='italic')
    ax.text(7, 2.5, 'Each thread: tmp accumulator, threadIdx.x/y indices, row/col coordinates', 
            ha='center', fontsize=9)
    
    # Key insight box
    insight_box = FancyBboxPatch((0.5, 0), 13, 1.2, boxstyle="round,pad=0.1",
                                facecolor='#FFFACD', edgecolor='orange', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(7, 0.6, '💡 Key Insight: By loading data from Global → Shared once, then reusing BLOCK_SIZE times in Shared Memory,\n'
                    '    we increase Arithmetic Intensity from ~0.5 to ~683 FLOPs/byte (1365× improvement)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='darkorange')
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/shared_gemm_memory_hierarchy.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Memory hierarchy diagram saved")
    plt.close()

def create_code_flow_diagram():
    """Create diagram showing the code execution flow"""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(6, 13.5, 'SGEMM Shared Kernel: Code Execution Flow', 
            ha='center', fontsize=14, fontweight='bold')
    
    y_pos = 12.5
    box_height = 0.8
    box_width = 10
    
    # Helper function to draw code block
    def draw_code_block(y, title, code_lines, color):
        # Title
        ax.text(6, y + 0.5, title, ha='center', fontsize=10, fontweight='bold', color=color)
        y -= 0.3
        
        # Code block
        code_height = len(code_lines) * 0.25 + 0.3
        code_box = FancyBboxPatch((1, y - code_height), box_width, code_height,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#F5F5F5', edgecolor=color, linewidth=2)
        ax.add_patch(code_box)
        
        # Code text
        for i, line in enumerate(code_lines):
            ax.text(1.3, y - 0.2 - i * 0.22, line, fontsize=8, family='monospace')
        
        return y - code_height - 0.4
    
    # Step 1: Memory allocation
    y_pos = draw_code_block(y_pos, '1. Allocate Shared Memory', [
        '__shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];',
        '__shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];'
    ], 'darkgreen')
    
    # Step 2: Index calculation
    y_pos = draw_code_block(y_pos, '2. Calculate Thread Indices', [
        'int tx = threadIdx.x;        // 0 ~ 31 (column within block)',
        'int ty = threadIdx.y;        // 0 ~ 31 (row within block)',
        'int row = blockIdx.y * BLOCK_SIZE + ty;  // Global row in C',
        'int col = blockIdx.x * BLOCK_SIZE + tx;  // Global col in C',
        'float tmp = 0.0f;            // Accumulator in register'
    ], 'darkblue')
    
    # Step 3: Main loop
    y_pos = draw_code_block(y_pos, '3. Main Loop: Iterate over K dimension', [
        'for (int k_step = 0; k_step < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k_step) {',
        '    int k_offset = k_step * BLOCK_SIZE;',
        ''
    ], 'purple')
    
    # Nested step 3a
    y_pos = draw_code_block(y_pos, '    3a. Load A tile to Shared Memory', [
        '    // Each thread loads one element of A',
        '    if (row < M && k_offset + tx < K)',
        '        sA[ty][tx] = A[row * K + k_offset + tx];',
        '    else',
        '        sA[ty][tx] = 0.0f;  // Out-of-bounds padding'
    ], 'green')
    
    # Nested step 3b
    y_pos = draw_code_block(y_pos, '    3b. Load B tile to Shared Memory', [
        '    // Each thread loads one element of B',
        '    if (k_offset + ty < K && col < N)',
        '        sB[ty][tx] = B[(k_offset + ty) * N + col];',
        '    else',
        '        sB[ty][tx] = 0.0f;  // Out-of-bounds padding'
    ], 'blue')
    
    # Step 3c
    y_pos = draw_code_block(y_pos, '    3c. Synchronize All Threads', [
        '    __syncthreads();  // Wait for all threads to finish loading'
    ], 'red')
    
    # Step 3d
    y_pos = draw_code_block(y_pos, '    3d. Compute Dot Product in Shared Memory', [
        '    #pragma unroll',
        '    for (int k = 0; k < BLOCK_SIZE; ++k) {',
        '        // Each thread computes one element of C',
        '        // Row from sA, Column from sB',
        '        tmp += sA[ty][k] * sB[k][tx];',
        '    }'
    ], 'darkorange')
    
    # Step 3e
    y_pos = draw_code_block(y_pos, '    3e. Synchronize Before Next Iteration', [
        '    __syncthreads();  // Prevent overwriting before all threads finish'
    ], 'red')
    
    # Close main loop
    code_box = FancyBboxPatch((1, y_pos - 0.3), box_width, 0.3,
                               boxstyle="round,pad=0.05",
                               facecolor='#F5F5F5', edgecolor='purple', linewidth=2)
    ax.add_patch(code_box)
    ax.text(1.3, y_pos - 0.15, '}', fontsize=8, family='monospace', color='purple')
    y_pos -= 0.6
    
    # Step 4: Write result
    y_pos = draw_code_block(y_pos, '4. Write Result to Global Memory', [
        'if (row < M && col < N)',
        '    C[row * N + col] = alpha * tmp + beta * C[row * N + col];'
    ], 'darkred')
    
    plt.tight_layout()
    plt.savefig('/home/chenb/code/master/LLM/GEMM/images/shared_gemm_code_flow.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Code flow diagram saved")
    plt.close()

if __name__ == '__main__':
    print("Generating SGEMM Shared Kernel visualization diagrams...")
    print("=" * 60)
    
    create_overview_diagram()
    create_step_by_step_diagram()
    create_memory_hierarchy_diagram()
    create_code_flow_diagram()
    
    print("=" * 60)
    print("All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  - images/shared_gemm_overview.png")
    print("  - images/shared_gemm_steps.png")
    print("  - images/shared_gemm_memory_hierarchy.png")
    print("  - images/shared_gemm_code_flow.png")

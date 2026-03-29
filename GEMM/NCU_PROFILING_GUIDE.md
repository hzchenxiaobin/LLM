# Nsight Compute (ncu) Profiling Guide for GEMM Operators

This guide explains how to run Nsight Compute profiling on all GEMM operators in this project.

## Prerequisites

1. **CUDA Toolkit** (11.8 or higher recommended)
   - Download: https://developer.nvidia.com/cuda-downloads

2. **NVIDIA Nsight Compute**
   - Usually included with CUDA Toolkit
   - Can be downloaded separately: https://developer.nvidia.com/nsight-compute

3. **NVIDIA GPU** with Compute Capability >= 7.0
   - Volta (sm_70), Turing (sm_75), Ampere (sm_80/sm_86), Ada (sm_89), or Hopper (sm_90)

## Quick Start

### Method 1: Using Python Script (Recommended)

```powershell
# Build and profile all kernels
python run_ncu.py

# Build only
python run_ncu.py --build-only

# Profile only (requires pre-built executable)
python run_ncu.py --profile-only

# Profile specific kernel
python run_ncu.py --kernel sgemm_register_kernel

# List available kernels
python run_ncu.py --list-kernels
```

### Method 2: Using PowerShell Script

```powershell
# Run the complete profiling pipeline
.\run_ncu_profiling.ps1
```

### Method 3: Manual Build + Profile

#### Step 1: Build the Project

```powershell
# Create build directory
mkdir build
cd build
mkdir gemm

# Compile (adjust sm_70 to your GPU architecture)
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/main.cu -o main.obj
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/gemm/sgemm_cublas.cu -o gemm/sgemm_cublas.obj
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/gemm/sgemm_naive.cu -o gemm/sgemm_naive.obj
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/gemm/sgemm_shared.cu -o gemm/sgemm_shared.obj
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/gemm/sgemm_register.cu -o gemm/sgemm_register.obj
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/gemm/sgemm_register_vectorized.cu -o gemm/sgemm_register_vectorized.obj
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/gemm/sgemm_register_bank_conflict.cu -o gemm/sgemm_register_bank_conflict.obj
nvcc -O3 -std=c++14 -arch=sm_70 -c ../src/gemm/sgemm_register_vec_bank.cu -o gemm/sgemm_register_vec_bank.obj

# Link
nvcc -O3 -std=c++14 -arch=sm_70 main.obj gemm/*.obj -o benchmark_gemm.exe -lcublas

cd ..
```

#### Step 2: Run Nsight Compute Profiling

```powershell
# Create output directory
mkdir ncu_results

# Profile each kernel
ncu --kernel-name sgemm_naive_kernel --launch-count 1 -o ncu_results/sgemm_naive_kernel.ncu-rep build/benchmark_gemm.exe
ncu --kernel-name sgemm_shared_kernel --launch-count 1 -o ncu_results/sgemm_shared_kernel.ncu-rep build/benchmark_gemm.exe
ncu --kernel-name sgemm_register_kernel --launch-count 1 -o ncu_results/sgemm_register_kernel.ncu-rep build/benchmark_gemm.exe
ncu --kernel-name sgemm_register_vectorized_kernel --launch-count 1 -o ncu_results/sgemm_register_vectorized_kernel.ncu-rep build/benchmark_gemm.exe
ncu --kernel-name sgemm_register_bank_conflict_kernel --launch-count 1 -o ncu_results/sgemm_register_bank_conflict_kernel.ncu-rep build/benchmark_gemm.exe
ncu --kernel-name sgemm_register_kernel_vec_bank --launch-count 1 -o ncu_results/sgemm_register_kernel_vec_bank.ncu-rep build/benchmark_gemm.exe
```

## Available Kernels

| Kernel Name | Description |
|------------|-------------|
| `sgemm_naive_kernel` | Naive implementation (baseline) |
| `sgemm_shared_kernel` | Shared memory tiling (32x32 blocks) |
| `sgemm_register_kernel` | Register tiling (8x8 per thread, 128x128 per block) |
| `sgemm_register_vectorized_kernel` | Register tiling + float4 vectorized loads/stores |
| `sgemm_register_bank_conflict_kernel` | Register tiling + shared memory padding (bank conflict fix) |
| `sgemm_register_kernel_vec_bank` | Register tiling + vectorized + padding (optimized) |

## Viewing Results

### Nsight Compute GUI

```powershell
# Open a specific report
ncu-ui ncu_results/sgemm_register_kernel.ncu-rep

# Or open Nsight Compute and load the file manually
```

### Command Line Export

```powershell
# Export to CSV for analysis
ncu --import ncu_results/sgemm_register_kernel.ncu-rep --csv > results.csv
```

## Key Metrics to Analyze

When profiling GEMM kernels, focus on these metrics:

1. **Performance**
   - `gpu__time_duration.sum` - Total execution time
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed` - SM utilization %

2. **Memory Bandwidth**
   - `dram__bytes.sum` - Total DRAM bytes transferred
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed` - DRAM bandwidth utilization
   - `l1tex__t_bytes.avg` - L1/TEX cache bytes

3. **Occupancy & Parallelism**
   - `sm__warps_active.avg.pct_of_peak_sustained_active` - Active warps %

4. **Memory Access Patterns**
   - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` - Global load sectors

## Troubleshooting

### "ncu not found"

Add Nsight Compute to your PATH:

```powershell
# Find your ncu installation
$env:PATH += ";C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0"
```

### "nvcc not found"

Add CUDA to your PATH:

```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
```

### Architecture Mismatch

Adjust `-arch=sm_XX` to match your GPU:
- V100: `sm_70`
- RTX 20 series: `sm_75`
- A100/RTX 30 series: `sm_80`
- RTX 40 series: `sm_89`
- H100: `sm_90`

### Permission Issues

Run PowerShell as Administrator if needed.

## Batch File (Simple)

A simple batch file is also provided for quick profiling:

```powershell
.\run_ncu_simple.bat
```

This assumes the project is already built.

## Expected Output Structure

After running profiling, you should have:

```
GEMM/
├── build/
│   └── benchmark_gemm.exe
├── ncu_results/
│   ├── sgemm_naive_kernel.ncu-rep
│   ├── sgemm_shared_kernel.ncu-rep
│   ├── sgemm_register_kernel.ncu-rep
│   ├── sgemm_register_vectorized_kernel.ncu-rep
│   ├── sgemm_register_bank_conflict_kernel.ncu-rep
│   └── sgemm_register_kernel_vec_bank.ncu-rep
└── ...
```

## Tips for Analysis

1. **Compare kernels**: Load multiple `.ncu-rep` files in Nsight Compute GUI to compare side-by-side
2. **Look for bottlenecks**: Check if kernel is memory-bound or compute-bound
3. **Verify optimizations**: Bank conflict fixes should show reduced replay overhead
4. **Vectorization impact**: Compare `sgemm_register_kernel` vs `sgemm_register_vectorized_kernel`

## Additional Resources

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GEMM Optimization Guide](docs/sgemm_register_analysis.md)

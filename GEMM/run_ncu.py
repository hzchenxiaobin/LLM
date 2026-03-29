#!/usr/bin/env python3
"""
Script to run Nsight Compute profiling on all GEMM kernels.

Requirements:
    - CUDA Toolkit installed (nvcc)
    - Nsight Compute installed (ncu)
    - GPU with Compute Capability >= 7.0

Usage:
    python run_ncu.py [--build-only] [--profile-only] [--kernel <kernel_name>]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Configuration
BUILD_DIR = Path("build")
TARGET = BUILD_DIR / "benchmark_gemm.exe"
NCU_OUTPUT_DIR = Path("ncu_results")

# Kernel names to profile (must match kernel names in the CUDA code)
KERNELS = [
    "sgemm_naive_kernel",
    "sgemm_shared_kernel",
    "sgemm_register_kernel",
    "sgemm_register_vectorized_kernel",
    "sgemm_register_bank_conflict_kernel",
    "sgemm_register_kernel_vec_bank",
]

# Source files to compile
KERNEL_FILES = [
    ("src/gemm/sgemm_cublas.cu", BUILD_DIR / "gemm" / "sgemm_cublas.obj"),
    ("src/gemm/sgemm_naive.cu", BUILD_DIR / "gemm" / "sgemm_naive.obj"),
    ("src/gemm/sgemm_shared.cu", BUILD_DIR / "gemm" / "sgemm_shared.obj"),
    ("src/gemm/sgemm_register.cu", BUILD_DIR / "gemm" / "sgemm_register.obj"),
    ("src/gemm/sgemm_register_vec_bank.cu", BUILD_DIR / "gemm" / "sgemm_register_vec_bank.obj"),
    ("src/gemm/sgemm_register_bank_conflict.cu", BUILD_DIR / "gemm" / "sgemm_register_bank_conflict.obj"),
    ("src/gemm/sgemm_register_vectorized.cu", BUILD_DIR / "gemm" / "sgemm_register_vectorized.obj"),
]

# Compilation settings
NVCC_FLAGS = ["-O3", "-std=c++14", "-arch=sm_70"]
LIBS = ["-lcublas"]


def find_executable(name: str, hints: Optional[List[str]] = None) -> Optional[str]:
    """Find an executable in PATH or common locations."""
    # Try PATH first
    try:
        result = subprocess.run(
            ["where", name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return result.stdout.strip().split("\n")[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try common Windows paths
    if hints:
        for hint in hints:
            if os.path.isfile(hint):
                return hint

    return None


def build_project() -> bool:
    """Build the GEMM benchmark executable."""
    print("=" * 60)
    print("Building GEMM Project")
    print("=" * 60)

    # Find nvcc
    nvcc_hints = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe",
    ]

    nvcc = find_executable("nvcc", nvcc_hints)
    if not nvcc:
        print("[ERROR] nvcc not found. Please install CUDA Toolkit.")
        return False

    print(f"[INFO] Using nvcc: {nvcc}")

    # Create build directories
    BUILD_DIR.mkdir(exist_ok=True)
    (BUILD_DIR / "gemm").mkdir(exist_ok=True)

    # Compile main.cu
    print("[INFO] Compiling main.cu...")
    main_obj = BUILD_DIR / "main.obj"
    cmd = [nvcc] + NVCC_FLAGS + ["-c", "src/main.cu", "-o", str(main_obj)]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to compile main.cu:\n{result.stderr}")
        return False

    # Compile kernel files
    obj_files = [str(main_obj)]
    for src, obj in KERNEL_FILES:
        print(f"[INFO] Compiling {src}...")
        cmd = [nvcc] + NVCC_FLAGS + ["-c", src, "-o", str(obj)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            print(f"[ERROR] Failed to compile {src}:\n{result.stderr}")
            return False
        obj_files.append(str(obj))

    # Link executable
    print("[INFO] Linking executable...")
    cmd = [nvcc] + NVCC_FLAGS + obj_files + ["-o", str(TARGET)] + LIBS
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to link:\n{result.stderr}")
        return False

    print(f"[SUCCESS] Build complete: {TARGET}")
    return True


def find_ncu() -> Optional[str]:
    """Find ncu executable."""
    hints = [
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.0\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.2.0\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.1.0\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2022.4.0\ncu.exe",
    ]
    for p in hints:
        if os.path.isfile(p):
            return p
    nsight_root = Path(r"C:\Program Files\NVIDIA Corporation")
    if nsight_root.is_dir():
        for install in sorted(nsight_root.glob("Nsight Compute *"), reverse=True):
            target = install / "target" / "windows-desktop-win7-x64" / "ncu.exe"
            if target.is_file():
                return str(target)
            legacy = install / "ncu.exe"
            if legacy.is_file():
                return str(legacy)
    return find_executable("ncu", None)


def cuda_bin_for_child_env() -> Optional[str]:
    """Directory containing cudart / cublas DLLs so ncu-launched apps can start."""
    nvcc_hints = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe",
    ]
    nvcc = find_executable("nvcc", nvcc_hints)
    if nvcc:
        return str(Path(nvcc).parent)
    toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():
        for vdir in sorted(toolkit.glob("v*"), reverse=True):
            b = vdir / "bin"
            if b.is_dir():
                return str(b)
    return None


def profile_kernel(kernel_name: str, ncu_path: str) -> bool:
    """Profile a single kernel using ncu."""
    output_file = NCU_OUTPUT_DIR / f"{kernel_name}.ncu-rep"

    # Metrics to collect
    metrics = [
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "l1tex__t_bytes.avg",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "dram__bytes.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ]

    cmd = [
        ncu_path,
        "-f",
        "--kernel-name", kernel_name,
        "--launch-count", "1",
        "--metrics", ",".join(metrics),
        "-o", str(output_file),
        str(TARGET),
    ]

    print(f"[PROFILING] {kernel_name}...")
    env = os.environ.copy()
    cuda_bin = cuda_bin_for_child_env()
    if cuda_bin:
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode == 0:
        size = output_file.stat().st_size / 1024
        print(f"[OK] {kernel_name} -> {output_file} ({size:.1f} KB)")
        return True
    else:
        print(f"[ERROR] Failed to profile {kernel_name}")
        print(result.stderr)
        return False


def run_profiling(specific_kernel: Optional[str] = None) -> bool:
    """Run ncu profiling on all or specific kernels."""
    print("=" * 60)
    print("Running Nsight Compute Profiling")
    print("=" * 60)

    # Check executable exists
    if not TARGET.exists():
        print(f"[ERROR] Executable not found: {TARGET}")
        print("Please build first with: python run_ncu.py --build-only")
        return False

    # Find ncu
    ncu_path = find_ncu()
    if not ncu_path:
        print("[ERROR] ncu not found. Please install Nsight Compute.")
        print("Common installation path:")
        print("  C:\\Program Files\\NVIDIA Corporation\\Nsight Compute <version>\\ncu.exe")
        return False

    print(f"[INFO] Using ncu: {ncu_path}")

    # Create output directory
    NCU_OUTPUT_DIR.mkdir(exist_ok=True)

    # Determine which kernels to profile
    kernels_to_profile = [specific_kernel] if specific_kernel else KERNELS

    # Profile each kernel
    success_count = 0
    for kernel in kernels_to_profile:
        if kernel not in KERNELS:
            print(f"[WARNING] Unknown kernel: {kernel}")
            print(f"[INFO] Available kernels: {', '.join(KERNELS)}")
            continue

        if profile_kernel(kernel, ncu_path):
            success_count += 1

    # Print summary
    print("\n" + "=" * 60)
    print("Profiling Summary")
    print("=" * 60)
    print(f"Total kernels: {len(kernels_to_profile)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(kernels_to_profile) - success_count}")
    print(f"\nResults saved to: {NCU_OUTPUT_DIR.absolute()}\\")
    print("\nTo view results:")
    print(f"  ncu-ui {NCU_OUTPUT_DIR}\\<kernel_name>.ncu-rep")

    return success_count == len(kernels_to_profile)


def main():
    parser = argparse.ArgumentParser(description="Run Nsight Compute profiling on GEMM kernels")
    parser.add_argument("--build-only", action="store_true", help="Only build the project")
    parser.add_argument("--profile-only", action="store_true", help="Only run profiling (skip build)")
    parser.add_argument("--kernel", type=str, help="Profile specific kernel only")
    parser.add_argument("--list-kernels", action="store_true", help="List available kernels")

    args = parser.parse_args()

    if args.list_kernels:
        print("Available kernels:")
        for k in KERNELS:
            print(f"  - {k}")
        return 0

    success = True

    # Build phase
    if not args.profile_only:
        success = build_project()
        if not success:
            return 1

    # Profile phase
    if not args.build_only:
        success = run_profiling(args.kernel) and success

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

# PowerShell script to run Nsight Compute profiling on all GEMM kernels
# Usage: .\run_ncu_profiling.ps1

$ErrorActionPreference = "Stop"

# Ensure CUDA DLLs are visible to the app when launched under ncu
$cudaBinCandidates = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
)
foreach ($dir in $cudaBinCandidates) {
    if (Test-Path $dir) {
        $env:PATH = "$dir;$env:PATH"
        break
    }
}

# Configuration
$BUILD_DIR = "build"
$TARGET = "$BUILD_DIR\benchmark_gemm.exe"
$NCU_OUTPUT_DIR = "ncu_results"

# List of kernel names to profile (must match kernel names in the code)
$KERNELS = @(
    "sgemm_naive_kernel",
    "sgemm_shared_kernel",
    "sgemm_register_kernel",
    "sgemm_register_vectorized_kernel",
    "sgemm_register_bank_conflict_kernel",
    "sgemm_register_kernel_vec_bank"
)

# Function to print colored messages
function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Green
}

function Write-Error($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

# Step 1: Build the project
Write-Info "Step 1: Building the GEMM project..."
if (-not (Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
}
if (-not (Test-Path "$BUILD_DIR\gemm")) {
    New-Item -ItemType Directory -Path "$BUILD_DIR\gemm" | Out-Null
}

# Build commands (using nvcc directly)
$NVCC = "nvcc"
$NVCC_FLAGS = "-O3 -std=c++14 -arch=sm_70"
$LIBS = "-lcublas"

# Compile main.cu
Write-Info "Compiling main.cu..."
& $NVCC $NVCC_FLAGS.Split(" ") -c "src/main.cu" -o "$BUILD_DIR/main.obj"

# Compile each kernel
$KERNEL_FILES = @(
    @("src/gemm/sgemm_cublas.cu", "$BUILD_DIR/gemm/sgemm_cublas.obj"),
    @("src/gemm/sgemm_naive.cu", "$BUILD_DIR/gemm/sgemm_naive.obj"),
    @("src/gemm/sgemm_shared.cu", "$BUILD_DIR/gemm/sgemm_shared.obj"),
    @("src/gemm/sgemm_register.cu", "$BUILD_DIR/gemm/sgemm_register.obj"),
    @("src/gemm/sgemm_register_vec_bank.cu", "$BUILD_DIR/gemm/sgemm_register_vec_bank.obj"),
    @("src/gemm/sgemm_register_bank_conflict.cu", "$BUILD_DIR/gemm/sgemm_register_bank_conflict.obj"),
    @("src/gemm/sgemm_register_vectorized.cu", "$BUILD_DIR/gemm/sgemm_register_vectorized.obj")
)

foreach ($file in $KERNEL_FILES) {
    $src = $file[0]
    $obj = $file[1]
    Write-Info "Compiling $src..."
    & $NVCC $NVCC_FLAGS.Split(" ") -c $src -o $obj
}

# Link
Write-Info "Linking executable..."
$OBJ_FILES = @(
    "$BUILD_DIR/main.obj",
    "$BUILD_DIR/gemm/sgemm_cublas.obj",
    "$BUILD_DIR/gemm/sgemm_naive.obj",
    "$BUILD_DIR/gemm/sgemm_shared.obj",
    "$BUILD_DIR/gemm/sgemm_register.obj",
    "$BUILD_DIR/gemm/sgemm_register_vec_bank.obj",
    "$BUILD_DIR/gemm/sgemm_register_bank_conflict.obj",
    "$BUILD_DIR/gemm/sgemm_register_vectorized.obj"
)
& $NVCC $NVCC_FLAGS.Split(" ") $OBJ_FILES -o $TARGET $LIBS.Split(" ")

Write-Info "Build completed: $TARGET"

# Step 2: Create output directory
if (-not (Test-Path $NCU_OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $NCU_OUTPUT_DIR | Out-Null
}

# Step 3: Check if ncu is available
Write-Info "Checking for ncu (Nsight Compute)..."
$ncuPath = (Get-Command ncu -ErrorAction SilentlyContinue).Source
if (-not $ncuPath) {
    # Try common installation paths
    $possiblePaths = @(
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.0\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.2.0\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.1.0\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2022.4.0\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2022.3.0\ncu.exe"
    )
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $ncuPath = $path
            break
        }
    }
}

if (-not $ncuPath) {
    Write-Error "ncu not found. Please install Nsight Compute or add it to PATH."
    exit 1
}

Write-Info "Found ncu at: $ncuPath"

# Step 4: Run ncu profiling for each kernel
Write-Info "Step 4: Running Nsight Compute profiling..."

foreach ($kernel in $KERNELS) {
    Write-Info "Profiling kernel: $kernel"

    $outputFile = "$NCU_OUTPUT_DIR\$kernel.ncu-rep"

    # ncu command to profile specific kernel
    # --kernel-name: filter by kernel name
    # --launch-count: number of kernel launches to profile (set to 1 for faster profiling)
    # -o: output file
    & $ncuPath -f --kernel-name $kernel `
               --launch-count 1 `
               --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes.avg,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed `
               -o $outputFile `
               $TARGET

    if ($LASTEXITCODE -eq 0) {
        Write-Info "Successfully profiled $kernel -> $outputFile"
    } else {
        Write-Error "Failed to profile $kernel (exit code: $LASTEXITCODE)"
    }
}

# Step 5: Generate summary report
Write-Info "Step 5: Generating summary..."
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Nsight Compute Profiling Complete!" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: $NCU_OUTPUT_DIR\" -ForegroundColor Yellow
Write-Host ""
Write-Host "To view results:" -ForegroundColor White
Write-Host "  ncu-ui $NCU_OUTPUT_DIR\<kernel_name>.ncu-rep" -ForegroundColor Gray
Write-Host ""
Write-Host "Profiled kernels:" -ForegroundColor White
foreach ($kernel in $KERNELS) {
    $file = "$NCU_OUTPUT_DIR\$kernel.ncu-rep"
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1KB
        Write-Host "  [OK] $kernel.ncu-rep ($([math]::Round($size, 2)) KB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $kernel.ncu-rep" -ForegroundColor Red
    }
}

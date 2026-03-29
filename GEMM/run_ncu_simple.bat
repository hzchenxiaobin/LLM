@echo off
chcp 65001 >nul
REM Simple batch script to run Nsight Compute profiling on all GEMM kernels

set "BUILD_DIR=build"
set "TARGET=%BUILD_DIR%\benchmark_gemm.exe"
set "NCU_OUTPUT_DIR=ncu_results"

REM Check if executable exists
if not exist "%TARGET%" (
    echo [ERROR] Executable not found: %TARGET%
    echo Please build the project first using: make
    exit /b 1
)

REM Create output directory
if not exist "%NCU_OUTPUT_DIR%" mkdir "%NCU_OUTPUT_DIR%"

echo ==========================================
echo Nsight Compute Profiling for GEMM Kernels
echo ==========================================
echo.

REM List of kernels to profile
set KERNELS=sgemm_naive_kernel sgemm_shared_kernel sgemm_register_kernel sgemm_register_vectorized_kernel sgemm_register_bank_conflict_kernel sgemm_register_kernel_vec_bank

REM Find ncu
set "NCU=ncu"
where ncu >nul 2>&1
if %ERRORLEVEL% neq 0 (
    REM Try common paths
    if exist "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.exe" (
        set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.exe"
    ) else if exist "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.exe" (
        set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.exe"
    ) else if exist "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.exe" (
        set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.exe"
    ) else if exist "C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.0\ncu.exe" (
        set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.0\ncu.exe"
    ) else (
        echo [ERROR] ncu not found. Please install Nsight Compute.
        exit /b 1
    )
)

echo Found ncu: %NCU%
echo.

REM Profile each kernel
for %%K in (%KERNELS%) do (
    echo [PROFILING] %%K
    "%~dp0%NCU%" -f --kernel-name %%K --launch-count 1 -o "%NCU_OUTPUT_DIR%\%%K.ncu-rep" "%TARGET%"
    if %ERRORLEVEL% equ 0 (
        echo [OK] %%K profiled successfully
    ) else (
        echo [WARN] Failed to profile %%K
    )
    echo.
)

echo ==========================================
echo Profiling Complete!
echo ==========================================
echo.
echo Results saved to: %NCU_OUTPUT_DIR%\
echo.
echo To view a report, run:
echo   ncu-ui %NCU_OUTPUT_DIR%\sgemm_naive_kernel.ncu-rep
echo.

pause

#!/bin/bash

# Nsight Compute Profiling Script for LLM CUDA Operators
# This script compiles and profiles all CUDA operators in the repository

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPORTS_DIR="nsight_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_BASE_DIR="${REPORTS_DIR}/${TIMESTAMP}"

# Nsight Compute options
# Default metrics: DRAM bytes, GPU time, SM throughput, etc.
NCU_METRICS="--metrics dram__bytes.sum,gpu__time_duration.avg,smsp__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak,score__pipe_fp32_cycles_active.avg.pct_of_peak"

# Parse command line arguments
NCU_ARGS=""
SKIP_BUILD=false
SKIP_PROFILE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-profile)
            SKIP_PROFILE=true
            shift
            ;;
        --metrics)
            NCU_METRICS="$2"
            shift 2
            ;;
        --set)
            NCU_ARGS="$NCU_ARGS --set $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    print_error "Nsight Compute (ncu) not found. Please install Nsight Compute first."
    exit 1
fi

print_info "Nsight Compute found: $(ncu --version | head -1)"

# Create reports directory
mkdir -p "$REPORT_BASE_DIR"
print_info "Reports will be saved to: $REPORT_BASE_DIR"

# Function to profile an operator
profile_operator() {
    local operator_name=$1
    local build_dir=$2
    local executable=$3
    local working_dir=$4
    
    print_info "=========================================="
    print_info "Profiling: $operator_name"
    print_info "=========================================="
    
    if [ "$SKIP_BUILD" = false ]; then
        print_info "Building $operator_name..."
        cd "$working_dir"
        if make clean && make; then
            print_info "Build successful for $operator_name"
        else
            print_error "Build failed for $operator_name"
            cd - > /dev/null
            return 1
        fi
        cd - > /dev/null
    else
        print_warning "Skipping build for $operator_name"
    fi
    
    if [ "$SKIP_PROFILE" = false ]; then
        print_info "Running Nsight Compute profiling..."
        cd "$working_dir"
        
        local report_name="${REPORT_BASE_DIR}/${operator_name}"
        
        if ncu -o "$report_name" $NCU_METRICS $NCU_ARGS "$executable"; then
            print_info "Profiling completed for $operator_name"
            print_info "Report saved to: ${report_name}.ncu-rep"
        else
            print_error "Profiling failed for $operator_name"
            cd - > /dev/null
            return 1
        fi
        cd - > /dev/null
    else
        print_warning "Skipping profiling for $operator_name"
    fi
    
    print_info "=========================================="
    echo ""
}

# Main script
print_info "Starting Nsight Compute profiling for all operators"
print_info "Timestamp: $TIMESTAMP"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Profile all operators
profile_operator "gemm" "build" "./build/benchmark_gemm" "./GEMM"
profile_operator "scan" "build" "./build/benchmark" "./scan"
profile_operator "reduction" "." "./benchmark" "./reduction"
profile_operator "transpose" "." "./transpose_benchmark" "./transpose"
profile_operator "softmax" "." "./benchmark_all" "./softmax"
profile_operator "topk" "build" "./build/topk_benchmark" "./topk"
profile_operator "flashattention" "build" "./build/benchmark_flashattention" "./flashattention"
profile_operator "batch_gemm" "build" "./build/benchmark" "./batch_gemm"

# Summary
print_info "=========================================="
print_info "Profiling Summary"
print_info "=========================================="
print_info "All operators profiled successfully!"
print_info "Reports directory: $REPORT_BASE_DIR"
print_info ""
print_info "To view reports:"
print_info "  ncu-ui ${REPORT_BASE_DIR}/*.ncu-rep"
print_info ""
print_info "To compare reports:"
print_info "  ncu-ui ${REPORT_BASE_DIR}/gemm.ncu-rep ${REPORT_BASE_DIR}/flashattention.ncu-rep"
print_info ""
print_info "To export metrics to CSV:"
print_info "  ncu export --type csv --file output.csv ${REPORT_BASE_DIR}/*.ncu-rep"
print_info "=========================================="

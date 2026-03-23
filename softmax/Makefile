# Makefile for Softmax Benchmark Suite

# Compiler settings
NVCC := nvcc
NVCC_FLAGS := -O3 -use_fast_math -std=c++17 -I.

# Detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/$$/0/')

# Detect CUDA version
CUDA_VERSION := $(shell $(NVCC) --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
CUDA_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d. -f1)
CUDA_MINOR := $(shell echo $(CUDA_VERSION) | cut -d. -f2)

# Architecture support by CUDA version:
# sm_100 (Blackwell) - CUDA 12.8+
# sm_90  (Hopper)    - CUDA 11.8+
# sm_89  (Ada)       - CUDA 11.8+
# sm_80  (Ampere)    - CUDA 11.0+
# sm_75  (Turing)    - CUDA 10.0+

# Set default based on CUDA version if GPU detection fails
ifeq ($(GPU_ARCH),)
    # CUDA 12.8+ supports sm_100 (Blackwell/RTX 5090)
    ifeq ($(shell [ $(CUDA_MAJOR) -gt 12 -o \( $(CUDA_MAJOR) -eq 12 -a $(CUDA_MINOR) -ge 8 \) ] && echo 1),1)
        GPU_ARCH := 100
    else
        # CUDA 12.0+ supports up to sm_90
        ifeq ($(shell [ $(CUDA_MAJOR) -ge 12 ] && echo 1),1)
            GPU_ARCH := 80
        else
            GPU_ARCH := 75
        endif
    endif
endif

ARCH_FLAG := -arch=sm_$(GPU_ARCH)

# Allow override from command line
ifdef ARCH
    ARCH_FLAG := -arch=$(ARCH)
endif

# Source files
SRC_DIR := src
SOURCES_ALL := $(SRC_DIR)/softmax_v1_naive.cu $(SRC_DIR)/softmax_v2_shared_memory.cu $(SRC_DIR)/softmax_v3_warp_reduction.cu $(SRC_DIR)/softmax_v4_vectorized.cu $(SRC_DIR)/softmax_v5_online.cu $(SRC_DIR)/softmax_v5_vec_ultimate.cu $(SRC_DIR)/benchmark_all.cu
SOURCES_SINGLE := $(SRC_DIR)/softmax_common.h $(SRC_DIR)/softmax_v1_naive.cu $(SRC_DIR)/softmax_v2_shared_memory.cu $(SRC_DIR)/softmax_v3_warp_reduction.cu $(SRC_DIR)/softmax_v4_vectorized.cu $(SRC_DIR)/softmax_v5_online.cu $(SRC_DIR)/softmax_v5_vec_ultimate.cu

# Targets
TARGET_ALL := benchmark_all
TARGET_MONO := softmax_benchmark

.PHONY: all clean run run-all run-mono profile help

# Default: build both (clean first)
all: clean $(TARGET_ALL) $(TARGET_MONO)

# Build all-in-one benchmark (clean first)
$(TARGET_ALL): $(SOURCES_ALL)
	@rm -f $@  # Clean previous build
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAG) -o $@ $(SOURCES_ALL)
	@echo "Built: $@ ($(ARCH_FLAG))"

# Build single-file version (monolithic) (clean first)
$(TARGET_MONO): softmax_benchmark.cu
	@rm -f $@  # Clean previous build
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAG) -o $@ $<
	@echo "Built: $@ ($(ARCH_FLAG))"

# Run the modular version (clean and rebuild)
run run-all: clean $(TARGET_ALL)
	./$(TARGET_ALL)

# Run the monolithic version (clean and rebuild)
run-mono: clean $(TARGET_MONO)
	./$(TARGET_MONO)

# Profile with Nsight Compute (clean and rebuild)
profile: clean $(TARGET_ALL)
	ncu -o profile_report --metrics dram__bytes.sum,gpu__time_duration.avg ./$(TARGET_ALL)

# Clean build artifacts
clean:
	@rm -f $(TARGET_ALL) $(TARGET_MONO) *.o
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "Softmax Benchmark Makefile - Optimized for RTX 5090 (Blackwell)"
	@echo ""
	@echo "Build Targets: (auto-clean before build)"
	@echo "  make all       - Clean and build both versions (default)"
	@echo "  $(TARGET_ALL)      - Clean and build modular version"
	@echo "  $(TARGET_MONO)  - Clean and build monolithic version"
	@echo ""
	@echo "Run Targets: (clean + build + run)"
	@echo "  make run       - Clean, build and run modular version"
	@echo "  make run-all   - Same as 'make run'"
	@echo "  make run-mono  - Clean, build and run monolithic version"
	@echo ""
	@echo "Other Targets:"
	@echo "  make profile   - Clean, build and profile with Nsight Compute"
	@echo "  make clean     - Remove all build artifacts"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "Build Options:"
	@echo "  ARCH=sm_XX     - Override GPU architecture (e.g., ARCH=sm_100 for RTX 5090)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Clean and build both versions"
	@echo "  make ARCH=sm_100        # Clean and build for RTX 5090"
	@echo "  make run ARCH=sm_100    # Clean, build for RTX 5090 and run"
	@echo "  make ARCH=sm_89         # Clean and build for RTX 4090"
	@echo ""
	@echo "Architecture Reference:"
	@echo "  sm_100  - Blackwell (RTX 5090, GDDR7, 1792 GB/s)"
	@echo "  sm_90a  - Blackwell/Ada (RTX 4090 Ti, backward compat)"
	@echo "  sm_89   - Ada (RTX 4090/4080)"
	@echo "  sm_80   - Ampere (A100/3090)"
	@echo "  sm_75   - Turing (RTX 2080 Ti)"
	@echo "  sm_70   - Volta (V100)"

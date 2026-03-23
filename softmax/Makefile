# Makefile for Softmax Benchmark Suite

# Compiler settings
NVCC := nvcc
NVCC_FLAGS := -O3 -use_fast_math -std=c++17 -Isrc

# Detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

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
SOURCES := $(SRC_DIR)/main.cu \
           $(SRC_DIR)/softmax/v1_naive.cu \
           $(SRC_DIR)/softmax/v2_shared_memory.cu \
           $(SRC_DIR)/softmax/v3_warp_reduction.cu \
           $(SRC_DIR)/softmax/v4_vectorized.cu \
           $(SRC_DIR)/softmax/v5_online.cu \
           $(SRC_DIR)/softmax/v5_vec_ultimate.cu \
           $(SRC_DIR)/benchmark/benchmark.cu

# Targets
TARGET := benchmark_all

.PHONY: all clean run profile help tree

# Default: build the benchmark
all: $(TARGET)

# Build the benchmark
$(TARGET): $(SOURCES)
	@rm -f $@  # Clean previous build
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAG) -o $@ $(SOURCES)
	@echo "Built: $@ ($(ARCH_FLAG))"

# Run the benchmark (clean, build and run)
run: clean $(TARGET)
	./$(TARGET)

# Profile with Nsight Compute (clean, build and profile)
profile: clean $(TARGET)
	ncu -o profile_report --metrics dram__bytes.sum,gpu__time_duration.avg ./$(TARGET)

# Show directory tree
tree:
	@echo "Source directory structure:"
	@find $(SRC_DIR) -type f -name "*.cu" -o -name "*.h" | sort

# Clean build artifacts
clean:
	@rm -f $(TARGET) *.o
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "Softmax Benchmark Makefile - Optimized for RTX 5090 (Blackwell)"
	@echo ""
	@echo "Build Targets:"
	@echo "  make all       - Build the benchmark (default)"
	@echo "  make run       - Clean, build and run the benchmark"
	@echo ""
	@echo "Other Targets:"
	@echo "  make profile   - Clean, build and profile with Nsight Compute"
	@echo "  make tree      - Show source directory structure"
	@echo "  make clean     - Remove all build artifacts"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "Build Options:"
	@echo "  ARCH=sm_XX     - Override GPU architecture (e.g., ARCH=sm_100 for RTX 5090)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build the benchmark"
	@echo "  make ARCH=sm_100        # Build for RTX 5090"
	@echo "  make run ARCH=sm_100    # Build for RTX 5090 and run"
	@echo "  make ARCH=sm_89         # Build for RTX 4090"
	@echo ""
	@echo "Architecture Reference:"
	@echo "  sm_100  - Blackwell (RTX 5090, GDDR7, 1792 GB/s)"
	@echo "  sm_90a  - Blackwell/Ada (RTX 4090 Ti, backward compat)"
	@echo "  sm_89   - Ada (RTX 4090/4080)"
	@echo "  sm_80   - Ampere (A100/3090)"
	@echo "  sm_75   - Turing (RTX 2080 Ti)"
	@echo "  sm_70   - Volta (V100)"

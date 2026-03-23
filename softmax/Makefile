# Makefile for Softmax Benchmark Suite

# Compiler settings
NVCC := nvcc
NVCC_FLAGS := -O3 -use_fast_math -std=c++17 -I.

# Detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/$$/0/')

# RTX 5090 Blackwell Architecture (sm_100) - GDDR7, 1792 GB/s bandwidth
# Default to sm_100 for RTX 5090, fallback to sm_80
ifeq ($(GPU_ARCH),)
    ARCH_FLAG := -arch=sm_100
else
    ARCH_FLAG := -arch=sm_$(GPU_ARCH)
endif

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

# Default: build both
all: $(TARGET_ALL) $(TARGET_MONO)

# Build all-in-one benchmark
$(TARGET_ALL): $(SOURCES_ALL)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAG) -o $@ $(SOURCES_ALL)
	@echo "Built: $@ ($(ARCH_FLAG))"

# Build single-file version (monolithic)
$(TARGET_MONO): softmax_benchmark.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAG) -o $@ $<
	@echo "Built: $@ ($(ARCH_FLAG))"

# Run the modular version
run run-all: $(TARGET_ALL)
	./$(TARGET_ALL)

# Run the monolithic version
run-mono: $(TARGET_MONO)
	./$(TARGET_MONO)

# Profile with Nsight Compute
profile: $(TARGET_ALL)
	ncu -o profile_report --metrics dram__bytes.sum,gpu__time_duration.avg ./$(TARGET_ALL)

# Clean build artifacts
clean:
	rm -f $(TARGET_ALL) $(TARGET_MONO) *.o

# Help
help:
	@echo "Softmax Benchmark Makefile - Optimized for RTX 5090 (Blackwell)"
	@echo ""
	@echo "Build Targets:"
	@echo "  make all       - Build both modular and monolithic versions (default)"
	@echo "  $(TARGET_ALL)      - Build modular version (separate files)"
	@echo "  $(TARGET_MONO)  - Build monolithic version (single file)"
	@echo ""
	@echo "Run Targets:"
	@echo "  make run       - Build and run modular version"
	@echo "  make run-all   - Same as 'make run'"
	@echo "  make run-mono  - Build and run monolithic version"
	@echo ""
	@echo "Other Targets:"
	@echo "  make profile   - Run with Nsight Compute profiling"
	@echo "  make clean     - Remove all build artifacts"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "Build Options:"
	@echo "  ARCH=sm_XX     - Override GPU architecture (e.g., ARCH=sm_100 for RTX 5090)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Auto-detect and build both versions"
	@echo "  make ARCH=sm_100        # Build for RTX 5090 (Blackwell sm_100)"
	@echo "  make run ARCH=sm_100    # Build for RTX 5090 and run"
	@echo "  make ARCH=sm_89         # Build for RTX 4090 (Ada)"
	@echo ""
	@echo "Architecture Reference:"
	@echo "  sm_100  - Blackwell (RTX 5090, GDDR7, 1792 GB/s)"
	@echo "  sm_90a  - Blackwell/Ada (RTX 4090 Ti, backward compat)"
	@echo "  sm_89   - Ada (RTX 4090/4080)"
	@echo "  sm_80   - Ampere (A100/3090)"
	@echo "  sm_75   - Turing (RTX 2080 Ti)"
	@echo "  sm_70   - Volta (V100)"

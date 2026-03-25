# FlashAttention CUDA Makefile
# RTX 5090 Optimized (Compute Capability 10.0 for Blackwell)

# CUDA paths
CUDA_PATH ?= /usr/local/cuda
NVCC = /usr/bin/nvcc

# RTX 5090 (Blackwell) - Compute Capability 10.0
# RTX 4090 (Ada) - Compute Capability 8.9
ARCH_FLAGS = -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90

# Compiler flags
NVCC_FLAGS = -O3 -std=c++14 $(ARCH_FLAGS) \
    -Xptxas -v \
    -Xcompiler -fopenmp \
    -lineinfo \
    -use_fast_math \
    -I$(CUDA_PATH)/include

# Linker flags
LD_FLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcurand -lgomp

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source files
SOURCES = $(wildcard $(SRC_DIR)/flashattention/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/flashattention/%.cu=$(BUILD_DIR)/%.o)

# Targets
TARGET = $(BUILD_DIR)/benchmark_flashattention

.PHONY: all clean dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)

# Main benchmark executable
$(TARGET): $(SRC_DIR)/benchmark.cu $(SRC_DIR)/common.h $(SRC_DIR)/flashattention/kernels.h $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(OBJECTS) $(LD_FLAGS)

# Compile kernel sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/flashattention/%.cu $(SRC_DIR)/flashattention/kernels.h $(SRC_DIR)/common.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

# Debug build
debug: NVCC_FLAGS = -O0 -g -G -std=c++17 $(ARCH_FLAGS) -I$(CUDA_PATH)/include
debug: all

# Profiling build (for Nsight Compute)
profile: NVCC_FLAGS = -O3 -std=c++17 $(ARCH_FLAGS) -lineinfo -Xptxas -v -I$(CUDA_PATH)/include
profile: all

# CUDA Top-K Benchmark Makefile

# CUDA Compiler
NVCC = nvcc

# Compute Capability (根据你的GPU调整)
# sm_70: V100, sm_80: A100, sm_90: H100
ARCH = -arch=sm_70

# Compiler Flags
NVCC_FLAGS = -O3 $(ARCH) -std=c++14

# Directories
SRC_DIR = ..
INC_DIR = $(SRC_DIR)/include
KERNEL_DIR = $(SRC_DIR)/src
BUILD_DIR = $(SRC_DIR)/build

# Target
TARGET = $(BUILD_DIR)/topk_benchmark

# Source
SRC = topk_benchmark.cu

# Dependencies
DEPS = $(INC_DIR)/topk_common.h \
       $(KERNEL_DIR)/topk_v1_thread_per_row.cu \
       $(KERNEL_DIR)/topk_v2_block_shared_memory.cu \
       $(KERNEL_DIR)/topk_v3_warp_shuffle.cu

.PHONY: all clean run run-quick run-llm run-compare

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(SRC_DIR) $(SRC) -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR)

# 运行所有测试
run: $(TARGET)
	$(TARGET) all

# 运行快速测试
run-quick: $(TARGET)
	$(TARGET) quick

# 运行LLM相关测试
run-llm: $(TARGET)
	$(TARGET) llm

# 运行对比测试套件
run-compare: $(TARGET)
	$(TARGET) compare

# 自定义测试 (可修改参数)
run-custom: $(TARGET)
	$(TARGET) custom 128 32000 16

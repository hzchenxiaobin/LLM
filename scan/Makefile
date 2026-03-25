# CUDA Scan Benchmark Makefile

# CUDA 编译器
NVCC = nvcc

# 编译选项
# CUDA 12.0 支持的架构：Maxwell (52), Pascal (61), Volta/Turing (70), Ampere (80), Hopper (90), Blackwell (120)
NVCC_FLAGS = -O3 -std=c++17 \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_120,code=sm_120 \
    --use_fast_math

# 包含目录
INCLUDES = -I./include

# 源文件
SOURCES = src/scan/v1_hillis_steele.cu \
          src/scan/v2_blelloch.cu \
          src/scan/v3_bank_free.cu \
          src/scan/v4_warp_primitive.cu

BENCHMARK_SRC = src/scan/benchmark.cu

# 目标文件
BUILD_DIR = build
OBJECTS = $(patsubst src/scan/%.cu,$(BUILD_DIR)/%.o,$(SOURCES))
BENCHMARK_OBJ = $(BUILD_DIR)/benchmark.o

# 可执行文件
TARGET = $(BUILD_DIR)/benchmark

# 默认目标：先清理再编译
all: clean_build $(TARGET)

# 清理并创建 build 目录
clean_build:
	@rm -rf $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)

# 创建 build 目录（仅当不存在时）
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# 编译各个版本的 scan 为 relocatable device code
$(BUILD_DIR)/%.o: src/scan/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# 编译 benchmark
$(BENCHMARK_OBJ): $(BENCHMARK_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# 链接所有目标文件（使用 device link）
$(TARGET): $(OBJECTS) $(BENCHMARK_OBJ)
	$(NVCC) $(NVCC_FLAGS) -dlink $(OBJECTS) $(BENCHMARK_OBJ) -o $(BUILD_DIR)/link.o
	$(NVCC) $(OBJECTS) $(BENCHMARK_OBJ) $(BUILD_DIR)/link.o -o $@

# 运行基准测试
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 重新构建
rebuild: clean all

# 调试构建（包含调试信息）
debug: NVCC_FLAGS += -g -G -lineinfo
debug: clean all

.PHONY: all clean clean_build run rebuild debug

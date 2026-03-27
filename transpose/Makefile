# CUDA Transpose Benchmark Makefile

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -Xptxas -v  # sm_80 for Ampere, adjust for your GPU
NVCC_FLAGS += -lineinfo  # Enable line info for profiling
NVCC_FLAGS += -std=c++14

# Include paths
INCLUDES = -I./src/include

# Source files
SRC_DIR = src
TRANSPOSE_SRCS = $(wildcard $(SRC_DIR)/transpose/*.cu)
BENCHMARK_SRCS = $(wildcard $(SRC_DIR)/benchmark/*.cu)
MAIN_SRC = $(SRC_DIR)/main.cu

# Object files
OBJ_DIR = obj
TRANSPOSE_OBJS = $(patsubst $(SRC_DIR)/transpose/%.cu,$(OBJ_DIR)/transpose_%.o,$(TRANSPOSE_SRCS))
BENCHMARK_OBJS = $(patsubst $(SRC_DIR)/benchmark/%.cu,$(OBJ_DIR)/benchmark_%.o,$(BENCHMARK_SRCS))
MAIN_OBJ = $(OBJ_DIR)/main.o

# Target executable
TARGET = transpose_benchmark

# Default target
.PHONY: all clean run profile

all: clean $(TARGET)

# Create object directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile transpose sources
$(OBJ_DIR)/transpose_%.o: $(SRC_DIR)/transpose/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile benchmark sources
$(OBJ_DIR)/benchmark_%.o: $(SRC_DIR)/benchmark/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile main
$(MAIN_OBJ): $(MAIN_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link
$(TARGET): $(TRANSPOSE_OBJS) $(BENCHMARK_OBJS) $(MAIN_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Run with default settings (all 10 test cases)
run: $(TARGET)
	./$(TARGET) -a

# Run with different matrix sizes (single test)
run-single: $(TARGET)
	./$(TARGET)

run-small: $(TARGET)
	./$(TARGET) -n 4096

run-large: $(TARGET)
	./$(TARGET) -n 16384

# Profile with Nsight Compute
profile: $(TARGET)
	ncu -o profile_report --set full ./$(TARGET) -n 8192 -b 10

# Clean
clean:
	rm -rf $(OBJ_DIR) $(TARGET) *.ncu-rep

# Print source files
print-srcs:
	@echo "Transpose sources: $(TRANSPOSE_SRCS)"
	@echo "Benchmark sources: $(BENCHMARK_SRCS)"
	@echo "Main source: $(MAIN_SRC)"

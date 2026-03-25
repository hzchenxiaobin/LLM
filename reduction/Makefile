# CUDA Reduction 算子性能测试 - Makefile

# 编译器
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_100 -Xcompiler -O3

# 路径
CUDA_PATH ?= /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64
INCLUDE_DIR = include
KERNEL_DIR = kernels
SRC_DIR = src

# 源文件
BENCHMARK_SRC = $(SRC_DIR)/benchmark.cu
KERNEL_SRCS = $(KERNEL_DIR)/common.cu \
              $(KERNEL_DIR)/reduce_v1_interleaved.cu \
              $(KERNEL_DIR)/reduce_v2_strided.cu \
              $(KERNEL_DIR)/reduce_v3_sequential.cu \
              $(KERNEL_DIR)/reduce_v4_first_add.cu \
              $(KERNEL_DIR)/reduce_v5_warp_shuffle.cu \
              $(KERNEL_DIR)/reduce_v6_vectorized.cu \
              $(KERNEL_DIR)/reduce_cub.cu \
              $(KERNEL_DIR)/run_kernel.cu

# 目标
BENCHMARK_BIN = benchmark
LIBREDUCTION = libreduction.so

# 查找 CUB
CUB_INCLUDE :=
CUB_PATHS = $(CUDA_INCLUDE)/cub /usr/include/cub $(HOME)/cub $(HOME)/cuda/cub
$(foreach path,$(CUB_PATHS),$(if $(wildcard $(path)),$(eval CUB_INCLUDE := -I$(dir $(path))),))

.PHONY: all clean test run compare help

# 默认目标：清理并编译所有
all:
	@$(MAKE) clean
	@$(MAKE) $(BENCHMARK_BIN)

# 编译基准测试可执行文件
$(BENCHMARK_BIN): $(BENCHMARK_SRC) $(KERNEL_SRCS)
	@echo "编译: $@"
	$(NVCC) $(NVCC_FLAGS) $(CUB_INCLUDE) -I$(INCLUDE_DIR) -o $@ $(BENCHMARK_SRC) $(KERNEL_SRCS) -I$(CUDA_INCLUDE) -L$(CUDA_LIB) -lcudart
	@echo "完成!"

# 编译共享库 (用于 Python 绑定)
$(LIBREDUCTION): $(KERNEL_SRCS)
	@echo "编译共享库: $@"
	$(NVCC) $(NVCC_FLAGS) -shared -Xcompiler -fPIC $(CUB_INCLUDE) -I$(INCLUDE_DIR) -o $@ $(KERNEL_SRCS) -I$(CUDA_INCLUDE) -L$(CUDA_LIB) -lcudart
	@echo "完成!"

# 清理
clean:
	@rm -f $(BENCHMARK_BIN) $(LIBREDUCTION)
	@rm -f *.o *.png
	@rm -rf __pycache__ .pytest_cache
	@find . -name "*.o" -delete
	@echo "清理完成"

# 快速测试
test: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --quick --sizes 1M --versions all

# 完整测试 (多种规模)
run: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 1M 10M 100M --versions all

# 对比测试 (v1 vs v6)
compare: $(BENCHMARK_BIN)
	@echo "对比: v1 (朴素) vs v6 (向量化)"
	@./$(BENCHMARK_BIN) --sizes 10M 100M --versions 1 6

# 测试单个版本
test-v1: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 10M --versions 1

test-v2: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 10M --versions 2

test-v3: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 10M --versions 3

test-v4: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 10M --versions 4

test-v5: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 10M --versions 5

test-v6: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 10M --versions 6

test-cub: $(BENCHMARK_BIN)
	@./$(BENCHMARK_BIN) --sizes 10M --versions 7

# 帮助
help:
	@echo "CUDA Reduction 性能测试"
	@echo ""
	@echo "目录结构:"
	@echo "  include/   - 头文件 (common.h, reduction_kernels.h)"
	@echo "  kernels/   - CUDA kernel 实现"
	@echo "  src/       - 主程序 (benchmark.cu)"
	@echo "  python/    - Python 脚本"
	@echo "  scripts/   - Shell 脚本"
	@echo ""
	@echo "目标:"
	@echo "  make              - 编译基准测试程序"
	@echo "  make clean        - 清理编译产物"
	@echo "  make test         - 快速测试 (1M 元素，所有版本)"
	@echo "  make run          - 完整测试 (1M, 10M, 100M)"
	@echo "  make compare      - 对比 v1 和 v6"
	@echo "  make test-vN      - 测试版本 N (N=1-7)"
	@echo "  make help         - 显示此帮助"
	@echo ""
	@echo "自定义测试:"
	@echo "  ./benchmark --sizes 10M 100M --versions 3 4 5 --quick"
	@echo "  ./benchmark -s 1G -v all -w 5 -i 50"
	@echo ""
	@echo "版本说明:"
	@echo "  1: interleaved     - 朴素版本 (Warp Divergence)"
	@echo "  2: strided         - 解决分支发散 (Bank Conflict)"
	@echo "  3: sequential      - 解决 Bank Conflict"
	@echo "  4: first_add       - 加载时相加"
	@echo "  5: warp_shuffle    - Warp Shuffle"
	@echo "  6: vectorized      - 向量化访存"
	@echo "  7: cub             - NVIDIA CUB 库 (基准)"

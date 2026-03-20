# 编译器设置
NVCC = nvcc

# 编译选项: -O3 开启最高级别优化，-arch=native 让 nvcc 自动适配你的 RTX 5090 架构
NVCC_FLAGS = -O3 -arch=native -std=c++14

# 链接库: 必须链接 cuBLAS
LIBS = -lcublas

# 编译输出目录
BUILD_DIR = build

# 目标执行文件 (放在 build/ 目录下)
TARGET = $(BUILD_DIR)/benchmark

# 所有的源文件和头文件 (在 src/ 目录下)
SRCS = src/benchmark_bmm.cu src/kernel_naive.cu src/kernel_shared_memory.cu
HEADERS = src/kernels.cuh

# 默认编译规则
all: $(BUILD_DIR) $(TARGET)

# 创建 build 目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译目标
$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# 清理生成的文件
clean:
	rm -rf $(BUILD_DIR)

# 运行程序
run: all
	./$(TARGET)

# 清理后重新编译
rebuild: clean all

# 编译器设置
NVCC = nvcc

# 编译选项: -O3 开启最高级别优化，-arch=native 让 nvcc 自动适配你的 RTX 5090 架构
NVCC_FLAGS = -O3 -arch=native -std=c++14

# 链接库: 必须链接 cuBLAS
LIBS = -lcublas

# 目标执行文件
TARGET = benchmark

# 所有的源文件和头文件
SRCS = benchmark_bmm.cu kernel_naive.cu kernel_shared_memory.cu
HEADERS = kernels.cuh

# 默认编译规则
all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# 清理生成的文件
clean:
	rm -f $(TARGET)
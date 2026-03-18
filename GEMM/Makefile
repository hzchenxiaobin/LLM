# 编译器设置
NVCC = nvcc
# 编译参数：开启最高级别优化
NVCC_FLAGS = -O3 
# 链接器参数：链接 cuBLAS 库
LIBS = -lcublas

# 源代码目录
SRC_DIR = src

# 目标可执行文件名称
TARGET = benchmark_gemm

# 需要编译的源文件对应的目标对象文件
OBJS = $(SRC_DIR)/main.o $(SRC_DIR)/sgemm_cublas.o $(SRC_DIR)/sgemm_naive.o $(SRC_DIR)/sgemm_shared.o $(SRC_DIR)/sgemm_register.o $(SRC_DIR)/sgemm_register_v2.o

# 默认构建目标
all: $(TARGET)

# 链接生成可执行文件
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET) $(LIBS)

# 编译主函数文件
$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/main.cu -o $(SRC_DIR)/main.o

# 编译 cuBLAS GEMM 文件
$(SRC_DIR)/sgemm_cublas.o: $(SRC_DIR)/sgemm_cublas.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_cublas.cu -o $(SRC_DIR)/sgemm_cublas.o

# 编译朴素 GEMM 文件
$(SRC_DIR)/sgemm_naive.o: $(SRC_DIR)/sgemm_naive.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_naive.cu -o $(SRC_DIR)/sgemm_naive.o

# 编译共享内存 GEMM 文件
$(SRC_DIR)/sgemm_shared.o: $(SRC_DIR)/sgemm_shared.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_shared.cu -o $(SRC_DIR)/sgemm_shared.o

# 编译寄存器分块 GEMM 文件
$(SRC_DIR)/sgemm_register.o: $(SRC_DIR)/sgemm_register.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_register.cu -o $(SRC_DIR)/sgemm_register.o

# 编译优化版寄存器分块 GEMM 文件
$(SRC_DIR)/sgemm_register_v2.o: $(SRC_DIR)/sgemm_register_v2.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_register_v2.cu -o $(SRC_DIR)/sgemm_register_v2.o

# 清理编译产物
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

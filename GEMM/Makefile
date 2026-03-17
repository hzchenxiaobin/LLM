# 编译器设置
NVCC = nvcc
# 编译参数：开启最高级别优化
NVCC_FLAGS = -O3 
# 链接器参数：链接 cuBLAS 库
LIBS = -lcublas

# 目标可执行文件名称
TARGET = benchmark_gemm

# 需要编译的源文件对应的目标对象文件
OBJS = main.o sgemm_cublas.o sgemm_naive.o sgemm_shared.o sgemm_register.o

# 默认构建目标
all: $(TARGET)

# 链接生成可执行文件
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET) $(LIBS)

# 编译主函数文件
main.o: main.cu common.h gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu

# 编译 cuBLAS GEMM 文件
sgemm_cublas.o: sgemm_cublas.cu common.h gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c sgemm_cublas.cu

# 编译朴素 GEMM 文件
sgemm_naive.o: sgemm_naive.cu common.h gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c sgemm_naive.cu

# 编译共享内存 GEMM 文件
sgemm_shared.o: sgemm_shared.cu common.h gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c sgemm_shared.cu

# 编译寄存器分块 GEMM 文件
sgemm_register.o: sgemm_register.cu common.h gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c sgemm_register.cu

# 清理编译产物
clean:
	rm -f *.o $(TARGET)
# 编译器设置
NVCC = nvcc
# 编译参数：开启最高级别优化，启用 WMMA Tensor Core (需要 sm_70+)
NVCC_FLAGS = -O3 -std=c++14 -arch=sm_70
# 链接器参数：链接 cuBLAS 库
LIBS = -lcublas

# 源代码目录
SRC_DIR = src

# 目标可执行文件名称
TARGET = benchmark_gemm

# CUTLASS：默认 third_party/cutlass，可通过环境变量 CUTLASS_DIR 覆盖
CUTLASS_DIR ?= $(abspath $(CURDIR)/third_party/cutlass)
CUTLASS_HEADER := $(CUTLASS_DIR)/include/cutlass/gemm/device/gemm.h

ifeq ($(wildcard $(CUTLASS_HEADER)),)
  GEMM_HAVE_CUTLASS := 0
  CUTLASS_SRC := $(SRC_DIR)/sgemm_cutlass_disabled.cu
  CUTE_SRC := $(SRC_DIR)/sgemm_cute_disabled.cu
  CUTLASS_NVCC_FLAGS := $(NVCC_FLAGS)
  CUTE_NVCC_FLAGS := $(NVCC_FLAGS)
  $(info [GEMM] CUTLASS 未找到: $(CUTLASS_HEADER))
  $(info       克隆: git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass)
  $(info       说明见 docs/cutlass_build.md 和 docs/cute_build.md)
else
  GEMM_HAVE_CUTLASS := 1
  CUTLASS_SRC := $(SRC_DIR)/sgemm_cutlass.cu
  CUTE_SRC := $(SRC_DIR)/sgemm_cute.cu
  # CUTLASS/CuTe 需 C++17；避免与 NVCC_FLAGS 中 -std=c++14 重复
  CUTLASS_NVCC_FLAGS := $(filter-out -std=c++14,$(NVCC_FLAGS)) -std=c++17 -I$(CUTLASS_DIR)/include --expt-relaxed-constexpr
  CUTE_NVCC_FLAGS := $(filter-out -std=c++14,$(NVCC_FLAGS)) -std=c++17 -I$(CUTLASS_DIR)/include --expt-relaxed-constexpr
endif

# main.cu 需与是否启用 CUTLASS 一致
MAIN_CPPFLAGS := -DGEMM_HAVE_CUTLASS=$(GEMM_HAVE_CUTLASS)

# 需要编译的源文件对应的目标对象文件
OBJS = $(SRC_DIR)/main.o $(SRC_DIR)/sgemm_cublas.o $(SRC_DIR)/sgemm_naive.o $(SRC_DIR)/sgemm_shared.o $(SRC_DIR)/sgemm_register.o $(SRC_DIR)/sgemm_register_v2.o $(SRC_DIR)/sgemm_register_v3.o $(SRC_DIR)/sgemm_register_bank_conflict.o $(SRC_DIR)/sgemm_wmma.o $(SRC_DIR)/sgemm_wmma_v2.o $(SRC_DIR)/sgemm_cutlass.o $(SRC_DIR)/sgemm_cute.o

# 默认构建目标: 先清理，再编译
all: clean $(TARGET)

# 链接生成可执行文件
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET) $(LIBS)

# 编译主函数文件
$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(MAIN_CPPFLAGS) -c $(SRC_DIR)/main.cu -o $(SRC_DIR)/main.o

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

# 编译双缓冲优化版寄存器分块 GEMM 文件
$(SRC_DIR)/sgemm_register_v3.o: $(SRC_DIR)/sgemm_register_v3.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_register_v3.cu -o $(SRC_DIR)/sgemm_register_v3.o

# 编译Bank Conflict优化版寄存器分块 GEMM 文件
$(SRC_DIR)/sgemm_register_bank_conflict.o: $(SRC_DIR)/sgemm_register_bank_conflict.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_register_bank_conflict.cu -o $(SRC_DIR)/sgemm_register_bank_conflict.o

# 编译 Tensor Core WMMA GEMM 文件
$(SRC_DIR)/sgemm_wmma.o: $(SRC_DIR)/sgemm_wmma.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_wmma.cu -o $(SRC_DIR)/sgemm_wmma.o

# 编译 Tensor Core WMMA v2 GEMM 文件
$(SRC_DIR)/sgemm_wmma_v2.o: $(SRC_DIR)/sgemm_wmma_v2.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/sgemm_wmma_v2.cu -o $(SRC_DIR)/sgemm_wmma_v2.o

# CUTLASS SGEMM（或占位实现）
$(SRC_DIR)/sgemm_cutlass.o: $(CUTLASS_SRC) $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(CUTLASS_NVCC_FLAGS) -c $(CUTLASS_SRC) -o $(SRC_DIR)/sgemm_cutlass.o

# CuTe SGEMM（或占位实现）
$(SRC_DIR)/sgemm_cute.o: $(CUTE_SRC) $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(CUTE_NVCC_FLAGS) -c $(CUTE_SRC) -o $(SRC_DIR)/sgemm_cute.o

# 清理编译产物
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

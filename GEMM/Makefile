# 编译器设置
NVCC = nvcc
# 编译参数：开启最高级别优化，启用 WMMA Tensor Core (需要 sm_70+)
NVCC_FLAGS = -O3 -std=c++14 -arch=sm_70
# 链接器参数：链接 cuBLAS 库
LIBS = -lcublas

# 源代码目录
SRC_DIR = src
GEMM_DIR = $(SRC_DIR)/gemm

# 目标可执行文件名称
TARGET = benchmark_gemm

# CUTLASS：默认 third_party/cutlass，可通过环境变量 CUTLASS_DIR 覆盖
CUTLASS_DIR ?= $(abspath $(CURDIR)/third_party/cutlass)
CUTLASS_HEADER := $(CUTLASS_DIR)/include/cutlass/gemm/device/gemm.h

ifeq ($(wildcard $(CUTLASS_HEADER)),)
  GEMM_HAVE_CUTLASS := 0
  $(info [GEMM] CUTLASS 未找到: $(CUTLASS_HEADER))
  $(info       克隆: git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass)
  $(info       说明见 docs/cutlass_build.md 和 docs/cute_build.md)
else
  GEMM_HAVE_CUTLASS := 1
  $(info [GEMM] CUTLASS 已找到: $(CUTLASS_HEADER))
endif

# main.cu 需与是否启用 CUTLASS 一致
MAIN_CPPFLAGS := -DGEMM_HAVE_CUTLASS=$(GEMM_HAVE_CUTLASS)

# 基础 GEMM 目标对象文件
BASE_GEMM_OBJS = $(GEMM_DIR)/sgemm_cublas.o $(GEMM_DIR)/sgemm_naive.o $(GEMM_DIR)/sgemm_shared.o \
                 $(GEMM_DIR)/sgemm_register.o $(GEMM_DIR)/sgemm_register_v2.o $(GEMM_DIR)/sgemm_register_v3.o \
                 $(GEMM_DIR)/sgemm_register_bank_conflict.o $(GEMM_DIR)/sgemm_wmma.o $(GEMM_DIR)/sgemm_wmma_v2.o

# 条件添加 CUTLASS/CuTe 对象文件
ifeq ($(GEMM_HAVE_CUTLASS),1)
  GEMM_OBJS = $(BASE_GEMM_OBJS) $(GEMM_DIR)/sgemm_cutlass.o $(GEMM_DIR)/sgemm_cute.o
  # CUTLASS/CuTe 需 C++17；避免与 NVCC_FLAGS 中 -std=c++14 重复
  CUTLASS_NVCC_FLAGS := $(filter-out -std=c++14,$(NVCC_FLAGS)) -std=c++17 -I$(CUTLASS_DIR)/include --expt-relaxed-constexpr
else
  GEMM_OBJS = $(BASE_GEMM_OBJS)
endif

# 主程序对象文件
MAIN_OBJ = $(SRC_DIR)/main.o

# 所有对象文件
ALL_OBJS = $(MAIN_OBJ) $(GEMM_OBJS)

# 默认构建目标: 先清理，再编译单矩阵 GEMM
all: clean $(TARGET)

# 链接生成单矩阵 GEMM 可执行文件
$(TARGET): $(MAIN_OBJ) $(GEMM_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(MAIN_OBJ) $(GEMM_OBJS) -o $(TARGET) $(LIBS)

# ==========================================
# 编译主函数文件
# ==========================================
$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(MAIN_CPPFLAGS) -c $(SRC_DIR)/main.cu -o $(SRC_DIR)/main.o

# ==========================================
# 编译 GEMM 算子文件 (单矩阵)
# ==========================================

# 编译 cuBLAS GEMM 文件
$(GEMM_DIR)/sgemm_cublas.o: $(GEMM_DIR)/sgemm_cublas.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_cublas.cu -o $(GEMM_DIR)/sgemm_cublas.o

# 编译朴素 GEMM 文件
$(GEMM_DIR)/sgemm_naive.o: $(GEMM_DIR)/sgemm_naive.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_naive.cu -o $(GEMM_DIR)/sgemm_naive.o

# 编译共享内存 GEMM 文件
$(GEMM_DIR)/sgemm_shared.o: $(GEMM_DIR)/sgemm_shared.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_shared.cu -o $(GEMM_DIR)/sgemm_shared.o

# 编译寄存器分块 GEMM 文件
$(GEMM_DIR)/sgemm_register.o: $(GEMM_DIR)/sgemm_register.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_register.cu -o $(GEMM_DIR)/sgemm_register.o

# 编译优化版寄存器分块 GEMM 文件
$(GEMM_DIR)/sgemm_register_v2.o: $(GEMM_DIR)/sgemm_register_v2.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_register_v2.cu -o $(GEMM_DIR)/sgemm_register_v2.o

# 编译双缓冲优化版寄存器分块 GEMM 文件
$(GEMM_DIR)/sgemm_register_v3.o: $(GEMM_DIR)/sgemm_register_v3.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_register_v3.cu -o $(GEMM_DIR)/sgemm_register_v3.o

# 编译Bank Conflict优化版寄存器分块 GEMM 文件
$(GEMM_DIR)/sgemm_register_bank_conflict.o: $(GEMM_DIR)/sgemm_register_bank_conflict.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_register_bank_conflict.cu -o $(GEMM_DIR)/sgemm_register_bank_conflict.o

# 编译 Tensor Core WMMA GEMM 文件
$(GEMM_DIR)/sgemm_wmma.o: $(GEMM_DIR)/sgemm_wmma.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_wmma.cu -o $(GEMM_DIR)/sgemm_wmma.o

# 编译 Tensor Core WMMA v2 GEMM 文件
$(GEMM_DIR)/sgemm_wmma_v2.o: $(GEMM_DIR)/sgemm_wmma_v2.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_wmma_v2.cu -o $(GEMM_DIR)/sgemm_wmma_v2.o

# CUTLASS SGEMM（仅在 CUTLASS 存在时编译）
ifeq ($(GEMM_HAVE_CUTLASS),1)
$(GEMM_DIR)/sgemm_cutlass.o: $(GEMM_DIR)/sgemm_cutlass.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(CUTLASS_NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_cutlass.cu -o $(GEMM_DIR)/sgemm_cutlass.o

# CuTe SGEMM（仅在 CUTLASS 存在时编译）
$(GEMM_DIR)/sgemm_cute.o: $(GEMM_DIR)/sgemm_cute.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h
	$(NVCC) $(CUTLASS_NVCC_FLAGS) -c $(GEMM_DIR)/sgemm_cute.cu -o $(GEMM_DIR)/sgemm_cute.o
endif

# 清理编译产物
clean:
	rm -f $(SRC_DIR)/*.o $(GEMM_DIR)/*.o $(TARGET)

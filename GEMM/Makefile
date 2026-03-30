# 编译器设置
NVCC = nvcc
# 编译参数：开启最高级别优化
NVCC_FLAGS = -O3 -std=c++14 -arch=sm_70
# 链接器参数：链接 cuBLAS 库
LIBS = -lcublas

# 源代码目录
SRC_DIR = src
GEMM_DIR = $(SRC_DIR)/gemm

# 编译产物目录
BUILD_DIR = build
BUILD_GEMM_DIR = $(BUILD_DIR)/gemm

# 目标可执行文件名称
TARGET = $(BUILD_DIR)/benchmark_gemm

# GEMM 算子对象文件
GEMM_OBJS = $(BUILD_GEMM_DIR)/v0_cublas.o $(BUILD_GEMM_DIR)/v1_naive.o $(BUILD_GEMM_DIR)/v2_shared.o \
            $(BUILD_GEMM_DIR)/v3_register.o $(BUILD_GEMM_DIR)/v4_vectorized.o \
            $(BUILD_GEMM_DIR)/v5_bank_conflict.o $(BUILD_GEMM_DIR)/v6_vec_bank.o

# 主程序对象文件
MAIN_OBJ = $(BUILD_DIR)/main.o

# 所有对象文件
ALL_OBJS = $(MAIN_OBJ) $(GEMM_OBJS)

# 默认构建目标: 先清理，再编译
all: clean $(TARGET)

# 创建编译目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_GEMM_DIR): $(BUILD_DIR)
	mkdir -p $(BUILD_GEMM_DIR)

# 链接生成可执行文件
$(TARGET): $(MAIN_OBJ) $(GEMM_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(MAIN_OBJ) $(GEMM_OBJS) -o $(TARGET) $(LIBS)

# ==========================================
# 编译主函数文件
# ==========================================
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu $(SRC_DIR)/common.h $(SRC_DIR)/gemm_kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/main.cu -o $(BUILD_DIR)/main.o

# ==========================================
# 编译 GEMM 算子文件
# ==========================================

# 编译 cuBLAS GEMM 文件 (v0)
$(BUILD_GEMM_DIR)/v0_cublas.o: $(GEMM_DIR)/v0_cublas.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h | $(BUILD_GEMM_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/v0_cublas.cu -o $(BUILD_GEMM_DIR)/v0_cublas.o

# 编译朴素 GEMM 文件 (v1)
$(BUILD_GEMM_DIR)/v1_naive.o: $(GEMM_DIR)/v1_naive.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h | $(BUILD_GEMM_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/v1_naive.cu -o $(BUILD_GEMM_DIR)/v1_naive.o

# 编译共享内存 GEMM 文件 (v2)
$(BUILD_GEMM_DIR)/v2_shared.o: $(GEMM_DIR)/v2_shared.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h | $(BUILD_GEMM_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/v2_shared.cu -o $(BUILD_GEMM_DIR)/v2_shared.o

# 编译寄存器分块 GEMM 文件 (v3)
$(BUILD_GEMM_DIR)/v3_register.o: $(GEMM_DIR)/v3_register.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h | $(BUILD_GEMM_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/v3_register.cu -o $(BUILD_GEMM_DIR)/v3_register.o

# 编译向量化访存优化版 GEMM 文件 (v4)
$(BUILD_GEMM_DIR)/v4_vectorized.o: $(GEMM_DIR)/v4_vectorized.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h | $(BUILD_GEMM_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/v4_vectorized.cu -o $(BUILD_GEMM_DIR)/v4_vectorized.o

# 编译 Bank Conflict 优化版 GEMM 文件 (v5)
$(BUILD_GEMM_DIR)/v5_bank_conflict.o: $(GEMM_DIR)/v5_bank_conflict.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h | $(BUILD_GEMM_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/v5_bank_conflict.cu -o $(BUILD_GEMM_DIR)/v5_bank_conflict.o

# 编译向量化+Padding 优化版 GEMM 文件 (v6)
$(BUILD_GEMM_DIR)/v6_vec_bank.o: $(GEMM_DIR)/v6_vec_bank.cu $(SRC_DIR)/common.h $(GEMM_DIR)/gemm_kernels.h | $(BUILD_GEMM_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(GEMM_DIR)/v6_vec_bank.cu -o $(BUILD_GEMM_DIR)/v6_vec_bank.o

# 清理编译产物
clean:
	rm -rf $(BUILD_DIR)

# 构建并运行
run: $(TARGET)
	./$(TARGET)

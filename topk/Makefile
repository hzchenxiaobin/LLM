# Top-K Benchmark Makefile - Optimized for RTX 5090 (Blackwell)

NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17 -Isrc --extended-lambda -rdc=true

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

ifeq ($(GPU_ARCH),)
    GPU_ARCH := 80
endif

ARCH_FLAG := -arch=sm_$(GPU_ARCH)

ifdef ARCH
    ARCH_FLAG := -arch=$(ARCH)
endif

SRC_DIR := src
# topk_host.cu includes v1-v5 kernel implementations
SOURCES := $(SRC_DIR)/main.cu \
           $(SRC_DIR)/topk/topk_host.cu \
           $(SRC_DIR)/benchmark/benchmark.cu

TARGET := topk_benchmark

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SOURCES)
	@rm -f $@
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAG) -o $@ $(SOURCES)
	@echo "Build complete: $@ ($(ARCH_FLAG))"

run: clean $(TARGET)
	./$(TARGET)

clean:
	@rm -f $(TARGET) *.o
	@echo "Cleaned build artifacts"

help:
	@echo "Top-K Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all    - Build benchmark"
	@echo "  make run    - Clean, build and run"
	@echo "  make clean  - Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  ARCH=sm_XX  - Override GPU architecture"

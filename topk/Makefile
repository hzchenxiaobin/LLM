# Top-K CUDA Benchmark - Top-level Makefile
# 用法: make [target] (在 topk2/ 目录下执行)

.PHONY: all clean run run-quick run-llm run-compare build

all: build

build:
	$(MAKE) -C benchmark all

clean:
	$(MAKE) -C benchmark clean

run: build
	$(MAKE) -C benchmark run

run-quick: build
	$(MAKE) -C benchmark run-quick

run-llm: build
	$(MAKE) -C benchmark run-llm

run-compare: build
	$(MAKE) -C benchmark run-compare

run-custom: build
	$(MAKE) -C benchmark run-custom

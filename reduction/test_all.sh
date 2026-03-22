#!/bin/bash
# CUDA Reduction 测试脚本 - 运行所有测试

set -e  # 出错时退出

echo "=========================================="
echo "CUDA Reduction 性能测试框架"
echo "=========================================="
echo ""

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: 找不到 nvcc，请确保 CUDA 已安装"
    exit 1
fi

echo "CUDA 版本:"
nvcc --version | grep "release"
echo ""

# 编译
if [ ! -f "./benchmark" ]; then
    echo "编译 benchmark..."
    make clean 2>/dev/null || true
    make
    echo "编译完成!"
    echo ""
fi

# 显示 GPU 信息
echo "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi 不可用"
echo ""

# 快速测试所有版本
echo "=========================================="
echo "快速测试 (1M 元素，所有版本)"
echo "=========================================="
./benchmark --quick --sizes 1M --versions all

echo ""
echo "=========================================="
echo "完整测试 (1M, 10M, 100M)"
echo "=========================================="
./benchmark --sizes 1M 10M 100M --versions all

echo ""
echo "=========================================="
echo "对比测试 (v1 vs v6)"
echo "=========================================="
./benchmark --sizes 10M 100M --versions 1 6

echo ""
echo "=========================================="
echo "所有测试完成!"
echo "=========================================="

cmake_minimum_required(VERSION 3.18)
project(cuda_scan LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA 包
find_package(CUDA REQUIRED)

# 设置 CUDA 架构（根据你的 GPU 调整）
# RTX 5090 (Blackwell) 使用 compute_100 或更高
# RTX 4090/3090 (Ampere/Ada) 使用 compute_80/89
# 这里设置支持从 Maxwell 到 Blackwell 的多种架构
set(CMAKE_CUDA_ARCHITECTURES 70 80 89 100)

# CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 源文件
set(SCAN_SOURCES
    src/scan/v1_hillis_steele.cu
    src/scan/v2_blelloch.cu
    src/scan/v3_bank_free.cu
    src/scan/v4_warp_primitive.cu
)

# 创建静态库
add_library(scan_lib STATIC ${SCAN_SOURCES})
set_target_properties(scan_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 创建基准测试可执行文件
add_executable(benchmark src/scan/benchmark.cu)
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(benchmark scan_lib)

# 启用详细输出（可选）
# set(CMAKE_VERBOSE_MAKEFILE ON)

# 安装目标
install(TARGETS benchmark DESTINATION bin)
install(TARGETS scan_lib DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

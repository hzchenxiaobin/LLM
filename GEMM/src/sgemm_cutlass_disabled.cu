// 当未检测到 CUTLASS 头文件时由 Makefile 编译此文件，仅用于满足链接符号。
// 若误调用会 abort（正常流程下 main 在 GEMM_HAVE_CUTLASS=0 时不会调用）。

#include "common.h"
#include "gemm_kernels.h"

#include <cstdlib>
#include <iostream>

void run_sgemm_cutlass(int /*M*/, int /*N*/, int /*K*/, float /*alpha*/, const float * /*A*/,
                       const float * /*B*/, float /*beta*/, float * /*C*/) {
    std::cerr << "[错误] 未编译 CUTLASS 版 GEMM。请按 GEMM/docs/cutlass_build.md 克隆 CUTLASS 后重新 make。"
              << std::endl;
    std::abort();
}

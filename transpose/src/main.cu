#include "transpose_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 外部函数声明
void run_full_benchmark(int M, int N, int warmup_runs, int benchmark_runs);

// 打印使用说明
void print_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("\n");
    printf("Options:\n");
    printf("  -m ROWS       Matrix rows (M), default: 8192\n");
    printf("  -n COLS       Matrix columns (N), default: 8192\n");
    printf("  -w WARMUP     Warmup iterations, default: 10\n");
    printf("  -b BENCH      Benchmark iterations, default: 100\n");
    printf("  -h            Show this help\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                  Run with default settings (M=8192, N=8192)\n", program);
    printf("  %s -m 4096 -n 8192  Run with M=4096, N=8192 (non-square matrix)\n", program);
    printf("  %s -m 16384 -n 4096 -b 50  Run with M=16384, N=4096, 50 benchmark iterations\n", program);
}

int main(int argc, char **argv) {
    // 默认参数
    int M = 8192;
    int N = 8192;
    int warmup_runs = 10;
    int benchmark_runs = 100;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup_runs = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            benchmark_runs = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // 验证参数
    if (M <= 0 || M > 65536) {
        printf("Error: M (rows) must be between 1 and 65536\n");
        return 1;
    }

    if (N <= 0 || N > 65536) {
        printf("Error: N (cols) must be between 1 and 65536\n");
        return 1;
    }

    if (warmup_runs < 0 || benchmark_runs <= 0) {
        printf("Error: Invalid warmup or benchmark iterations\n");
        return 1;
    }

    // 打印 GPU 信息
    print_gpu_info();

    // 运行基准测试
    run_full_benchmark(M, N, warmup_runs, benchmark_runs);

    return 0;
}

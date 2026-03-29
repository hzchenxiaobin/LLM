#!/usr/bin/env python3
"""
Build CUDA benchmarks under LLM/ and run Nsight Compute for each project's kernels.

Outputs: LLM/ncu_results/<project>/<kernel>.ncu-rep
Builds:  LLM/build_ncu/<project>/<exe>

Usage:
    python run_ncu_all.py              # build + profile all
    python run_ncu_all.py --no-build   # profile only (exes must exist)
    python run_ncu_all.py --only softmax transpose
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build_ncu"
RESULTS = ROOT / "ncu_results"

NVCC_ARCH = ["-arch=native"]
# CUDA 12.x host compiler check may reject newer MSVC (e.g. VS 2026).
NVCC_HOST = ["-allow-unsupported-compiler", "-Xcompiler", "/utf-8"]
SUBPROC_KW = dict(encoding="utf-8", errors="replace")


def find_executable(name: str, hints: Optional[Sequence[str]] = None) -> Optional[str]:
    try:
        r = subprocess.run(
            ["where", name],
            capture_output=True,
            text=True,
            check=True,
            **SUBPROC_KW,
        )
        return r.stdout.strip().split("\n")[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if hints:
        for h in hints:
            if os.path.isfile(h):
                return h
    return None


def find_nvcc() -> Optional[str]:
    hints = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe",
    ]
    return find_executable("nvcc", hints)


def find_ncu() -> Optional[str]:
    hints = [
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.exe",
    ]
    for p in hints:
        if os.path.isfile(p):
            return p
    base = Path(r"C:\Program Files\NVIDIA Corporation")
    if base.is_dir():
        for install in sorted(base.glob("Nsight Compute *"), reverse=True):
            cand = install / "target" / "windows-desktop-win7-x64" / "ncu.exe"
            if cand.is_file():
                return str(cand)
            leg = install / "ncu.exe"
            if leg.is_file():
                return str(leg)
    return find_executable("ncu", None)


def cuda_bin_for_env(nvcc_path: str) -> str:
    return str(Path(nvcc_path).parent)


def find_vcvars64() -> Optional[str]:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def run_with_msvc_env(cmd: List[str], cwd: Optional[Path]) -> subprocess.CompletedProcess:
    """On Windows, run via vcvars64.bat so nvcc can invoke cl.exe."""
    if sys.platform == "win32":
        vc = find_vcvars64()
        if vc:
            body = 'call "{}"\nset CL=/utf-8\n{}\n'.format(
                vc.replace("/", "\\"),
                subprocess.list2cmdline(cmd),
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".bat",
                delete=False,
                encoding="utf-8",
            ) as bf:
                bf.write(body)
                bat = bf.name
            try:
                return subprocess.run(
                    ["cmd", "/c", bat],
                    cwd=cwd,
                    capture_output=True,
                    **SUBPROC_KW,
                )
            finally:
                try:
                    os.unlink(bat)
                except OSError:
                    pass
    return subprocess.run(cmd, cwd=cwd, capture_output=True, **SUBPROC_KW)


def child_env(nvcc: str) -> dict:
    env = os.environ.copy()
    bin_dir = cuda_bin_for_env(nvcc)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def nvcc_link(
    nvcc: str,
    cwd: Path,
    sources: List[Path],
    out_exe: Path,
    includes: List[str],
    extra_flags: List[str],
    libs: List[str],
) -> bool:
    out_exe.parent.mkdir(parents=True, exist_ok=True)
    inc_flags = []
    for i in includes:
        inc_flags.append("-I" + str(i))
    cmd = (
        [nvcc]
        + NVCC_HOST
        + ["-O3", "-lineinfo"]
        + NVCC_ARCH
        + extra_flags
        + inc_flags
        + [str(s) for s in sources]
        + ["-o", str(out_exe)]
        + libs
    )
    print(f"[BUILD] {out_exe.name} <- {cwd}")
    r = run_with_msvc_env(cmd, cwd)
    if r.returncode != 0:
        print(r.stderr or r.stdout or "(no output)")
        return False
    return True


def _kernel_report_basename(ncu_filter: str) -> str:
    if ncu_filter.startswith("regex:"):
        return ncu_filter[len("regex:") :].strip("^$").replace("\\", "_")[:120] or "kernel"
    return ncu_filter


def profile_kernels(
    ncu: str,
    nvcc: str,
    exe: Path,
    kernels: Sequence[str],
    out_subdir: Path,
    app_args: Optional[Sequence[str]] = None,
) -> Tuple[int, int]:
    out_subdir.mkdir(parents=True, exist_ok=True)
    env = child_env(nvcc)
    ok = 0
    app_args = list(app_args or [])
    for k in kernels:
        base = _kernel_report_basename(k)
        outp = out_subdir / f"{base}.ncu-rep"
        cmd = [
            ncu,
            "-f",
            "--check-exit-code",
            "0",
            "--kernel-name",
            k,
            "--launch-count",
            "1",
            "-o",
            str(outp),
            str(exe),
        ] + app_args
        print(f"[NCU] {out_subdir.name}/{base}.ncu-rep")
        r = subprocess.run(cmd, env=env, capture_output=True, **SUBPROC_KW)
        if outp.is_file() and outp.stat().st_size > 0:
            ok += 1
        else:
            print((r.stderr or r.stdout or "")[:800])
    return ok, len(kernels)


# --- Per-project build + kernel lists ---

def build_gemm(nvcc: str) -> Optional[Path]:
    g = ROOT / "GEMM"
    b = BUILD / "gemm"
    b.mkdir(parents=True, exist_ok=True)
    (b / "gemm").mkdir(exist_ok=True)
    flags = ["-std=c++14"]
    objs: List[Path] = []
    main_o = b / "main.obj"
    r = run_with_msvc_env(
        [nvcc] + NVCC_HOST + flags + NVCC_ARCH + ["-c", str(g / "src/main.cu"), "-o", str(main_o)],
        g,
    )
    if r.returncode != 0:
        print(r.stderr)
        return None
    objs.append(main_o)
    for name in (
        "sgemm_cublas.cu",
        "sgemm_naive.cu",
        "sgemm_shared.cu",
        "sgemm_register.cu",
        "sgemm_register_vec_bank.cu",
        "sgemm_register_bank_conflict.cu",
        "sgemm_register_vectorized.cu",
    ):
        src = g / "src/gemm" / name
        o = b / "gemm" / (name.replace(".cu", ".obj"))
        r = run_with_msvc_env(
            [nvcc] + NVCC_HOST + flags + NVCC_ARCH + ["-c", str(src), "-o", str(o)],
            g,
        )
        if r.returncode != 0:
            print(r.stderr)
            return None
        objs.append(o)
    out = b / "benchmark_gemm.exe"
    r = run_with_msvc_env(
        [nvcc] + NVCC_HOST + flags + NVCC_ARCH + [str(x) for x in objs] + ["-o", str(out), "-lcublas"],
        g,
    )
    if r.returncode != 0:
        print(r.stderr)
        return None
    return out


def build_softmax(nvcc: str) -> Optional[Path]:
    d = ROOT / "softmax"
    sources = [
        d / "src/main.cu",
        d / "src/softmax/v1_naive.cu",
        d / "src/softmax/v2_shared_memory.cu",
        d / "src/softmax/v3_warp_reduction.cu",
        d / "src/softmax/v4_vectorized.cu",
        d / "src/softmax/v5_online.cu",
        d / "src/softmax/v5_vec_ultimate.cu",
        d / "src/benchmark/benchmark.cu",
    ]
    out = BUILD / "softmax" / "benchmark_all.exe"
    if nvcc_link(
        nvcc,
        d,
        sources,
        out,
        [d / "src"],
        ["-use_fast_math", "-std=c++17"],
        [],
    ):
        return out
    return None


def build_transpose(nvcc: str) -> Optional[Path]:
    d = ROOT / "transpose"
    trans = sorted((d / "src/transpose").glob("v*.cu"))
    host = d / "src/transpose/transpose_host.cu"
    sources = [d / "src/main.cu", host, d / "src/benchmark/benchmark.cu"] + trans
    out = BUILD / "transpose" / "transpose_benchmark.exe"
    if nvcc_link(nvcc, d, sources, out, [d / "src/include"], ["-std=c++14"], []):
        return out
    return None


def build_topk(nvcc: str) -> Optional[Path]:
    d = ROOT / "topk" / "benchmark"
    out = BUILD / "topk" / "topk_benchmark.exe"
    if nvcc_link(
        nvcc,
        d,
        [d / "topk_benchmark.cu"],
        out,
        [d.parent],
        ["-std=c++14"],
        [],
    ):
        return out
    return None


def build_scan(nvcc: str) -> Optional[Path]:
    d = ROOT / "scan"
    sources = [
        d / "src/scan/v1_hillis_steele.cu",
        d / "src/scan/v2_blelloch.cu",
        d / "src/scan/v3_bank_free.cu",
        d / "src/scan/v4_warp_primitive.cu",
        d / "src/scan/benchmark.cu",
    ]
    out = BUILD / "scan" / "benchmark_scan.exe"
    if nvcc_link(nvcc, d, sources, out, [d / "include"], ["-std=c++17", "--use_fast_math"], []):
        return out
    return None


def build_reduction(nvcc: str) -> Optional[Path]:
    d = ROOT / "reduction"
    kdir = d / "kernels"
    sources = [
        d / "src/benchmark.cu",
        kdir / "common.cu",
        kdir / "reduce_v1_interleaved.cu",
        kdir / "reduce_v2_strided.cu",
        kdir / "reduce_v3_sequential.cu",
        kdir / "reduce_v4_first_add.cu",
        kdir / "reduce_v5_warp_shuffle.cu",
        kdir / "reduce_v6_vectorized.cu",
        kdir / "reduce_cub.cu",
        kdir / "run_kernel.cu",
    ]
    out = BUILD / "reduction" / "benchmark_reduction.exe"
    if nvcc_link(
        nvcc,
        d,
        sources,
        out,
        [d / "include"],
        ["-std=c++14"],
        [],
    ):
        return out
    return None


def build_flashattention(nvcc: str) -> Optional[Path]:
    d = ROOT / "flashattention"
    srcs = sorted((d / "src/flashattention").glob("*.cu"))
    sources = [d / "src/benchmark.cu"] + srcs
    out = BUILD / "flashattention" / "benchmark_flashattention.exe"
    if nvcc_link(
        nvcc,
        d,
        sources,
        out,
        [d / "src"],
        ["-std=c++14", "-use_fast_math"],
        ["-lcublas", "-lcurand"],
    ):
        return out
    return None


def build_batch_gemm(nvcc: str) -> Optional[Path]:
    d = ROOT / "batch_gemm"
    sources = [
        d / "src/benchmark_bmm.cu",
        d / "src/kernel_naive.cu",
        d / "src/kernel_shared_memory.cu",
    ]
    out = BUILD / "batch_gemm" / "benchmark_bmm.exe"
    if nvcc_link(nvcc, d, sources, out, [], ["-std=c++14"], ["-lcublas"]):
        return out
    return None


# If build_ncu compile fails or --no-build, use these when present.
FALLBACK_EXE: Dict[str, Path] = {
    "gemm": ROOT / "GEMM" / "build" / "benchmark_gemm.exe",
}


PROJECTS: Dict[str, Dict] = {
    "gemm": {
        "build": build_gemm,
        "kernels": [
            "sgemm_naive_kernel",
            "sgemm_shared_kernel",
            "sgemm_register_kernel",
            "sgemm_register_vectorized_kernel",
            "sgemm_register_bank_conflict_kernel",
            "sgemm_register_kernel_vec_bank",
        ],
        "args": [],
    },
    "softmax": {
        "build": build_softmax,
        "kernels": [
            "kernel_max_v1",
            "kernel_sum_v1",
            "kernel_div_v1",
            "softmax_v2_kernel",
            "softmax_v3_warp_kernel",
            "softmax_v4_vectorized_kernel",
            "softmax_v5_online_kernel",
            "softmax_v5_vec_kernel",
        ],
        "args": [],
    },
    "transpose": {
        "build": build_transpose,
        "kernels": [
            "transpose_naive_kernel",
            "transpose_shared_kernel",
            "transpose_shared_pad_kernel",
            "transpose_optimized_kernel",
            "transpose_vectorized_kernel",
        ],
        "args": ["-m", "4096", "-n", "4096", "-w", "2", "-b", "2"],
    },
    "topk": {
        "build": build_topk,
        "kernels": [
            "topk_v1_kernel",
            "topk_v2_kernel",
            "topk_v3_kernel",
            "topk_v4_kernel",
            "topk_v4_warp_kernel",
        ],
        "args": ["custom", "5000", "8"],
    },
    "scan": {
        "build": build_scan,
        "kernels": [
            # v1 host wrapper launches hillis_steele_exclusive_kernel, not hillis_steele_kernel.
            "regex:hillis_steele_exclusive_kernel",
            "regex:blelloch_scan_kernel",
            "regex:bank_free_scan_kernel",
            "regex:warp_scan_kernel",
            "regex:warp_scan_single_block_kernel",
        ],
        "args": [],
    },
    "reduction": {
        "build": build_reduction,
        "kernels": [
            "reduce_v1",
            "reduce_v2",
            "reduce_v3",
            "reduce_v4",
            "reduce_v5",
            "reduce_v6",
        ],
        "args": [],
    },
    "flashattention": {
        "build": build_flashattention,
        "kernels": [
            "flash_attention_v1_naive_kernel",
            "flash_attention_v2_shared_kv_kernel",
            "flash_attention_v3_q_tiling_kernel",
            "flash_attention_v4_vectorized_kernel",
            "flash_attention_v5_fa2_kernel",
        ],
        # Smaller head dim keeps V3 shared memory within per-block limits on consumer GPUs.
        "args": ["1", "512", "32", "2"],
    },
    "batch_gemm": {
        "build": build_batch_gemm,
        "kernels": ["bmm_naive_kernel", "bmm_shared_memory_kernel"],
        "args": [],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Nsight Compute all LLM CUDA benchmarks")
    ap.add_argument("--no-build", action="store_true", help="Skip nvcc; use existing build_ncu exes")
    ap.add_argument(
        "--only",
        nargs="*",
        metavar="PROJECT",
        help=f"Subset: {', '.join(sorted(PROJECTS))}",
    )
    args = ap.parse_args()

    nvcc = find_nvcc()
    ncu = find_ncu()
    if not ncu:
        print("[ERROR] ncu not found")
        return 1
    if not nvcc:
        nvcc = find_nvcc()
    if not args.no_build and not nvcc:
        print("[ERROR] nvcc not found")
        return 1
    if not nvcc:
        print("[ERROR] nvcc needed for CUDA DLL path when profiling")
        return 1

    print(f"[INFO] nvcc: {nvcc}")
    print(f"[INFO] ncu:  {ncu}")

    selected = args.only if args.only else list(PROJECTS.keys())
    for name in selected:
        if name not in PROJECTS:
            print(f"[ERROR] unknown project: {name}")
            return 1

    total_ok = total_k = 0
    for name in selected:
        cfg = PROJECTS[name]
        exe = BUILD / name / (
            "benchmark_gemm.exe"
            if name == "gemm"
            else "benchmark_all.exe"
            if name == "softmax"
            else "transpose_benchmark.exe"
            if name == "transpose"
            else "topk_benchmark.exe"
            if name == "topk"
            else "benchmark_scan.exe"
            if name == "scan"
            else "benchmark_reduction.exe"
            if name == "reduction"
            else "benchmark_flashattention.exe"
            if name == "flashattention"
            else "benchmark_bmm.exe"
        )
        if not args.no_build:
            if not nvcc:
                return 1
            fn = cfg["build"]
            built = fn(nvcc)
            if built:
                exe = built
            elif name in FALLBACK_EXE and FALLBACK_EXE[name].is_file():
                exe = FALLBACK_EXE[name]
                print(f"[INFO] {name}: build failed, using {exe}")
            else:
                print(f"[SKIP] {name}: build failed")
                continue
        elif not exe.is_file():
            if name in FALLBACK_EXE and FALLBACK_EXE[name].is_file():
                exe = FALLBACK_EXE[name]
            else:
                print(f"[SKIP] {name}: missing {exe}")
                continue

        # reduction: each kernel needs matching --versions N
        if name == "topk":
            for k in cfg["kernels"]:
                base = _kernel_report_basename(k)
                outp = (RESULTS / name) / f"{base}.ncu-rep"
                (RESULTS / name).mkdir(parents=True, exist_ok=True)
                if k == "topk_v4_warp_kernel":
                    targs = ["custom", "512", "8"]
                else:
                    targs = list(cfg["args"])
                cmd = [
                    ncu,
                    "-f",
                    "--check-exit-code",
                    "0",
                    "--kernel-name",
                    k,
                    "--launch-count",
                    "1",
                    "-o",
                    str(outp),
                    str(exe),
                ] + targs
                print(f"[NCU] {name}/{base}.ncu-rep")
                r = subprocess.run(cmd, env=child_env(nvcc), capture_output=True, **SUBPROC_KW)
                if outp.is_file() and outp.stat().st_size > 0:
                    total_ok += 1
                else:
                    print((r.stderr or r.stdout or "")[:600])
                total_k += 1
            continue

        if name == "reduction":
            base_args = ["--quick", "--sizes", "1048576"]
            for i, k in enumerate(cfg["kernels"], start=1):
                out_sub = RESULTS / name
                base = _kernel_report_basename(k)
                outp = out_sub / f"{base}.ncu-rep"
                out_sub.mkdir(parents=True, exist_ok=True)
                cmd = [
                    ncu,
                    "-f",
                    "--check-exit-code",
                    "0",
                    "--kernel-name",
                    k,
                    "--launch-count",
                    "1",
                    "-o",
                    str(outp),
                    str(exe),
                ] + base_args + ["--versions", str(i)]
                print(f"[NCU] {name}/{base}.ncu-rep (v{i})")
                r = subprocess.run(cmd, env=child_env(nvcc), capture_output=True, **SUBPROC_KW)
                if outp.is_file() and outp.stat().st_size > 0:
                    total_ok += 1
                else:
                    print((r.stderr or r.stdout or "")[:600])
                total_k += 1
            continue

        kerns: List[str] = list(cfg["kernels"])
        app_args = cfg["args"]
        ok, nk = profile_kernels(ncu, nvcc, exe, kerns, RESULTS / name, app_args)
        total_ok += ok
        total_k += nk

    print(f"\n[DONE] {total_ok}/{total_k} reports -> {RESULTS}")
    return 0 if total_ok == total_k else 1


if __name__ == "__main__":
    sys.exit(main())

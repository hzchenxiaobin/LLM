# CUDA Toolkit 12.8 + Nsight Compute 安装指南 (Windows)

## 系统信息确认

你的系统配置：
- **GPU**: NVIDIA GeForce RTX 5090 D (Blackwell 架构, sm_100)
- **驱动版本**: 591.44 ✓ (已满足要求)
- **操作系统**: Windows 10/11
- **所需 CUDA 版本**: 12.8 或更高 (为了支持 RTX 5090 的 sm_100)

---

## 第一步: 下载 CUDA Toolkit 12.8

### 方法 1: 官方下载页面 (推荐)

1. 访问: https://developer.nvidia.com/cuda-12-8-0-download-archive
2. 选择以下选项:
   - **Operating System**: Windows
   - **Architecture**: x86_64
   - **Version**: 11 或 10/11 (根据你的系统选择)
   - **Installer Type**: 
     - `exe (local)` - 完整离线安装包 (约 3.5GB，推荐)
     - `exe (network)` - 小型在线安装程序

3. 点击 **Download** 按钮

### 方法 2: 直接下载链接

```powershell
# CUDA Toolkit 12.8.0 - Windows 11 x86_64
# 本地安装包 (完整版)
https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe

# 网络安装包 (较小，需要联网)
https://developer.download.nvidia.com/compute/cuda/12.8.0/network_installers/cuda_12.8.0_windows_network.exe
```

> **注意**: RTX 5090 需要 CUDA 12.8+ 才能支持 sm_100 架构。旧版本 CUDA 无法识别你的显卡。

---

## 第二步: 安装 CUDA Toolkit

### 2.1 运行安装程序

1. 双击下载的 `cuda_12.8.0_xxx_windows.exe`
2. 选择提取路径 (默认即可)，点击 **OK** 等待提取完成
3. 安装程序启动后，选择 **"同意并继续"**

### 2.2 选择安装类型

选择 **"自定义 (高级)"**，这样可以选择具体安装哪些组件：

```
☑ CUDA
  ☑ CUDA Runtime (必须)
  ☑ CUDA Development (必须)
  ☑ CUDA Documentation (可选)
  ☑ Visual Studio Integration (如果你使用 VS)
  
☑ Driver components
  ☑ Display Driver (如果驱动版本较旧，建议勾选)
  
☑ Other components
  ☑ Nsight Compute (必须 - 用于性能分析)
  ☑ Nsight Systems (推荐 - 系统级性能分析)
  ☑ Nsight Visual Studio Edition (可选)
  ☑ Nsight Monitor (可选)
```

### 2.3 选择安装位置

默认路径:
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v12.8`

**建议保持默认路径**，这样环境变量配置更简单。

### 2.4 等待安装完成

安装过程约需 10-30 分钟，取决于选择的组件和磁盘速度。

---

## 第三步: 验证安装

### 3.1 检查 CUDA 版本

打开 PowerShell 或 CMD，运行：

```powershell
# 检查 nvcc 版本
nvcc --version
```

预期输出：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.8, V12.8.xxx
```

### 3.2 检查 Nsight Compute

```powershell
# 检查 ncu 是否可用
ncu --version

# 或者查看完整路径
Get-Command ncu
```

预期输出：
```
NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2025 NVIDIA Corporation
Version 2025.x.x.x (build xxx)
```

### 3.3 运行样例测试

```powershell
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\demo_suite"

# 运行设备查询
.\deviceQuery.exe

# 运行带宽测试
.\bandwidthTest.exe
```

---

## 第四步: 配置环境变量 (如需要)

如果 `nvcc` 或 `ncu` 命令不可用，手动添加环境变量：

### 4.1 打开环境变量设置

```powershell
# 快速打开系统属性
sysdm.cpl
```

然后：
1. 点击 **"高级"** 标签
2. 点击 **"环境变量"** 按钮
3. 在 **"系统变量"** 区域找到 **Path**，点击 **编辑**

### 4.2 添加以下路径

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\  (或其他版本号)
```

### 4.3 添加 CUDA_PATH 变量

在 **"系统变量"** 中点击 **新建**：
- **变量名**: `CUDA_PATH`
- **变量值**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

### 4.4 验证环境变量

重启 PowerShell 后运行：

```powershell
$env:CUDA_PATH
$env:PATH -split ";" | Select-String "CUDA"
```

---

## 第五步: 运行 GEMM 性能分析

安装完成后，回到 GEMM 目录运行 profiling：

```powershell
cd d:\github\LLM\GEMM

# 一键构建并分析
python run_ncu.py

# 或手动步骤
nvcc -O3 -std=c++14 -arch=sm_100 -c src/main.cu -o build/main.obj
# ... 编译其他文件 ...
ncu --kernel-name sgemm_register_kernel --launch-count 1 -o results.ncu-rep build/benchmark_gemm.exe
ncu-ui results.ncu-rep
```

> **注意**: RTX 5090 使用 `sm_100` 架构，不是 `sm_90` 或其他。

---

## 常见问题解决

### Q1: 安装程序提示 "无法找到兼容的驱动程序"

**解决**: 你的驱动 591.44 已经很新，直接跳过驱动安装，只安装 CUDA Toolkit。

### Q2: "nvcc 不是内部或外部命令"

**解决**: 环境变量未配置。按照第四步手动添加 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` 到 PATH。

### Q3: Nsight Compute 未安装或版本不对

**解决**: 
1. 重新运行 CUDA 安装程序
2. 选择 "自定义安装"
3. 确保勾选 **Nsight Compute** 组件

### Q4: RTX 5090 不被识别

**解决**: 必须使用 CUDA 12.8 或更高版本。旧版本 CUDA 不支持 Blackwell 架构。

### Q5: VS Integration 安装失败

**解决**: 如果不使用 Visual Studio，可以在安装时取消勾选 "Visual Studio Integration" 组件。

---

## 验证清单

安装完成后，确认以下命令都能正常运行：

```powershell
# 1. GPU 驱动和 CUDA 运行时
nvidia-smi

# 2. CUDA 编译器
nvcc --version

# 3. Nsight Compute 命令行
ncu --version

# 4. Nsight Compute GUI
ncu-ui --version
```

---

## 下一步

1. ✅ 安装 CUDA Toolkit 12.8
2. ✅ 验证 `nvcc` 和 `ncu` 可用
3. 🎯 运行 `python run_ncu.py` 开始 GEMM 性能分析

如需帮助，请查看 `NCU_PROFILING_GUIDE.md` 文件了解如何进行性能分析。

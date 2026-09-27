# RTX 4060 笔记本 CUDA 实践环境搭建

适用环境：Windows 11 + NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存）。

编写与资料核对日期：2026-09-27。

目标：在本机建立 Linux CUDA C++ 开发环境，完成编译、GPU 计算、结果校验和内存检查，随后开始向量加法、并行归约、数据搬运与 Stream 实验。

本文保留从零安装的步骤，示例输出用于对照。2026-09-27 后续验收已在本机 WSL2 / Ubuntu 24.04 / CUDA 12.8 上完成：向量加法八组输入通过，Compute Sanitizer memcheck 为 0 errors。

环境已经搭好时，可以直接进入[第一课](lesson-01-vector-add.md)。换电脑或使用 GPU 服务器时，参考[远端 GPU 接续](remote-gpu.md)，获取仓库中的源码并重新编译。

## 1. 本机现状与安装方案

此前在 Windows 中实际检查到：

| 项目 | 当前状态 | 后续动作 |
| --- | --- | --- |
| 操作系统 | Windows 11，Build 26200 | 满足 WSL2 的系统版本条件 |
| 显卡 | RTX 4060 Laptop GPU | 用于本地单卡实验 |
| 显存 | NVIDIA-SMI 显示 8188 MiB | 入门实验足够 |
| NVIDIA 驱动 | 610.88，NVIDIA-SMI 正常 | 先沿用现有驱动 |
| 驱动侧 CUDA 信息 | CUDA UMD Version 13.3 | 不等于已安装 Toolkit |
| CUDA 编译器 | PATH、默认安装目录及安装记录中未发现 | 在 Ubuntu 中安装 |
| WSL | 系统提示尚未安装 | 安装 WSL2 和 Ubuntu |
| VS Code | 已安装 Windows 版本 | 增加 WSL 扩展 |

采用如下分工：

```text
Windows 11
├── NVIDIA Windows 驱动：负责访问 RTX 4060
├── VS Code：编辑代码，通过 WSL 扩展连接 Linux
└── WSL2 / Ubuntu 24.04
    ├── GCC / G++：编译 CPU 端 C++ 代码
    ├── CUDA Toolkit 12.8：nvcc、头文件、运行库和开发工具
    ├── CMake / Ninja：组织构建
    └── ~/cuda-labs：源代码、构建产物和实验记录
```

本文固定使用 CUDA **12.8 系列**作为学习基线，APT 会选取仓库中该系列可用的补丁版本。它足以覆盖当前实验；这里不要求安装最新 CUDA。Ubuntu 24.04 和 GCC 13 属于 CUDA 12.8 的支持范围，具体发行版验证信息见 [CUDA 12.8 Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)。

WSL 中使用 Windows 提供的 GPU 驱动接口，只安装 Linux Toolkit。不要在 WSL 中安装 `nvidia-driver-*`、`cuda-drivers`，也不要将下文的 `cuda-toolkit-12-8` 换成带驱动依赖的 `cuda` 或 `cuda-12-8`。这是 NVIDIA 对 WSL 安装方式的明确要求。[CUDA on WSL 官方说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## 2. 先区分两种终端

| 标记 | 在哪里执行 | 典型提示符 |
| --- | --- | --- |
| Windows PowerShell | Windows Terminal 的 PowerShell 标签页 | `PS C:\Users\...>` |
| Ubuntu / Bash | 启动 WSL 后，或 VS Code 的 WSL 终端 | `name@computer:~$` |

所有代码块只复制命令本身。`sudo apt`、`export`、`source` 和 Linux 路径在 Ubuntu 中使用。

本文出现的 `~` 表示 Linux 用户主目录，例如 `/home/你的Linux用户名`。

## 3. 安装前准备

1. 接通笔记本电源。
2. 保存正在编辑的内容，WSL 安装可能要求重启 Windows。
3. 建议在 WSL 所在磁盘预留 30—40 GB 空间，用于 Ubuntu、Toolkit、缓存和实验。这是规划余量，不是安装包精确大小。
4. 在任务管理器 → 性能 → CPU 中查看“虚拟化”。若为“已禁用”，需要在 BIOS/UEFI 中启用 Intel VT-x 或 AMD SVM，具体菜单按电脑厂商说明操作。

**Windows PowerShell：**

```powershell
nvidia-smi
wsl --list --online
```

第一条应显示 RTX 4060。若第二条因尚未安装 WSL 而失败，直接进入下一步。

Windows 家庭版可以使用 WSL2；不需要为了学习 CUDA 升级到专业版。

## 4. 安装 WSL2 与 Ubuntu 24.04

右键开始菜单 → 终端（管理员），打开 PowerShell。

**Windows PowerShell，管理员：**

```powershell
wsl --install -d Ubuntu-24.04
```

按安装器提示完成安装。如提示需要重启，保存工作后重启，再从开始菜单启动 Ubuntu 24.04。

如果发行版名称提示不可用，运行 `wsl --list --online` 核对名称；本文后续命令中的发行版名称应与列表中的实际名称保持一致。

如果下载长时间停留在 0%，可尝试微软提供的另一种下载方式：

```powershell
wsl --install --web-download -d Ubuntu-24.04
```

首次启动 Ubuntu 时，创建一个 Linux 用户名和密码。输入密码时不显示字符是正常行为。该密码用于之后的 `sudo`，与 Windows 登录密码独立。

安装完成后回到 **Windows PowerShell**：

```powershell
wsl --update
wsl --list --verbose
```

验收重点：`Ubuntu-24.04` 对应的 `VERSION` 必须为 `2`。`STATE` 是 `Running` 或 `Stopped` 都正常。

如果显示 `1`，执行：

```powershell
wsl --set-version Ubuntu-24.04 2
```

之后进入 Ubuntu：

```powershell
wsl -d Ubuntu-24.04
```

安装、发行版选择和版本检查命令参考 [微软 WSL 安装指南](https://learn.microsoft.com/en-us/windows/wsl/install)。

## 5. 检查 Ubuntu 和 GPU 接入

从这一节开始，除特别标明外，命令都在 **Ubuntu / Bash** 中执行。

```bash
cat /etc/os-release
uname -m
uname -r
nvidia-smi
```

检查以下结果：

- Ubuntu 版本为 24.04。
- CPU 架构为 `x86_64`。
- 内核版本带有 WSL2 / Microsoft 相关标识。
- `nvidia-smi` 能识别 RTX 4060 Laptop GPU。

如果 `nvidia-smi` 提示找不到命令，尝试：

```bash
/usr/lib/wsl/lib/nvidia-smi
```

这个绝对路径能运行，说明 GPU 接入正常，只是 PATH 需要补充。后面的环境配置会处理它。WSL 的 NVIDIA-SMI 部分监控字段可能与 Windows 不同。[NVIDIA WSL 限制说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications)

如果两个命令都无法访问 GPU，先处理第 12 节的 WSL/GPU 故障，再继续安装 Toolkit。

## 6. 安装基础开发工具

**Ubuntu / Bash：**

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build git wget ca-certificates
```

APT 会显示要安装的包与磁盘用量，确认后继续。

```bash
gcc --version
g++ --version
cmake --version
ninja --version
git --version
```

这些命令应分别输出版本。CUDA 的 `nvcc` 需要调用主机 C++ 编译器，因此仅安装 CUDA 运行库并不足以编译程序。

本方案使用 Ubuntu 中的 GCC/G++；Windows 上未安装 MSVC 不影响这条开发路线。

## 7. 安装 CUDA Toolkit 12.8

### 7.1 配置 NVIDIA 的 WSL 软件源

**Ubuntu / Bash：**

```bash
mkdir -p ~/Downloads/cuda-setup
cd ~/Downloads/cuda-setup
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

每条命令成功后再继续。若下载失败或 APT 报签名错误，先排查网络与软件源，不要跳过签名校验。

安装 keyring 的方法见 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。本文使用的 keyring 文件和 `cuda-toolkit-12-8` 包已在 [NVIDIA WSL 软件仓库](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/) 核对。

### 7.2 检查并安装指定系列

```bash
apt-cache policy cuda-toolkit-12-8
```

确认 `Candidate` 后面有版本号，而不是 `(none)`，再执行：

```bash
sudo apt install cuda-toolkit-12-8
```

下载时间由网络决定；这是完整 Toolkit，包含多种开发组件。安装结束后先用绝对路径检查：

```bash
/usr/local/cuda-12.8/bin/nvcc --version
```

应能看到 `release 12.8`。最后的小版本和构建号不必与其他教程一致。

### 7.3 配置 Bash 环境

以下代码会在 `~/.bashrc` 末尾添加一段配置；标记已存在时不会重复追加。按整块复制：

```bash
if ! grep -q '^# CUDA learning environment 12.8$' ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'

# CUDA learning environment 12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/usr/lib/wsl/lib:$PATH"
EOF
fi
source ~/.bashrc
```

这段配置适用于 Ubuntu 默认 Bash。如果以后改用 Zsh，需要将相应配置放到 `~/.zshrc`。

验证：

```bash
command -v nvcc
nvcc --version
nvidia-smi
command -v compute-sanitizer
```

`nvcc` 路径应指向 `/usr/local/cuda-12.8/bin/nvcc`。

本例使用 `nvcc` 默认的 CUDA Runtime 链接方式，不需要先全局设置 `LD_LIBRARY_PATH`。后续若使用动态运行库且明确提示缺少 `.so`，再按实际缺失的库设置路径；不要把 `lib64/stubs` 放进运行时搜索路径。

`nvidia-smi` 的驱动侧 CUDA 信息与 `nvcc --version` 的 Toolkit 版本可以不同。前者描述驱动，后者描述实际用于编译的工具链。最终通过编译并运行示例验证整个组合。

## 8. 准备实验目录与 VS Code

### 8.1 在 Linux 文件系统中保存代码

**Ubuntu / Bash：**

```bash
mkdir -p ~/cuda-labs/00-environment
cd ~/cuda-labs/00-environment
pwd
```

预期路径形如 `/home/你的Linux用户名/cuda-labs/00-environment`。

当前 Windows 仓库 `D:\学习\linkforfuture` 在 WSL 中通常对应 `/mnt/d/学习/linkforfuture`，可从那里读取本篇文档。实验代码放在 `~/cuda-labs`，有利于 Linux 工具的文件访问性能。[微软关于 WSL 文件存储的建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)

在 Ubuntu 中运行以下命令，可用 Windows 文件资源管理器打开当前目录：

```bash
explorer.exe .
```

### 8.2 连接 VS Code

1. 打开已安装的 Windows VS Code。
2. 在扩展面板安装 Microsoft 发布的 **WSL**，扩展 ID 为 `ms-vscode-remote.remote-wsl`。
3. 回到 Ubuntu 的实验目录，运行：

```bash
code .
```

首次连接会下载 VS Code Server。打开后确认左下角显示 WSL 和 Ubuntu 相关标识。

在此窗口的扩展面板中，将 **C/C++**（Microsoft）安装到 WSL 环境。使用 CMake 时，可选装 **CMake Tools**（Microsoft）。编译先以终端命令为准。

在 VS Code 新建终端，检查：

```bash
pwd
which g++
which nvcc
```

应分别指向 Linux 实验目录、Linux 编译器和 `/usr/local/cuda-12.8/bin/nvcc`。如果仍是 `PS C:\...>`，说明终端没有进入 WSL。

如果 `code .` 不可用，在 Windows VS Code 中按 `Ctrl+Shift+P`，选择 **WSL: Connect to WSL using Distro...**，连接 Ubuntu 24.04 后打开 Linux 实验目录。操作方式参考 [VS Code WSL 文档](https://code.visualstudio.com/docs/remote/wsl)。

## 9. 编译并运行第一个 CUDA 程序

这一节验证显存分配、数据拷贝、kernel 启动、GPU 完成同步、结果校验和资源释放。

### 9.1 创建源文件

在 `~/cuda-labs/00-environment` 中创建 `vector_add.cu`，完整内容如下。也可以在 Ubuntu 中执行 `nano vector_add.cu` 后粘贴，按 `Ctrl+O`、回车保存，再按 `Ctrl+X` 退出。

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: %s failed: %s\n",
                     file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static bool runCase(int n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 97) * 0.25f;
        b[i] = static_cast<float>(i % 31) * 0.5f;
    }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&da), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&db), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dc), bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (!std::isfinite(c[i]) || std::fabs(c[i] - expected) > 1e-6f) {
            std::fprintf(stderr, "N=%d index=%d got=%g expected=%g\n",
                         n, i, c[i], expected);
            passed = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::printf("N=%d: %s\n", n, passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device found\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\nCompute capability: %d.%d\nVRAM: %.0f MiB\n",
                prop.name, prop.major, prop.minor,
                static_cast<double>(prop.totalGlobalMem) / (1024 * 1024));

    const int sizes[] = {1, 31, 32, 33, 255, 256, 257, 1000003};
    bool allPassed = true;
    for (int n : sizes) {
        if (!runCase(n)) {
            allPassed = false;
        }
    }
    std::puts(allPassed ? "ALL TESTS PASSED" : "TEST FAILED");
    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

### 9.2 编译

**Ubuntu / Bash，在源文件所在目录：**

```bash
nvcc -std=c++17 -O2 -lineinfo -arch=sm_89 vector_add.cu -o vector_add
```

| 参数 | 用途 |
| --- | --- |
| `-std=c++17` | 固定 C++ 标准 |
| `-O2` | 启用优化 |
| `-lineinfo` | 保留 GPU 代码的源码行信息，方便定位 |
| `-arch=sm_89` | 为本机 RTX 4060 的计算能力 8.9 编译 |
| `-o vector_add` | 指定输出可执行文件 |

RTX 4060 的计算能力见 [NVIDIA GPU 列表](https://developer.nvidia.com/cuda/gpus)。后续换远程 GPU 时，需要按对应设备调整架构参数。

编译成功通常没有输出，接着运行：

```bash
./vector_add
echo $?
```

预期看到 GPU 名称、计算能力 `8.9`、八组 `PASS` 和 `ALL TESTS PASSED`；最后退出码为 `0`。显存输出以运行时结果为准。

这里故意包含非 256 整数倍的数据规模，用于验证尾块处理。该程序是环境与正确性检查，还没有做性能计时；本节通过不代表已经完成性能分析。

### 9.3 用 Compute Sanitizer 检查

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./vector_add
echo $?
```

预期程序仍然全部通过，工具报告 `ERROR SUMMARY: 0 errors`，退出码为 `0`。`--error-exitcode 1` 让工具发现错误时返回非零状态，便于后续自动化。

内存检查适合验证越界与非对齐访问；分析工具运行会带来额外开销，不要拿它报告的运行时长作为性能成绩。后续涉及共享内存与同步时，再分别使用 `racecheck`、`synccheck`。[Compute Sanitizer 官方文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

## 10. 可选：使用 CMake 构建

先完成 `nvcc` 直接编译，再做这一节。在同一目录创建 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(cuda_environment_check LANGUAGES CXX CUDA)

add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 89
)
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
```

**Ubuntu / Bash：**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build
./build/vector_add
```

预期同样输出 `ALL TESTS PASSED`。如果以后切换 Toolkit，使用新的构建目录，例如 `build-cuda-new`，避免旧 CMake 缓存继续引用原来的编译器。

## 11. 可选：验证 Nsight Systems 时间线采集

这个步骤用于后续 Stream 与拷贝重叠实验。工具是否可以完整采集，还取决于具体工具版本、WSL 和驱动的组合；先以向量加法及 memcheck 通过作为基础环境验收。

**Ubuntu / Bash：**

```bash
nsys --version
```

如果命令可用：

```bash
mkdir -p reports
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o reports/vector-add ./vector_add
nsys stats reports/vector-add.nsys-rep
```

预期生成 `.nsys-rep` 报告，并在统计中看到 CUDA API、kernel 或内存传输记录。这次只验证采集链路；小程序的时间线不能代替后续规范的性能实验。重复运行时使用不同的输出名称，例如 `reports/vector-add-02`。

若没有 `nsys`，可以检查已安装的 Nsight 包与工具位置：

```bash
dpkg -l | grep -i nsight
find /opt/nvidia /usr/local/cuda-12.8 -type f -name nsys 2>/dev/null
```

若找到文件，用返回的绝对路径替换命令中的 `nsys`。若未安装，根据 [Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 选择支持当前 WSL 环境的 Linux CLI 版本。Windows GUI 可用于打开采集报告，读取时优先使用与采集端相同或兼容的版本。

## 12. 常见问题排查

### 12.1 WSL 安装提示虚拟化、0x80370102 或虚拟机无法启动

先确认已完成安装要求的 Windows 重启，再检查任务管理器中的虚拟化状态。若 BIOS 虚拟化已开启但仍失败，核对 Windows 的“虚拟机平台”组件是否启用，并按 [微软 WSL 故障排查](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting) 对照具体错误处理。

### 12.2 Windows 能看到 GPU，Ubuntu 看不到

在 **Windows PowerShell** 运行：

```powershell
nvidia-smi
wsl --list --verbose
wsl --update
```

确认发行版使用 WSL2。保存所有 WSL 会话中的工作后，可以重启 WSL：

```powershell
wsl --shutdown
wsl -d Ubuntu-24.04
```

`wsl --shutdown` 会停止所有运行中的 WSL 发行版和其中的进程。重新进入 Ubuntu 后运行 `/usr/lib/wsl/lib/nvidia-smi`。不要通过在 WSL 内安装 Linux 显卡驱动来修复这个问题。

### 12.3 `nvcc: command not found`

**Ubuntu / Bash：**

```bash
ls -l /usr/local/cuda-12.8/bin/nvcc
/usr/local/cuda-12.8/bin/nvcc --version
source ~/.bashrc
command -v nvcc
```

文件存在且绝对路径可运行：检查第 7.3 节的 PATH。文件不存在：检查 `sudo apt install cuda-toolkit-12-8` 是否真正安装成功。

### 12.4 `Unable to locate package cuda-toolkit-12-8`

```bash
dpkg -s cuda-keyring
grep -R 'developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu' /etc/apt/sources.list.d/
sudo apt update
apt-cache policy cuda-toolkit-12-8
```

检查 keyring 是否安装、源是否配置，以及 `apt update` 是否有下载错误。不要为了绕过问题同时加入多套教程里的 CUDA 源。

### 12.5 下载超时或 DNS 解析失败

```bash
getent hosts developer.download.nvidia.com
wget --spider https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
```

第一条检查域名解析，第二条检查 HTTPS 访问。Windows 浏览器能访问不意味着 WSL 中的软件源一定能访问。如果使用代理，检查 WSL 的代理配置；先保留具体错误，再区分 DNS、连接超时和证书问题。

### 12.6 编译时报 `unsupported GNU version`

```bash
g++ --version
nvcc --version
```

本文基线使用 Ubuntu 24.04 默认的 GCC 13。若后来手动切换了 GCC，应按 CUDA 12.8 的支持范围选择主机编译器；不要把 `--allow-unsupported-compiler` 当成常规解决办法。

### 12.7 编译成功，但运行报 `no CUDA-capable device` 或驱动不足

先检查第 5 节的 GPU 接入，再检查当前 `nvcc` 是否被其他安装覆盖。可收集：

```bash
nvidia-smi
which nvcc
nvcc --version
ls -l /dev/dxg
```

如果 Windows 驱动本身异常，应在 Windows 侧修复或升级驱动，并重启；不要将驱动问题与缺少 Toolkit 混在一起处理。

### 12.8 `no kernel image is available` 或 `invalid device function`

核对目标设备与架构参数。本文的 `sm_89` 和 CMake 的 `89` 针对 RTX 4060；复制到其他型号 GPU 后需要重新编译。设备信息打印部分能帮助确认计算能力。

### 12.9 VS Code 找不到头文件，但终端可以编译

确认窗口连接到 WSL，C/C++ 扩展也安装在 WSL 一侧。若仅编辑器有红线，在 C/C++ 配置中设置 Linux 编译器路径 `/usr/bin/g++`，并补充 `/usr/local/cuda-12.8/include`。编辑器索引错误与真实的 `nvcc` 编译失败应分开判断。

### 12.10 性能工具提示计数器权限不足

这与 CUDA kernel 能否运行是不同问题。记录完整错误，按对应工具的官方权限说明处理；GPU 计数器权限可能需要在 Windows NVIDIA 控制面板侧配置，单纯在 Ubuntu 中加 `sudo` 未必解决。入门阶段可以先完成正确性检查和 CUDA Event 计时。

## 13. 最终验收记录

完成后逐项勾选：

- [ ] Windows `nvidia-smi` 正常。
- [ ] `wsl --list --verbose` 显示 Ubuntu 使用 WSL2。
- [ ] Ubuntu 可以访问 RTX 4060。
- [ ] `nvcc --version` 显示 CUDA 12.8。
- [ ] GCC/G++、CMake、Ninja 可用。
- [ ] VS Code 连接到 WSL，终端目录位于 `~/cuda-labs`。
- [ ] `vector_add.cu` 编译成功，八组输入全部 PASS。
- [ ] Compute Sanitizer memcheck 报告 0 errors。
- [ ] 可选：CMake 构建成功。
- [ ] 可选：Nsight Systems 生成包含 CUDA 记录的报告。

建议在 **Ubuntu / Bash** 保存一份环境快照，方便之后对比：

```bash
mkdir -p ~/cuda-labs/environment
{
    date -Is
    cat /etc/os-release
    uname -a
    nvidia-smi
    nvcc --version
    g++ --version
    cmake --version
    dpkg-query -W 'cuda-toolkit*' 'cuda-nvcc*' 'cuda-cudart*'
} > ~/cuda-labs/environment/setup-info.txt 2>&1
```

安装后第一轮学习就继续修改本节的向量加法：解释线程索引和尾块，比较 128/256/512 的 block size，再加入 CUDA Event 计时。每次只改一个主要因素，保留运行结果和解释。

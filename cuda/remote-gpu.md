# 换台电脑，在远端 GPU 上继续学习

学习材料、起点源码和笔记都在个人主页的 Git 仓库中。换设备后，先阅读网页，再在有 GPU 的 Linux 环境拉取代码并编译。网站本身不执行 CUDA，也不需要上传服务器账号来阅读课程。

## 1. 选择今天的运行环境

| 环境 | 怎样开始 |
| --- | --- |
| 本机 RTX 4060 + WSL2 | 打开 Ubuntu 终端，使用已安装的 CUDA 12.8 |
| 远端 Linux GPU 服务器 | SSH 登录，或使用平台提供的终端，先检查工具链 |
| 当前只有手机或平板 | 先阅读课程、手算索引、整理笔记；有服务器网页终端时再运行实验 |

如果租用服务器，选择带 CUDA **开发工具**的环境，确认有 `nvcc`；只有 CUDA runtime 的镜像可能无法编译 `.cu`。第一课单张 NVIDIA GPU 即可。

服务器环境以平台或管理员提供的配置为准，不直接套用 WSL 的驱动安装步骤。容器环境中的 GPU 需要由平台正确暴露给容器。

## 2. 检查远端环境

在运行实验的 Linux 终端执行：

```bash
nvidia-smi
nvcc --version
g++ --version
cmake --version
git --version
```

确认 GPU 可见，有可用的 CUDA Toolkit、兼容的主机编译器和 CMake 3.24 以上。检查工具可单独确认：

```bash
compute-sanitizer --version
```

`nvcc` 不可用时，先查看服务器是否提供 CUDA 环境模块、开发容器或安装路径。多套 CUDA 并存时，按平台说明切换版本；无需为了与笔记本完全一致而重装驱动。

课程起点在 CUDA 12.8 上验证，其他版本以实际编译和运行结果为准。换 GPU 后重新构建，不复制本机二进制文件到服务器。

## 3. 获取同一套课程和源码

第一次使用这台 Linux 机器：

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/linkforfuture/linkforfuture.github.io.git linkforfuture
cd linkforfuture
```

如果 `~/src/linkforfuture` 已存在，进入原来的仓库，不要覆盖目录：

```bash
cd ~/src/linkforfuture
git status --short
```

没有尚未提交的修改时，再更新：

```bash
git pull --ff-only
```

有修改时，先提交自己的实验，或保存到另一个明确的位置后再同步。若 Git 报分支分叉，先查看两侧提交再处理，不用强制覆盖来消除提示。

## 4. 编译并运行第一课

在仓库根目录：

```bash
cmake -S cuda/labs/01-vector-add -B cuda/labs/01-vector-add/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda/labs/01-vector-add/build
./cuda/labs/01-vector-add/build/vector_add
```

构建文件默认使用 `CMAKE_CUDA_ARCHITECTURES=native`，检测这台机器可见 GPU 的架构。首次应看到八组 `PASS` 与 `ALL TESTS PASSED`。[CMake 架构配置说明](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)

若 CUDA 编译器不在 PATH，但已知其路径，可以在首次配置时补充：

```bash
cmake -S cuda/labs/01-vector-add -B cuda/labs/01-vector-add/build-explicit -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build cuda/labs/01-vector-add/build-explicit
./cuda/labs/01-vector-add/build-explicit/vector_add
```

`/usr/local/cuda/bin/nvcc` 是示例，换成服务器实际路径。若只能显式指定架构，先查 [NVIDIA GPU 计算能力列表](https://developer.nvidia.com/cuda/gpus)：A100 为 `80`、A10 为 `86`、RTX 4060 为 `89`、H100 为 `90`。在 CMake 配置命令后增加相应的 `-DCMAKE_CUDA_ARCHITECTURES=数字`，并确保 Toolkit 支持该型号。

正确性检查：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./cuda/labs/01-vector-add/build/vector_add
```

如果之前用了 `build-explicit` 等其他目录，运行和检查命令中的路径也要一起替换。之后按[第一课](lesson-01-vector-add.md)修改与实验，每次修改后重新构建。

## 5. 在熟悉的编辑器中操作

有 VS Code 的电脑可安装 Microsoft 的 **Remote - SSH** 扩展，通过平台提供的主机名、用户名和端口连接，在远端打开 `~/src/linkforfuture`。编辑器的终端也必须连接到远端，CUDA 编译与运行才会发生在服务器上。[VS Code Remote-SSH 官方文档](https://code.visualstudio.com/docs/remote/ssh)

本机 WSL 使用 WSL 扩展。没有 VS Code 时，服务器网页终端或 SSH 终端中的编辑器也可以完成全部实验。

## 6. 保存进度，方便下一台设备接续

将代码修改保存在 `cuda/labs/`，将结论保存在 `cuda/notes/`。构建产物、Nsight 报告和缓存目录已通过 `.gitignore` 排除；要分享数据时，把必要的汇总表和解释写进实验记录。

先检查变更：

```bash
git status --short
git diff -- cuda/
```

编辑 `cuda/notes/README.md`，写清楚今天完成了什么、下次从哪里开始。新增笔记时，同时在进度页和 `cuda/_sidebar.md` 中添加入口。

下面是提交某次第一课记录的示例，文件名应替换为本次实际文件：

```bash
git add cuda/labs/01-vector-add/vector_add.cu cuda/notes/2026-09-27-vector-add.md cuda/notes/README.md cuda/_sidebar.md
git diff --cached
git commit -m "docs(cuda): record vector addition experiment"
git push origin main
```

上面的推送示例适用于在 `main` 上工作的个人仓库。若正在学习分支，应推送对应分支，合并到网站发布分支后网页才更新。公开仓库可以匿名拉取，推送需要你在当前设备上配置有写权限的 GitHub 身份；根据设备与平台选择登录方式，不把令牌写进仓库文件或命令 URL。

暂时不能推送时，可以先保存提交，之后从这台机器同步；服务器将被释放前，要把源码与笔记保存到可持续访问的位置。不要把浏览器里尚未保存的文字作为唯一的学习记录。

当前站点通过 GitHub Pages 发布。向发布分支推送后，在[仓库 Actions](https://github.com/linkforfuture/linkforfuture.github.io/actions)确认部署完成，再打开[CUDA 学习入口](https://linkforfuture.github.io/cuda/)。流程说明见 [GitHub Pages 发布文档](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。

## 7. 每次离开前留下这三项

- 当前源码与记录已经保存到哪里、是否已推送。
- GPU 型号、Toolkit 版本和本次运行的提交号。
- 下次要回答的问题与第一条要执行的命令。

性能结果必须同时记录硬件和测量条件。笔记本与服务器的数据可以帮助理解差异，但不能只按耗时大小判断某次代码修改是否有效。

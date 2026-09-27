# CUDA 第一课：从向量加法理解线程与数据

建议用时：60—90 分钟。练习基于你已经完成的环境检查程序。

本次已在你的 WSL2 / Ubuntu 24.04 中重新编译运行：CUDA 12.8，RTX 4060 Laptop GPU，计算能力 8.9；原有八组输入全部通过，Compute Sanitizer memcheck 报告 0 errors。验证日期：2026-09-27。

课程起点源码已收入本站仓库的 `cuda/labs/01-vector-add`。本机之前准备的 `~/cuda-labs/01-vector-add` 仍可使用；为了换设备后继续学习，建议以后在仓库内做实验，把修改和笔记一起提交。下面的练习需要自己修改与观察，尚未替你完成。

新电脑或远端服务器先按[远端 GPU 接续](remote-gpu.md)获取仓库并检查环境。源码和通用构建命令见[第一课源码与构建](labs/01-vector-add/README.md)。

## 1. 打开程序，先跑一次

以下命令都在 Ubuntu 终端执行：

```bash
cd ~/src/linkforfuture/cuda/labs/01-vector-add
code .
```

这里假设按接续文档克隆到了 `~/src/linkforfuture`；已有仓库可以使用自己的实际路径。也可以通过 VS Code WSL 或 Remote-SSH 打开目录。终端应处于运行 CUDA 的 Linux 环境。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/vector_add
```

这份 CMake 默认检测当前可见 GPU 的架构，适用于本机与远端机器。确认最后输出 `ALL TESTS PASSED`。每次修改后运行 `cmake --build build`；编译失败时先修复错误，避免误把旧可执行文件当成新结果。

## 2. 先看懂数据去了哪里（15 分钟）

按这个顺序阅读 `runCase`，在源码旁写出自己的注释：

| 代码 | 执行位置与作用 |
| --- | --- |
| `std::vector<float> a, b, c` | CPU 端容器，用于输入和结果校验 |
| `cudaMalloc` | CPU 调用 API，在设备上申请数组空间 |
| `cudaMemcpy(...HostToDevice)` | 将输入复制到 GPU |
| `vectorAdd<<<blocks, threads>>>` | CPU 提交 kernel，让 GPU 线程处理数组 |
| `cudaGetLastError()` | 检查 CUDA 错误状态，包括启动配置等错误 |
| `cudaDeviceSynchronize()` | 等待此前设备工作完成，并检查异步执行错误 |
| `cudaMemcpy(...DeviceToHost)` | 将结果复制回 CPU |
| 校验循环与 `cudaFree` | 检查结果并释放显存 |

在这个程序中，GPU 计算发生在 `__global__ void vectorAdd(...)` 内。`runCase()`、`main()` 和校验循环都在 CPU 上运行。CPU 侧持有设备地址，并不意味着可以把它当作普通 CPU 数组直接解引用。

请自己写下两点：

- CPU 数组 `a` 与 GPU 数组 `da` 分别在哪里？哪条语句把数据从前者送到后者？
- 如果漏掉从设备拷回结果的步骤，CPU 校验读到的是什么？

这一阶段先使用显式分配与拷贝，方便把数据流看清楚。

## 3. 理解线程怎样对应元素（20 分钟）

重点看这三行：

```cpp
constexpr int threads = 256;
const int blocks = (n + threads - 1) / threads;
vectorAdd<<<blocks, threads>>>(da, db, dc, n);
```

以及 kernel 中的：

```cpp
const int i = blockIdx.x * blockDim.x + threadIdx.x;
```

本例使用一维 grid 和 block：`blocks` 是 block 数量，`threads` 是每个 block 的线程数量；`blockIdx.x` 是当前 block 编号，`threadIdx.x` 是块内线程编号，`blockDim.x` 是每块线程数。全部 block 构成本次 kernel 的 grid。[CUDA 12.8：Thread Hierarchy](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#thread-hierarchy)

先用更小的例子手算：`N = 10`，每块 4 个线程。

| block 编号 | 块内线程编号 | 对应全局索引 i | 真正参与加法的索引 |
| --- | --- | --- | --- |
| 0 | 0、1、2、3 | 0、1、2、3 | 0、1、2、3 |
| 1 | 0、1、2、3 | 4、5、6、7 | 4、5、6、7 |
| 2 | 0、1、2、3 | 8、9、10、11 | 8、9 |

需要 3 个 block，共启动 12 个线程。末尾两个线程仍然执行了索引计算和条件判断，但不会进入 `if (i < n)` 中访问数组。

这与 HCCL 中处理切片尾块有联系：分配的执行单元数量可以超过有效数据量，边界条件负责限制实际访问范围。

现在自己填写这张表，暂时不要运行代码：

| N | 每块线程数 | block 数 | 总启动线程数 | 因 i >= N 跳过加法的线程数 |
| --- | --- | --- | --- | --- |
| 255 | 256 | 待填写 | 待填写 | 待填写 |
| 256 | 256 | 待填写 | 待填写 | 待填写 |
| 257 | 256 | 待填写 | 待填写 | 待填写 |
| 1000 | 256 | 待填写 | 待填写 | 待填写 |

然后在 host 代码中、kernel 启动前插入：

```cpp
std::printf("N=%d blocks=%d threads_per_block=%d launched=%d\n",
            n, blocks, threads, blocks * threads);
```

在 `sizes` 数组中添加 `1000`，重新编译运行，核对预测。这里打印的是启动配置，不是 GPU 线程实际执行顺序；block 编号不保证调度先后。

## 4. 亲手改三个版本（20 分钟）

每次只做一个改动，编译运行后记录结论。完成后将当前版本保存下来，再做下一项。

### 实验 A：更改每块线程数

把 `threads` 依次设为 `128`、`256`、`512`，保持其余代码不变。保留小输入与 `257`、`1000`、`1000003` 等非整除输入。

记录：计算结果是否改变？block 数是否改变？为什么仍然可以完整覆盖数组？

这一轮只研究正确性与分工。程序还没有规范计时，不能根据终端“感觉快慢”判断哪种配置性能更好。

### 实验 B：更改 GPU 计算

把计算改为 `c[i] = 2.0f * a[i] + b[i]`，并同步修改 CPU 校验中的 `expected`。

先想一想：如果只改 kernel，为什么检查应该失败？然后实际做一次这个中间版本，再补上 CPU 参考结果，使所有输入重新通过。

这个实验的重点是建立习惯：每次修改 GPU 算法，都用独立的 CPU 参考检查结果。

### 实验 C：观察边界错误

从当前正确版本复制出一个专门的故障练习文件：

```bash
cp -n vector_add.cu vector_add_oob.cu
```

若同名文件已存在，换一个新文件名。只在故障文件里删除 `if (i < n)` 的保护，保留数组加法。不要更改原始练习文件。

```bash
nvcc -std=c++17 -O2 -lineinfo -arch=native vector_add_oob.cu -o vector_add_oob
compute-sanitizer --tool memcheck --error-exitcode 1 ./vector_add_oob
```

上面的 `-arch=native` 需要当前 nvcc 支持并能够检测 GPU。若不支持，按服务器 GPU 型号指定架构，例如 RTX 4060 为 `sm_89`，不要把本机参数直接套在远端设备上。先预测哪些输入会发生越界，再阅读报告中的访问类型、源码位置和线程信息。即使普通运行碰巧输出正确，也不能证明越界访问安全；本实验以检查工具的诊断为依据。

最后给故障副本恢复边界保护，重新编译并运行同一检查命令，确认错误消失。工具用法参考 [Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

## 5. 留下一份自己的实验记录（10 分钟）

用[实验记录模板](notes/template.md)在 `cuda/notes/` 下新建按日期命名的 Markdown，并在[进度页](notes/README.md)登记。按下面格式记录，不必写成长文章：

```text
实验：
我改了什么：
运行前的预测：
实际输出：
预测是否正确，原因是什么：
仍然不理解的地方：
```

今天完成的标准是你能用自己的话解释：

1. 为什么 GPU kernel 里没有遍历整个数组的普通 for 循环，却能算完所有元素？
2. 当 N=257、每块 256 个线程时，最后一个 block 中哪些线程访问数组？
3. 为什么 block 数使用向上取整？
4. `if (i < n)` 保护了哪几次数组访问？
5. `cudaGetLastError()` 与 `cudaDeviceSynchronize()` 分别帮助发现哪类问题？

## 6. 下一次再加入计时

第一课完成后，下一课继续使用这份程序：预热，在同一 Stream 上用 CUDA Event 包住重复 kernel，比较不同 N 与 block size；另外单独记录包含输入输出拷贝的耗时。

kernel 启动对 CPU 通常是异步的，计时必须明确等待哪个工作完成。性能数据中还要区分单次启动开销、缓存影响和大数据访问成本。[CUDA 12.8：性能计时与带宽](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-best-practices-guide/index.html#performance-metrics)

先把线程索引、数据流和边界弄明白，下一次的性能曲线才有可解释的基础。

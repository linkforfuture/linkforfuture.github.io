# 第一课源码与构建

这是第一课的起点程序：CPU 准备输入，GPU 做向量加法，CPU 验证八组大小不同的数组。练习任务见[第一课](/lesson-01-vector-add.md)。

<p><a href="labs/01-vector-add/vector_add.cu" download="vector_add.cu" data-no-router>下载 vector_add.cu</a> · <a href="labs/01-vector-add/CMakeLists.txt" download="CMakeLists.txt" data-no-router>下载 CMakeLists.txt</a></p>

[在线阅读源码](https://github.com/linkforfuture/linkforfuture.github.io/blob/main/cuda/labs/01-vector-add/vector_add.cu)

推荐通过 Git 获取完整仓库，下载单个文件适合临时实验。下面在已经获取的仓库根目录执行：

```bash
cmake -S cuda/labs/01-vector-add -B cuda/labs/01-vector-add/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda/labs/01-vector-add/build
./cuda/labs/01-vector-add/build/vector_add
```

要求 CMake 3.24 或更高，当前机器上有可见的 NVIDIA GPU、兼容的驱动和 CUDA Toolkit。默认使用 `native` 编译当前可见 GPU 的架构。若当前工具链检测不到设备或需要交叉编译，可显式传入架构；例如 RTX 4060：

```bash
cmake -S cuda/labs/01-vector-add -B cuda/labs/01-vector-add/build-89 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build cuda/labs/01-vector-add/build-89
./cuda/labs/01-vector-add/build-89/vector_add
```

修改源码后执行 `cmake --build` 即可。不同 GPU 或工具链使用新的构建目录，避免复用旧缓存。架构参数定义参考 [CMake CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)。

预期输出八组 `PASS`，最后为 `ALL TESTS PASSED`。检查内存：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./cuda/labs/01-vector-add/build/vector_add
```

应报告 `ERROR SUMMARY: 0 errors`。这是正确性基线，尚未加入性能计时。

基线已于 2026-09-27 在 RTX 4060 Laptop GPU、WSL2、CUDA 12.8 上验证；其他 GPU 需要在对应机器上重新验证。

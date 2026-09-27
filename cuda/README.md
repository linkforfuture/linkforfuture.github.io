# CUDA 实践学习

从 HCCL 集合通信研发出发，通过可运行的小实验学习 GPU 编程。每一课都留下源码、验证方法与自己的解释。

<div class="learning-intro">
<p><strong>现在开始：线程与向量加法</strong><br>用 60—90 分钟看懂 CPU/GPU 数据流，手算线程索引，修改 block 大小，再定位一次越界访问。</p>
<a href="#/lesson-01-vector-add">进入第一课 →</a>
</div>

## 从哪里继续

| 当前情况 | 学习入口 |
| --- | --- |
| 本机已有 CUDA 环境 | [第一课：线程与向量加法](lesson-01-vector-add.md) |
| Windows 笔记本需要搭建环境 | [WSL2 + Ubuntu + CUDA 安装文档](environment-setup-wsl2.md) |
| 使用另一台电脑或远端 GPU | [获取源码、检测环境、编译并继续学习](remote-gpu.md) |
| 想直接运行例子 | [第一课源码与构建说明](labs/01-vector-add/README.md) |
| 想记录今天学到了什么 | [进度与实验记录](notes/README.md) |

## 学习路线

按实验结果推进，每次先预测、再修改、最后解释。后续课程随学习过程补充。

| 阶段 | 动手做什么 | 完成标准 | 材料状态 |
| --- | --- | --- | --- |
| 00 环境 | 编译 CUDA、访问 GPU、检查内存 | 程序正确运行，memcheck 无错误 | 文档已提供 |
| 01 线程与数据 | 向量加法、线程索引、尾块 | 能解释每个线程访问哪个元素 | 课程与源码已提供 |
| 02 性能计时 | CUDA Event、预热、数据规模扫描 | 区分 kernel 与端到端耗时 | 待学习时补充 |
| 03 内存访问 | 显存拷贝、矩阵转置、共享内存 | 用测量解释布局对性能的影响 | 待学习时补充 |
| 04 并行归约 | 分块求和与线程协作 | 正确处理同步、尾块和数值误差 | 待学习时补充 |
| 05 流水线 | Stream / Event 与分块传输 | 从时间线确认是否实际重叠 | 待学习时补充 |
| 06 通信相关实验 | 打包、重排、拷贝与归约 | 比较中间缓冲区和直接输出的代价 | 待学习时补充 |

## 网页阅读，GPU 上运行

手机、平板或另一台电脑都可以阅读这套文档。CUDA 代码在本机 WSL 或你连接的 GPU 服务器上运行；网页提供课程、源码入口和可复制命令。

源码与笔记存放在同一个 Git 仓库。换设备时拉取仓库，按当前 GPU 重新编译；完成实验后提交源码和 Markdown 记录，推送后即可在其他设备接着阅读。网页不把浏览器内的勾选状态当成跨设备进度，学习进度以[仓库中的记录](notes/README.md)为准。

```text
cuda/
├── environment-setup-wsl2.md   本机安装与验收
├── lesson-01-vector-add.md    第一课
├── remote-gpu.md              换设备与远端运行
├── labs/01-vector-add/        CUDA 源码和 CMake
└── notes/                    进度、实验结论和记录模板
```

[在 GitHub 查看全部学习材料](https://github.com/linkforfuture/linkforfuture.github.io/tree/main/cuda)

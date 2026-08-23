# HCCL 代码入口梳理

## 1. 对外接口文件

HCCL 对用户暴露的接口主要在：

| 文件 | 作用 |
| --- | --- |
| [include/hccl.h](D:/CANN/hccl/include/hccl.h:35) | 常规通信算子 API：`HcclAllReduce`、`HcclBroadcast`、`HcclAllGather`、`HcclSend`、`HcclRecv` 等。 |
| [include/hccl_mc2.h](D:/CANN/hccl/include/hccl_mc2.h:82) | MC2/KFC 资源上下文相关接口，如 `HcclCreateOpResCtx`。 |

这两个头文件会被安装到包内的 `x86_64-linux/include/hccl/`，对应 CMake 配置见 [CMakeLists.txt](D:/CANN/hccl/CMakeLists.txt:153)。

## 2. 算子入口文件

每个算子的 Host API 实现入口在 `src/ops/<op>/<op>_op.cc`。这些文件负责参数校验、组装 `OpParam`、选择执行模式，然后进入公共执行链。

| 算子 | 入口实现文件 |
| --- | --- |
| AllReduce | [src/ops/all_reduce/all_reduce_op.cc](D:/CANN/hccl/src/ops/all_reduce/all_reduce_op.cc:23) |
| Broadcast | [src/ops/broadcast/broadcast_op.cc](D:/CANN/hccl/src/ops/broadcast/broadcast_op.cc:23) |
| ReduceScatter | [src/ops/reduce_scatter/reduce_scatter_op.cc](D:/CANN/hccl/src/ops/reduce_scatter/reduce_scatter_op.cc:23) |
| ReduceScatterV | [src/ops/reduce_scatter_v/reduce_scatter_v_op.cc](D:/CANN/hccl/src/ops/reduce_scatter_v/reduce_scatter_v_op.cc:24) |
| Scatter | [src/ops/scatter/scatter_op.cc](D:/CANN/hccl/src/ops/scatter/scatter_op.cc:30) |
| AllGather | [src/ops/all_gather/all_gather_op.cc](D:/CANN/hccl/src/ops/all_gather/all_gather_op.cc:24) |
| AllGatherV | [src/ops/all_gather_v/all_gather_v_op.cc](D:/CANN/hccl/src/ops/all_gather_v/all_gather_v_op.cc:22) |
| AlltoAll / AlltoAllV / AlltoAllVC | [src/ops/all_to_all_v/all_to_all_v_op.cc](D:/CANN/hccl/src/ops/all_to_all_v/all_to_all_v_op.cc:23) |
| Reduce | [src/ops/reduce/reduce_op.cc](D:/CANN/hccl/src/ops/reduce/reduce_op.cc:23) |
| Send | [src/ops/send/send_op.cc](D:/CANN/hccl/src/ops/send/send_op.cc:48) |
| Recv | [src/ops/recv/recv_op.cc](D:/CANN/hccl/src/ops/recv/recv_op.cc:46) |
| BatchSendRecv | [src/ops/batch_send_recv/batch_send_recv_op.cc](D:/CANN/hccl/src/ops/batch_send_recv/batch_send_recv_op.cc:23) |

通用目录读法：

| 目录 | 作用 |
| --- | --- |
| `selector/` | 算法选择，注册宏是 `REGISTER_SELECTOR_BY_OPTYPE`。 |
| `executor/` | 资源计算和算法编排，注册宏是 `REGISTER_EXECUTOR_IMPL`。 |
| `template/aicpu/` | AICPU/TS 展开时使用的算法模板。 |
| `template/aiv/` | AIV 算法模板和 AIV kernel。 |
| `template/ccu/` | CCU 相关模板和 kernel。 |
| `op_common/` | Host 侧公共执行、资源申请、拓扑、注册表、AICPU/AIV/CCU 分发。 |

## 3. Host 侧公共入口

算子自己的 `HcclXxx` 函数只做入口包装；真正公共执行链在 [src/ops/op_common/op_common.cc](D:/CANN/hccl/src/ops/op_common/op_common.cc:85)。

核心函数：

| 函数 | 位置 | 作用 |
| --- | --- | --- |
| `Selector` | [op_common.cc:85](D:/CANN/hccl/src/ops/op_common/op_common.cc:85) | 计算拓扑，运行 `ExecuteSelector`，得到算法名和执行引擎。 |
| `HcclExecOp` | [op_common.cc:494](D:/CANN/hccl/src/ops/op_common/op_common.cc:494) | Host 侧最重要的公共执行入口：取 executor、申请资源、按 engine 分发到 AICPU/AIV/CCU/CPU。 |
| `HcclGetAlgRes` | [op_common.cc:869](D:/CANN/hccl/src/ops/op_common/op_common.cc:869) | 统一申请算法资源、线程、channel、上下文。 |
| `HcclAicpuKernelEntranceLaunch` | [op_common.cc:651](D:/CANN/hccl/src/ops/op_common/op_common.cc:651) | Host 侧进入 AICPU kernel 前的通知、上下文准备和 profiling 上报。 |
| `AicpuKernelLaunch` | [op_common.cc:685](D:/CANN/hccl/src/ops/op_common/op_common.cc:685) | 通过 `aclrtBinaryGetFunction` 找到 `HcclLaunchAicpuKernel` 并 launch。 |

典型链路：

```text
include/hccl.h
  -> src/ops/<op>/<op>_op.cc::HcclXxx
  -> Selector(...)
  -> HcclExecOp(...)
  -> CollAlgExecRegistryV2::GetAlgExec(...)
  -> executor->CalcRes / executor->Orchestrate
  -> template/aicpu 或 template/aiv 或 template/ccu
```

## 4. AICPU 入口

AICPU 真正导出的入口函数是：

| 文件 | 作用 |
| --- | --- |
| [src/ops/op_common/template/aicpu/kernel_launch.cc:258](D:/CANN/hccl/src/ops/op_common/template/aicpu/kernel_launch.cc:258) | `extern "C" unsigned int HcclLaunchAicpuKernel(OpParam *param)`，AICPU 侧入口。 |
| [src/ops/op_common/template/aicpu/load_kernel.cc:38](D:/CANN/hccl/src/ops/op_common/template/aicpu/load_kernel.cc:38) | `LoadAICPUKernel()`，加载 `libscatter_aicpu_kernel.json`。 |
| [src/ops/scatter/scatter_aicpu_kernel.ini](D:/CANN/hccl/src/ops/scatter/scatter_aicpu_kernel.ini:1) | 生成 AICPU json 的源配置，声明 `functionName=HcclLaunchAicpuKernel`、`kernelSo=libscatter_aicpu_kernel.so`。 |

安装后的 json 内容指向：

```json
{
  "HcclLaunchAicpuKernel": {
    "opInfo": {
      "functionName": "HcclLaunchAicpuKernel",
      "kernelSo": "libscatter_aicpu_kernel.so",
      "opKernelLib": "KFCKernel"
    }
  }
}
```

## 5. 编包后的动态库

当前 `build_out` staging 里能看到这些关键产物：

| 产物 | 安装位置 | 作用 |
| --- | --- | --- |
| `libhccl.so` | `x86_64-linux/lib64/` | Host 侧主库，对外承载 `HcclXxx` 算子接口，包含算子入口、公共执行链、selector/executor/template 等 Host 逻辑。CMake target 见 [src/CMakeLists.txt:435](D:/CANN/hccl/src/CMakeLists.txt:435)。 |
| `libhccl_compat.so` | `x86_64-linux/lib64/` | Host 侧兼容/dlsym 适配层，封装 HCOMM/HCCL 相关动态符号。CMake target 见 [hcomm_dlsym/CMakeLists.txt:80](D:/CANN/hccl/src/common/hcomm_dlsym/CMakeLists.txt:80)。 |
| `aicpu_hccl.tar.gz` | `opp/built-in/op_impl/aicpu/kernel/` | AICPU kernel 包。CMake 打包目标见 [CMakeLists.txt:121](D:/CANN/hccl/CMakeLists.txt:121)。 |
| `libscatter_aicpu_kernel.so` | 在 `aicpu_hccl.tar.gz` 内 | AICPU 侧 kernel so，包含 `HcclLaunchAicpuKernel`。target 见 [src/CMakeLists.txt:228](D:/CANN/hccl/src/CMakeLists.txt:228)。 |
| `libhccl_kernel_compat.so` | 在 `aicpu_hccl.tar.gz` 内 | Device/AICPU 侧兼容/dlsym 适配层，供 AICPU kernel 调 HCOMM device/profiling/diag 等能力。target 见 [hcomm_dlsym/CMakeLists.txt:159](D:/CANN/hccl/src/common/hcomm_dlsym/CMakeLists.txt:159)。 |

`aicpu_hccl.tar.gz` 内实际文件：

```text
aicpu_kernels_device/
aicpu_kernels_device/libhccl_kernel_compat.so
aicpu_kernels_device/bin_hash.cfg
aicpu_kernels_device/libscatter_aicpu_kernel.so
```

一句话记忆：`libhccl.so` 是 Host 主入口，`libhccl_compat.so` 是 Host 兼容符号层，`aicpu_hccl.tar.gz` 是 Device AICPU 包，其中 `libscatter_aicpu_kernel.so` 承载 AICPU 入口，`libhccl_kernel_compat.so` 给 AICPU 侧补底层动态符号适配。
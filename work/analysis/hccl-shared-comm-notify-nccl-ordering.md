# HCCL 同一通信域跨流卡死定位与 NCCL 保序机制

AllGather 与 BatchSendRecv 使用同一个 HCCL 通信域、在不同用户流上重叠执行时，8 卡用例出现本地 Notify 等待超时。分离通信域或等待 AllGather 完成后再启动 P2P 均可通过。本次定位把范围收敛到 A5 AICPU 路径的 Thread 和 Notify 复用，并通过 NCCL 源码解释了为什么相似的 `shared` 用例可以运行成功：NCCL 在库内维护同一 communicator 的跨流完成依赖。

> 记录日期：2026-10-16。适用场景：A5、8 个 rank、AllGather 与批量点对点通信、非图捕获执行。复现结果来自实际运行反馈，源码分析基于下表提交；这些提交不直接代表复现环境实际加载的二进制版本。

| 源码仓 | 对照提交 | 用途 |
| --- | --- | --- |
| HCCL | `4ec9456f4026` | 算子入口、资源上下文、Host 与 Device 握手 |
| HCOMM | `d1c2b18c2a8b` | Thread 池、普通 Notify 和 A5 SQE |
| torch_npu | `7f767e5d1e52` | ProcessGroup 通信流、Work 等待与提交队列 |
| NCCL | `3e46c089f4ae`，版本 2.32.3 | 同一 communicator 的跨流保序 |

## 1 复现场景与对照结果

每个进程绑定一张 NPU，共 8 个 rank。AllGather 使用 WORLD 进程组，单 rank 的输入分别为 1MiB、2MiB 和 384KiB，类型为 `uint8`。P2P 按 `0 → 1 → … → 7` 发送，每条消息来自一个 `128 × 512` 的 BF16 张量，共 131072 字节。

P2P 直接通过 `ctypes` 调用当前进程已加载的 `libhccl.so` 中的 `HcclBatchSendRecv`，并传入指定进程组的原生通信域句柄。

| 模式 | 通信域关系 | 第二批 AllGather 与 P2P 的依赖 | 反馈结果 |
| --- | --- | --- | --- |
| `shared` | 复用 WORLD 的同一个 comm | P2P 只等待 `current`，未等待 `ag_done` | 卡死，随后出现 EI0002 |
| `separate` | P2P 使用独立的新进程组 | 允许两批通信重叠 | 通过 |
| `serial` | 复用同一个 comm | CPU 等待 `ag_stream` 完成后才调用 P2P | 通过 |
| `shared` 加事件依赖 | 仍复用同一个 comm | `p2p_stream.wait_event(ag_done)` | 8 个 rank 全部 PASS，重复两次 |

加事件后的完整用例反馈耗时约 38 秒。这是整段运行耗时，不能当作通信性能或并行加速指标。

### 1.1 为什么脚本中的同步调用仍会留下重叠窗口

torch_npu 的 AllGather 实际提交到 ProcessGroup 管理的 `hcclStream`。默认情况下，`async_op=False` 对应的 `Work.wait()` 在调用者当前流上插入通信完成事件等待；它不等同于 CPU 已经等待设备通信结束。开启 blocking wait 时才有额外的 CPU 等待逻辑。

第二批 AllGather 调用结束后，`ag_stream` 上的 `ag_done` 能表示该批通信完成，但脚本中的 P2P 只执行：

```python
p2p_stream.wait_stream(current)
```

它等待的是调用时 `current` 已经排入的工作，没有覆盖 `ag_stream` 上第二批 AllGather 的完成点。脚本形成的设备依赖是：

```text
第一批 AllGather 完成
        ├── 第二批 AllGather ── ag_done
        └── BatchSendRecv   ── p2p_done
```

应用层这张依赖图没有明显的环。风险来自两个分支共同访问 HCCL 内部资源。

## 2 algTag 区分上下文不等于底层资源独占

A5 的 `HcclGetAlgRes` 使用 `param.algTag` 获取和保存算法资源上下文。这保证不同算子、算法可以维护不同的描述信息，但上下文中的 Thread 句柄和 CCL Buffer 地址仍可能相同。

### 2.1 AICPU Thread 从公共池返回

HCCL 的 `HcclGetThreadWithConfig` 调用：

```cpp
HcclThreadAcquireWithConfig(
    comm, COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS,
    threadConfigs.data(), threads.data());
```

该接口没有接收 `algTag`。在对应 HCOMM 实现中，`ThreadMgr::HcclThreadAcquireV2` 按 `(engine, type)` 查找当前通信域的池，并返回池首的句柄：

```cpp
auto it = engineToThreadsMap_.find(std::make_pair(engine, type));
auto& threadVec = it->second;
// 补充 Thread 和 Notify 后
for (uint32_t idx = 0; idx < threadNum; idx++) {
    threads[idx] = threadVec[idx].handle;
}
```

因此，不同 `algTag` 的 `resCtx->threads[0]` 可以引用同一个 device Thread。Thread 上的 Notify 也就需要遵守共同的执行顺序，不能只根据各自上下文推断独占关系。

另外，AICPU 展开流按 `commName + "_unfold"` 缓存；CCL Buffer 通过 `HcclGetHcclBuffer(comm, ...)` 从通信域获取。它们也不是每个 `algTag` 分配一份。

### 2.2 启动 Notify 与算法同步 Notify 的索引可能重合

BatchSendRecv SoleMesh 的资源请求为：

```cpp
resourceRequest.slaveThreadNum = 1;
resourceRequest.notifyNumPerThread = {1};
resourceRequest.notifyNumOnMainThread = 1;
```

HCCL 给主 Thread 多申请一个用于 Host 与 Device 握手的 Notify，并将数量保存为 `1 + 1 = 2`。启动阶段按保存的数量减一发送通知，逻辑索引为 1：

```cpp
HcommThreadNotifyRecordOnThread(
    cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread - 1);
```

这个 Record 位于 P2P 用户流上。device Thread 上随后生成对应的启动 Wait。

AllGather Mesh 的从流完成通知也会使用主 Thread 的多个索引。以 8 卡、每对 rank 单通道为例，Thread 数为 7，6 个从流使用主 Thread 的索引 `0…5`。其中索引 1 与 BatchSendRecv 的启动索引相同。

**逻辑 `notifyIdx` 与设备日志中的 `notifyId` 必须分开理解。**上面的 1 是 Thread 内的索引，本次日志中的 6 是实际 Notify ID；二者需要通过 Thread 的 Notify 映射关联，不能因为数值不同就排除冲突，也不能把 `notifyId[6]` 直接读成“数组第 6 槽”。

### 2.3 普通 Notify 的失配风险

A5 普通 Notify 的 SQE 配置使用 `cntFlag=false`，Wait 使用 `clrFlag=true`。它采用普通通知状态，不保证为任意多个独立 Record 累积相同数量的可消费通知。

在共享主 Thread 和重合槽位的条件下，一种可能的设备执行时序是：

```text
P2P 用户流： Record(T0, idx=1)  // BatchSendRecv 启动
AG 从流：    Record(T0, idx=1)  // AllGather 从流完成

主 Thread：  Wait(idx=1)        // AllGather 消费并清除通知
             ……后续任务……
主 Thread：  Wait(idx=1)        // BatchSendRecv 再等待启动通知
```

如果两个 Record 在第一次 Wait 前合并，后一个 Wait 就可能失去应有的通知。也可能出现通知被另一个用途的 Wait 提前消费的情况。下面的日志和对照实验支持共享 Notify 同步失配，但尚不足以单独还原最先发生的那一次消费或合并。

## 3 关键日志与证据边界

以下日志来自故障运行反馈，省略与判断无关的字段。日志仅包含时分秒，没有据此补造复现日期。

### 3.1 AllGather 任务引用 notifyId6

第二批 AllGather 的诊断记录包含：

```text
14:10:39.809.763 [FillSlotNotify_] sqId[71] taskId[2147484753] TASK_NOTIFY_WAIT   notifyId[5]
14:10:39.809.774 [FillSlotNotify_] sqId[71] taskId[2147484754] TASK_NOTIFY_WAIT   notifyId[6]
14:10:39.809.786 [FillSlotNotify_] sqId[71] taskId[2147484755] TASK_NOTIFY_WAIT   notifyId[7]
14:10:39.810.102 [FillSlotNotify_] sqId[72] taskId[2147485728] TASK_NOTIFY_RECORD notifyId[5]
14:10:39.810.375 [FillSlotNotify_] sqId[73] taskId[2147486747] TASK_NOTIFY_RECORD notifyId[6]
14:10:39.810.653 [FillSlotNotify_] sqId[74] taskId[2147487771] TASK_NOTIFY_RECORD notifyId[7]
```

这证明任务中存在引用同一 Notify ID 的 Wait 和 Record。`FillSlotNotify_` 填充任务诊断元数据，其打印时刻不是设备完成该任务的时刻，不能直接把这些时间当作 Notify 在硬件上的占用区间。

### 3.2 BatchSendRecv 启动 Wait 也引用 notifyId6

预热 BatchSendRecv 在 `14:10:39.757` 提交。故障测试中的调用为：

```text
14:10:39.834.084 Entry-HcclLaunchAicpuKernel
    commName[group_name_0]
    tag[BatchSendRecv_group_name_0]
    algTag[BatchSendRecv_..._AicpuBatchSendRecvSoleMesh_device]
14:10:39.835.411 [DumpSqeContent] type[7] rtStreamId[1217]
    notifyId[6] subType[1] timeout[68]
14:10:39.835.653 HcclLaunchAicpuKernel success
    tag[BatchSendRecv_group_name_0] commName[group_name_0]
```

A5 定义中，`type=7` 是 `NOTIFY_WAIT`，`subType=1` 是 `SINGLE_NOTIFY_WAIT`。这条 SQE 是等待任务，不能用它证明“启动 Record 写入或覆盖了 AllGather 的 Notify”。真正的启动 Record 来自 Host 用户流，需要继续关联其目标 Thread、Notify 索引和实际 Notify ID。

### 3.3 最终异常指向 BatchSendRecv 的启动 Wait

```text
14:12:08.508.742 ReportOfOne ... sqId:71 sqHead:193 ...
14:12:08.508.867 [TaskException][AICPU]
    streamID(sqId):[71]
    taskID(sqeId):[2147484865]
    taskType:[TASK_NOTIFY_WAIT]
    notify id:[6] value:[1]
    remote rank:[local] localRank:[7]
    sqHead:193 sqTail:197
```

相邻的算子上下文记录为：

| opIndex | 算子 | count |
| --- | --- | ---: |
| 8 | AllGather | 1048576 |
| 9 | AllGather | 2097152 |
| 10 | AllGather | 393216 |
| 11 | BatchSendRecv | 131072 |

反馈的任务序列为 `OP(11): NW(/,6)`，最终明确归属到 BatchSendRecv 的 Wait。该 taskId 也不同于前面 AllGather 的 `2147484754`，不能直接写成“同一个 AllGather Wait 一直未完成”。

`sqHead=193、sqTail=197` 表示队列尚有未推进的任务；结合 `remote rank=local` 和 Notify Wait 异常，排查重点应放在本地握手及共享资源顺序。单看队列差值不等于所有后续任务都已确认具备执行条件。

### 3.4 超时配置与实际时间

本次反馈使用 `HCCL_EXEC_TIMEOUT=68`。当前源码 `DeriveAicpuTimeout` 派生的值为：

| 字段 | 计算 | 秒 |
| --- | --- | ---: |
| `waitTimeout` | 原始执行超时 | 68 |
| `fullTimeout` | 执行超时加 20 | 88 |
| `kernelLaunchTimeout` | 执行超时加 30 | 98 |
| `hostNotifyTimeout` | 执行超时加 50 | 118 |

这些值服务于不同等待位置，不是按顺序累加的多个确认阶段。从 `14:10:39.835411` 到 `14:12:08.508867` 约为 88.67 秒，接近 `fullTimeout=88`；具体哪个超时先触发，仍应关联最早的异常位置，不能仅凭配置中的 118 秒推断实测耗时。

8 个 rank 最终都反馈了 EI0002。结合模式对照、local Notify 等待和后续规避结果，本例支持“同一通信域跨流重叠引发内部同步失配”的定位；没有必要先把它解释为物理链路故障。

## 4 规避方法与验证

在所有 rank 上，保留脚本原有逻辑，将正式 P2P 调用之前的分支改为：

```python
if case == "serial":
    ag_stream.synchronize()
elif case == "shared":
    p2p_stream.wait_event(ag_done)

p2p_done = p2p(recv)
```

这样保持同一个 comm，同时建立设备依赖：

```text
第二批 AllGather 完成
    → ag_done
    → p2p_stream 继续
    → BatchSendRecv 启动 Record
    → device Thread 执行 P2P
```

启动 Record 本身位于 `p2p_stream`，也会被事件挡住。这里的“Host stream”指 Host 提交任务的设备流，Record 不是 CPU 当场直接修改 Notify。

原生调用使用的 `p2p_stream.npu_stream` getter 在对照的 torch_npu 版本中会排空相应 Host 提交队列，使先前的流等待下发到运行时。排空提交队列与等待设备任务执行完成是两个不同动作。

反馈结果是：仍使用原来的同一个通信域，无 CPU 通信完成同步，重复两次均为 8 个 rank 全部 PASS。该修改取消了第二批 AllGather 与 P2P 的设备执行重叠。需要保留两者重叠时，本用例已经验证通过的做法是 `separate`。

用 `ag_done.synchronize()` 也可以等待通信完成后再调用 P2P，但会阻塞 CPU，效果接近原有 `serial` 模式。新增一条用户流或只给 Host API 加互斥锁，不会自动让已经下发的设备任务串行。

## 5 为什么相似的 NCCL shared 用例能够通过

前提是 CUDA 侧确实传入同一个 `ncclComm_t`，且采用本地源码中的非图捕获提交路径。NCCL 也有共享的内部资源，但会在前后两次提交之间建立完整的通信完成依赖。

### 5.1 启动前让用户流等待内部流

`ncclLaunchPrepare` 获取 `comm->sharedRes->deviceStream`，随后执行：

```cpp
ncclStreamWaitStream(
    launchStream, deviceStream, comm->sharedRes->scratchEvent);
```

`ncclStreamWaitStream(a, b, event)` 的实现是：

```cpp
cudaEventRecord(event, b);
cudaStreamWaitEvent(a, event, 0);
```

因此，本次 kernel 的用户流必须等待内部流已经积累的完成依赖。

### 5.2 提交后让内部流等待本次通信

`ncclLaunchFinish` 在当前用户流的 kernel 后记录完成事件：

```cpp
cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
cudaEventRecord(finishedEvent, launchStream);
ncclStreamAdvanceToEvent(
    planner->capturingGraph, deviceStream, finishedEvent);
```

非图捕获路径下，`ncclStreamAdvanceToEvent` 直接调用 `cudaStreamWaitEvent`。前一次通信的完成点于是被保存在内部 `deviceStream` 上，供下一次提交等待：

```text
AG 实际通信流： AllGather kernel ── Record(完成事件)
                                           │
comm 内部流：                  Wait(AG完成) ── Record(内部流事件)
                                                              │
P2P 用户流：                                  Wait(内部流事件) ── P2P kernel
```

CPU 可以在 AG 仍执行时提交 P2P，但 P2P kernel 会在设备上等待此前的 AG 完成。本地 NCCL 源码由此解释了 `shared` 能通过的原因；运行成功本身不能证明两个通信 kernel 实际重叠。

### 5.3 Collective 与 P2P 都经过同一套保序

`ncclAllGather`、`ncclSend` 和 `ncclRecv` 都进入 `ncclEnqueueCheck`。统一的 Group 提交流程调用：

```text
ncclLaunchPrepare
    → ncclLaunchKernel
    → ncclLaunchFinish
```

因此，即使 AG 和 P2P 属于前后两次独立提交，仍会串接到同一条内部完成链。这不要求把前面的 AllGather 与后面的 Send/Recv 放进同一个显式 Group。

同一个 Group 内的多个用户流另外还有汇合与分发依赖，批量 Send/Recv 也可以被合并调度；不能将“跨次提交保序”推广为“Group 内每个收发任务都只能逐个执行”。

### 5.4 事件复用与普通 Notify 的区别

NCCL 可以反复使用 `scratchEvent`，因为 `cudaStreamWaitEvent` 等待调用时该事件最近一次 Record 所对应的工作，后续重新 Record 不改变已经提交的等待。它不依赖普通 Notify 那种由 Wait 消费并清除槽位状态的配对方式。

此外，同一 comm 的 `deviceStream` 完成链，与 `NCCL_LAUNCH_ORDER_IMPLICIT` 管理的同设备跨 communicator 下发顺序是不同层次。本例的同 comm 依赖无需依靠后者才能建立。

| 对照项 | 本例 HCCL AICPU 路径 | 本地 NCCL 非图路径 |
| --- | --- | --- |
| 同一 comm 的跨用户流提交 | 启动 Record 可在前次通信结束前执行 | 当前用户流先等待内部完成链 |
| 内部资源复用 | 不同 algTag 仍可引用同一 Thread 和 Notify | 共享内部流及事件，通过完成依赖保护 |
| 调用端新增的 `wait_event` | 挡住 P2P 提前执行启动通知 | 在本例中实现了与库内完成依赖相似的效果 |
| CPU 是否必须等待通信完成 | 事件规避无需 CPU 完成同步 | 内部事件依赖无需 CPU 完成同步 |

## 6 后续修复与复查要点

若在 HCCL 库内参照 NCCL 增加保护，应让同一通信域形成“前次通信完成 → 本次启动”的依赖闭环，并把等待放在本次启动 Notify 的 Record 之前。仅让展开 kernel 按序进入内部流，无法阻止另一条用户流上的 Record 提前执行。

如果选择支持同一通信域内的实际并发，就需要为并发实例处理 Thread、Notify、Channel 和中转内存的隔离与生命周期，不能仅新增一个 `algTag` 缓存项。这些是修复方向，本次验证的是调用端事件规避，没有据此宣称库内已修复。

为了确认首个失配动作，后续任务轨迹至少要关联：

1. Host 启动 Record 的源流、目标 Thread、`notifyIdx` 与实际 `notifyId`。
2. 同一 Notify ID 上各 Record 和 Wait 的设备执行先后及所属 `opIndex`。
3. 首个未推进的任务，以及它的 Record 是否尚未执行、已被消费或发生状态合并。

最后异常任务是受阻位置，不一定就是最先出错的位置。其他 AIV 同 comm 并发案例可以沿同样方法检查资源和依赖；AICPU 与 AIV 的具体同步机制仍应分别验证。

## 7 源码导航与相关记录

以下路径均相对各仓根目录，行号对应开头的源码对照提交。函数名用于后续版本搜索定位。

| 仓库 | 路径 | 重点函数或字段 |
| --- | --- | --- |
| HCCL | `src/ops/op_common/op_common.cc:1377` | 按 algTag 获取 EngineCtx |
| HCCL | `src/ops/op_common/op_common.cc:1788` | `HcclGetThreadWithConfig`、主 Thread 额外握手 Notify |
| HCCL | `src/ops/op_common/op_common.cc:2000` | `GetUnfoldThreadInfo` |
| HCCL | `src/ops/op_common/op_common.cc:1129` | 用户流上的启动 Record，及后续展开与完成等待 |
| HCCL | `src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:288` | `OpOrchestrate` 的启动 Wait |
| HCCL | `src/ops/batch_send_recv/algorithm/executor/ins_v2_batch_send_recv_sole_executor.cc:61` | BatchSendRecv 的 Thread 和 Notify 资源数量 |
| HCCL | `src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc:359` | `GetNotifyIdxSubToMain` |
| HCCL | `src/ops/op_common/aicpu_timeout.h:49` | `DeriveAicpuTimeout` |
| HCOMM | `src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads/thread_manager.cc:230` | 按 engine/type 返回池首 Thread |
| HCOMM | `src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_v82.h:28` | WAIT/RECORD 与 subType 定义 |
| HCOMM | `src/legacy/ascend950/unified_platform/resource/stream/aicpu/hccl_sqe_v82.cc:51` | 普通 Notify 的非计数及清除配置 |
| HCOMM | `src/base_comm/primitives/aicpu/task_cache/aicpu_task_cache_entry.cc:1461` | `FillSlotNotify_` 是诊断元数据填充 |
| torch_npu | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp:3994` | ProcessGroup 的内部通信流 |
| torch_npu | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp:996` | `WorkHCCL::synchronizeInternal` |
| torch_npu | `torch_npu/csrc/core/npu/NPUStream.cpp:504` | 原生流句柄 getter 的提交队列处理 |
| NCCL | `src/collectives.cc:130` | AllGather、Send、Recv 的统一 enqueue 入口 |
| NCCL | `src/group.cc:427` | `doLaunches` |
| NCCL | `src/enqueue/enqueue.cc:1792` | `ncclLaunchPrepare` 的跨流等待 |
| NCCL | `src/enqueue/enqueue.cc:2037` | `ncclLaunchFinish` 的完成依赖发布 |
| NCCL | `src/misc/strongstream.cc:386` | `ncclStreamWaitStream` 和 `ncclStreamAdvanceToEvent` |

相关记录：[AICPU 执行超时分析](analysis/aicpu-exec-timeout-analysis.md)、[Checker Barrier 分析](analysis/checker-barrier-analysis.md)、[HCOMM Channel 建链流程](hcomm/hcomm-channel-connection-flow.md)。

## 附录 复现脚本

下面保存原始复现逻辑，`shared` 分支尚未加入第 4 节的事件规避。每个模式分别启动全新的 8 个进程：

```bash
torchrun --standalone --nproc-per-node=8 hccl_shared_comm_repro.py shared
torchrun --standalone --nproc-per-node=8 hccl_shared_comm_repro.py separate
torchrun --standalone --nproc-per-node=8 hccl_shared_comm_repro.py serial
```

脚本检查 `world_size=8` 且 global rank 与 local rank 相同，对应单机 8 卡运行方式。P2P 使用 `dtype=0`（INT8）配合字节数，按原始字节搬运 BF16 张量。

```python
import ctypes as C
import datetime
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch_npu

case = sys.argv[1] if len(sys.argv) > 1 else "shared"
assert case in ("shared", "separate", "serial")
rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
device = torch.device("npu", rank)
dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=150))
assert dist.get_world_size() == 8 and dist.get_rank() == rank
pg = dist.group.WORLD
p2p_pg = dist.new_group(list(range(8)), backend="hccl") if case == "separate" else pg
pg._get_backend(device).get_hccl_comm(rank)
comm = p2p_pg._get_backend(device).get_hccl_comm(rank)


class Item(C.Structure):
    _fields_ = [("kind", C.c_int), ("buf", C.c_void_p), ("count", C.c_uint64),
                ("dtype", C.c_int), ("peer", C.c_uint32)]


paths = {s.split()[-1] for s in Path("/proc/self/maps").read_text().splitlines()
         if "/libhccl.so" in s}
assert len(paths) == 1, paths
lib = C.CDLL(paths.pop())
lib.HcclBatchSendRecv.argtypes = [C.POINTER(Item), C.c_uint32, C.c_void_p, C.c_void_p]
lib.HcclBatchSendRecv.restype = C.c_int
current = torch.npu.current_stream()
ag_stream = torch.npu.Stream(priority=-1)
p2p_stream = torch.npu.Stream()
inputs = [torch.full((n,), rank, dtype=torch.uint8, device=device)
          for n in (1048576, 2097152, 393216)]
outputs = [torch.empty(x.numel() * 8, dtype=x.dtype, device=device) for x in inputs]
outputs2 = [torch.empty_like(x) for x in outputs]
send = torch.full((128, 512), rank + 1, dtype=torch.bfloat16, device=device)
recv = torch.empty_like(send)
warm_recv = torch.empty_like(send)


def allgather(outputs):
    for src, dst in zip(inputs, outputs):
        dist.all_gather_into_tensor(dst, src, group=pg, async_op=False)


def p2p(dst):
    nbytes = send.numel() * send.element_size()
    items = []
    if rank < 7:
        items.append(Item(0, send.data_ptr(), nbytes, 0, rank + 1))
    if rank > 0:
        items.append(Item(1, dst.data_ptr(), nbytes, 0, rank - 1))
    batch = (Item * len(items))(*items)
    p2p_stream.wait_stream(current)
    with torch.npu.stream(p2p_stream):
        rc = lib.HcclBatchSendRecv(batch, len(items), comm, p2p_stream.npu_stream)
        assert rc == 0, rc
        done = p2p_stream.record_event()
    current.wait_event(done)
    return done


allgather(outputs)
torch.npu.synchronize()
p2p(warm_recv)
torch.npu.synchronize()
assert rank == 0 or (warm_recv == rank).all().item()
expected = torch.arange(8, device=device).to(torch.uint8)[:, None]
assert all((x.view(8, -1) == expected).all().item() for x in outputs)

allgather(outputs)
ag_stream.wait_stream(current)
with torch.npu.stream(ag_stream):
    allgather(outputs2)
    ag_done = ag_stream.record_event()
if case == "serial":
    ag_stream.synchronize()
p2p_done = p2p(recv)

store = dist.distributed_c10d._get_default_store()
store.set(f"submitted/{rank}", "1")
store.wait([f"submitted/{r}" for r in range(8)])
if rank == 0:
    store.set("deadline", str(time.monotonic() + 60))
deadline = float(store.get("deadline"))
completed = [f"done/{r}" for r in range(8)]
while True:
    ready = ag_done.query(), p2p_done.query()
    if all(ready):
        store.set(completed[rank], "1")
    if store.check(completed):
        break
    if time.monotonic() >= deadline:
        print(f"rank={rank} case={case} TIMEOUT: AG={ready[0]} P2P={ready[1]}", flush=True)
        os._exit(42)
    time.sleep(0.05)

assert all((x.view(8, -1) == expected).all().item() for x in outputs + outputs2)
assert rank == 0 or (recv == rank).all().item()
store.set(f"checked/{rank}", "1")
store.wait([f"checked/{r}" for r in range(8)])
print(f"rank={rank} case={case} PASS", flush=True)
dist.destroy_process_group()
```

这里的 60 秒 deadline 在提交与 Store 汇合之后才建立，不能覆盖预热、原生接口、流句柄 getter 或 Store 等待阶段。定位“为什么没有打印脚本 TIMEOUT”时，应在这些调用前后增加带 rank 的即时日志，再与设备侧超时记录对齐。

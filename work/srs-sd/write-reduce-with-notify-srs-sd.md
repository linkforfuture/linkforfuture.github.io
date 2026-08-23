# 1. SRS：write-reduce 数据搬运与完成通知融合（use write reduce with notify substitute write reduce）

> 对应 PR：!2046（commitId：3e9090c6）

## 1.1 介绍

HCCL wrapper 层（`src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc`）原有的 write-reduce 流程为：等待对端 ACK → 逐 slice 调用 `HcommWriteReduceOnThread` 将本端数据归约写入对端 hcclBuffer → **单独下发一条** `HcommChannelNotifyRecordOnThread(NOTIFY_IDX_DATA_SIGNAL)` 通知对端写入完成。

该机制存在两个问题：

1. **同步正确性风险**：reduce 搬运任务与 notify 任务是两个独立的 device 任务，无法严格保证 notify 在 reduce 数据真正落盘之后生效，对端收到 DATA_SIGNAL 后可能读取到未完成归约的旧数据（本 PR 类型标签为 Bug 修复）。
2. **任务下发开销**：每个 batch 需要额外下发一条独立的 notify 任务。

本需求使用 HCOMM 新提供的融合原语 `WRITE_REDUCE_WITH_NOTIFY`（完成"归约写入"后由底层保证再"记录通知"）替代原有的「WRITE_REDUCE + 独立 NOTIFY_RECORD」组合，在消除同步竞态的同时减少一次任务下发。同步的原子性语义由 HCOMM 底层保证，HCCL 侧通过 dlsym 动态加载调用（符合 HCCL/HCOMM 两仓解耦约束）。

## 1.2 输入

| 输入 | 说明 |
| ---- | ---- |
| 数据切片信息 srcSlices / dstSlices | 每 slice 含基地址（本端 hcclBuffer/userIn 或对端 hcclBuffer 远端地址）、offset、size、count |
| dataType | 数据类型（HcclDataType） |
| reduceOp | 归约操作（HcclReduceOp：SUM/MAX/MIN/PROD 等） |
| thread / channel | 执行线程句柄、通信通道句柄（ChannelHandle） |
| notifyIdx | 完成通知索引，固定为 NOTIFY_IDX_DATA_SIGNAL（=1，定义于 `src/ops/op_common/inc/alg_param.h:81`） |
| execTimeout | 执行超时时间（来自 ExecTimeoutManager） |

## 1.3 处理

1. **批量路径融合**：新增 `FuseNotifyToLastWriteReduceDesc`，将批量传输描述符数组中**最后一个** `WRITE_REDUCE` 描述符改写为 `WRITE_REDUCE_WITH_NOTIFY`（携带 notifyIdx）；若数组为空或末尾描述符不是 WRITE_REDUCE，则融合失败，回退为独立 NotifyRecord，保证通知语义不丢失。
2. **非批量回退路径融合**：新增 `RunWriteReduceAndNotify`，最后一个非空 slice 调用融合原语 `HcommWriteReduceWithNotifyOnThread`，其余 slice 仍调用普通 `HcommWriteReduceOnThread`；全部 slice 为空时回退独立 NotifyRecord。
3. **对外接口启用融合**：`SendRecvBatchWriteReduce`、`SendBatchWriteReduce` 开启 fusePostNotify；`SendWriteReduce`、`SendRecvWriteReduce`（非批量回退路径）改用 `RunWriteReduceAndNotify`，并复用公共的 slice 校验函数 `CheckReduceSlicePair` 消除重复代码。
4. **算法模板切换到批量接口**：AllReduce NHR 的 RunAllGather 阶段 `SendRecvWrite` → `SendRecvBatchWrite`；AllGather Mesh1D 的 `SendRecvRead` → `SendRecvBatchRead`，统一收敛到批量传输路径。
5. **ST 桩适配**：`HcommWriteReduceWithNotifyOnThread` 桩函数由直接返回 -1（not support）改为「WriteReduce + NotifyRecord」顺序组合模拟，使 ST 可验证融合路径。

## 1.4 输出

1. 对端 hcclBuffer 中完成归约的数据。
2. 由最后一次写归约任务原子携带的 DATA_SIGNAL 完成通知（不再单独下发 notify 任务）；对端据此可安全读取归约结果。
3. 返回值 HcclResult：HCCL_SUCCESS 或相应错误码。

## 1.5 约束分析

| 支持的算子名称 | 走 write-reduce wrapper 路径的算子：AllReduce、ReduceScatter、Reduce；另有 AllGather（Mesh1D）与 AllReduce（NHR）批量接口切换 |
| -------------- | ---- |
| 支持的算法名称 | NHR（含 omnipipe NHR 变体）、Mesh1D（含 two_shot mesh chunk、meshchunk 变体） |
| 支持的芯片类型 | 继承原 aicpu 通信模板支持的芯片，无新增芯片约束 |
| 支持的展开模式 | aicpu 模板（wrapper 数据搬运路径） |
| 支持的拓扑形态 | 单机（mesh/PCIe 框内）点对点直写场景，与原路径一致 |
| 支持的调用类型 | 单算子、图模式均支持（不改变对外 API） |
| 支持的数据类型 | 继承 write-reduce 支持的数据类型（int8/int16/int32/int64/fp16/bf16/fp32 等，需满足 count × dataTypeSize = size 校验） |
| 支持的数据量 | 无变化，与原路径一致 |
| 是否支持绕路 | 不涉及（不改变路由行为） |
| 是否支持确定性计算 | 不改变 |

其他约束：

| 约束 | 说明 |
| ---- | ---- |
| HCOMM 版本依赖 | 需要 HCOMM 库导出 `HcommWriteReduceWithNotifyOnThread` 符号且批量传输支持 `HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY`（type=3） |
| 回退机制 | HCOMM 不支持批量传输时（`IsHcommBatchTransferOnThreadSupported` 为 false）回退非批量路径；融合失败时回退独立 NotifyRecord，功能语义不变 |
| slice 校验 | 非批量路径要求 src/dst slice 满足 count × DATATYPE_SIZE_TABLE[dataType] == size，否则返回 HCCL_E_INTERNAL |

# 2. SD：write-reduce 与 notify 融合设计

## 2.1 功能描述

在 `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc` 中实现四层功能：

1. **描述符融合函数** `FuseNotifyToLastWriteReduceDesc`（:102）：把批量描述符数组末尾的 `WRITE_REDUCE` 描述符改写为 `WRITE_REDUCE_WITH_NOTIFY`，搬运参数（count/dst/src/reduceOp/dataType）原样迁移并附加 notifyIdx；不满足融合条件返回 false。
2. **批量执行函数** `RunBatchTransferAndNotify`（:166）：调用 `RunBatchTransfer` 时尝试融合；融合成功则不再单独下发 notify，融合失败则补发独立 NotifyRecord。
3. **非批量执行函数** `RunWriteReduceAndNotify`（:202）：先扫描出最后一个非空 slice；循环下发普通 write-reduce，最后一个非空 slice 改用 `HcommWriteReduceWithNotifyOnThread` 融合原语；全空 slice 时单独下发 NotifyRecord。
4. **对外接口启用**：`SendRecvBatchWriteReduce`（:469）/`SendBatchWriteReduce`（:529）传入 fusePostNotify=true；`SendWriteReduce`（:516）/`SendRecvWriteReduce`（:578）重构为调用 `RunWriteReduceAndNotify`，公共校验抽取为 `CheckReduceSlicePair`（:182）。

同时调整两个算法模板的接口选择（详见 2.2），并同步修改 ST 桩。

## 2.2 流程描述

**批量融合路径（SendRecvBatchWriteReduce / SendBatchWriteReduce）：**

```text
NotifyWait(sendChannel, ACK, timeout)          // 等对端 hcclBuffer 就绪
    ↓
遍历 slice 构建批量描述符数组（跳过 size==0 的 slice，逐条校验 count×dataTypeSize==size）
    ↓
FuseNotifyToLastWriteReduceDesc(descs, NOTIFY_IDX_DATA_SIGNAL)
    ├─ 末尾为 WRITE_REDUCE → 改写为 WRITE_REDUCE_WITH_NOTIFY（融合成功）
    └─ 数组为空 / 末尾非 WRITE_REDUCE → 融合失败
    ↓
HcclHcommBatchTransferOnThread(thread, channel, descs, n)   // 一次性下发批量任务
    ↓
notifyFused == false 时：HcommChannelNotifyRecordOnThread(thread, channel, DATA_SIGNAL)  // 补发通知
    ↓
（SendRecv 场景）NotifyWait(recvChannel, DATA_SIGNAL, timeout)
```

**非批量回退路径（SendWriteReduce / SendRecvWriteReduce）：**

```text
NotifyWait(sendChannel, ACK, timeout)
    ↓
扫描得到 lastValidIdx（最后一个非空 slice）
    ↓
遍历 slice：
  ├─ size==0 → WARNING 跳过
  ├─ i != lastValidIdx → HcommWriteReduceOnThread(...)
  └─ i == lastValidIdx → HcommWriteReduceWithNotifyOnThread(..., NOTIFY_IDX_DATA_SIGNAL)  // 融合原语
    ↓
lastValidIdx < 0（全空）时：HcommChannelNotifyRecordOnThread(...)  // 补发通知
```

**批量接口收敛（本 PR 附带）：**

| 模板文件 | 修改前 | 修改后 |
| -------- | ------ | ------ |
| `src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_nhr.cc` RunAllGather（非 dmaRead 分支） | SendRecvWrite | SendRecvBatchWrite |
| `src/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.cc` RunAllGatherMesh | SendRecvRead | SendRecvBatchRead |

**批量能力探测**：`IsHcommBatchTransferOnThreadSupported()` 为 false 时，`DoSendRecvBatchTx` / `DoSendBatchTx` 直接回退非批量 fallback 函数（即上面的非批量路径）。

## 2.3 数据描述

**批量传输描述符 `HcclHcommBatchTransferDesc`**（`src/common/hcomm_dlsym/hcomm_primitives_dl.h:39`，HCCL 侧私有 ABI 兼容类型）：

| 字段 | 说明 |
| ---- | ---- |
| transType | 传输类型枚举；本需求使用 `HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE`（=1）与 `HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY`（=3） |
| transferInfo.writeReduceWithNotify.count | 归约元素个数（uint64_t） |
| transferInfo.writeReduceWithNotify.dst | 目的地址（对端 hcclBuffer） |
| transferInfo.writeReduceWithNotify.src | 源地址（本端数据） |
| transferInfo.writeReduceWithNotify.reduceOp | 归约操作（HcommReduceOp） |
| transferInfo.writeReduceWithNotify.dataType | 数据类型（HcommDataType） |
| transferInfo.writeReduceWithNotify.notifyIdx | 完成通知索引（本需求固定 NOTIFY_IDX_DATA_SIGNAL=1） |

**数据切片 `DataSlice`**：`addr_`（基地址）+ `offset_` + `size_`（字节）+ `count_`（元素个数），实际地址 = addr_ + offset_（`GetSliceAddr`）。

**通知索引常量**（`src/ops/op_common/inc/alg_param.h:80`）：`NOTIFY_IDX_ACK = 0`、`NOTIFY_IDX_DATA_SIGNAL = 1`。

## 2.4 依赖性描述

| 依赖项 | 说明 |
| ------ | ---- |
| HCOMM 融合原语 | `HcommWriteReduceWithNotifyOnThread`（int32_t (\*)(ThreadHandle, ChannelHandle, void\* dst, const void\* src, uint64_t count, HcommDataType, HcommReduceOp, uint32_t remoteNotifyIdx)），由 `src/common/hcomm_dlsym/hcomm_primitives_dl.cc` dlsym 动态加载，带支持标志（INIT_SUPPORT_FLAG）；底层保证"归约完成后再记录远端 notify"的原子语义 |
| HCOMM 批量传输接口 | `HcclHcommBatchTransferOnThread` 及 `WRITE_REDUCE_WITH_NOTIFY` 描述符类型支持；不支持时整体回退非批量路径 |
| HCCL 内部模块 | ExecTimeoutManager（超时时间）、DATATYPE_SIZE_TABLE（数据类型大小校验）、alg_data_trans_wrapper 对外接口被各算子 aicpu 模板调用 |
| 架构约束 | 跨仓调用仅经 dlsym 符号表（`src/common/hcomm_dlsym/`），无编译期对 cann/hcomm 的硬依赖，符合两仓解耦约束 |
| 下游受影响模板 | AllReduce：NHR、Mesh1D two_shot_mesh_chunk；ReduceScatter：NHR、omnipipe NHR、Mesh1D meshchunk；Reduce：NHR（以上走 write-reduce 路径）；AllGather Mesh1D、AllReduce NHR RunAllGather（批量接口切换） |

## 2.5 接口描述

| 函数原型 | bool FuseNotifyToLastWriteReduceDesc(std::vector\<HcclHcommBatchTransferDesc\>& descs, uint32_t notifyIdx) |
| ---------- | ---- |
| 函数功能   | 将描述符数组中最后一个 WRITE_REDUCE 描述符融合为 WRITE_REDUCE_WITH_NOTIFY |
| 输入说明   | descs：批量传输描述符数组；notifyIdx：完成通知索引 |
| 输出说明   | descs：末尾描述符被改写为融合类型并携带 notifyIdx |
| 返回值说明 | true：融合成功；false：数组为空或末尾描述符非 WRITE_REDUCE（不融合） |

| 函数原型 | HcclResult RunBatchTransferAndNotify(const ThreadHandle& thread, const ChannelInfo& sendChannel, const std::vector\<DataSlice\>& srcSlices, const std::vector\<DataSlice\>& dstSlices, const char\* funcName, const char\* transType, ProcessSliceFunc processSlice, bool fusePostNotify) |
| ---------- | ---- |
| 函数功能   | 批量下发传输任务并保证 DATA_SIGNAL 通知（优先融合进最后一个 reduce 描述符，失败补发独立 notify） |
| 输入说明   | thread/sendChannel：执行线程与通道；srcSlices/dstSlices：收发切片；processSlice：slice→描述符构建回调；fusePostNotify：是否尝试融合通知 |
| 输出说明   | 对端完成数据写入；对端 DATA_SIGNAL 通知被记录 |
| 返回值说明 | HCCL_SUCCESS / HCCL_E_INTERNAL 等错误码 |

| 函数原型 | HcclResult RunWriteReduceAndNotify(const ThreadHandle& thread, const ChannelInfo& sendChannel, const std::vector\<DataSlice\>& srcSlices, const std::vector\<DataSlice\>& dstSlices, HcclDataType dataType, HcclReduceOp reduceOp, const char\* funcName) |
| ---------- | ---- |
| 函数功能   | 非批量下发 write-reduce：最后一个非空 slice 用融合原语，其余用普通 write-reduce；全空时补发独立 notify |
| 输入说明   | thread/sendChannel：执行线程与通道；srcSlices/dstSlices：收发切片；dataType/reduceOp：数据类型与归约操作 |
| 输出说明   | 对端完成归约写入；对端 DATA_SIGNAL 通知被记录 |
| 返回值说明 | HCCL_SUCCESS / HCCL_E_INTERNAL（slice 校验失败）等错误码 |

| 函数原型 | int32_t HcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void\* dst, const void\* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx) |
| ---------- | ---- |
| 函数功能   | HCOMM 提供的融合原语：归约写入对端 dst 完成后记录对端 remoteNotifyIdx 通知（dlsym 加载，HCCL 侧不实现） |
| 输入说明   | thread/channel：线程与通道；dst/src：目的与源地址；count：元素个数；dataType/reduceOp：数据类型与归约操作；remoteNotifyIdx：远端通知索引 |
| 输出说明   | 对端完成归约写入并记录远端通知 |
| 返回值说明 | 0 成功；非 0 失败（ST 桩中模拟为 WriteReduce + NotifyRecord 顺序组合） |

对外接口 `SendWriteReduce` / `SendRecvWriteReduce` / `SendBatchWriteReduce` / `SendRecvBatchWriteReduce`（`alg_data_trans_wrapper.h`）函数签名不变，仅内部实现变更，对算子模板层透明。

## 2.6 约束分析

| 支持的算子名称 | AllReduce、ReduceScatter、Reduce（write-reduce 路径）；AllGather（Mesh1D）、AllReduce（NHR）批量接口切换 |
| -------------- | ---- |
| 支持的算法名称 | NHR（含 omnipipe 变体）、Mesh1D（含 two_shot mesh chunk、meshchunk 变体） |
| 支持的芯片类型 | 继承原 aicpu 通信模板支持的芯片 |
| 支持的展开模式 | aicpu 模板 |
| 支持的拓扑形态 | 单机（mesh/PCIe 框内）点对点直写场景 |
| 支持的调用类型 | 单算子、图模式 |
| 支持的数据类型 | 继承 write-reduce 支持类型（int8/int16/int32/int64/fp16/bf16/fp32 等） |
| 支持的数据量 | 无变化 |
| 是否支持绕路 | 不涉及 |
| 是否支持确定性计算 | 不改变 |

其他约束：

| 约束 | 说明 |
| ---- | ---- |
| 融合条件 | 仅批量描述符数组末尾为 WRITE_REDUCE 时融合；非批量路径仅最后一个非空 slice 用融合原语 |
| 融合失败回退 | 描述符为空 / 末尾非 WRITE_REDUCE / 全空 slice：补发独立 NotifyRecord，语义与原实现等价 |
| slice 校验 | count × DATATYPE_SIZE_TABLE[dataType] 必须等于 size，否则报 HCCL_E_INTERNAL |
| HCOMM 能力 | 依赖 `HcommWriteReduceWithNotifyOnThread` 符号与批量传输支持；不支持时回退原路径 |

## 2.7 DFX设计

| 校验内容 | 级别 | 搜索内容 |
| -------- | ---- | -------- |
| 融合成功（验证融合路径生效） | DEBUG | `FuseNotifyToLastWriteReduceDesc: fused last descriptor to WRITE_REDUCE_WITH_NOTIFY` |
| 批量传输下发成功 | DEBUG | `BATCH_WRITE_REDUCE] totalSliceNum` |
| slice 参数不一致（异常场景） | ERROR | `is not mate to slice size`（返回 HCCL_E_INTERNAL） |

## 2.8 资料描述

不涉及（无对外 API、环境变量、使用方式变更，不需要更新用户文档）。

## 2.9 性能&&质量

- **性能**：每个 write-reduce batch 少下发一条独立 notify 任务，降低 host→device 任务下发开销；批量接口收敛（AllGather Mesh1D / AllReduce NHR RunAllGather）进一步减少逐 slice 下发次数。
- **质量（正确性修复）**：融合原语由 HCOMM 底层保证"归约写入完成后才记录通知"，消除原两任务机制下 notify 先于数据生效导致对端读到旧数据的竞态；所有融合失败场景均有独立 NotifyRecord 回退，功能语义与原实现等价。
- **测试**：ST 桩已适配融合原语（WriteReduce + NotifyRecord 顺序模拟），可通过 ST 回归验证；建议覆盖多 slice、含空 slice、count×size 不匹配、dmaRead/非 dmaRead 分支等场景。

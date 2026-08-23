# HCCL WriteReduceWithNotify 改造设计

## 1. 背景

HCCL 正常 WriteReduce wrapper 改造前采用以下方式通知远端数据写入完成：

```text
HcommWriteReduceOnThread
    -> HcommChannelNotifyRecordOnThread(DATA_SIGNAL)
```

WriteReduce 和 NotifyRecord 在底层是两个独立硬件操作。即使两个任务在同一条流上保序下发，也只能保证下发顺序，不能保证 WriteReduce 的数据操作一定在 NotifyRecord 到达远端前完成。远端收到 `DATA_SIGNAL` 后可能提前使用尚未完成规约的数据，形成正确性风险。

HCOMM 已提供融合接口：

```cpp
HcommWriteReduceWithNotifyOnThread(..., remoteNotifyIdx)
```

该接口将 WriteReduce 和 Notify 融合为一个操作，能够保证本次 WriteReduce 完成后再通知远端。`HcommBatchTransferOnThread` 也已支持 `HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY`。

## 2. 改造目标

1. HCCL 正常 WriteReduce wrapper 不再使用独立的 `WriteReduce + NotifyRecord` 作为数据完成通知。
2. Normal 路径只在最后一个有效 slice 调用 `HcommWriteReduceWithNotifyOnThread`，此前有效 slice 保持普通 WriteReduce。
3. Batch 路径仅将最后一个有效 descriptor 改为 `WRITE_REDUCE_WITH_NOTIFY`。
4. 保持原有 ACK、反向 DATA_SIGNAL wait、超时和错误处理语义不变。
5. 保持 HCCL 与 HCOMM 的 dlsym 解耦，不引入 HCOMM 私有头文件或编译期硬依赖。
6. 每轮 WriteReduce 只产生一次 `DATA_SIGNAL`；全零 slice 继续使用独立通知，避免对端永久等待。

## 3. 改造前实现

主要代码位于：

```text
src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc
```

涉及接口：

- `SendWriteReduce`
- `SendRecvWriteReduce`
- `SendBatchWriteReduce`
- `SendRecvBatchWriteReduce`

改造前时序如下：

```text
Wait ACK
    -> WriteReduce × N
    -> ChannelNotifyRecord(DATA_SIGNAL)
```

其中：

- Normal 路径逐个调用 `HcommWriteReduceOnThread`，循环结束后单独 record。
- Batch 路径生成多个 `WRITE_REDUCE` descriptor，batch 完成下发后单独 record。
- `RecvWriteReduce` 仅负责发送 ACK 和等待 DATA_SIGNAL，不执行 WriteReduce，无需修改。

## 4. 当前实现时序

### 4.1 单个有效 slice

```text
Wait ACK
    -> WriteReduceWithNotify(DATA_SIGNAL)
```

不再额外调用 `HcommChannelNotifyRecordOnThread(DATA_SIGNAL)`。

### 4.2 Normal 多个有效 slice

```text
Wait ACK
    -> WriteReduce(slice 0)
    -> WriteReduce(slice 1)
    -> ...
    -> WriteReduceWithNotify(last valid slice, DATA_SIGNAL)
```

只有最后一个有效 slice 携带 notify，前面的有效 slice 继续调用 `HcommWriteReduceOnThread`。该实现依赖 HCOMM 对同一 channel 上最终融合 WQE 的 fence/强序语义，相关接口契约见第 11 节待讨论问题。

### 4.3 Batch 多个有效 slice

```text
Wait ACK
    -> BatchTransfer[
           WriteReduce,
           WriteReduce,
           ...,
           WriteReduceWithNotify(DATA_SIGNAL)
       ]
```

只有最后一个有效 descriptor 携带 notify。HCOMM batch 实现会对最后一个 WQE 设置 strong/completion order，从而保证前面的 WriteReduce 完成后再执行最终融合通知。

禁止让每个 slice 都携带 notify，否则远端可能在第一个 slice 完成后被提前唤醒。

### 4.4 全部 slice 长度为零

没有数据操作可以承载融合通知，此时保留一次独立通知：

```text
Wait ACK
    -> ChannelNotifyRecord(DATA_SIGNAL)
```

否则远端会永久等待 DATA_SIGNAL。

### 4.5 SendRecv 后同步

`SendRecvWriteReduce` 和 `SendRecvBatchWriteReduce` 在发送端完成信号之后，仍等待反向通道的 `DATA_SIGNAL`。本次改造只融合发送侧 WriteReduce 与发送侧完成通知，不改变双向同步协议。

## 5. 详细设计

### 5.1 完整重建融合 descriptor

当前实现在匿名命名空间增加：

```cpp
bool FuseNotifyToLastWriteReduceDesc(
    std::vector<HcclHcommBatchTransferDesc> &descs,
    uint32_t notifyIdx);
```

函数处理过程如下：

1. descriptor 为空时返回 `false`；
2. 最后一个 descriptor 不是 `WRITE_REDUCE` 时返回 `false`；
3. 先读取最后一个普通 reduce descriptor 的 `count/dst/src/reduceOp/dataType`；
4. 零初始化新的 `HcclHcommBatchTransferDesc`；
5. 通过 `writeReduceWithNotify` 成员完整写入所有字段和 `notifyIdx`；
6. 用新 descriptor 替换 `descs.back()`。

这里不能只修改 `transType` 后写入另一个 union 成员的 `notifyIdx`。完整重建能够避免依赖 inactive union 成员残留字节，也避免将来两个结构公共前缀发生变化时静默传错参数。

该问题的发现与修复过程是一次典型的“人审 AI”：AI 首版采用直接切换 `transType` 并设置 `notifyIdx` 的写法；人在代码 Review 中指出 C++14 union active member 切换涉及对象生命周期，不能跨 inactive member 复用残留字段。AI 随后独立核对 C++14 union 语义、内存布局和 ABI 演进风险，确认审查意见成立，再按上述方案完整重建 descriptor。该风险不是 AI 自审发现的。

### 5.2 最后一个有效 slice 的识别

wrapper 会跳过 `size_ == 0` 的 slice，因此不能直接以数组最后一个元素作为融合对象。

- Normal 路径先扫描 `srcSlices`，记录最后一个 `size_ != 0` 的下标 `lastValidIdx`；
- Batch 路径先收集所有有效普通 descriptor，再处理 `transferDescs.back()`；
- 没有有效 slice 时，`lastValidIdx == -1` 或 descriptor 为空，最终保留一次独立 DATA_SIGNAL。

该处理能够覆盖尾部零 slice、中间零 slice 和全零 slice。

### 5.3 Batch 通用模板调整

当前实现在以下内部模板增加 `bool fusePostNotify = false` 参数：

- `RunBatchTransfer`
- `DoSendBatchTx`
- `DoSendRecvBatchTx`

同时新增 `RunBatchTransferAndNotify`，统一封装“执行 BatchTransfer，并在 notify 未融合时发送独立 DATA_SIGNAL”的公共控制流，避免 `DoSendBatchTx` 和 `DoSendRecvBatchTx` 重复实现。

`RunBatchTransfer` 还接收 `notifyIdx` 和 `bool *notifyFused`：

```text
先构造所有有效 descriptor
    -> fusePostNotify 为 true 时尝试融合最后一个 descriptor
    -> 调用 HcclHcommBatchTransferOnThread
    -> 将是否融合通过 notifyFused 返回
```

`RunBatchTransferAndNotify` 仅在 `notifyFused == false` 时发送独立 DATA_SIGNAL；两类入口仍分别保留自己的 ACK 和反向 wait 协议。

| 调用方 | fusePostNotify | Batch 后独立 DATA_SIGNAL |
|---|---:|---|
| `SendBatchWrite` | false | 是 |
| `SendRecvBatchWrite` | false | 是 |
| `SendBatchWriteReduce` | true | 仅全零或 descriptor 转换未成功时 |
| `SendRecvBatchWriteReduce` | true | 仅全零或 descriptor 转换未成功时 |

普通 Write 本次不改造成 WriteWithNotify。

### 5.4 Normal 路径

当前 `SendWriteReduce` 和 `SendRecvWriteReduce` 的实现为：

1. 原有 ACK wait 保持不变；
2. 两个入口调用公共内部函数 `RunWriteReduceAndNotify`；
3. 公共函数扫描获得 `lastValidIdx`，并通过 `CheckReduceSlicePair` 统一校验源、目标 slice；
4. 遍历 slice：
   - 零长度 slice 继续跳过；
   - 非最后有效 slice 调用 `HcommWriteReduceOnThread`；
   - 最后有效 slice 调用 `HcommWriteReduceWithNotifyOnThread(..., NOTIFY_IDX_DATA_SIGNAL)`；
5. 仅当 `lastValidIdx < 0`，即没有有效 slice 时发送独立 DATA_SIGNAL。

融合接口或此前任意 WriteReduce 返回失败时立即向上传递错误，不补发独立 DATA_SIGNAL。

### 5.5 SendRecv 后同步

`SendRecvWriteReduce` 和 `SendRecvBatchWriteReduce` 在发送完成信号后，还会等待反向通道的 `DATA_SIGNAL`：

```cpp
HcommChannelNotifyWaitOnThread(
    thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout);
```

该 wait 必须保留。改造只融合本端 WriteReduce 与向发送对端的 DATA_SIGNAL，不改变双向同步协议。

## 6. dlsym 与版本前提

已确认 `HcommWriteReduceWithNotifyOnThread` 与 `HcommWriteReduceOnThread` 属于同一时间基线，本次不增加新的接口可用性判断，也不新增资源上下文 capability 字段。

HCCL 当前动态加载层已经包含：

- `HcommWriteReduceWithNotifyOnThread` weak 定义；
- `HcommPrimitivesDlInit` 中的符号初始化；
- Batch descriptor 的私有 ABI 兼容定义，其中已经包含 `WRITE_REDUCE_WITH_NOTIFY`。

当前补丁不修改 `src/common/hcomm_dlsym/`，也不直接 include HCOMM 私有头文件，继续满足 HCCL/HCOMM 通过 dlsym 解耦的架构约束。

本设计的运行前提是目标 HCOMM 和目标设备均支持融合接口及 Batch 融合 descriptor；如后续产品重新提出跨版本或跨设备兼容要求，需要单独设计能力协商，不在当前改造范围内。

## 7. 文件级改动清单

### 7.1 当前已修改

`src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc`

- 增加 `FuseNotifyToLastWriteReduceDesc`；
- 增加 `RunBatchTransferAndNotify`，收敛 BatchTransfer 后通知逻辑；
- 增加 `CheckReduceSlicePair` 和 `RunWriteReduceAndNotify`，收敛 Normal WriteReduce 的校验、遍历和通知逻辑；
- 扩展 Batch 通用模板的融合通知控制；
- 修改四个 WriteReduce 发送接口；
- 保留零长度、ACK、反向 wait 和错误传播语义。

`test/st/algorithm/utils/src/hccl_proxy/hccl_stub.cc`

- `HcommWriteReduceWithNotifyOnThread` 已实现为：先复用 `HcommWriteReduceOnThread` 生成 WriteReduce 模拟任务，成功后调用 `HcommChannelNotifyRecordOnThread` 生成 Post 模拟任务；
- 保留首个模拟任务失败时立即返回、不追加 Post 的错误传播语义。

### 7.2 尚待补充

- 当前 `HcommIsSupportHcommBatchTransferOnThread` 仍固定返回 `false`；
- 模型 ST 只进入 Normal fallback，Batch descriptor 仍需 wrapper UT 或支持 BatchTransfer 的模型测试；
- A5 上板测试仍需验证真实硬件完成时序。

当前不需要修改：

```text
src/common/hcomm_dlsym/
src/ops/op_common/inc/alg_param.h
src/ops/op_common/op_common.cc
src/ops/op_common/template/aicpu/kernel_launch.cc
```

### 7.3 可选扩展范围

仓库中还存在两处 experimental 直调组合：

```text
experimental/ops/reduce_scatter/birs/template/reduce_scatter_birs.cc
experimental/ops/reduce_scatter/birs/template/reduce_scatter_birs_inter.cc
```

两处均为单次 `HcommWriteReduceOnThread + HcommChannelNotifyRecordOnThread`，存在同类风险。如果需求范围是整个 HCCL 仓，应一并替换；如果仅针对正常商用 wrapper，可拆分为独立补丁。

CCU 的 `ccu::WriteReduce`、ReadReduce 和普通 Write 不在本次改造范围内。

## 8. 异常处理

1. 融合接口返回失败时立即向上传递错误，不再补发独立 DATA_SIGNAL。
2. BatchTransfer 返回失败时立即返回，不做旧路径重试。
3. 参数校验逻辑保持不变，包括：
   - `count * datatypeSize == size`；
   - 源、目标 slice 的 count/size 匹配；
   - 空 slice 跳过；
   - timeout 获取和 wait 错误传播。
4. 融合通知使用现有 `NOTIFY_IDX_DATA_SIGNAL`，不改变 notify 索引协议。

## 9. 测试方案

### 9.1 当前测试状态

当前测试代码和执行结果如下：

1. 模型 ST 中 `HcommWriteReduceWithNotifyOnThread` 已完成模拟实现：先生成 WriteReduce 模拟任务，成功后追加 NotifyRecord/Post 模拟任务；
2. 相关目标编译通过，模型 ST 已执行并 20/20 通过；
3. trace 日志可见同一 rank/remoteRank 上连续 WriteReduce 后仅跟一次 Post。例如可将 `rankIdx:16`、`remoteRank=15` 的一个任务段压缩表示为 `WriteReduce × 8 -> Post × 1`；
4. `HcommIsSupportHcommBatchTransferOnThread` 仍固定返回 `false`，因此上述 ST 只覆盖 Normal fallback，无法验证 Batch descriptor 是否正确转换。

结论：融合接口桩和 Normal fallback 已获得模型 ST 与 trace 证据；Batch descriptor 仍需独立自动化覆盖，A5 上板仍需验证真实硬件完成时序。

### 9.2 Wrapper 单元测试

至少覆盖：

1. 单有效 slice：
   - 调用一次 `HcommWriteReduceWithNotifyOnThread`；
   - 不再调用独立 DATA_SIGNAL record。

2. 多有效 slice：
   - Normal 路径前 N-1 个调用普通 WriteReduce，最后一个调用融合接口；
   - Normal 路径总共只发送一次 DATA_SIGNAL；
   - descriptor 数量正确；
   - 只有最后一个 descriptor 为 `WRITE_REDUCE_WITH_NOTIFY`；
   - `notifyIdx` 为 `NOTIFY_IDX_DATA_SIGNAL`。

3. 中间或尾部零长度 slice：
   - 最后一个有效 descriptor 携带 notify，而不是数组最后一个元素；
   - Normal 路径的 `lastValidIdx` 指向最后一个非零 slice。

4. 全零 slice：
   - 不调用融合数据接口；
   - 独立发送一次 DATA_SIGNAL。

5. SendRecv：
   - 发送侧通知已融合；
   - 反向 DATA_SIGNAL wait 保留。

6. 错误场景：
   - count/size 不匹配；
   - 融合接口失败；
   - BatchTransfer 失败。

7. Batch 构造约束：
   - 同一次 BatchTransfer 中的 descriptor 操作类型保持一致；
   - 非空 WriteReduce Batch 的最后一个 descriptor 必然可融合；
   - 全零 slice 导致的空 Batch 使用独立 DATA_SIGNAL。

### 9.3 模型 ST

运行涉及 WriteReduce wrapper 的算子和算法：

- AllReduce：NHR、Mesh/TwoShot 等规约阶段；
- ReduceScatter：NHR、Mesh、Omnipipe 等；
- Reduce：NHR。

重点覆盖 `nSlices > 1` 的场景，验证数据结果和同步图不发生变化。

模型 stub 的融合接口不能只返回 `HCCL_SUCCESS`，还必须模拟 WriteReduce 数据任务和远端 DATA_SIGNAL，否则对端 wait 无法反映真实同步协议。

本次执行结果为 20/20 通过。除 pass/fail 外，还应检查 trace 中的任务序列，确认连续 WriteReduce 后只出现一次 Post，避免重复通知或漏通知。

### 9.4 A5 上板验证

模拟器无法真实模拟硬件操作完成乱序，因此必须进行 A5 上板验证：

1. BatchTransfer 开启；
2. 多 slice、大数据量；
3. 多轮重复执行；
4. AllReduce、ReduceScatter、Reduce 结果校验；
5. 检查超时、提前唤醒和数据偶现错误；
6. 对比改造前后的任务数量和性能，确认融合操作没有引入明显回退。

## 10. 当前验证状态与后续验证

### 10.1 已完成

- 当前 CANN 9.1 AICPU 编译参数下语法检查通过；
- `git diff --check` 通过；
- 相关模型库和 ST 目标编译通过；
- 模型 ST 20/20 通过；
- trace 日志确认连续 WriteReduce 后仅跟一次 Post，Normal fallback 的任务编排符合预期；
- ST 融合接口桩已完成，不再是固定失败打桩；
- 已对照 HCOMM A5 单次融合和 Batch 融合实现检查字段布局、notifyIdx 和最后 WQE 的强序配置。

### 10.2 尚未完成

- 未执行完整 `--pkg` 构建；
- 未执行 wrapper 专项 UT；
- 模型 ST 未覆盖 Batch descriptor 路径；
- 未执行 A5 上板验证；
- 当前环境没有可用的 clang-format 命令，未完成格式工具检查。

### 10.3 建议命令

```bash
bash build.sh --pkg
bash build.sh -u
bash build.sh -s
```

优先执行 wrapper 相关 UT 和 WriteReduce 相关 ST，再执行完整构建。代码修改后使用仓库 `.clang-format` 规则格式化，并通过静态检查。

## 11. 已确认事项与待讨论问题

### 11.1 已确认事项

1. `HcommWriteReduceWithNotifyOnThread` 与 `HcommWriteReduceOnThread` 属于同一时间基线，不需要新增接口可用性判断。
2. 一轮 wrapper 只能产生一次 DATA_SIGNAL，不能让每个 slice 都携带 notify。
3. 全零 slice 必须保留独立 DATA_SIGNAL。
4. SendRecv 的反向 DATA_SIGNAL wait 必须保留。
5. Batch 融合 descriptor 必须完整重建，不能依赖 inactive union 成员残留值。
6. 本次只修改 WriteReduce 发送路径，不扩展普通 Write、ReadReduce 或 CCU WriteReduce。

### 11.2 Batch 融合失败分支说明（已确认）

本节是对当前代码构造不变量的静态说明，不是本次人机协作过程中实际发生的争论、故障或 Review 案例。

当前 wrapper 中，每次 `HcclHcommBatchTransferOnThread` 调用生成的 descriptor 操作类型是一致的：

- SendBatchWrite/SendRecvBatchWrite 全部为 `WRITE`；
- SendBatchWriteReduce/SendRecvBatchWriteReduce 全部为 `WRITE_REDUCE`；
- Read 路径全部为 `READ`；
- ReadReduce 路径全部为 `READ_REDUCE`。

只有两个 WriteReduce Batch 调用方会传入 `fusePostNotify=true`。因此，只要`transferDescs` 非空，其最后一个 descriptor 就必然是 `WRITE_REDUCE`，
`FuseNotifyToLastWriteReduceDesc` 必然能将其重建为 `WRITE_REDUCE_WITH_NOTIFY`。该融合过程只是本地 descriptor 字段转换，不调用 HCOMM，也没有资源申请，不存在运行时偶发转换失败。

当前可达的 `notifyFused=false` 场景只有所有 slice 均为零，此时 `transferDescs` 为空，没有数据操作需要等待，独立下发 DATA_SIGNAL 是正确行为。“非空 Batch 但最后一个 descriptor 不是 WriteReduce”只属于未来修改破坏当前构造不变量后的防御性异常，不是本次实现中的实际待讨论场景，当前不需要设计报错或降级策略。如后续引入混合类型 Batch，则需要重新检视融合规则；不应在无时序保证的情况下静默降级为 `WriteReduce + NotifyRecord`。

### 11.3 待讨论问题

#### 问题一：Normal 多 slice 的跨调用 fence 是否属于 HCOMM 接口契约

当前 Normal 实现为前 N-1 个普通 WriteReduce、最后一个 WriteReduceWithNotify。HCOMM 当前 A5 内部 `SqeConfigLite` 默认启用 place order、completion order 和 fence，因此从现有实现看能够约束此前同 channel WQE；但公开接口描述主要保证融合操作自身的 WriteReduce 与 Notify。

需要 HCOMM/SE 确认：最终融合调用是否正式承诺等待此前同 channel 的独立 WriteReduce 完成。如果该保证只属于当前实现而不是接口契约，需要决定是否补充接口说明或改为其他编排方式。

#### 问题二：测试补充方式

需要确定由哪类测试负责检查 Batch descriptor：

- 新增 wrapper UT，mock `HcclHcommBatchTransferOnThread` 并检查 descriptor；
- 或扩展模型 stub，使其支持 BatchTransfer；
- A5 板上 ST 仍然必须保留，因为模型无法复现真实硬件完成乱序。

#### 问题三：日志是否区分融合类型

当前最后一个 Normal slice 的 HCCL trace 仍记录为 `WRITE_REDUCE`，Batch summary 仍记录为 `BATCH_WRITE_REDUCE`，而 HCOMM profiling 会显示融合任务。

需要决定是否将最后一个有效 slice/Batch 的日志标记为 `WRITE_REDUCE_WITH_NOTIFY`，便于问题定位。该问题不影响功能正确性。

#### 问题四：experimental BIRS 是否纳入当前补丁

`experimental/ops/reduce_scatter/birs/` 仍有两处直接使用普通 WriteReduce 加独立 DATA_SIGNAL。需要明确本需求只覆盖正式商用 wrapper，还是覆盖整个 HCCL 仓。

## 12. 最终结论

本次改造应以“一轮 WriteReduce 只产生一次、且严格发生在全部有效数据操作完成后的 DATA_SIGNAL”为设计原则：

- 单 slice 使用 `HcommWriteReduceWithNotifyOnThread`；
- Normal 多 slice 使用前 N-1 个普通 WriteReduce、最后一个有效 slice 使用融合接口；
- Batch 多 slice 将最后一个有效 descriptor 完整重建为 `WRITE_REDUCE_WITH_NOTIFY`；
- 全零 slice 保留独立通知；
- 当前每个 Batch 内的操作类型一致，非空 WriteReduce Batch 不存在无法融合的可达分支；
- SendRecv 的反向 wait 保持不变；
- 不新增 capability 判断，继续沿用现有 dlsym 解耦结构。

当前生产代码主逻辑和模型 ST 融合接口桩已完成，union descriptor 构造问题已修正，模型 ST 20/20 通过并有 trace 任务序列佐证。后续仍需完成 Batch descriptor 自动化覆盖和 A5 上板验证，并确认 Normal 多 slice 的跨调用 fence 契约及 experimental 范围。

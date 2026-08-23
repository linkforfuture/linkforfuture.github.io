# Checker 对 ReduceScatter MeshChunk 屏障校验失败的分析

## 背景

`InsTempReduceScatterMesh1DMeshChunk` 的步间屏障原本按 `PreSyncInterThreads -> PostSyncInterThreads` 调用。该顺序存在半屏障风险：slave 线程完成 `PostSync` 的 Record 后即可继续下一步，未必等待其余 slave 完成。

HCCL 已将步间顺序调整为 `PostSyncInterThreads -> PreSyncInterThreads`，与 hcomm legacy 的 `PostSyncInterQueues -> PreSyncInterQueues` 保持一致。修复后，8 卡 FP32、512MB ReduceScatter 的 runner 语义校验与 `hccl_test` 数据校验均成功，但 checker 仍报 `ErrorCode: 302`。

最初假设是 A5 路径的 `LocalNotifyRecord/LocalNotifyWait` 未被 proxy 采集。通过本次任务计数和 DAG 路径追踪确认，该假设不成立：本地通知任务已经进入 DB/DAG；真正的问题是 checker 对重复 local `notifyId` 的配对发生了代际错位。

## 算法侧本地屏障

`PreSyncInterThreads` 对每个 slave 执行：

```text
main Record -> slave Wait
```

`PostSyncInterThreads` 对每个 slave 执行：

```text
slave Record -> main Wait
```

对于 8 卡 MeshChunk，`templateRankSize = 8`，每个屏障有 6 个 slave。单次 `MeshChunk::KernelRun` 的本地同步数量为：

```text
初始 PreSync:       6 对
6 个步间 Post+Pre:  6 x 2 x 6 = 72 对
结束 PostSync:      6 对
合计:               84 对 Record/Wait
```

当前测试的每个 rank、每个 `syncIter` 有两次 `MeshChunk::KernelRun`，故算法主体产生 168 次 Record 和 168 次 Wait；运行日志实际观测到 169 次 `HcommThreadNotifyRecordOnThread` 和 169 次 `HcommThreadNotifyWaitOnThread`，多出的 1 对来自算子外围辅助同步。

## 任务数对比：节点并未缺失

测试归档：`hccl_vm_install/archive/20260709_205842`。

| 层级                                             | 每 rank、每 syncIter 的 Record / Wait 数 |
| ------------------------------------------------ | ---------------------------------------: |
| Hcomm 实际 `HcommThreadNotify*OnThread` 调用     |                                169 / 169 |
| SQLite DB 中全部 `NOTIFY_RECORD` / `NOTIFY_WAIT` |                                366 / 366 |
| Checker DAG 中全部 Record / Wait 节点            |                                366 / 366 |
| Checker DAG 中 local Record / Wait 节点          |                                170 / 170 |
| Checker DAG 中跨 rank Record / Wait 节点         |                                196 / 196 |

DB 有两个 `syncIter`，每轮共 2928 个 `NOTIFY_RECORD`、2928 个 `NOTIFY_WAIT`；checker 对每轮加载 6672 个业务任务并构建相同数量的业务 DAG 节点。每 rank 有 170 条 local Record->Wait 边和 196 条跨 rank Record->Wait 边。

因此，不能再用“算法 LocalNotify 调用数大于 DAG local 节点数”解释本问题。修改 `PostSync -> PreSync` 只改变任务顺序和依赖关系，不会增加本地通知的数量；local edge 数量在修复前后保持不变是预期行为。

## ErrorCode 302 的直接路径

Checker 报告两个对 rank 0 CCL buffer `[0x0, 0x100000)` 的写操作可能并行：

```text
node 1818: rank 2, stream 73, TaskReduce
node 2669: rank 3, stream 97, TaskReduce
```

跨 rank 的 Notify 边本身匹配正确：

```text
1818
  -> 1819  rank2 stream73 Record, notify 898, rank2 -> rank0
  -> 307   rank0 stream66 Wait,   notify 898, rank2 -> rank0

408
  -> 2668  rank3 stream97 Wait,   notify 885, rank0 -> rank3
  -> 2669  rank3 stream97 TaskReduce
```

但从 node 1818 在 rank 0 内经 local Notify 建立的实际 DAG 路径是：

```text
1819 (rank2 -> rank0, notify 898)
  -> 307  (rank0 stream66 Wait 898)
  -> 308  (rank0 stream66 Local Wait 404)
  -> 309  (rank0 stream66 Local Record 360)
  -> 159  (rank0 stream65 Local Wait 360)
  -> 171  (rank0 stream65 Local Record 413)
  -> 413  (rank0 stream67 Local Wait 413)
  -> 415  (rank0 stream67 Record, notify 885, rank0 -> rank3)
  -> 2685 (rank3 stream97 Wait 885)
```

目标 node 2669 所在链则由更早一代的 `notifyId=413` 驱动：

```text
154 (rank0 stream65 Local Record 413)
  -> 406 (rank0 stream67 Local Wait 413)
  -> 407 (rank0 stream67 Local Record 361)
  -> 408 (rank0 stream67 Record, notify 885, rank0 -> rank3)
  -> 2668
  -> 2669
```

即，node 1818 的路径到达了下一代 `415 -> 2685`，而非约束冲突任务所需的 `408 -> 2668 -> 2669`。`1818 -> 2669` 在当前 DAG 中不可达，故 MemConflict 将其归类为 `parallelCandidatePairs=1` 并报 `ErrorCode: 302`。

## 根因：local Notify 匹配键不完整

Checker 的 `AddLocalNotifyEdges` 将已见的 local Record 放入 FIFO 队列，并且只以 `notifyId` 判断 Record/Wait 是否匹配：

```cpp
bool IsNotifyIdPeer(const AicpuNotify &recordNotify, const AicpuNotify &waitNotify)
{
    return recordNotify.notifyId == waitNotify.notifyId;
}
```

MeshChunk 在多个轮次、多条 slave stream 上重复使用 local `notifyId`，例如上例中的 `413`。仅凭 `notifyId` FIFO 匹配无法区分哪一代 Record 应唤醒哪一代 Wait，最终将 node 171 配给 node 413，而冲突路径所需的是前一代 `154 -> 406`。

因此，本问题是 **local Notify 的代际/目标 stream 匹配错误**，而非 local Notify 节点缺失。跨 rank Notify 的目标 rank 与配对在冲突链上是正确的。

## 相关实现风险

设备 SQE parser 中 `ParseDavidNotifySqe` 将 `dstRankId` 固定写为 0：

```cpp
taskMeta.taskData.notify.dstRankId = 0;  // 目前暂不区分notify的目的rank，默认为0
```

该实现会丢失一部分 Notify 对端信息；当前 checker 会再依据 `notifyId` 推断 peer。该信息缺失会放大重复 `notifyId` 场景中的误匹配风险，但本次 1818/2669 路径表明首要问题仍是 local stream/代际没有进入匹配键。

## 修复建议

### 方案 A：补充 local Notify 元数据并精确匹配（推荐）

Proxy 在采集 local Notify 时记录源/目标 thread 或 stream；checker 使用如下键匹配：

```text
(rankId, notifyId, recordStreamId, waitStreamId, generation)
```

其中 `generation` 可按相同 `(rankId, notifyId, recordStreamId, waitStreamId)` 的出现序号生成。这样既能处理同一 notifyId 的循环复用，也不会混淆不同 slave stream。

### 方案 B：为 local Notify 生成唯一 ID

在 proxy 侧为 local Notify 构造包含 rank、源 stream、目标 stream 和代次的 synthetic notifyId。这样可复用 checker 现有逻辑，但仍需避免 `StorageManager` 对 local Notify 进行跨 rank peer 推断。

### 方案 C：checker 侧按任务模式推断

根据 MeshChunk 的 stream/task 序列补充语义边。该方案侵入较小，但算法模式耦合强，不作为首选。

## 验证方法

修复后检查以下条件：

1. `node 1818` 可以到达 `node 2669`。
2. rank 0 上 `notifyId=413` 的 local 边按正确代次配对，冲突链应经过 `154 -> 406 -> 408`，而不是偏移到 `171 -> 413 -> 415`。
3. `parallelCandidatePairs` 从 1 降为 0。
4. Checker 从 `Checker failed` 变为 `Checker Success`。


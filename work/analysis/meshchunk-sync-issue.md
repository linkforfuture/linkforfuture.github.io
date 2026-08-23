# HCCL MeshChunk难复现同步问题定位与修复举证材料

## 1. 举证信息

| 项目 | 内容 |
|---|---|
| 能力要求 | 能够定位难以复现的问题 |
| 问题模块 | AICPU MeshChunk多线程同步 |
| 涉及算子 | AllReduce、ReduceScatter |
| Commit | `f455ca50d8ac55bd934515eecc455820dc5ec337` |
| MR | !1913 |
| 提交标题 | `fix meshchunk post/prec` |
| 作者 | weixin_43960572（yuantangzhi@huawei.com） |
| 合入时间 | 2026-07-10 |
| 修改规模 | 2个文件，5行新增、5行删除 |

## 2. 问题背景

AllReduce和ReduceScatter的AICPU MeshChunk算法将通信任务分配到主线程和多个从线程并行执行。每轮Chunk处理结束后，各线程需要通过Notify完成步间同步：

1. `PostSync`：从线程上报当前轮完成，主线程等待全部从线程。
2. `PreSync`：主线程确认全部完成后，统一通知从线程进入下一轮。

该过程既要保证主从线程的执行顺序，也要保证外层同步和MeshChunk内部同步使用不同的Notify资源。任意一处关系错误，都可能造成跨轮执行、Notify误匹配或同步等待异常。

## 3. 问题为何难以复现

该问题仅在以下条件组合下进入风险路径：

- 算法选择AllReduce或ReduceScatter MeshChunk模板。
- 使用AICPU多线程执行，`threadNum_ > 1`。
- 当前不是最后一个通信Step，需要进行步间同步。
- 不同从线程的任务完成时间存在差异。

问题是否暴露受Device负载、Channel链路时延、数据分片大小和线程调度顺序影响。线程完成时间接近时，错误同步顺序可能不产生可见异常；当某个从线程明显快于其他线程时，才可能跨轮执行并放大问题。

因此，该问题具有典型的异步时序特征：

- 相同用例重复执行不一定每次触发。
- 小数据量或低负载场景可能长期正常。
- 负载波动或链路时延差异增大后更容易暴露。
- 最终可能表现为后续同步等待失败或超时，报错位置与真正破坏时序的位置存在距离。

## 4. 定位方法

针对偶现同步问题，不能只分析最终报错点，需要还原不同Thread上的任务队列及Notify依赖关系。本次定位重点检查：

1. 每轮通信任务提交到哪些主、从线程。
2. 主线程何时等待所有从线程完成。
3. 从线程何时获得进入下一轮的许可。
4. 外层同步与MeshChunk内部同步分别使用哪些Notify索引。
5. Notify申请数量、分组范围和实际访问索引是否一致。

通过把主线程和从线程的Record/Wait操作展开成时序关系，最终识别出同步顺序和Notify索引两个相互独立的问题。

## 5. 根因分析

### 5.1 根因一：步间屏障顺序错误

原实现每轮MeshChunk通信任务提交后，先执行`PreSyncInterThreads`，再执行`PostSyncInterThreads`。

原同步顺序可抽象为：

```text
主线程：发送“可以继续”通知 → 等待各从线程上报完成
从线程：等待主线程通知     → 上报自身完成 → 可能进入下一轮
```

在该顺序下，每个从线程上报自身完成后，就没有新的全局屏障阻止其执行下一轮任务。较快的从线程可能已经进入下一轮，而主线程仍在等待其他较慢线程完成上一轮。

也就是说，原逻辑实现的是“每个从线程分别与主线程同步”，没有建立“所有线程完成后再统一进入下一轮”的全局步间屏障。

### 5.2 根因二：ReduceScatter内部Notify索引偏移一位

设从线程数量为`S`。主线程为外层同步和MeshChunk内部同步共申请`2S`个Notify，因此合法索引范围为：

```text
[0, 2S-1]
```

外层同步使用第一组Notify：

```text
[0, S-1]
```

MeshChunk内部同步应使用第二组Notify：

```text
[S, 2S-1]
```

原实现使用`threadNum`作为第二组起始偏移。对ReduceScatter MeshChunk而言：

```text
threadNum = S + 1
```

因此，原实现得到的实际索引范围为：

```text
[S+1, 2S]
```

该范围跳过了合法索引`S`，同时最后一个索引`2S`超出了已申请范围，存在Notify资源错配及同步异常风险。

## 6. 修复方案

### 6.1 重建完整的步间全局屏障

将同步顺序调整为先`PostSyncInterThreads`，再`PreSyncInterThreads`：

```text
从线程上报当前轮完成
        ↓
主线程等待所有从线程完成
        ↓
主线程统一发送下一轮许可
        ↓
所有从线程进入下一轮
```

修复后的关键不变量为：

> 任意线程进入第N+1轮之前，主线程必须确认所有线程均已完成第N轮。

该修改同时应用到：

- `src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc`
- `src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc`

### 6.2 修正Notify索引分组

ReduceScatter MeshChunk将内部同步索引偏移从`threadNum`修正为`notifyNum`：

```text
修复前：notifyIdx + threadNum
修复后：notifyIdx + notifyNum
```

修复后的第二组索引准确落入：

```text
[S, 2S-1]
```

从而保证外层同步和MeshChunk内部同步各自使用独立且合法的Notify区间。

## 7. 修复结果

本次修改以5行新增、5行删除的小范围补丁完成以下闭环：

- 恢复“当前轮全部线程完成后才能进入下一轮”的全局同步关系。
- 消除快线程跨轮执行、慢线程仍停留在上一轮的时序风险。
- 修正ReduceScatter内部Notify索引偏移，消除访问未申请Notify的风险。
- 同步修复AllReduce和ReduceScatter两个MeshChunk模板，避免同类问题遗漏。
- 提交通过MR !1913合入`master`，Git空白检查无告警。

## 8. 任职能力体现

该案例能够证明以下能力：

1. 能够分析异步多线程、跨Channel通信中的复杂时序关系。
2. 能够从下游同步失败或超时反向追踪到步间屏障设计问题。
3. 能够识别依赖线程调度、链路时延和负载窗口的难复现问题。
4. 能够同时检查同步时序和资源布局，发现隐藏的Notify索引越界问题。
5. 能够将局部现象抽象为系统不变量，并据此验证修复正确性。
6. 能够使用小范围、低风险、可审查的补丁完成多个同类模板的闭环。

## 9. 任职材料精简表述

> 负责定位并修复AICPU MeshChunk算法多线程同步偶现问题。该问题仅在多线程、中间通信轮次及线程执行存在时延差异时触发，复现受Device负载、Channel时延和数据分片影响。通过还原主从线程Notify时序，识别出原实现先PreSync、后PostSync，导致快线程可能在其他线程完成前跨轮执行；同时发现ReduceScatter内部Notify索引偏移一位，存在超出已申请资源范围的风险。最终将同步顺序调整为“从线程完成上报→主线程等待全部完成→统一释放下一轮”，并修正Notify索引区间，以5行新增、5行删除的小范围补丁同步完成AllReduce和ReduceScatter MeshChunk修复，相关修改通过MR !1913合入主干。

## 10. 证据索引

| 证据 | 内容 |
|---|---|
| Commit `f455ca50` | 修复同步顺序及ReduceScatter Notify索引 |
| MR !1913 | Bug修复评审与合入记录 |
| AllReduce修改文件 | `src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc` |
| ReduceScatter修改文件 | `src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc` |

> 证据边界：提交记录未包含具体复现日志、测试次数和长稳结果。正式举证时建议补充故障日志、触发场景、修复前后复现概率、ST或长稳测试结果及MR评审截图，不在缺少原始记录时推定这些过程数据。

# 问题单：MeshChunk 算法步间屏障 PreSync/PostSync 顺序反转导致 Reduce 非确定性

## 一、问题现象

在 8 卡 512MB fp32 ReduceScatter 场景下，`InsTempReduceScatterMesh1DMeshChunk` 算法产生 **Reduce 结果非确定性**：多个 rank 在不同 step 中向同一 CCL buffer 区域并行写入（`TaskReduce` 写-写冲突），导致 reduce 结果依赖于线程调度顺序，无法保证正确性。

Checker V3 检测到 `ErrorCode: 302`（两个任务可能并行访问同一内存区域，且至少一个为写操作），`parallelCandidatePairs=1`。

## 二、根因分析

### 2.1 算法背景

`InsTempReduceScatterMesh1DMeshChunk` 将每卡数据切分为 `N-1` 个子块，通过 `N-1` 步 mesh 流水线完成 ReduceScatter。每步内，main 线程向各 slave 线程分发 `SendRecvBatchWriteReduce` 任务，步末通过 `PreSyncInterThreads` / `PostSyncInterThreads` 进行线程间屏障同步。

`OrchestrateLoop` 按每卡数据量和 CCL buffer 容量将数据拆分为多个 loop。512MB 场景下每卡 64MB，拆分为 loop0（57MB）+ loop1（7MB），CCL buffer 从 offset 0 复用。

### 2.2 步间偏移回绕

`DoMeshChunk` 中接收偏移 `sliceRecvOffset_` 从 `sliceRecvBaseOffset` 逐步递减，最后一个子块（`i = N-2`）的接收偏移回绕到 `0x0`。同时，所有 rank 在各自第 `k-1` 步以 `sliceSendOffset_=0` 向 rank 0 的 CCL `[0x0, 0x100000)` 发送 `TaskReduce`（写操作）。

| 写入 rank | 所在 step | 写入目标（rank 0 CCL） |
|-----------|----------|----------------------|
| rank 1 | step 0 | [0x0, 0x100000) |
| rank 2 | step 1 | [0x0, 0x100000) |
| rank 3 | step 2 | [0x0, 0x100000) |
| ... | ... | ... |
| rank 7 | step 6 | [0x0, 0x100000) |

这些写入必须按 step 顺序串行化，否则同一区域被并行写入产生数据竞争。

### 2.3 屏障顺序错误

**原始代码**（`DoMeshChunk` 步间屏障）：

```cpp
PreSyncInterThreads(...);    // 1. main → slaves：main 发信号，slaves 等待
PostSyncInterThreads(...);   // 2. slaves → main：slaves 发信号，main 等待
```

`PreSyncInterThreads` 语义：main 线程 Record 通知各 slave，slave Wait 等待 main 信号。

`PostSyncInterThreads` 语义：main 线程 Wait 等待各 slave 完成，slave Record 通知 main。

**原始顺序的问题**：main 先执行 PreSync（发信号给 slaves），此时 slaves 可能尚未完成当前 step 的工作；再执行 PostSync（等待所有 slave 完成）。但 slave 在完成 PostSync 的 Record 后即可立即进入下一步，**无需等待其他 slave 也完成**。这是一个**半屏障**：

- ✅ main 等待所有 slave 完成（PostSync 的 Wait）
- ❌ slave 之间不互相等待——slave A 完成 PostSync Record 后即可进入下一步，而 slave B 可能仍在执行当前 step 的任务

这导致 rank 0 的 slave-1（处理 rank 2 通信）和 slave-2（处理 rank 3 通信）之间缺少时序约束：slave-2 可以在 slave-1 完成前就开始下一步，向 rank 3 发出 "ready" 信号，rank 3 随即写入 rank 0 CCL `[0x0, 0x100000)`，而此时 rank 2 的写入可能尚未完成——**写-写冲突**。

### 2.4 正确顺序

**修复后代码**：

```cpp
PostSyncInterThreads(...);   // 1. slaves → main：main 等待所有 slave 完成
PreSyncInterThreads(...);    // 2. main → slaves：main 确认全部完成后，发信号让 slave 进入下一步
```

此顺序构成**全屏障**：
1. main 等待所有 slave 完成当前 step（PostSync Wait）
2. main 确认全部完成后，发信号让 slave 进入下一步（PreSync Record）
3. slave 必须等待 main 的 PreSync 信号才能进入下一步，此时所有 slave 均已完成

这与 hcomm legacy 版本（`PostSyncInterQueues → PreSyncInterQueues`）的顺序一致。

### 2.5 同类问题排查

对 hccl 全代码库扫描所有 `PreSyncInterThreads` / `PostSyncInterThreads` 调用（共 68 个文件），区分为两类模式：

| 模式 | 说明 | 是否有 bug |
|------|------|-----------|
| **相邻屏障**（PreSync 和 PostSync 之间无工作） | 步间屏障，需全屏障顺序 | PreSync→PostSync = **bug** |
| **Fork-Join**（PreSync → work → PostSync） | 每步内 fork-join，步间自然形成 PostSync→PreSync | **无 bug** |

仅以下两个文件存在相邻屏障写反的问题：

| 文件 | 修复位置 |
|------|---------|
| `ins_temp_reduce_scatter_mesh_1D_meshchunk.cc` | `DoMeshChunk` 函数步间屏障 |
| `ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc` | `ReduceScatterMeshChunk` 函数步间屏障 |

其余 9 处 `PreSync→PostSync` 均为 Fork-Join 模式（中间有实际工作），步间屏障由 `PostSync(step N) → PreSync(step N+1)` 自然构成全屏障，**无需修改**。

### 2.6 附带修复：NotifyIdxSubToMainInMeshChunk 越界

代码检视中发现 `NotifyIdxSubToMainInMeshChunk` 的 notify 索引计算存在 OOB：

```cpp
// 修复前（OOB）：notifyIdx + threadNum，最大索引 = 2*threadNum - 2 = notifyNumOnMainThread（越界 1）
notifyIdxSubToMain.push_back(notifyIdx + threadNum);

// 修复后：notifyIdx + notifyNum，最大索引 = 2*notifyNum - 1 = notifyNumOnMainThread - 1（合法）
notifyIdxSubToMain.push_back(notifyIdx + notifyNum);
```

与 AllReduce 的 `NotifyIdxSubToMainInRSMeshChunk`（已使用 `notifyIdx + notifyNum`）保持一致。

## 三、修改内容

### 3.1 修改文件清单

| 仓库 | 文件 | 修改内容 |
|------|------|---------|
| `/home/ytz/CANN/hccl` | `src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc` | 步间屏障 swap + notify 索引越界修复 |
| `/home/ytz/CANN/hccl` | `src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc` | 步间屏障 swap |
| `/home/ytz/CANN/hccl_900` | 同上两个文件 | 同步修改 |

### 3.2 修改详情

**ReduceScatter `DoMeshChunk` 步间屏障**：

```diff
- NotifyIdxMainToSubInMeshChunk(notifyIdxMainToSub_);
- CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
  NotifyIdxSubToMainInMeshChunk(notifyIdxSubToMain_);
  CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
+ NotifyIdxMainToSubInMeshChunk(notifyIdxMainToSub_);
+ CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
```

**AllReduce `ReduceScatterMeshChunk` 步间屏障**：

```diff
- NotifyIdxMainToSubInRSMeshChunk(notifyIdxMainToSub_);
- CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
  NotifyIdxSubToMainInRSMeshChunk(notifyIdxSubToMain_);
  CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
+ NotifyIdxMainToSubInRSMeshChunk(notifyIdxMainToSub_);
+ CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
```

**NotifyIdxSubToMainInMeshChunk 越界修复**：

```diff
- notifyIdxSubToMain.push_back(notifyIdx + threadNum);
+ notifyIdxSubToMain.push_back(notifyIdx + notifyNum);
```

## 四、测试验证

### 4.1 测试环境

- 仿真工具: HCCL-VM（CheckerL2），`hccl-vm start ascend950_cluster_1_server_hf`
- 拓扑: 1 server, 8 rank（Ascend950 HF 组网）
- 测试用例: `reduce_scatter_test -b 512M -e 512M -d fp32 -o sum -w 0 -n 1 -c 1`
- 构建: `HCCL_CODE_HOME=/home/ytz/CANN/hccl bash build_pkg.sh --install hccl`
- 关键环境变量: `export HWLOC_COMPONENTS=-gl,-opencl`

### 4.2 修复前结果

Checker V3 报内存冲突：

```
[ErrorCode: 302] Two tasks may access the same memory range in parallel, and at least one access is a write.
  Conflict memory : rank 0, CCL
  Overlap range    : [0x0, 0x100000)
  Conflict task 1: node 2669, rank=3, TaskReduce, dst=rank 0, CCL, [0x0, 0x100000)
  Conflict task 2: node 1818, rank=2, TaskReduce, dst=rank 0, CCL, [0x0, 0x100000)

parallelCandidatePairs=1   ← 存在 1 对并行写冲突
orderedCandidatePairs=2
op[0] Checker failed
op[1] Checker failed
```

### 4.3 修复后结果

Checker V3 全部通过：

```
MemConflict: status=success, parallelCandidatePairs=0, orderedCandidatePairs=7600
SemanticCheck: status=success, normalSemanticCount=80
op[0] Checker Success
op[1] Checker Success
```

### 4.4 修复前后对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| parallelCandidatePairs | 1 | **0** |
| orderedCandidatePairs | 2 | **7600** |
| SemanticCheck | 跳过（MemConflict 失败） | **success**（80 个语义检查通过） |
| 最终结果 | **Checker failed** | **Checker Success** |

### 4.5 日志文件

| 文件 | 说明 |
|------|------|
| `hccl_rs_8_fp32_512m_runner.log` | mpirun 日志（0 error） |
| `checker.log` | Checker V3 完整日志 |
| `hvm_session.log` | HCCL-VM 会话日志 |

日志路径: `/home/ytz/CANN/ZZ/杂项/`

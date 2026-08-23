# InsTempAllGatherNHR 模板调用关系与末步并行优化分析

## 1. 模板定义

| 项目 | 内容 |
|------|------|
| 类名 | `InsTempAllGatherNHR` |
| 头文件 | `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.h` |
| 实现文件 | `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc` |
| 基类 | `InsAlgTemplateBase` |
| 引擎 | AICPU |
| 子类 | `InsTempAllGatherOmniPipeNHR`（重写 `KernelRun`/`RunAllGatherNHR`，不复用本优化） |

> **注意**：`InsTempAllGatherNHRDPU` / `InsTempAllGatherNHRDPUInter` 是独立类（直接继承 `InsAlgTemplateBase`），不是 `InsTempAllGatherNHR` 的子类，不携带本优化，不在本文档范围内。

---

## 2. 调用该模板的 Executor 一览

通过 `REGISTER_EXEC_V2` / `REGISTER_EXECUTOR_BY_TWO_TEMPS` / `REGISTER_EXECUTOR_BY_FOUR_TEMPS` / `REGISTER_EXEC_V2_MULTI` 宏注册，直接调用方涉及 AllGather、AllReduce、Broadcast 和 Reduce。独立 Scatter 使用 `InsTempScatterNHR`，不直接调用本模板。

### 2.1 AllGather 类

| Executor | 算法名 | 拓扑匹配 | NHR 在算法中的角色 | 注册位置 |
|---|---|---|---|---|
| `InsV2AllGatherSoleExecutor` | `InsAllGatherNHR` | `TopoMatch1D` | 单模板 | `ins_v2_all_gather_sole_executor.cc:266` |
| `InsV2AllGatherParallelExecutor` | `InsAllGatherParallelMesh1DNHR` | `TopoMatchMultilevel` | inter 模板（配 Mesh1D） | `ins_v2_all_gather_parallel_executor.cc:603` |
| 同上 | `InsAllGatherParallelMesh1DNHRMultiJetty` | `TopoMatchUBX` | inter 模板 | `:607` |
| 同上 | `InsAllGatherParallelMesh1DNHRPcie` | `TopoMatchPcieMix` | inter 模板 | `:610` |
| 同上 | `InsAllGatherParallelMesh1DNHRUboe` | `TopoMatchSqueeze2D` | temp0 + temp1 均为 NHR | `:614` |
| `InsV2AllGatherSequenceExecutorAicpu` | `InsAllGatherSequenceNHRMesh1D` | `TopoMatchMultilevel` | inter 模板（配 Mesh1D1DZAxisDetour） | `ins_v2_all_gather_sequence_executor_aicpu.cc:431` |
| `InsV2AllGatherConcurrentExecutor` | `InsAllGatherConcurrentMesh1DNHR` | `TopoMatchUBX` | inter 模板（配 Mesh1D） | `ins_v2_all_gather_concurrent_executor.cc:439` |
| `InsV2AllGatherSequenceExecutor3Level` | `InsAllGatherSequenceNHRNHRMesh1D` | `TopoMatchMultilevel` | level1、level2 均为 NHR | `ins_v2_all_gather_sequence_executor_3level.cc:330` |

### 2.2 Broadcast 类

Broadcast = Scatter（前两个模板）+ AllGather（后两个模板），NHR 作为 AllGather 段出现。

| Executor | 算法名 | 拓扑匹配 | NHR 角色 | 注册位置 |
|---|---|---|---|---|
| `InsBroadcastParallelExecutor` | `InsBroadcastParallelMesh1DNHR` | `TopoMatchMultilevel` | 第 4 模板（AllGather inter） | `ins_v2_broadcast_parallel_executor.cc:1102` |
| 同上 | `InsBroadcastParallelMesh1DNHRUBX` | `TopoMatchUBX` | 同上 | `:1104` |
| 同上 | `InsBroadcastParallelMesh1DNHRPcie` | `TopoMatchPcieMix` | 同上 | `:1107` |
| 同上 | `InsBroadcastParallelNHRNHRUboe` | `TopoMatchSqueeze2D` | 第 3、4 模板均为 NHR | `:1109` |

### 2.3 AllReduce 与 Reduce 类

| 算子 | Executor | 算法 | NHR 角色 |
|---|---|---|---|
| AllReduce | `InsAllReduceParallelExecutor` | `InsAllReduceParallelRSAG` / `Pcie` / `UBX` / `Dpu` / `Uboe` | ReduceScatter 之后的 AllGather 阶段 |
| AllReduce | `InsV2AllReduceSequenceExecutorAicpu` | `InsAllReduceSequenceMesh1DNhr` | 第 3 模板（AllGather） |
| AllReduce | `InsV2AllReduceSequenceExecutorAicpu3Level` | `InsV2AllReduceSequenceMesh1DNHRNHR` | level1、level0 AllGather |
| AllReduce | `InsV2AllReduceOrderPreservedExecutor` | `AllReduceOrderPreservedGroup` | ReduceScatter 之后的 AllGather |
| Reduce | `ReduceParallelExecutor` | `ReduceParallelMesh1DNHR` / `UBX` / `Pcie` / `Uboe` | ReduceScatter 之后的 AllGather 阶段 |

AllReduce/Reduce 的完整 `TemplateDataParams` 配置需在实施前按阶段审计；模板级放宽会对其中所有 `outBuffType == OUTPUT` 的路径生效，不能只验证 AllGather 单算子。

---

## 3. 末步 read + localcopy 并行优化

### 3.1 正常流程

`InsTempAllGatherNHR::KernelRun`（`ins_temp_all_gather_nhr.cc:91`）的常规执行路径：

1. `LocalDataCopy`：input buffer → scratch buffer（hcclBuff）。
2. `RunAllGatherNHR`：逐 step 执行 NHR 环形交换，每步收到的数据写入 **scratch buffer**。
3. `PostLocalCopy`：全部 step 完成后，在**同一线程** `threads[channelIdx]` 上把所有分片从 scratch 拷到 output。

### 3.2 优化流程

当满足触发条件时，最后一个 step 走 `RunLastStepWriteThenRead`（`:227`），将该步拆为 3 个子操作、分布在 2 条线程上：

```
线程 threads[channelIdx]                        线程 threads[channelsPerRank_ + channelIdx]
┌─────────────────────────────┐                ┌──────────────────────────────┐
│ ① SendRecvBatchWrite (发送)  │                │                              │
│   仅 i==0 的 tx 切片 → 对端   │                │                              │
│                              │   PreSync      │                              │
│                              │ ──────────────>│ ② PostLocalCopy              │
│ ③ SendRecvBatchRead (接收)   │                │   scratch → output           │
│   收到的数据直写 output       │                │   (跳过自身分片与末步直读分片) │
│   (不再经 scratch)            │                │                              │
└─────────────────────────────┘                └──────────────────────────────┘
                     ↑                                ↑
                     └──── ② 与 ③ 并行执行 ────────────┘
```

**优化收益**：
- 末步的网络接收（③ Read）与本地 DMA 拷贝（② PostLocalCopy）在两条线程上**并行执行**，隐藏拷贝延迟。
- 末步接收的数据**直写 output buffer**（不再经 scratch 中转），省去一次拷贝。
- `postLocalCopyLaunched` 置位后，外层 `KernelRun` 不再重复调 `PostLocalCopy`。

**资源开销**：
- `GetThreadNum()` 返回 `channelsPerRank_ * 2`（线程数翻倍，`:67`），多出一倍的流用于 PostLocalCopy。
- 每线程配 2 个 notify（`:60`），多出的一个用于 ② 与 ③ 的前同步。

### 3.3 代码位置索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `CanReadLastStepToOutput` | `:143` | 判断是否满足 buffer 条件 |
| `RunStepNHR` | `:270` | 逐 step 执行，末步走优化分支 |
| `RunLastStepWriteThenRead` | `:227` | 优化的末步拆分执行 |
| `BuildStepSlices` (mode=`LAST_STEP_WRITE_THEN_READ`) | `:185` / `:199` | 构造末步切片（rx 目标为 output） |
| `PostLocalCopy` | `:406` | scratch → output，跳过已直读分片 |
| `IsLastStepReadSlice` | `:154` | 判断某分片是否已被末步直读 |

---

## 4. 优化的生效条件

### 4.1 条件总览

入口判断在 `RunStepNHR`（`:291`）：

```cpp
const bool readLastStepToOutput = readLastStepToOutput_
                               && step == nSteps - 1
                               && stepInfo.nSlices > 1;
```

其中 `readLastStepToOutput_` 由 `CanReadLastStepToOutput()`（`:143`）在 `KernelRun` 中计算：

```cpp
bool InsTempAllGatherNHR::CanReadLastStepToOutput() const
{
    return !isDmaRead_ && !enableRemoteMemAccess_ &&
           tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr &&
           tempAlgParams_.buffInfo.inBuffType == BufferType::OUTPUT &&
           tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT &&
           tempAlgParams_.buffInfo.inBuffBaseOff == tempAlgParams_.buffInfo.outBuffBaseOff &&
           tempAlgParams_.inputSliceStride == tempAlgParams_.outputSliceStride &&
           tempAlgParams_.inputRepeatStride == tempAlgParams_.outputRepeatStride;
}
```

| # | 条件 | 含义 | 来源 |
|---|------|------|------|
| 1 | `!isDmaRead_` | 链路**非 PCIe**（须为 RoCE/UB 网络链路）。`isDmaRead_ = IsPcieProtocol(channels)` | `:110`、`:145` |
| 2 | `!enableRemoteMemAccess_` | **非 OFFLOAD/图模式**（`param.opMode != OpMode::OFFLOAD`） | `:107`、`:145` |
| 3 | `inputPtr == outputPtr` | **in-place**：输入输出同一块 buffer | `:146` |
| 4 | `inBuffType == OUTPUT && outBuffType == OUTPUT` | 输入输出 buffer 类型均为 OUTPUT | `:147-148` |
| 5 | `inBuffBaseOff == outBuffBaseOff` | 基址偏移相同 | `:149` |
| 6 | `inputSliceStride == outputSliceStride` | slice 步长一致 | `:150` |
| 7 | `inputRepeatStride == outputRepeatStride` | repeat 步长一致 | `:151` |
| 8 | `step == nSteps - 1` | NHR 的最后一个 step | `:291` |
| 9 | `stepInfo.nSlices > 1` | 最后一步切片数 > 1（等价 `templateRankSize_ >= 4`，见 4.2） | `:291` |

### 4.2 `nSlices > 1` 与 rank 数的关系

`GetNHRStepNum`（`template_utils.cc:24`）返回 `ceil(log2(rankSize))`。最后一步（`step = nSteps - 1`）的切片数：

```
nSlices = (rankSize - 1 + (1 << 0)) / (1 << 1) = rankSize / 2
```

| rankSize | nSteps | 末步 nSlices | nSlices > 1? |
|----------|--------|-------------|-------------|
| 2 | 1 | 1 | ❌ |
| 3 | 2 | 1 | ❌ |
| 4 | 2 | 2 | ✅ |
| 5 | 3 | 2 | ✅ |
| 8 | 3 | 4 | ✅ |

即 **`templateRankSize_ >= 4`** 时末步才有 >1 片，优化方有可能触发。

---

## 5. AllGather/Broadcast Executor 对 NHR 的参数配置与生效判定

优化条件 3–7 取决于 executor 如何构造 `tempAlgParams`。逐个核查：

### 5.1 不生效的 Executor

| Executor | NHR 的 `inBuffType`/`outBuffType` | `inputPtr` vs `outputPtr` | 判定 |
|---|---|---|---|
| `InsV2AllGatherSoleExecutor` (`:110-116`) | `INPUT`/`OUTPUT` | `inputPtr`≠`outputPtr` | ❌ 条件 4 不满足 |
| `InsV2AllGatherConcurrentExecutor` (`:131-137`) | `INPUT`/`OUTPUT` | `inputPtr`≠`outputPtr` | ❌ 条件 4 不满足 |
| `InsV2AllGatherSequenceExecutorAicpu` (`:249-259`) | inter NHR: `INPUT`/`OUTPUT`(CCU) 或 `INPUT`/`HCCL_BUFFER` | `inputPtr`≠`outputPtr` | ❌ 条件 4 不满足 |
| `InsV2AllGatherSequenceExecutor3Level` (`:220-313`) | Level0: `HCCL_BUFFER`/`OUTPUT`; Level1: `HCCL_BUFFER`/`HCCL_BUFFER`; Level2: `INPUT`/`HCCL_BUFFER` | 各层均不等 | ❌ 三层均条件 4 不满足 |
| `InsBroadcastParallelExecutor` (`:941`等) | AllGather NHR: `HCCL_BUFFER`/`HCCL_BUFFER` | `cclMem.addr`==`cclMem.addr` | ❌ 条件 4 不满足（broadcast 用 INPUT/HCCL_BUFFER，无 OUTPUT） |

### 5.2 本节范围内唯一可生效的 Executor：`InsV2AllGatherParallelExecutor`

该 executor 的 `OrchestrateLoop`（`:455`）分两个 phase 执行，NHR 在两个 phase 使用不同参数函数：

| Phase | 执行顺序 | NHR 使用的参数函数 | `inBuffType`/`outBuffType` | 生效? |
|-------|---------|-------------------|---------------------------|-------|
| Phase 1 | Intra0 → **Inter1** | `GenTemplateAlgParamsInter1` (`:199`) | `INPUT`/`OUTPUT` | ❌ |
| Phase 2 | **Inter0** → Intra1 | `GenTemplateAlgParamsInter0` (`:164`) | `OUTPUT`/`OUTPUT` | ✅ |

`GenTemplateAlgParamsInter0`（`:164-196`）的完整配置：

```cpp
inputPtr  = param.outputPtr;          outputPtr = param.outputPtr;        // 条件 3 ✓
inBuffType  = BufferType::OUTPUT;      outBuffType  = BufferType::OUTPUT;  // 条件 4 ✓
inBuffBaseOff = dataOffset;            outBuffBaseOff = dataOffset;        // 条件 5 ✓
inputSliceStride  = dataSize_*rankSizeLevel0_;  outputSliceStride  = 同;   // 条件 6 ✓
inputRepeatStride = dataSize_;         outputRepeatStride = dataSize_;     // 条件 7 ✓
```

**buffer 条件 3–7 全部满足。**

> 补充：`GenTemplateAlgParamsIntra1`（`:233`）同样满足全部 buffer 条件，但仅 `InsAllGatherParallelMesh1DNHRUboe`（NHR+NHR）的 temp0=NHR 会用到 Intra1 参数；其余 3 个注册的 temp0=Mesh1D 用 Intra1，与 NHR 优化无关。

### 5.3 Parallel Executor 各注册算法的生效判定

| 算法名 | 拓扑 | Phase 2 NHR 参数 | 链路（`isDmaRead_`） | 能否生效 |
|---|---|---|---|---|
| `InsAllGatherParallelMesh1DNHR` | `TopoMatchMultilevel` | Inter0（inter=server间） | 网络 → `false` | **✅ 可生效** |
| `InsAllGatherParallelMesh1DNHRMultiJetty` | `TopoMatchUBX` | Inter0 | 网络 → `false` | **✅ 可生效** |
| `InsAllGatherParallelMesh1DNHRPcie` | `TopoMatchPcieMix` | Inter0 | PcieMix 下 NHR 若分到 PCIe 通道 → `true` | **视运行时链路而定** |
| `InsAllGatherParallelMesh1DNHRUboe` | `TopoMatchSqueeze2D` | temp1 NHR: Inter0（inter） | 网络 → `false` | **✅ 可生效** |
| 同上 | 同上 | temp0 NHR: Intra1（intra） | Squeeze2D 两层均可为网络 | **视运行时链路而定** |

---

## 6. 结论

### 6.1 优化可生效的完整条件

优化在以下条件**同时满足**时生效：

1. **executor**：在第 5 节已审计的 AllGather/Broadcast 路径中，为 `InsV2AllGatherParallelExecutor`（Phase 2 的 inter NHR，使用 `GenTemplateAlgParamsInter0` 参数）。AllReduce/Reduce 调用方需另行按阶段判定。
2. **算法**：`InsAllGatherParallelMesh1DNHR` / `InsAllGatherParallelMesh1DNHRMultiJetty` / `InsAllGatherParallelMesh1DNHRUboe`（PcieMix 视链路而定）。
3. **链路**：NHR 实际通道为网络（非 PCIe），即 `isDmaRead_ = false`。
4. **模式**：非 OFFLOAD/图模式（`param.opMode != OpMode::OFFLOAD`）。
5. **rank 数**：通信域 `templateRankSize_ >= 4`（保证末步 `nSlices > 1`）。
6. **时机**：NHR 的最后一个 step。

### 6.2 不生效的 executor 及原因汇总

| Executor | 不生效根因 |
|---|---|
| `InsV2AllGatherSoleExecutor` | NHR 配 `inBuffType = INPUT` |
| `InsV2AllGatherConcurrentExecutor` | NHR 配 `inBuffType = INPUT` |
| `InsV2AllGatherSequenceExecutorAicpu` | inter NHR 配 `inBuffType = INPUT` |
| `InsV2AllGatherSequenceExecutor3Level` | 三层 NHR 分别配 `INPUT`/`HCCL_BUFFER`，均非 `OUTPUT` |
| `InsBroadcastParallelExecutor` | AllGather NHR 配 `inBuffType = HCCL_BUFFER`（broadcast 无 OUTPUT buffer） |
| `InsV2AllGatherParallelExecutor` Phase 1 | NHR 用 Inter1 参数，`inBuffType = INPUT` |

**核心判定**：`CanReadLastStepToOutput()` 要求 `inBuffType == OUTPUT`。在第 5 节已审计的 AllGather/Broadcast 路径中，只有 `InsV2AllGatherParallelExecutor` 的 `GenTemplateAlgParamsInter0`（Phase 2 inter NHR）满足完整条件。由于 AllReduce/Reduce 也直接注册该模板，实施时必须将它们纳入参数审计与回归，不应将该结论外推为全仓唯一生效路径。

---

## 7. 需求分析与影响面修订

目标是将 `RunLastStepWriteThenRead` 扩展到更多 `outBuffType == OUTPUT` 的非原地场景，同时保持当前已生效路径的行为和性能。

### 7.1 直接调用方

全仓精确搜索表明，`InsTempAllGatherNHR` 被以下四类算子直接注册：

| 算子 | 直接使用 | 主要 executor |
|---|---|---|
| AllGather | ✅ | sole / parallel / concurrent / sequence / 3level |
| AllReduce | ✅ | parallel / sequence / order-preserved / 3level |
| Broadcast | ✅ | parallel，作为 Scatter 之后的 AllGather 阶段 |
| Reduce | ✅ | parallel |
| Scatter | ❌ | 独立 Scatter 使用 `InsTempScatterNHR` |

Broadcast 的四模板结构为 `Scatter intra → Scatter inter → AllGather intra → AllGather inter`，其后两个模板可为 `InsTempAllGatherNHR`。独立 Scatter 不是本优化的直接影响面，但 Broadcast 中的 Scatter 前置阶段必须随整体流程回归。

`InsTempAllGatherOmniPipeNHR` 虽继承本类，但重写 `KernelRun` / `RunAllGatherNHR`，不复用本优化；`InsTempAllGatherNHRDPU` 和 `InsTempAllGatherNHRDPUInter` 是独立类，也不在修改范围内。

### 7.2 核心问题

当前代码把两个不同问题绑在 `CanReadLastStepToOutput()` 中：

1. 末步 Read 是否可以直写最终 output。
2. `PostLocalCopy` 是否可以跳过自身分片。

非原地场景下，问题 1 可能成立，但问题 2 不成立。因此不能只删除现有条件；否则自身分片会既不被末步 Read 写入，也被 `PostLocalCopy` 跳过，导致 output 数据不完整。

## 8. 总体解决方案

### 8.1 拆分资格判定

保留 `CanReadLastStepToOutput()`，另增一个精确表示“自身分片已在最终 output 位置”的判定：

```cpp
bool CanReadLastStepToOutput() const;
bool CanSkipOwnSliceCopy() const;

bool readLastStepToOutput_{false};
bool skipOwnSliceCopy_{false};
```

不建议命名为 `isInPlace_`，因为指针相等不代表分片地址布局相等。

`CanReadLastStepToOutput()` 只判断末步 Read 的目标是否为可用的最终 output：

```cpp
return !isDmaRead_ && !enableRemoteMemAccess_ &&
       tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT &&
       tempAlgParams_.buffInfo.outputPtr != tempAlgParams_.buffInfo.hcclBuff.addr;
```

`CanSkipOwnSliceCopy()` 保留旧路径的完整同布局约束：

```cpp
return tempAlgParams_.buffInfo.inBuffType == BufferType::OUTPUT &&
       tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT &&
       tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr &&
       tempAlgParams_.buffInfo.inBuffBaseOff == tempAlgParams_.buffInfo.outBuffBaseOff &&
       tempAlgParams_.inputSliceStride == tempAlgParams_.outputSliceStride &&
       tempAlgParams_.inputRepeatStride == tempAlgParams_.outputRepeatStride;
```

`KernelRun()` 中每次重新计算：

```cpp
readLastStepToOutput_ = CanReadLastStepToOutput();
skipOwnSliceCopy_ = readLastStepToOutput_ && CanSkipOwnSliceCopy();
```

### 8.2 修改 PostLocalCopy 分片归属

```cpp
if (readLastStepToOutput_ && skipOwnSliceCopy_ && algRank == myAlgRank) {
    continue;
}
if (readLastStepToOutput_ && IsLastStepReadSlice(algRank)) {
    continue;
}
```

| 分片来源 | 非原地场景 | 现有同布局场景 |
|---|---|---|
| 自身分片 | `LocalDataCopy` 先写 scratch，再由 `PostLocalCopy` 写 output | 已在 output，跳过 |
| 前面 step 收到的分片 | scratch → output | scratch → output |
| 末步 `i == 0` 收到的分片 | 对端 Write 到 scratch，再 copy | 同左 |
| 末步 `i > 0` 的分片 | Read 直写 output，copy 跳过 | 同左 |

### 8.3 地址、同步与资源正确性

- 末步 rx 目标使用 `outBuffBaseOff + rpt * outputRepeatStride + outputSliceStride * rxIdx + rxPartialOffset`，与普通 `PostLocalCopy` 的目标地址公式一致，不依赖 input 布局。
- `LocalDataCopy` 在 NHR 前已将自身分片写入 scratch，非原地场景具备自身分片的 copy 数据源。
- `lastStepReadSliceIdxs_` 在任务下发前已构造；Read 直写集合与 PostLocalCopy 写集合不相交，两线程无写覆盖。
- `GetRes()` 已按 `channelsPerRank_ * 2` 申请线程，`KernelRun()` 和 `RunLastStepWriteThenRead()` 已有数量与下标检查，无需在资格判定中重复检查。
- `outBuffType == HCCL_BUFFER` 的场景保持不启用；此时 output 就是 scratch，不存在需要隐藏的 PostLocalCopy。

### 8.4 修改范围与约束

最小修改限于：

- `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.h`
- `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc`

不修改 executor 参数，不改动 HCOMM 接口，不引入编译期跨仓依赖。现有 `OUTPUT/OUTPUT` 同布局路径的任务序列必须保持不变。

## 9. 回归测试方案

### 9.1 验收原则

每个正向用例必须同时满足：

1. 输出数据语义校验通过；可以保存完整数据时，与修改前基线做 bit-exact 对比。
2. 算法选择日志确认命中目标 executor 和 `InsTempAllGatherNHR`。
3. HCCL_DEBUG 日志确认 `Read last step to output[1]`。
4. 任务轨迹确认末步存在专职 PostLocalCopy 线程和 Read 线程，且直读分片没有被重复 copy。

仅看 `Read last step to output[1]` 不足以证明测到目标 executor；仅看 ST PASS 也不足以证明优化已生效。

### 9.2 分层回归流程

#### 第 1 层：静态检查与编译

```bash
git diff --check
bash build.sh --pkg
bash build.sh -u
```

重点检查新增成员每次 `KernelRun()` 都被重置，避免模板对象复用时残留上次状态。

#### 第 2 层：模板级定向测试

如现有 UT 框架可以直接构造 `TemplateDataParams`，应补充以下判定矩阵：

| 场景 | 末步直写 | 跳过自身 copy |
|---|---:|---:|
| `INPUT/OUTPUT`，指针不同 | ✅ | ❌ |
| `HCCL_BUFFER/OUTPUT`，指针不同 | ✅ | ❌ |
| `OUTPUT/OUTPUT`，指针和布局全相同 | ✅ | ✅ |
| 指针相同，但 base offset 不同 | ✅ | ❌ |
| 指针相同，但 slice/repeat stride 不同 | ✅ | ❌ |
| `outBuffType == HCCL_BUFFER` | ❌ | ❌ |
| PCIe / `isDmaRead_ == true` | ❌ | ❌ |
| OFFLOAD / remote memory access | ❌ | ❌ |

任务级断言应覆盖：非原地自身分片存在 scratch → output copy；末步 Read 分片的目标是 output；PostLocalCopy 不包含这些直读分片。

#### 第 3 层：算子 ST

先跑项目全量 ST：

```bash
bash build.sh -s
```

为避免每次手工修改 testcase，可安装本地 Auto ST runner。安装会修改 `test/st` 下的本地测试文件，需在执行回归时单独进行，不应与业务修改一起提交：

```bash
bash /home/ytz/.codex/skills/hccl-auto-st-runner/scripts/install_auto_st.sh /home/ytz/CANN/hccl
```

参数化用例示例：

```bash
bash test/st/algorithm/run_auto_st.sh --op allgather --data-size 1048577 --topo 4x2 --dtype fp16
bash test/st/algorithm/run_auto_st.sh --op allreduce --data-size 1048577 --topo 4x2 --dtype fp16 --reduce-op sum
bash test/st/algorithm/run_auto_st.sh --op broadcast --data-size 1048577 --topo 4x2 --dtype fp16 --root 0
bash test/st/algorithm/run_auto_st.sh --op reduce --data-size 1048577 --topo 4x2 --dtype fp16 --reduce-op sum --root 0
bash test/st/algorithm/run_auto_st.sh --op scatter --data-size 1048577 --topo 4x2 --dtype fp16 --root 0
```

Scatter 是间接回归：它不应出现 `InsTempAllGatherNHR` 或 `Read last step to output[1]`，用于确认模板改动没有污染独立 Scatter 路径。

Auto ST 只能参数化算子、数据量和拓扑，不保证必然选中某个 executor。需覆盖 sole / parallel / concurrent / sequence / 3level 时，应根据 selector 条件选择拓扑和数据量，并以算法选择日志作为最终判定。

### 9.3 数据与拓扑覆盖矩阵

| 维度 | 必测值 | 目的 |
|---|---|---|
| template rankSize | 2、3、4、5、8 | 2/3 不启用；4 起启用；5 覆盖非 2 的幂 |
| buffer 关系 | 同布局、非原地、同指针不同布局 | 验证自身分片跳过条件 |
| 数据量 | 0/极小、对齐块、非对齐块、跨 loop 大数据 | 覆盖空路径、tail、repeatNum |
| dtype | fp16、fp32、int8/int32 | 覆盖 sliceCount 换算 |
| channel | 单通道、多通道 | 覆盖 dataSplit/dataOffset |
| 协议 | RoCE/UB、PCIe | 正向启用和负向不启用 |
| 模式 | 非 OFFLOAD、OFFLOAD | 正向启用和负向不启用 |
| reduce op | sum、max/min | AllReduce/Reduce 语义回归 |
| root | 0、非 0、尾 rank | Broadcast/Reduce/Scatter root 路径 |

tail 用例至少覆盖两种角色：当前 rank 自身持有尾块，以及当前 rank 在末步直读尾块。两者分别验证 `LocalDataCopy/PostLocalCopy` 和 `rxPartialOffset/dataSplitTail_`。

### 9.4 四类直接调用方的回归重点

| 算子 | 正向覆盖 | 特别检查 |
|---|---|---|
| AllGather | sole / parallel phase1+phase2 / concurrent / sequence / 3level | `INPUT/OUTPUT`、`HCCL_BUFFER/OUTPUT`、现有 `OUTPUT/OUTPUT` |
| AllReduce | parallel / sequence / order-preserved / 3level | ReduceScatter 前置结果不被覆盖，reduce op 不变 |
| Broadcast | parallel 的 AllGather 阶段 | Scatter 前置阶段与 root 分片正确 |
| Reduce | parallel | 仅 root 的最终语义正确，多 root 值 |

对于 `outBuffType == HCCL_BUFFER`、PCIe、OFFLOAD、rankSize < 4 的负向用例，日志必须为 `Read last step to output[0]`，且任务轨迹保持原流程。

### 9.5 A/B 正确性与性能验证

1. 分别用修改前基线和修改后代码构建包，使用相同软硬件、拓扑、数据、warm-up 和迭代次数。
2. 正确性先以 ST 语义检查为准；真实数据环境中对 AllGather/Broadcast 做 bit-exact，AllReduce/Reduce 对整数类型做 bit-exact，浮点 reduce 使用项目已有容差标准。
3. 性能至少统计 p50/p90 和带宽，分开报告旧的 `OUTPUT/OUTPUT` 场景与新增非原地场景。
4. 现有 Phase 2 路径不得出现明显回退；新场景需在 timeline 中确认 PostLocalCopy 与末步 Read 存在重叠，不能只依赖端到端噪声数据判定优化有效。

## 10. 完成准入标准

- 两个资格判定解耦，非原地场景不再错误跳过自身分片。
- AllGather、AllReduce、Broadcast、Reduce 直接调用方的正向和负向用例通过。
- 独立 Scatter 不进入该优化，Broadcast 的 Scatter + AllGather 整体流程通过。
- rankSize 2/3/4/5/8、tail、repeatNum、多通道、PCIe 和 OFFLOAD 边界完成覆盖。
- `bash build.sh --pkg`、`bash build.sh -u`、`bash build.sh -s` 通过，无新增编译告警和静态检查问题。
- 旧 `OUTPUT/OUTPUT` 路径任务序列不变，性能无明显回退；新路径能证明 Read 与 PostLocalCopy 实际重叠。

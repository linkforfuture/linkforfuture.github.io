# Reduce类算子AICPU引擎64位数据类型与PROD规约兜底设计

- 文档状态：定稿（机制介绍）
- 更新日期：2026-08-20
- 适用仓库：`cann/hccl`、`cann/hcomm`
- 相关Issue：无（存量机制梳理）

---

## 1. 概要

HCCL的Reduce类算子（AllReduce、Reduce、ReduceScatter）在AICPU引擎下，当**数据类型为64位（INT64/UINT64/FP64）**或**规约操作为PROD**时（下称"特殊Reduce场景"），无法复用常规的"SDMA搬运+硬件归约"路径，需要切换为"AICPU软件归约"兜底方案。

本设计包含两个核心决策：

1. **规约执行位置迁移**：本地规约从SDMA任务队列（`HcommLocalReduceOnThread`）迁移到AICPU上的同步CPU计算（`AicpuReduce`），由通用搬运包装层`LocalReduce`统一分流。
2. **同步屏障插入**：由于AICPU软件规约直接读取内存、不感知异步任务队列，所有含本地规约的AICPU模板必须在"通信完成"与"本地规约"之间插入同步屏障（`HcommBatchModeEnd`/`HcommBatchModeStart` + `HcommThreadJoin`），保证规约读到的是已落盘的完整数据。

特殊Reduce场景的判定条件在仓库内统一为：

```cpp
Is64BitDataType(dataType) || reduceType == HCCL_REDUCE_PROD
// Is64BitDataType: INT64 / UINT64 / FP64，定义于 src/ops/op_common/selector/auto_selector_base.h:119
```

## 2. 背景与问题

### 2.1 根因：底层归约原语的能力边界

常规路径下，AICPU模板通过HCOMM数据面接口`HcommLocalReduceOnThread`下发本地归约（SDMA硬件归约）。HCOMM仓中该接口的合法性校验（`cann/hcomm` `src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc:53`）：

```cpp
bool IsSupportReduce(HcommDataType dataType, HcommReduceOp op)
{
    bool checkDataType = (dataType == HCOMM_DATA_TYPE_FP32 || dataType == HCOMM_DATA_TYPE_FP16
        || dataType == HCOMM_DATA_TYPE_INT8 || dataType == HCOMM_DATA_TYPE_INT16
        || dataType == HCOMM_DATA_TYPE_INT32 || dataType == HCOMM_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCOMM_REDUCE_SUM || op == HCOMM_REDUCE_MAX || op == HCOMM_REDUCE_MIN);
    return checkDataType && checkReduceType;
}
```

即SDMA硬件归约仅支持**32位及以下数据类型**与**SUM/MAX/MIN**三种操作。64位类型与PROD超出硬件能力，必须在软件层另行处理。

### 2.2 影响范围

| 维度 | 说明 |
|------|------|
| 算子 | AllReduce、Reduce、ReduceScatter、ReduceScatterV |
| 数据类型 | INT64、UINT64、FP64 |
| 规约操作 | HCCL_REDUCE_PROD |
| 引擎 | CCU与AIV引擎均不支持64位类型与PROD（见4.2节），此类场景只能落在AICPU引擎 |
| 归并模式 | MS模式不支持PROD |

## 3. 总体设计

### 3.1 设计思路

特殊Reduce场景带来两个连锁问题，设计上分别以"分流"和"屏障"应对：

**问题一：规约无硬件支持。**
将规约执行位置从"SDMA任务队列"搬到"AICPU同步计算"。由通用搬运包装层`LocalReduce`集中判断：命中特殊场景则转调`AicpuReduce`（C++模板逐元素计算），否则走原生的`HcommLocalReduceOnThread`。上层模板代码对此无感知，仍统一调用`LocalReduce`。

**问题二：执行模型从异步变为同步，破坏既有依赖关系。**
AICPU模板的通信任务（Send/Recv/LocalCopy）以批量模式攒批异步执行；而`AicpuReduce`在CPU上同步直读内存，**不经过任务队列，与队列中尚未执行完的搬运任务没有任何依赖关系**。若 Gather/LocalCopy 尚未真正执行，规约将读到脏数据。

解决方案是在"通信阶段结束、本地规约开始"的临界点插入同步屏障：

```cpp
HcommBatchModeEnd(algTag);     // 结束批量模式：强制提交队列中攒批的所有搬运任务
HcommBatchModeStart(algTag);   // 重新进入批量模式，后续任务继续攒批
for (const auto& thread : threads) {
    HcommThreadJoin(thread, CUSTOM_TIMEOUT);  // 阻塞等待所有线程队列任务执行完成
}
```

三个接口均为HCOMM数据面"批量下发设置接口"（`hcomm_primitives.h`），`CUSTOM_TIMEOUT`默认1836秒（`src/ops/op_common/inc/alg_param.h:83`）。

### 3.2 分层处理框架

特殊Reduce场景在四个层次被协同处理，各层职责单一：

```text
┌─────────────────────────────────────────────────────────────────┐
│ L1 参数校验层  CheckReduceOp                                    │
│    PROD仅允许 INT8/INT32/INT64/UINT64/FP16/FP32/FP64            │
├─────────────────────────────────────────────────────────────────┤
│ L2 算法选择层  AutoSelector + CostTable                         │
│    ① 必不选规则：64bit/PROD 排除全部 CCU、AIV 算法（回退AICPU）  │
│    ② 多级拓扑下直接选定专用 AicpuReduce 算法                     │
│    ③ 单级Mesh拓扑复用通用Mesh模板，由模板内部处理                 │
├─────────────────────────────────────────────────────────────────┤
│ L3 模板执行层  AICPU Templates                                  │
│    在"通信完成后、本地规约前"插入同步屏障：                       │
│    BatchModeEnd → BatchModeStart → ThreadJoin                   │
├─────────────────────────────────────────────────────────────────┤
│ L4 规约执行层  LocalReduce → AicpuReduce                        │
│    特殊场景分流到AICPU软件逐元素归约（SUM/PROD/MAX/MIN）          │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 详细设计

### 4.1 参数校验层

入口处`CheckReduceOp`（`src/ops/op_common/op_common.cc:2971`）限定PROD支持的数据类型为：INT8、INT32、INT64、UINT64、FP16、FP32、FP64。类型与操作不匹配时直接返回`HCCL_E_NOT_SUPPORT`，不进入算法选择。

### 4.2 算法选择层

**必不选规则（CostTable）。** `src/ops/op_common/selector/cost_table.cc:77-88`定义两条硬规则，命中即从候选集中剔除全部CCU与AIV算法，强制回退AICPU：

| 规则名 | 触发条件 | 剔除范围 |
|--------|----------|----------|
| `prod_skip_ccu_aiv` | `reduceType == HCCL_REDUCE_PROD` | 全部CCU + AIV算法 |
| `64bit_skip_ccu_aiv` | `Is64BitDataType(dataType)` | 全部CCU + AIV算法 |

各算子的AutoSelector在CCU/AIV/MS分支内有等价排除逻辑（如`all_reduce_auto_selector.cc:196、621`）。

**AICPU算法选择。** 各算子selector在选择AICPU算法时优先判断特殊场景：

| 算子 | 多级拓扑 | 单级Mesh1D拓扑 |
|------|----------|----------------|
| AllReduce | `AicpuAllReduceSoleNHRAicpuReduce`（专用） | `AicpuAllReduceSoleMeshOneShot`/`TwoShot`（通用，模板内部处理；阈值`AR_AICPU_1D_64DATATYPE_DATA_SIZE`） |
| Reduce | `AicpuReduceSoleNHRAicpuReduce`（专用） | `AicpuReduceSoleMesh`（通用） |
| ReduceScatter | `AicpuReduceScatterSoleNHRAicpuReduce`（专用） | `AicpuReduceScatterSoleMesh`（通用） |

两类算法的差别仅在通信编排（专用算法为NHR递推倍增，且规约阶段无条件插屏障；通用模板按条件插屏障），规约执行路径完全一致。

### 4.3 模板执行层：同步屏障

**插入位置。** 所有"先通信搬运、后本地规约"的AICPU模板，在规约前判定特殊场景并插屏障。判定条件与插入模式在各模板中高度一致：

| 模板（src/ops/.../template/aicpu/） | 插入点 | 判定条件 |
|--------------------------------------|--------|----------|
| all_reduce/ins_temp_all_reduce_mesh_1D_one_shot.cc:199 | `PostLocalReduce`入口 | INT64/UINT64/FP64/PROD |
| all_reduce/ins_temp_all_reduce_mesh_1D_two_shot.cc:217 | 规约前 | INT64/UINT64/FP64/PROD |
| all_reduce/ins_temp_all_reduce_aicpu_reduce_nhr.cc:239 | `RunReduce`中LocalCopy后 | 无条件（专用算法） |
| all_reduce/ins_temp_reduce_scatter_mesh_1D_intra.cc:99 | 规约前 | INT64/UINT64/FP64/PROD |
| all_reduce/ins_temp_reduce_scatter_mesh_1D_dpu_inter.cc:254 | 规约前 | INT64/UINT64/FP64/PROD |
| reduce/reduce_mesh_1D.cc:117 | Gather后、ReduceData前 | INT64/UINT64/FP64/PROD |
| reduce/reduce_mesh_1D_two_shot.cc:229 | 规约前 | INT64/UINT64/FP64/PROD |
| reduce/reduce_aicpu_reduce_nhr.cc:223 | 规约前 | 无条件（专用算法） |
| reduce_scatter/ins_temp_reduce_scatter_mesh_1D.cc:125、180 | `KernelRun`尾部及`PostCopy`每个repeat内 | INT64/UINT64/FP64/PROD |
| reduce_scatter/ins_temp_reduce_scatter_mesh_1d_dpu.cc:230 | 规约前 | INT64/FP64/PROD |
| reduce_scatter/ins_temp_reduce_scatter_aicpu_reduce_nhr.cc:70 | 每个slice的AllGather后 | 无条件（专用算法） |
| reduce_scatter/ins_temp_reduce_scatter_order_preserved_level1.cc:121 | AllToAll后、LocalReduce前 | FP64/PROD |
| reduce_scatter/ins_temp_reduce_scatter_order_preserved_group.cc:125 | 同上 | FP64/PROD |
| reduce_scatter_v/ins_temp_reduce_scatter_v_mesh_1D.cc:105 | 规约前 | INT64/PROD |

**判定条件的差异化说明。** 个别模板的判定条件窄于"INT64/UINT64/FP64/PROD"全集，均有上游保证闭环：

- 保序模板（order_preserved）只判断FP64/PROD：因为保序严格模式入口`IsNeedStrictModeForOrderPreserved`（`src/common/order_preserved_common.h:63`）本身仅允许FP16/FP32/BFP16/FP64与SUM/PROD，INT64/UINT64不会进入保序模板。
- reduce_scatter_v的Mesh1D只判断INT64/PROD：V类算子的数据类型支持范围由算子特性另行限定。
- 屏障判断条件是`LocalReduce`分流条件（见4.4节）的**子集时才有意义**：模板只要在"会走AicpuReduce的场景"插了屏障即可；漏插会导致脏读，多插只损失性能。

### 4.4 规约执行层：AicpuReduce

**统一分流点。** `LocalReduce`（`src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc:901`）是所有模板本地规约的统一入口，命中特殊场景转`AicpuReduce`，否则走`HcommLocalReduceOnThread`硬件路径：

```cpp
HcclResult LocalReduce(thread, srcSlice, dstSlice, dataType, reduceOp)
{
    if (dataType == INT64 || dataType == UINT64 || dataType == FP64
        || reduceOp == HCCL_REDUCE_PROD) {
        return AicpuReduce(thread, srcSlice, dstSlice, dataType, reduceOp);  // 软件兜底
    }
    // 常规路径：SDMA硬件归约
    return HcommLocalReduceOnThread(thread, dst, src, count, dataType, reduceOp);
}
```

**AICPU软件归约实现。** `AicpuReduce`（同文件:1254）按数据类型分发到C++模板`AicpuReduceTemplate<T>`（同文件:1314），在AICPU上逐元素完成SUM/PROD/MAX/MIN。要点：

- **同步执行**：`thread`参数被显式忽略（`(void)thread`），不向任何任务队列下发，直接读写内存。这正是4.3节必须插同步屏障的原因。
- **FP16软件浮点**：AICPU上无FP16算术，`AicpuReduceFp16`先将src/dst按位转换为FP32（`Fp16ToFp32`），用FP32完成规约后转回FP16（`Fp32ToFp16`，round-to-nearest-even舍入，含非规格化数与上溢饱和处理）。
- **INT8/INT32的PROD无符号截断**：先按无符号位宽相乘再截断回原类型，与硬件行为对齐（`AicpuReduceTemplate`内`std::is_same`特化分支）。

### 4.5 AICPU Task Cache排除

AICPU任务缓存（算子展开缓存）依赖"下发序列可复放"。特殊Reduce场景在执行序列中间插入了BatchModeEnd/Start与ThreadJoin等干预动作，无法纳入缓存复放，因此`AicpuTaskCachePolicy::IsOpTypeSupported`（`src/ops/op_common/template/aicpu/task_cache/aicpu_task_cache_policy.cc:185`）显式禁止AllReduce/Reduce/ReduceScatter在64位类型或PROD下使能task cache。

## 5. 典型执行流程

### 5.1 通用模板：AicpuAllReduceSoleMeshOneShot（单级Mesh，小数据量）

```text
rank r (root流程，其余rank类似)                     数据流
─────────────────────────────────────────────────────────────
1. 主流 LocalCopy:  userIn ──────────────────► userOut
2. 各从流 SendRecvBatchWrite: 本卡userIn ──► 各对端cclBuff槽位
3. PostSyncInterThreads: 主从线程栅格同步
4. [特殊场景屏障] BatchModeEnd → BatchModeStart
                  └─ ThreadJoin(所有线程)     等待2.真正落盘
5. 对每个远端rank k:
   LocalReduce(cclBuff[k], userOut)           走AicpuReduce，
                                              CPU逐元素规约 userOut = userOut ⊕ cclBuff[k]
```

步骤4的屏障保证步骤5读到的`cclBuff[k]`是对端在步骤2写入并已完成的数据。

### 5.2 专用算法：AicpuAllReduceSoleNHRAicpuReduce（多级拓扑）

单线程执行（`GetThreadNum()==1`）：

1. `PreCopy`：userIn拷贝到本卡cclBuff自己的槽位；
2. `RunGather`：按NHR递推倍增（`deltaRank = 1 << (nSteps-1-step)`）交换切片，log(N)步后每卡集齐全部rank数据于cclBuff；
3. `LocalCopy`：本卡slice从cclBuff拷到userOut；
4. 无条件屏障：BatchModeEnd → BatchModeStart → ThreadJoin；
5. 逐个远端rank执行`LocalReduce`（AICPU软件归约）累加到userOut。

该专用算法`CalcCostCoeff`固定返回`A=10, B=0, C=0`并注明"用aicpu做reduce，不参与性能排序"——算法由selector按场景直接选定，不参与代价模型竞争。

### 5.3 端到端时序图

以下时序覆盖"调用AllReduce(FP64, SUM) → 算法选择 → 模板执行 → 软件规约"全过程，Reduce/ReduceScatter同理：

```mermaid
sequenceDiagram
    participant FW as 框架调用层
    participant PC as 参数校验<br/>CheckReduceOp
    participant SE as SelectorEngine
    participant TM as AICPU模板<br/>(以OneShot为例)
    participant WR as LocalReduce<br/>(搬运包装层)
    participant HC as HCOMM数据面<br/>(dlsym)

    FW->>PC: AllReduce(userIn, userOut, count, FP64, SUM)
    PC->>PC: CheckReduceOp(FP64, SUM)
    Note over PC: FP64不在PROD校验表外，<br/>64位类型本身合法，通过

    PC->>SE: 参数校验通过，进入算法选择
    SE->>SE: CostTable必不选规则
    Note over SE: 64bit_skip_ccu_aiv 命中<br/>全部CCU/AIV算法被标记filtered
    SE-->>TM: 选定 AicpuAllReduceSoleMeshOneShot

    TM->>HC: LocalCopy(userIn→userOut)
    TM->>HC: SendRecvBatchWrite(本卡userIn→对端cclBuff槽位)
    Note over TM,HC: 任务进入批量队列，异步攒批执行

    TM->>HC: 【同步屏障】HcommBatchModeEnd(algTag)
    Note over HC: 强制提交队列中攒批的搬运任务
    TM->>HC: 【同步屏障】HcommBatchModeStart(algTag)
    TM->>HC: 【同步屏障】HcommThreadJoin(thread, 1836s)
    Note over TM,HC: 阻塞等待所有线程任务真正执行完成，<br/>保证cclBuff数据已落盘

    loop 遍历每个远端rank k
        TM->>WR: LocalReduce(cclBuff[k], userOut, FP64, SUM)
        WR->>WR: 命中64位类型 → 分流至AicpuReduce
        Note over WR: AicpuReduceTemplate&lt;double&gt;<br/>CPU逐元素 SUM，同步直读内存
    end

    TM-->>FW: AllReduce完成，userOut为规约结果
```

关键点：同步屏障（图中标"同步屏障"的三次调用）是异步搬运世界与同步软件规约世界之间的"边界"，缺少它规约将读到脏数据。

### 5.4 关键日志（现场定位用）

按执行顺序给出各层代表性日志，HCCL日志级别默认INFO，搬运层明细需DEBUG：

**① 算法选择层（SelectorEngine，INFO）**

```text
[SelectorEngine] SelectMinCost: costTable count=12, opType=0, dataSize=1048576.
[SelectorEngine] | idx | algName                          | engine   | cost         | status   |
[SelectorEngine] |   0 | AicpuAllReduceSoleMeshOneShot    | AICPU    |         0.05 | valid    |
[SelectorEngine] |   1 | CcuSchedAllReduceSoleMesh       | CCU      |        -1.00 | filtered |   ← 64bit/PROD规则剔除
[SelectorEngine] |   2 | AivAllReduceMesh1D...           | AIV      |        -1.00 | filtered |   ← 同上
[SelectorEngine] The opExecuteConfig is AICPU, the selected algo type is AicpuAllReduceSoleMeshOneShot
```

cost为负表示被必不选规则剔除（`filtered`）。若最终选中算法非AICPU引擎而场景为64bit/PROD，属于配置异常，需检查`HCCL_ALGO`等环境变量覆盖。

**② 模板执行层（INFO）**

```text
[InsTempAllReduceMesh1DOneShot] Run Start
[KernelRun] sliceSize: 262144, count_: 32768, typeSize: 8        ← typeSize=8即64位类型的直观特征
[InsTempAllReduceMesh1DOneShot][RunAllReduce] send/recv: rank[3]
[InsTempAllReduceMesh1DOneShot][RunAllReduce] reduce: rank[3]    ← 进入PostLocalReduce，屏障即将执行
[InsTempAllReduceMesh1DOneShot][KernelRun] AllReduceMesh1DOneShot finished: rank[3] end
```

专用算法的对应日志前缀为`[InsTempAllReduceAicpuReduceNHR]`，另有`Use Dma Read[0/1]`（PCIe链路走Read模式）。

**③ 搬运与规约层（DEBUG）**

```text
[AlgDataTransWrapper][AicpuReduce][AICPU_REDUCE] sliceIdx[0], sliceNum[1], srcBase[0x7f...], srcAddr[0x7f...], srcCount[32768], dstBase[0x7f...], dstCount[32768], len[262144], dataType[10], reduceOp[0].
```

`transType=AICPU_REDUCE`即为软件归约路径的直接证据（常规硬件路径为`LOCAL_REDUCE`）；`dataType[10]`为FP64枚举值（INT64=5、UINT64=6、FP64=10，`reduceOp[0]`=SUM）。开启DEBUG级别后，每次搬运/规约均有此明细（`TraceDataSlice`），数据量大时日志量可观，仅建议定位问题时开启。

**④ Task Cache排除（INFO）**

```text
[AicpuTaskCachePolicy][IsOpTypeSupported] opType[0] is not supported, dataType[10] reduceOp[0]
```

出现该日志说明本场景被task cache排除，属预期行为。

## 6. 设计约束与注意事项

1. **引擎能力矩阵**：64位类型与PROD在CCU、AIV引擎均不支持（CostTable必不选规则），MS模式不支持PROD。新增引擎若底层硬件支持64位/PROD归约，需同步修订`IsSupportReduce`（HCOMM仓）与CostTable规则。
2. **性能特征**（介绍或调优时需说明）：
   - 同步屏障切断"搬运-规约"流水线，模板执行期间AICPU需忙等全部通信任务完成；
   - 通信模式退化为"先Gather全量到本卡、再本地规约"，无法使用WriteReduce搬运与规约融合，cclBuff scratch需求为`rankSize × sliceSize`（各模板`CalcScratchMultiple`返回`templateRankSize_`）；
   - 规约为CPU逐元素循环，带宽远低于SDMA，仅作正确性兜底。
3. **新增模板指引**：任何AICPU模板只要调用`LocalReduce`且规约数据来自异步搬运（Send/Recv/LocalCopy），就必须在规约前插入屏障；屏障判定条件必须是`LocalReduce`分流条件（INT64/UINT64/FP64/PROD）在模板可达场景上的覆盖，并注明上游范围限定（参照4.3节差异化的先例）。
4. **Task Cache互斥**：特殊Reduce场景不使能AICPU task cache，两机制不可同时生效。
5. **跨仓接口**：`HcommBatchModeStart/End`、`HcommThreadJoin`、`HcommLocalReduceOnThread`均由HCCL经`src/common/hcomm_dlsym/`符号表dlsym动态加载，遵循HCCL与HCOMM解耦约束，无编译期依赖。

## 7. 关键代码索引

| 机制 | 位置 |
|------|------|
| 64位类型判定`Is64BitDataType` | `src/ops/op_common/selector/auto_selector_base.h:119` |
| PROD类型校验`CheckReduceOp` | `src/ops/op_common/op_common.cc:2971` |
| CostTable必不选规则 | `src/ops/op_common/selector/cost_table.cc:77` |
| 专用算法选择 | `src/ops/all_reduce/selector/all_reduce_auto_selector.cc:427`（reduce/reduce_scatter的selector同理） |
| 同步屏障调用（典型） | `src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc:199` |
| 规约分流`LocalReduce` | `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc:901` |
| AICPU软件归约`AicpuReduce` | `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc:1254` |
| 逐元素模板`AicpuReduceTemplate` | `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc:1314` |
| FP16软浮点转换 | `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc:1077`（Fp16ToFp32/Fp32ToFp16） |
| Task Cache排除 | `src/ops/op_common/template/aicpu/task_cache/aicpu_task_cache_policy.cc:185` |
| 硬件归约能力边界`IsSupportReduce` | `cann/hcomm` `src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc:53` |
| 批量模式接口定义 | `cann/hcomm` `include/hcomm_primitives.h:498` |

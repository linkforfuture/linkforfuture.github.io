# InsTempAllGatherNHR 末步全 Read 优化回归测试设计

## 1. 测试目标

本测试计划用于验证 `InsTempAllGatherNHR` 最后一个 NHR step 从“Write 到本端 CCL Buffer，再
LocalCopy 到 Output”改为“直接从 `fromRank` CCL Buffer Read 到本端 Output”后的：

1. 数据正确性。
2. 非 2 幂 rankSize 下的 NHR 数据可达性。
3. Read、PostLocalCopy 并行时的同步闭合。
4. 尾片、多 loop、多 Channel 下的地址和长度计算。
5. 未命中优化条件时，原 NHR 路径保持不变。
6. 复用该模板的 AllReduce、Broadcast、Reduce 不发生回归。

本计划只描述场景设计和验收标准，不包含实际执行结果。

### 1.1 CheckerL2 适用范围

本计划限定使用 HCCL algorithm ST 当前接入 CheckerL2 的方式执行：测试用例通过
`SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950)` 初始化仿真世界，由测试侧
`TopoModel` 根据 `TopoMeta` 构造 RankGraph，CheckerL2 承担任务仿真和正确性校验。

当前链路仅保留 CheckerL2 可稳定表达和验证的以下拓扑：

```text
T1：两层 MESH_1D + NHR
T2：每 Server 1 Device 的 Pure NHR
T5：三层普通 Topo
T7：rankSize > 32 的逻辑一维通信域
```

UBX Mesh/CLOS、PCIe Mix 和 UBoE/Squeeze2D 当前不能由这条测试链路完整构造，相关场景已从
本计划删除。

## 2. 影响范围

### 2.1 直接实例化模板的算子

| 算子 | 是否实例化 `InsTempAllGatherNHR` | 新末步全 Read 是否可能生效 | 测试级别 |
|---|---|---|---|
| AllGather | 是 | 是 | P0，核心功能测试 |
| AllReduce | 是 | Parallel 最终 AG 阶段和 OrderPreservedGroup 可能生效 | P0，核心功能测试 |
| Broadcast | 是 | 当前最终 AG 输出类型为 `INPUT`，通常不生效 | P1，保护性回归 |
| Reduce | 是 | AG 阶段输出在 CCL Buffer，通常不生效 | P1，保护性回归 |
| Scatter | 否 | 不涉及 | P2，只做全量冒烟时覆盖 |
| ReduceScatter | 否 | 不涉及 | P2，只做全量冒烟时覆盖 |

CCU、DPU 使用各自的 AllGather NHR 模板；`InsTempAllGatherOmniPipeNHR` 虽然继承
`InsTempAllGatherNHR`，但重写了 `KernelRun` 和 NHR step 执行过程。因此，它们不直接执行本次
修改的 `RunLastStepReadToOutput`，不属于本次定向功能验证范围。

### 2.2 新分支命中条件

末步全 Read 需要同时满足：

```text
通信协议不是 PCIe
enableRemoteMemAccess == false
outBuffType == BufferType::OUTPUT
outputPtr != hcclBuff.addr
最后一个 step 的 nSlices > 1
```

CheckerL2 当前不构造 950 PCIe Mix Topo，因此本计划不覆盖“PCIe 协议导致不命中”这一保护分支；
保护性回归通过 `enableRemoteMemAccess`、非 `OUTPUT` 以及 `nSlices==1` 等条件完成。

最后一个 NHR step 的 `nSlices` 约为 `floor(templateRankSize / 2)`，因此：

| NHR templateRankSize | 最后一步 nSlices | 预期 |
|---:|---:|---|
| 2 | 1 | 不命中全 Read |
| 3 | 1 | 不命中全 Read |
| 4 | 2 | 最小正向命中值 |
| 5 | 2 | 命中，非 2 幂奇数 |
| 6 | 3 | 命中，非 2 幂偶数 |
| 7 | 3 | 命中，非 2 幂奇数 |
| 8 | 4 | 命中，2 幂 |

## 3. CheckerL2 Topo 构造说明

### 3.1 TopoMeta 表达方式和初始化入口

algorithm ST 使用三维结构描述通信域：

```text
TopoMeta[superPod][server][devicePhysicalId]
```

测试用例完成 `TopoMeta` 构造后统一调用：

```cpp
SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
```

CheckerL2 测试侧根据结构自动确定网络层：

```text
只有 1 个 Server       → Level 0
同一 SuperPod 多 Server → Level 0 + Level 1
多个 SuperPod           → Level 0 + Level 1 + Level 2
```

本计划中的记法为：

```text
4×5 = 1 个 SuperPod，5 个 Server，每个 Server 4 个 Device，共 20 rank
2×4×4 = 2 个 SuperPod，每个 SuperPod 4 个 Server，每个 Server 4 个 Device，共 32 rank
```

### 3.2 T1：两层 MESH_1D + NHR

T1 是本次最重要的 Topo。每个 Server 选择连续物理卡 `[0, 1, 2, 3]`，CheckerL2 将其构造成
Server 内 `MESH_1D`；同一 SuperPod 内放置 N 个 Server 后形成 Level 1，匹配
`TopoMatchMultilevel`。Level 1 同序号卡组成 NHR 子通信域，因此 `templateRankSize=N`。

可使用下面的辅助函数构造 4×N：

```cpp
TopoMeta Build4xNTopo(uint32_t serverNum)
{
    SuperPodMeta superPod;
    for (uint32_t i = 0; i < serverNum; ++i) {
        superPod.push_back({0, 1, 2, 3});
    }
    return TopoMeta{superPod};
}

TopoMeta topo4x3 = Build4xNTopo(3);
TopoMeta topo4x4 = Build4xNTopo(4);
TopoMeta topo4x5 = Build4xNTopo(5);
TopoMeta topo4x6 = Build4xNTopo(6);
TopoMeta topo4x7 = Build4xNTopo(7);
TopoMeta topo4x8 = Build4xNTopo(8);
```

其中 4×3 是末步优化的不命中边界，4×4 是最小命中边界，4×5～4×8 用于覆盖非 2 幂和
2 幂路径。

### 3.3 T2：每 Server 1 Device 的 Pure NHR

每个 Server 只放物理卡 0，使 Level 0 退化；多 Server 的 rank 被 `TopoMatch1D` 展平成一维
通信域，直接运行 `InsAllGatherNHR`：

```cpp
TopoMeta BuildPureNhrTopo(uint32_t rankSize)
{
    SuperPodMeta superPod;
    for (uint32_t i = 0; i < rankSize; ++i) {
        superPod.push_back({0});
    }
    return TopoMeta{superPod};
}

TopoMeta topo3 = BuildPureNhrTopo(3);
TopoMeta topo4 = BuildPureNhrTopo(4);
TopoMeta topo5 = BuildPureNhrTopo(5);
TopoMeta topo8 = BuildPureNhrTopo(8);
```

### 3.4 T5：三层普通 Topo

多个 SuperPod 会使 CheckerL2 测试侧生成 Level 2。标准 `2×4×4` 使用：

```cpp
TopoMeta topoMeta;
GenTopoMeta(topoMeta, 2, 4, 4);
// 等价于：2 个 SuperPod，每个 SuperPod 4 个 Server，每个 Server 使用物理卡 0～3
```

该构造用于 `TopoMatchMultilevel` 三层 Sequence 保护性回归。这里限定使用 algorithm ST 的
`SimWorld/TopoModel` 接入；CheckerL2 `hccl-vm mock-comm` 自动生成 ranktable 的路径目前只处理
Level 0 和 Level 1，不用于本场景。

### 3.5 T7：rankSize > 32 的逻辑一维通信域

使用 1 个 SuperPod、33 或 36 个 Server、每个 Server 1 个 Device：

```cpp
TopoMeta topo33;
GenTopoMeta(topo33, 1, 33, 1);

TopoMeta topo36;
GenTopoMeta(topo36, 1, 36, 1);
```

不要使用“1 个 Server 内放 33/36 个 Device”的写法；950 测试模型会按 8×8 网格拆分物理卡，
从而形成二维拓扑，不再满足本场景要求。T7 除拓扑外还必须配置 `DETERMINISTIC_STRICT` 和受支持的
dtype/reduceOp，才能命中 `AllReduceOrderPreservedGroup`。

### 3.6 本次保留的 Topo 类型

| Topo 编号 | CheckerL2 构造 | 关键属性 | 匹配器 | 主要用途 |
|---|---|---|---|---|
| T1 | `Build4xNTopo(N)`，N=3～8 | `topoLevelNums=2`、`level0Topo=MESH_1D` | `TopoMatchMultilevel` | 4×N Parallel 主路径 |
| T2 | `BuildPureNhrTopo(N)`，N=3/4/5/8 | Level 0 退化，逻辑一维 | `TopoMatch1D` | `InsAllGatherNHR` Sole 路径 |
| T5 | `GenTopoMeta(topo, 2, 4, 4)` | `topoLevelNums=3` | `TopoMatchMultilevel` | 三层 Sequence 保护性回归 |
| T7 | `GenTopoMeta(topo, 1, 33或36, 1)` | `userRankSize > 32`、逻辑一维 | `TopoMatch1D` | AllReduce OrderPreservedGroup |

## 4. P0 最小必测集

以下 16 个场景构成建议的最小必测集。已有并确认有效的日志可以复用，不需要重复执行。

### 4.1 AllGather 核心场景

| ID | 算法 | Topo | Topo 类型 | 数据量/单 rank | dtype | 模式 | 全 Read 预期 | 覆盖点 |
|---|---|---|---|---:|---|---|---|---|
| AG-P-01 | `InsAllGatherParallelMesh1DNHR` | 4×3 | T1 Multilevel Mesh+NHR | 32 MiB | int8 | OPBASE，out-of-place | 否 | `nSlices==1` 保护边界 |
| AG-P-02 | 同上 | 4×4 | T1 | 32 MiB | int8 | OPBASE，out-of-place | 是 | 最小命中值 |
| AG-P-03 | 同上 | 4×5 | T1 | 32 MiB | int8 | OPBASE，out-of-place | 是 | 非 2 幂奇数 |
| AG-P-04 | 同上 | 4×6 | T1 | 32 MiB | int8 | OPBASE，out-of-place | 是 | 非 2 幂偶数 |
| AG-P-05 | 同上 | 4×7 | T1 | 32 MiB | int8 | OPBASE，out-of-place | 是 | 更长奇数路径 |
| AG-P-06 | 同上 | 4×8 | T1 | 32 MiB | int8 | OPBASE，out-of-place | 是 | 2 幂、多 rx slice |
| AG-P-07 | 同上 | 4×5 | T1 | `32 MiB + 3 B` | int8 | OPBASE，out-of-place | 是 | 非对齐尾片、多 loop |
| AG-P-08 | 同上 | 4×5 | T1 | 2 MiB | fp16 | OPBASE，out-of-place | 是 | dtype、较小数据、单/少 loop |

AG-P-02～AG-P-06 已由现有 `allgather_new_opt_4x4～4x8_32mb_int8.log` 覆盖。新增测试重点是
AG-P-01、AG-P-07 和 AG-P-08。

AG-P-01～AG-P-08 均按表中 Server 数调用 `Build4xNTopo(N)` 构造 CheckerL2 Topo。

> AG-P-07 使用 int8，`sendCount` 可直接设置为 `32 * 1024 * 1024 + 3`。

### 4.2 AllGather Sole NHR 边界场景

| ID | 算法 | Topo | Topo 类型 | sendCount | dtype | 全 Read 预期 | 覆盖点 |
|---|---|---|---|---:|---|---|---|
| AG-S-01 | `InsAllGatherNHR` | 1 Device/Server × 3 Server | T2 Pure NHR | 101 | fp32 | 否 | rankSize=3、`nSlices==1` |
| AG-S-02 | `InsAllGatherNHR` | 1 Device/Server × 4 Server | T2 | 101 | fp16 | 是 | Sole 最小正向命中 |
| AG-S-03 | `InsAllGatherNHR` | 1 Device/Server × 5 Server | T2 | 101 | int8 | 是 | Sole 非 2 幂、非对齐 count |
| AG-S-04 | `InsAllGatherNHR` | 1 Device/Server × 8 Server | T2 | 100 | bfp16 | 是 | Sole 2 幂、多 rx slice |

仓库已有 rankSize 3、4、8 的 Sole NHR ST，可在原用例基础上增加日志断言；rankSize 5 建议新增。
AG-S-01～AG-S-04 分别调用 `BuildPureNhrTopo(3/4/5/8)`。

### 4.3 AllReduce Parallel 核心场景

| ID | 算法 | Topo | Topo 类型 | dataCount/数据量 | dtype | reduceOp | 全 Read 预期 | 覆盖点 |
|---|---|---|---|---|---|---|---|---|
| AR-P-01 | `InsAllReduceParallelRSAG` | 4×3 | T1 | 64 MiB | fp16 | sum | 否 | AG NHR rankSize=3 保护边界 |
| AR-P-02 | 同上 | 4×4 | T1 | 64 MiB | fp16 | sum | 是 | AllReduce 最小命中 |
| AR-P-03 | 同上 | 4×5 | T1 | `32 Mi + 1` 个 fp16，约 64 MiB+2 B | fp16 | sum | 是 | 非 2 幂、尾片、非整除 count |
| AR-P-04 | 同上 | 4×8 | T1 | 64 MiB | fp32 | max | 是 | 2 幂、dtype/reduceOp 变化 |

AllReduce AICPU 自动选择 Parallel 的单 rank 数据量条件是大于 32 MiB，建议使用 64 MiB，并从
日志确认算法名；不要使用恰好 32 MiB 作为唯一 Parallel 用例。

AR-P-01～AR-P-04 分别按表中 Server 数调用 `Build4xNTopo(N)`。

## 5. P1 扩展和保护性回归

### 5.1 AllGather Topo 扩展

| ID | 目标算法 | Topo 类型 | CheckerL2 构造 | 数据量 | 全 Read 预期 | 目的 |
|---|---|---|---|---:|---|---|
| AG-X-01 | `InsAllGatherSequenceNHRMesh1D` | T1 | `Build4xNTopo(4)` | 由算法名强制或超过选择阈值 | 否 | NHR 输出到 CCL Buffer |
| AG-X-02 | `InsAllGatherSequenceNHRNHRMesh1D` | T5 | `GenTopoMeta(topo, 2, 4, 4)` | 非对齐 count | 否 | 三层 Sequence 保护 |

### 5.2 AllReduce 扩展

| ID | 目标算法 | Topo 类型 | CheckerL2 构造 | 参数 | 全 Read 预期 | 目的 |
|---|---|---|---|---|---|---|
| AR-X-01 | `InsAllReduceSequenceMesh1DNhr` | T1 | `Build4xNTopo(4)` | 强制算法或大于 Sequence 阈值 | 否 | NHR 中间结果仍在 CCL |
| AR-X-02 | `InsV2AllReduceSequenceMesh1DNHRNHR` | T5 | `GenTopoMeta(topo, 2, 4, 4)` | 非对齐 count、fp32/sum | 否 | 三层 Sequence 保护 |
| AR-X-03 | `AllReduceOrderPreservedGroup` | T7 | `GenTopoMeta(topo, 1, 33或36, 1)` | STRICT、fp16/sum、非整除 count | 是 | 大通信域和 tailSize |

`AllReduceOrderPreservedGroup` 需要：

```text
DETERMINISTIC_STRICT
rankSize > 32
dtype 为 fp16/fp32/bfp16/fp64
reduceOp 为 sum/prod
```

### 5.3 Broadcast 保护性回归

Broadcast Parallel 虽然复用 `InsTempAllGatherNHR`，但最终 AllGather 阶段使用
`outBuffType=INPUT`，不应进入末步全 Read。

| ID | 目标算法 | Topo | Topo 类型 | 数据量 | dtype | root | 全 Read 预期 |
|---|---|---|---|---:|---|---:|---|
| BR-R-01 | `InsBroadcastParallelMesh1DNHR` | 4×4 | T1 | 32 MiB | int8 | 0 | 否 |
| BR-R-02 | 同上 | 4×5 | T1 | `32 MiB + 3 B` | int8 | 7 | 否 |

BR-R-01 使用 `Build4xNTopo(4)`，BR-R-02 使用 `Build4xNTopo(5)`，是 CheckerL2 下建议必做的
Broadcast 最小回归。

### 5.4 Reduce 保护性回归

Reduce Parallel 的 AllGather 阶段输入、输出都在 CCL Buffer，最终由 root 做本地拷贝，因此不应
进入末步全 Read。

| ID | 目标算法 | Topo | Topo 类型 | 数据量 | dtype | reduceOp | root | 全 Read 预期 |
|---|---|---|---|---:|---|---|---:|---|
| RD-R-01 | `ReduceParallelMesh1DNHR` | 4×4 | T1 | 64 MiB | fp16 | sum | 0 | 否 |
| RD-R-02 | 同上 | 4×5 | T1 | `64 MiB + 4 B` | fp32 | max | 7 | 否 |

RD-R-01 使用 `Build4xNTopo(4)`，RD-R-02 使用 `Build4xNTopo(5)`，是 CheckerL2 下建议必做的
Reduce 最小回归。

### 5.5 OFFLOAD/RemoteMemAccess

| ID | 算子/算法 | Topo | 参数 | 预期 |
|---|---|---|---|---|
| MODE-01 | AllGather Parallel | 4×4，T1 | `opMode=OFFLOAD` | `enableRemoteMemAccess=true`，不命中全 Read |
| MODE-02 | AllGather Sole NHR | Pure NHR 4 rank，T2 | `opMode=OFFLOAD` | 不命中全 Read |
| MODE-03 | AllReduce Parallel | 4×4，T1 | `opMode=OFFLOAD` | 重点确认实际是否支持，以及参数是否正确传入 |

AllReduce Parallel 的数据参数构造中没有明显看到 `enableRemoteMemAccess` 的显式赋值。如果该算法
允许 OFFLOAD，MODE-03 应作为高风险用例：预期不能因为字段保持默认值而错误进入末步全 Read。

## 6. 数据维度覆盖原则

### 6.1 数据量

至少覆盖以下四类：

| 类型 | 建议值 | 目的 |
|---|---|---|
| 极小数据 | 100 或 101 elements | Sole NHR、边界地址 |
| Parallel 小/中数据 | AllGather 2 MiB | 单 loop 或较少 loop |
| Parallel 大数据 | AllGather 32 MiB；AllReduce 64 MiB | 多 loop、主优化路径 |
| 非对齐数据 | `32 MiB+3 B`、`32 Mi+1` fp16 elements | tailSize、偏移、最后一 loop |

需要从日志同时覆盖：

```text
loopTimes == 1
loopTimes > 1
```

实际 loop 边界由 CCL Buffer 大小、templateRankSize 和 Channel 数共同决定。若上述建议数据没有
形成两种 loop 数，应根据日志中的 `maxCountPerLoop` 调整数据量，而不是固定依赖某个字节阈值。

### 6.2 dtype

| dtype | 必要性 | 说明 |
|---|---|---|
| int8 | 必测 | 与现有 32 MiB 日志一致，offset 最直观 |
| fp16 | 必测 | AllReduce 主流规约类型，2 字节元素 |
| fp32 | 必测 | 4 字节元素，配合 max/sum |
| bfp16 | 建议 | 复用现有 Sole NHR 用例 |
| int64/fp64/fp8 | 可选 | 更多用于 selector 和 datatype 全量回归，不是本次搬运逻辑核心 |

### 6.3 Buffer 与执行模式

至少覆盖：

```text
out-of-place OPBASE
in-place（接口和算法支持时）
OFFLOAD/RemoteMemAccess 保护场景
```

## 7. 算法选择验收

不能只根据 Topo 和数据量推断算法。每个用例都应从日志确认实际 `algName` 与设计一致。

关键自动选择条件：

| 算子 | 场景 | 选择条件摘要 |
|---|---|---|
| AllGather | T1 两层 `MESH_1D` Parallel | 单 rank 数据量 `>1 MiB`，且 `dataSize × userRankSize <= 4 GiB` |
| AllGather | T1 两层 Sequence | `dataSize × userRankSize > 4 GiB` |
| AllGather | T2 Pure NHR | 每 Server 1 Device，Level 0 退化 |
| AllGather | T5 三层 Sequence | `topoLevelNums=3`，确认选择 `InsAllGatherSequenceNHRNHRMesh1D` |
| AllReduce | T1 Parallel | 单 rank 数据量 `>32 MiB` 且不超过 Sequence 阈值 |
| AllReduce | T1 Sequence | 单 rank 数据量 `>4 GiB` |
| AllReduce | T5 三层 Sequence | `topoLevelNums=3`，确认选择 `InsV2AllReduceSequenceMesh1DNHRNHR` |
| AllReduce | T7 OrderPreservedGroup | `DETERMINISTIC_STRICT`、`rankSize > 32`、dtype/reduceOp 满足限制 |

如果 ST 支持直接指定算法，优先直接指定目标 `algName`；如果依赖 AutoSelector，则用日志对选择结果
做强校验，选择到其他算法的用例不能计入本模板覆盖率。

## 8. 日志和任务验收标准

### 8.1 正向命中场景

应出现：

```text
[InsTempAllGatherNHR] Read last step to output[1]
```

并满足：

1. 最后一步所有 `rx[0..nSlices-1]` 都从同一个 `fromRank` 的远端 CCL Buffer Read。
2. Read 目标地址属于本地 Output，offset 与 `rxIdx` 对应。
3. 最后一步不再出现原 `i==0` 的 Write。
4. PostLocalCopy 跳过所有末步 Read slice。
5. PostLocalCopy 仍复制前序已进入本地 CCL、但未被 Read 直达 Output 的 slice。
6. 所有 rank 结果校验 PASS，无超时、死锁、越界或 notify 不闭合。

相对旧 Write+Read 方案，每个 repeat 预期：

```text
Write       -1
Read        +1
LocalCopy   -1
```

对应 Write 的 Channel Post/Wait 也应下降，而 PostLocalCopy 线程的 LocalPostTo/LocalWaitFrom 应保持
闭合。

### 8.2 保护性场景

本计划保留的保护性场景满足 RemoteMemAccess、非 Output 或 `nSlices==1` 任一条件时，应出现：

```text
[InsTempAllGatherNHR] Read last step to output[0]
```

或者虽计算出 `readLastStepToOutput_=true`，但因最后一步 `nSlices==1` 仍执行普通 NHR step。此时不应
观察到 `RunLastStepReadToOutput` 的全量 Read 任务结构。

### 8.3 正确性验收

| 算子 | 验收内容 |
|---|---|
| AllGather | 每个 rank 的 Output 按 rank 顺序包含所有输入分片 |
| AllReduce | 每个 rank 的 Output 都等于全 rank 规约结果 |
| Broadcast | 每个 rank 的结果与 root 输入一致 |
| Reduce | 只有 root Output 等于规约结果；非 root 不误写用户 Output |

## 9. 建议执行顺序

### 第一批：最快确认修改正确

```text
AG-P-01 ～ AG-P-08
AR-P-01 ～ AR-P-04
```

其中 AG-P-02～AG-P-06 已有日志，可优先补齐 AG-P-01、AG-P-07、AG-P-08 和全部 AllReduce
Parallel 场景。

### 第二批：共享模板回归

```text
AG-S-01 ～ AG-S-04
BR-R-01、BR-R-02
RD-R-01、RD-R-02
MODE-01、MODE-02、MODE-03
```

### 第三批：CheckerL2 支持的多层和大通信域扩展

```text
AG-X-01、AG-X-02
AR-X-01、AR-X-02、AR-X-03
```

## 10. 准出标准

满足以下条件后，可认为模板修改完成基础回归：

1. 4×3 不命中边界和 4×4 最小命中边界均正确。
2. 4×5、4×6、4×7 非 2 幂路径和 4×8 2 幂路径全部 PASS。
3. 至少一组非对齐数据和一组多 loop 数据 PASS。
4. AllGather Sole NHR 的 rankSize 3、4、5 至少各一组 PASS。
5. AllReduce Parallel 的 4×4、4×5、4×8 至少各一组 PASS。
6. Broadcast、Reduce 各完成 root=0 和非 0 root 的保护性回归。
7. 至少一组 RemoteMemAccess 保护场景确认未误入优化分支。
8. 正向用例的任务结构符合“Write -1、Read +1、LocalCopy -1”，且所有同步任务闭合。
9. 所有用例无超时、死锁、内存越界和结果校验失败。

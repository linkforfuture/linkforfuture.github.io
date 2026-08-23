# TopoMatchMultilevel三级HostDPU拓扑降级两级SRS-SD

- 文档状态：设计评审稿
- 编写日期：2026-07-28
- 适用仓库：`cann/hccl`
- 涉及模块：`src/ops/op_common/topo`
- 相关Issue/PR：待补充

---

## 1. 文档目的

本文档给出`TopoMatchMultilevel`在HostDPU场景下将3级物理topo适配为2级算法topo的需求规格和软件设计。

本文档以AllReduce算法`InsAllReduceSequenceMeshNhrDPU`为主要示例，同时覆盖所有复用
`TopoMatchMultilevel`的2级HostDPU算法。

## 2. 术语

| 术语 | 含义 |
|------|------|
| 物理层 | RankGraph提供的网络层，例如物理L0、物理L1、物理L2 |
| 算法层 | `AlgHierarchyInfoForAllLevel::infos`表达的算法通信层 |
| 3级topo | `topoInfo->topoLevelNums == 3`的物理拓扑 |
| HostDPU | 最高物理层仅存在Host Endpoint、没有Device Endpoint，并且最高层覆盖通信域全部Rank |
| 拓扑降级 | 不修改物理RankGraph和`TopoInfoWithNetLayerDetails`，仅把3级物理拓扑映射为2级算法拓扑 |
| 对称topo | 物理L0各实例大小一致，且物理L1各实例大小一致 |

## 3. 背景

### 3.1 软件分层

本方案位于HCCL集合通信算子侧的topo适配组件：

```text
HCCL算子入口
    ↓
算法选择器
    ↓
Executor
    ↓
TopoMatchMultilevel          ← 本方案修改位置
    ↓
通过dlsym查询HCOMM RankGraph
```

方案遵循[HCCL架构约束](./architecture-brief.md)：

- 不修改HCOMM代码，不引入HCOMM私有头文件。
- 继续通过现有dlsym接口查询RankGraph。
- 不改变控制面RankGraph数据，只在HCCL算子侧生成算法分层信息。
- 不修改对外API。

### 3.2 AllReduce调用链

```text
HcclAllReduce
  └─ AllReduceOutPlace
      └─ Selector
          ├─ HcclCalcTopoInfo
          └─ ExecuteSelector
              └─ AutoSelectorBase::Select
                  ├─ CheckHostDPUOnly
                  └─ AllReduceAutoSelector::SelectDPUAlgo
                      └─ InsAllReduceSequenceMeshNhrDPU
                          └─ InsV2AllReduceSequenceExecutor
                              ├─ CalcAlgHierarchyInfo
                              │   └─ TopoMatchMultilevel::MatchTopo
                              ├─ CalcRes
                              └─ Orchestrate
```

相关代码：

- [`all_reduce_op.cc`](../../../src/ops/all_reduce/all_reduce_op.cc)
- [`auto_selector_base.cc`](../../../src/ops/op_common/selector/auto_selector_base.cc)
- [`all_reduce_auto_selector.cc`](../../../src/ops/all_reduce/selector/all_reduce_auto_selector.cc)
- [`ins_v2_all_reduce_sequence_executor.cc`](../../../src/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor.cc)
- [`topo_match_multilevel.cc`](../../../src/ops/op_common/topo/topo_match_multilevel.cc)

## 4. 问题描述

### 4.1 Executor只有2个算法层

`InsV2AllReduceSequenceExecutor`的算法步骤为：

```text
算法L0：框内ReduceScatter Mesh1D
算法L1：框间ReduceScatter DPU
算法L1：框间AllGather DPU
算法L0：框内AllGather Mesh1D
```

该Executor只使用`algHierarchyInfo.infos[0]`和`algHierarchyInfo.infos[1]`，资源请求也只包含2层Channel：

```cpp
resourceRequest.channels = {
    resReqReduceScatterMesh1D.channels[0],
    resReqReduceScatterMesh1dDpu.channels[0]
};
```

### 4.2 当前TopoMatchMultilevel仍输出3层

当前HostDPU分支已经把算法L1使用的网络层切换到最高物理层：

```cpp
netLayer = topoInfo->netLayerDetails.netLayers[
    topoInfo->netLayerDetails.netLayerNum - 1];
```

因此Rank集合层面已经在尝试合并物理L1和物理L2。但是，当`topoLevelNums == 3`时，当前实现仍然：

1. 执行`algHierarchyInfo.infos.resize(3)`；
2. 生成算法L0；
3. 使用最高物理层生成算法L1；
4. 继续调用`TopoForLayer2`生成算法L2。

最终形成“2级Executor消费3级`AlgHierarchyInfo`”的不一致。

### 4.3 故障表现

不同Executor的故障表现可能不同：

- `InsV2ReduceScatterSequenceExecutor`会检查`infos.size() == 2`，当前会直接返回内部错误。
- AllReduce、AllGather、Reduce、Broadcast、Scatter和Barrier的2级Executor申请2层Channel，但
  `RestoreChannelMap`按`infos.size()`遍历。`infos.size() == 3`时会访问不存在的`channels[2]`。
- 即使未立即触发异常，多出的算法L2也没有对应DPU算法步骤，语义上无效。

## 5. SRS：需求规格

### 5.1 功能需求

#### SRS-FR-001 降级触发条件

仅当以下条件同时满足时执行3级到2级降级：

```text
hostDPUOnly == true
topoInfo->topoLevelNums == 3
```

不得仅根据`topoLevelNums == 3`执行降级。

#### SRS-FR-002 降级映射

降级方向固定为：

```text
算法L0 = 物理L0
算法L1 = 物理最高层，合并物理L1和物理L2
```

HostDPU最高层必须覆盖通信域全部Rank。算法L1继续按照相同物理L0序号筛选Rank。

#### SRS-FR-003 输出层数

降级成功后：

```cpp
algHierarchyInfo.infos.size() == 2;
```

不得生成`infos[2]`，不得调用`TopoForLayer2`。

#### SRS-FR-004 非对称topo

HostDPU不支持非对称topo。

当3级HostDPU拓扑满足以下任一条件时，应返回`HCCL_E_NOT_SUPPORT`：

- 物理L0各实例大小不一致；
- 物理L1各实例大小不一致。

对称性检查必须基于物理L0和物理L1，不能使用降级后的算法L1代替物理L1。最高物理层通常只有一个覆盖
全部Rank的实例，仅检查最高层会掩盖物理L1非对称。

#### SRS-FR-005 非HostDPU三级算法兼容

当`hostDPUOnly == false && topoLevelNums == 3`时，必须保持现有3级算法拓扑：

```text
算法L0 = 物理L0
算法L1 = 物理L1
算法L2 = 物理L2
```

输出`algHierarchyInfo.infos.size() == 3`。

#### SRS-FR-006 现有1级和2级场景兼容

现有1级、2级topo处理逻辑不得因本特性发生行为变化。2级HostDPU仍输出2个算法层。

#### SRS-FR-007 诊断日志

发生降级时，应记录INFO日志，至少包含：

- 物理topo层数；
- 输出算法层数；
- 算法L1使用的物理网络层；
- 当前Rank。

非对称topo被拒绝时，应记录物理L0、物理L1的对称性检查结果。

### 5.2 非功能需求

| 编号 | 要求 |
|------|------|
| SRS-NFR-001 | 不新增或修改HCCL对外API |
| SRS-NFR-002 | 不修改`TopoInfoWithNetLayerDetails::topoLevelNums`，该字段继续表达物理topo |
| SRS-NFR-003 | 不修改HCOMM RankGraph，不引入跨仓编译依赖 |
| SRS-NFR-004 | 不新增环境变量或特性开关 |
| SRS-NFR-005 | 降级判断只增加常数级控制逻辑，不引入额外通信或网络调用 |
| SRS-NFR-006 | 代码保持C++14兼容，并符合仓库`.clang-format` |

### 5.3 不在本需求范围内

- 为DPU新增真正的3级算法。
- 支持HostDPU非对称topo。
- 修改RankGraph层级或Endpoint定义。
- 使用`TopoMatchSqueeze2D`替代`TopoMatchMultilevel`。
- 无条件将所有3级topo降为2级。
- 重构所有Executor的层数校验机制。

## 6. SD：软件设计

### 6.1 总体方案

在`TopoMatchMultilevel::MatchTopo`内部区分3个概念：

```text
physicalLayer1NetLayer：物理L1，用于对称性检查
algLayer1NetLayer：算法L1实际使用的RankGraph网络层
needDowngrade：是否将3级物理topo输出为2级算法topo
```

普通场景：

```text
physicalLayer1NetLayer = netLayers[1]
algLayer1NetLayer      = netLayers[1]
needDowngrade          = false
```

3级HostDPU场景：

```text
physicalLayer1NetLayer = netLayers[1]
algLayer1NetLayer      = netLayers[layerNum - 1]
needDowngrade          = true
```

### 6.2 关键处理流程

```text
开始
  │
  ├─ 校验topoLevelNums范围
  │
  ├─ 查询netLayers
  │
  ├─ 查询物理L0实例大小并检查对称性
  │
  ├─ topoLevelNums > 1？
  │    ├─ 否：保持现有流程
  │    └─ 是：
  │         ├─ 使用netLayers[1]检查物理L1对称性
  │         └─ CheckHostDPUOnly
  │              ├─ false：算法L1使用物理L1
  │              └─ true：算法L1使用最高物理层
  │
  ├─ 3级且L0/L1非对称？
  │    ├─ 是：返回HCCL_E_NOT_SUPPORT
  │    └─ 否：继续
  │
  ├─ needDowngrade = hostDPUOnly && topoLevelNums == 3
  │
  ├─ 输出算法层数
  │    ├─ needDowngrade：2
  │    ├─ 普通3级：3
  │    └─ 其他：保持现有2级输出
  │
  ├─ 生成算法L0
  │
  ├─ 使用algLayer1NetLayer生成算法L1
  │
  └─ 普通3级且未降级？
       ├─ 是：生成算法L2
       └─ 否：结束
```

### 6.3 伪代码

```cpp
uint32_t physicalLayer1NetLayer = netLayers[1];
uint32_t algLayer1NetLayer = physicalLayer1NetLayer;

bool hostDPUOnly = false;
HcclResult dpuRet = CheckHostDPUOnly(comm, topoInfo, hostDPUOnly);
if (dpuRet == HCCL_SUCCESS && hostDPUOnly) {
    algLayer1NetLayer = netLayers[layerNum - 1];
}

bool layer0Symmetric = CheckPhysicalLayer0Symmetry();
bool layer1Symmetric = CheckLayerSymmetry(physicalLayer1NetLayer);

if (topoInfo->topoLevelNums == COMM_LAYER_SIZE_3 &&
    (!layer0Symmetric || !layer1Symmetric)) {
    return HCCL_E_NOT_SUPPORT;
}

bool needDowngrade =
    hostDPUOnly && topoInfo->topoLevelNums == COMM_LAYER_SIZE_3;

uint32_t algLayerNum = needDowngrade
    ? COMM_LAYER_SIZE_2
    : (topoInfo->topoLevelNums == COMM_LAYER_SIZE_3
        ? COMM_LAYER_SIZE_3
        : COMM_LAYER_SIZE_2);

algHierarchyInfo.infos.resize(algLayerNum);

TopoForLayer0(...);
TopoForLayer1(comm, algLayer1NetLayer, ...);

if (topoInfo->topoLevelNums == COMM_LAYER_SIZE_3 && !needDowngrade) {
    TopoForLayer2(...);
}
```

实际实现继续使用项目现有`CHK_RET`和`CHK_PRT_RET`错误处理宏。

### 6.4 Rank映射示例

物理拓扑为：

```text
2个超节点 × 每超节点2个Server × 每Server 4个Rank
```

Rank顺序为：

```text
超节点0：
  Server0：[0, 1, 2, 3]
  Server1：[4, 5, 6, 7]

超节点1：
  Server0：[8, 9, 10, 11]
  Server1：[12, 13, 14, 15]
```

对rank 0：

```text
算法L0：[0, 1, 2, 3]
算法L1：[0, 4, 8, 12]
```

对rank 5：

```text
算法L0：[4, 5, 6, 7]
算法L1：[1, 5, 9, 13]
```

降级前后对比：

| 项目 | 当前行为 | 修改后行为 |
|------|----------|------------|
| `topoInfo->topoLevelNums` | 3 | 3 |
| `infos.size()` | 3 | 2 |
| 算法L0 | 当前Server内Rank | 不变 |
| 算法L1 | 最高物理层同序号Rank | 不变 |
| 算法L2 | 通常退化为单Rank | 不再生成 |
| Channel层数 | 2 | 2 |

### 6.5 修改文件

#### 6.5.1 必须修改

[`src/ops/op_common/topo/topo_match_multilevel.cc`](../../../src/ops/op_common/topo/topo_match_multilevel.cc)

- 区分物理L1和算法L1使用的网络层。
- 物理L1对称性始终基于`netLayers[1]`检查。
- 增加`needDowngrade`判断。
- 降级时将`algHierarchyInfo.infos`调整为2层。
- 降级时跳过`TopoForLayer2`。
- 增加降级日志。

[`test/st/algorithm/testcase/all_reduce_dpu_testcase.cc`](../../../test/st/algorithm/testcase/all_reduce_dpu_testcase.cc)

- 将测试Rank总数计算改为遍历所有超节点，或复用`CalRankSize`。
- 新增3级对称HostDPU AllReduce用例。
- 新增3级非对称HostDPU拒绝用例。

#### 6.5.2 无需修改

- `topo_match_multilevel.h`：不新增成员和接口。
- `ins_v2_all_reduce_sequence_executor.cc`：仍按既有2级算法执行。
- `TopoInfoWithNetLayerDetails`：物理topo信息保持不变。
- HCOMM仓及dlsym符号表：不涉及新接口。

### 6.6 共享调用方影响

以下2级HostDPU算法复用`TopoMatchMultilevel`，会共同获得修正后的2级`AlgHierarchyInfo`：

| 算子 | HostDPU算法 |
|------|-------------|
| AllReduce | `InsAllReduceSequenceMeshNhrDPU` |
| AllGather | `InsAllGatherMeshNhrDPU` |
| ReduceScatter | `InsReduceScatterSequenceMeshMeshDPU` |
| Reduce | `InsReduceSequenceMeshNhrDPU` |
| Broadcast | `InsBroadcastSequenceMeshNhrDPU` |
| Scatter | `InsScatterSequenceMeshNhrDPU` |
| Barrier | `InsBarrierMeshNhrDPU` |

普通3级AICPU算法虽然也复用`TopoMatchMultilevel`，但因`hostDPUOnly == false`，仍输出3层，不进入降级分支。

## 7. 异常和边界场景

| 场景 | 预期行为 |
|------|----------|
| 2级HostDPU对称topo | 保持现有2级输出 |
| 3级HostDPU对称topo | 降级为2级算法topo |
| 3级HostDPU物理L0非对称 | 返回`HCCL_E_NOT_SUPPORT` |
| 3级HostDPU物理L1非对称 | 返回`HCCL_E_NOT_SUPPORT` |
| 3级非HostDPU对称topo | 保持3级算法topo |
| 3级非HostDPU非对称topo | 保持现有不支持行为 |
| 最高层不能覆盖全部Rank | `CheckHostDPUOnly`判定为false，不进入DPU降级 |
| 最高层存在Device Endpoint | `CheckHostDPUOnly`判定为false，不进入DPU降级 |
| `topoLevelNums > 3`或为0 | 保持现有参数校验错误 |

## 8. 测试方案

### 8.1 测试基础设施调整

现有`RunAllReduceDPUCase`只统计`topoInfo[0]`中的Rank，无法运行多超节点用例。修改为：

```cpp
u32 rankSize = CalRankSize(topoInfo);
```

或者等价地遍历所有超节点、Server和Rank。

### 8.2 新增功能用例

#### ST-DPU-3TO2-001 三级对称topo基本功能

```text
topo：4卡 × 2Server × 2超节点
构造：GenTopoMeta(topoMeta, 2, 2, 4)
数据类型：FP32
归约操作：SUM
count：1024
期望算法：InsAllReduceSequenceMeshNhrDPU
期望结果：AllReduce语义校验成功
期望分层：2个算法层
```

建议用例名：

```text
st_all_reduce_dpu_4x2x2_fp32_sum_1024
```

#### ST-DPU-3TO2-002 三级对称topo较大数据

```text
topo：4卡 × 2Server × 2超节点
数据类型：FP32
归约操作：SUM
count：301 × 1024
期望结果：AllReduce语义校验成功
```

该用例用于同时覆盖拓扑降级和现有DPU分片/循环处理。

#### ST-DPU-3TO2-003 物理L0非对称拒绝

示例拓扑：

```text
超节点0：Server0 4卡，Server1 2卡
超节点1：Server0 4卡，Server1 2卡
```

期望所有Rank的`HcclAllReduce`返回`HCCL_E_NOT_SUPPORT`，不得进入算法编排。

#### ST-DPU-3TO2-004 物理L1非对称拒绝

示例拓扑：

```text
超节点0：2个4卡Server
超节点1：1个4卡Server
```

物理L0实例大小均为4，但物理L1实例大小分别为8和4。该用例用于验证实现没有使用最高层的单实例结果
掩盖物理L1非对称。

期望所有Rank的`HcclAllReduce`返回`HCCL_E_NOT_SUPPORT`。

负向用例需要单独收集每个线程的`HcclAllReduce`返回值，不能继续使用遇错直接返回的正向测试辅助函数。

### 8.3 回归用例

| 回归范围 | 目的 |
|----------|------|
| 现有`all_reduce_dpu`全部用例 | 验证2级HostDPU行为不变 |
| 现有`all_reduce_multilevel`三级用例 | 验证非HostDPU三级算法仍输出3层 |
| `all_gather_dpu` | 验证共享TopoMatch调用方 |
| `reduce_scatter`中的DPU用例 | 验证严格要求2层的Executor |
| `reduce`中的DPU用例 | 验证Reduce共享调用方 |
| `broadcast_dpu` | 验证Broadcast共享调用方 |
| `scatter_dpu` | 验证Scatter共享调用方 |

### 8.4 建议验证命令

定向构建和运行：

```bash
cd test/st/algorithm
ST_TASKS=all_reduce_dpu bash build.sh
./build/testcase/st_all_reduce_dpu_test
```

三级非DPU回归：

```bash
cd test/st/algorithm
ST_TASKS=all_reduce_multilevel bash build.sh
./build/testcase/st_all_reduce_multilevel_test
```

实现完成后还应执行仓库要求的相关编译和静态检查。

## 9. 验收标准

| 编号 | 验收标准 | 对应用例 |
|------|----------|----------|
| AC-001 | 3级对称HostDPU拓扑下选择`InsAllReduceSequenceMeshNhrDPU`并正确完成AllReduce | ST-DPU-3TO2-001 |
| AC-002 | 降级后`AlgHierarchyInfoForAllLevel::infos.size() == 2` | ST-DPU-3TO2-001 |
| AC-003 | 2级Channel资源与2级`AlgHierarchyInfo`一致，不访问`channels[2]` | ST-DPU-3TO2-001 |
| AC-004 | 较大数据在降级拓扑下语义正确 | ST-DPU-3TO2-002 |
| AC-005 | 物理L0非对称HostDPU拓扑返回`HCCL_E_NOT_SUPPORT` | ST-DPU-3TO2-003 |
| AC-006 | 物理L1非对称HostDPU拓扑返回`HCCL_E_NOT_SUPPORT` | ST-DPU-3TO2-004 |
| AC-007 | 普通3级算法仍生成3个算法层并通过现有ST | `all_reduce_multilevel` |
| AC-008 | 现有2级DPU用例全部通过 | 现有`all_reduce_dpu` |

## 10. 风险与应对

### 10.1 共享TopoMatch影响多个算子

风险：修改`TopoMatchMultilevel`可能影响非AllReduce调用方。

应对：

- 使用`hostDPUOnly && topoLevelNums == 3`作为严格条件。
- 不通过算法名称或算子类型做特殊判断。
- 回归所有现有DPU ST和普通三级ST。

### 10.2 物理层与算法层混用

风险：使用最高物理层检查对称性会掩盖物理L1非对称。

应对：

- 使用不同变量表示`physicalLayer1NetLayer`和`algLayer1NetLayer`。
- 对称性检查固定查询物理L1。
- 增加物理L1非对称负向用例。

### 10.3 Rank取模映射假设

风险：`TopoForLayer1`通过`rankId % layer0Size`筛选同序号Rank，依赖对称实例和既有Rank编号布局。

应对：

- HostDPU继续限制为对称topo。
- 不在本需求中改变现有Rank映射规则。
- 使用多超节点用例覆盖跨物理L1、L2后的Rank分组。

### 10.4 资源缓存

风险：算法资源按算法Tag缓存，层数变化后复用旧进程中已经创建的资源可能产生不一致。

应对：

- 本特性随版本发布生效，不支持进程内热替换二进制。
- 新进程首次执行时按修正后的2级拓扑创建资源。
- 不修改算法Tag，避免不必要的缓存和兼容逻辑扩散。

## 11. 替代方案

### 11.1 新增TopoMatchMultilevelDPU

优点：DPU与普通三级算法完全隔离。

缺点：

- 与`TopoMatchMultilevel`重复大量逻辑；
- 需要修改多个DPU算法注册点；
- 后续对称性、Mesh1D和Mesh2D修复需要维护两份实现。

结论：不采用。

### 11.2 在每个2级Executor中删除infos[2]

优点：单个Executor改动直观。

缺点：

- topo matcher输出与算法契约仍不一致；
- 所有DPU Executor都要重复处理；
- 资源计算前后可能得到不同层数。

结论：不采用。

### 11.3 使用TopoMatchSqueeze2D

`TopoMatchSqueeze2D`采用：

```text
算法L0 = 合并物理L0和物理L1
算法L1 = 物理L2
```

这与DPU Sequence算法要求的“保留框内L0、由DPU合并框间层”方向相反。

结论：不采用。

## 12. 已确认决策

1. 降级方向为“保留物理L0，合并物理L1和物理L2”。
2. 仅在`hostDPUOnly && topoLevelNums == 3`时启用。
3. HostDPU不支持非对称topo。
4. 新增统一的三级HostDPU AllReduce ST并执行共享调用方回归。


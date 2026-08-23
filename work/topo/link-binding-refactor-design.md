# 物理建链统一重构需求设计

- 文档状态：草案
- 更新日期：2026-08-11
- 适用仓库：`cann/hccl`、`cann/hcomm`
- 上游设计：[Topo信息解析与标准化](./topo-parse-refactor-design.md)、[TopoMatch统一重构](./topo-match-refactor-design.md)

---

## 1. 概要

本文档只定义物理建链：Template根据算法步序生成Channel需求，LinkBindingResolver将每个需求绑定到标准化
PhysicalLevel中的实际Link，Channel模块据此申请资源。

```text
TopoInfoWithNetLayerDetails.physicalLevels
    + AlgHierarchyInfoForAllLevel
    + Template::CalcChannelRequirements
    + engine LinkSelectionPolicy
    -> LinkBindingResolver
    -> EdgeLinkBinding / HcclChannelDesc
    -> CalcRes / Channel
```

建链可以因协议、Endpoint和实际Link存在性选择不同物理范围，但不得回写或改变Hierarchy。

## 2. 背景与问题

当前建链逻辑分散在Template、Executor和`channel.cc`中，主要问题包括：

1. 不同路径使用固定NetLayer编号或`topoLevelNums - 1`推断层；
2. 部分逻辑默认取`links[0]`，MultiJetty等路径又有独立规则；
3. HostDPU检测会反向改变Hierarchy层数；
4. 相同算法层被假定固定使用一个NetLayer；
5. rank范围或协议标志被错误当作实际Link存在性；
6. 同一TopoInstance的Link归属可能只按协议判断；
7. Template的通信步序与Channel申请之间缺少统一需求对象；
8. src/dst两端的Level、Endpoint和Link排序可能不一致。

### 2.1 必须保留的现有行为

以下三条是现有实现的既定语义，重构后必须等价保留，否则Channel与Notify数量会发生非预期变化：

1. **协议偏好是有序的**。`GetProtocolByEngine`按engine返回有序协议列表（例如AICPU为
   `UBC_CTP > UBC_TP > PCIE > UBOE > UBG`），`ProcessLinkForProtocol`在**第一个存在Link的协议处停止**，
   不会把多种协议的Link混在同一条边上；
2. **同一src die只保留一条Link**。`ProcessLinkForProtocol`通过`ENDPOINT_ATTR_DIE_ID`对本端Endpoint去重；
3. **同一条边只使用一个NetLayer的Link**，找到即停止向上层查找。

## 3. 目标与非目标

### 3.1 目标

1. Template只声明实际需要访问的peer和方向；
2. Channel需求只有一份权威表示；
3. Link按rank边逐个绑定，不按整个算法层固定NetLayer；
4. PhysicalLevel按范围从小到大优先；
5. 同一Level内默认Device优先、Host兜底；
6. 每个rank对必须通过`HcclRankGraphGetLinks`确认实际Link；
7. TopoInstance来源必须通过本端Endpoint描述确认Link归属；
8. Link数量由显式策略控制，不隐式固定为第一条；
9. 同一算法层的不同rank边可以绑定不同PhysicalLevel；
10. src/dst两端使用确定性规则得到等价Binding；
11. 保留2.1节列出的协议顺序、die去重和单Level约束。

### 3.2 非目标

1. 解析或合并TopoInstance与NetInstance；
2. 计算算法Hierarchy或改变其层数与平面数；
3. 修改Mesh、NHR等Template的通信步序；
4. 修改HCOMM Link或Channel内部实现；
5. 在TopoMatch中决定协议或Endpoint；
6. 将全部rank对Link预先复制进Topo结构；
7. 改造Selector现有的能力探测入口（见第9章，只约束其数据来源与一致性）。

## 4. 输入与输出契约

### 4.1 ChannelRequirement

新增需求结构，不修改现有Topo和Hierarchy结构：

```cpp
enum class ChannelDirection : uint32_t {
    SEND,
    RECV,
    BIDIRECTIONAL,
};

// Template只声明"要访问谁、什么方向"
struct ChannelRequirement {
    uint32_t peerRank;
    ChannelDirection direction;
};

// 一次CalcRes内共享的上下文，不在每条Requirement中重复保存
struct ChannelRequirementScope {
    uint32_t srcRank;
    uint32_t planeIndex;                     // 该Template所处理的并行范围下标，单平面时为0
    const std::vector<uint32_t> *scopeRanks; // 该平面的rankList，用于候选过滤前的自检
};
```

**Requirement中不包含`algLevel`**。`AlgResourceRequest::channels`的外层下标由Executor在合并各Template的
`AlgResourceRequest`时分配，每个Template只填自己的`channels[0]`，Template内部无从得知自己属于哪个算法层。
把`algLevel`放进Requirement既填不出来，也会把Executor的合并逻辑卷进本次改造。

`planeIndex`对应Template收到的`subcommInfo`下标，即`AlgHierarchyInfoForAllLevel::infos[level][planeIndex]`
中的第二维——Template构造时收到的本来就是整个`infos[level]`。UBX等声明了并行范围的算法，不同平面对同一peer
可能有不同需求，由不同的`ChannelRequirementScope`分别表达。

`direction`的语义：

- `SEND`：查询`GetLinks(layer, myRank, peerRank)`，生成本端到对端的Channel；
- `RECV`：查询`GetLinks(layer, peerRank, myRank)`，生成对端到本端的Channel；
- `BIDIRECTIONAL`：集合通信默认值，按`SEND`方向查询并生成双向可用的Channel描述。

P2P（Send/Recv/BatchSendRecv）使用`SEND`或`RECV`，集合通信Template一律使用`BIDIRECTIONAL`。

Template生成Requirement是现有`CalcChannelRequest*`直接构造`HcclChannelDesc`路径的替代实现。迁移完成后不得
同时维护Requirement和另一份必需rank边列表。

### 4.2 LinkSelectionPolicy

```cpp
struct LinkSelectionPolicy {
    // 有序偏好，不是集合：命中第一个存在合规Link的协议后即停止
    std::vector<CommProtocol> protocolPreference;
    std::vector<EndpointLocType> endpointPreference;
    bool oneLinkPerSrcDie;    // 同一本端die只保留一条Link
    uint32_t minLinksPerEdge;
    uint32_t maxLinksPerEdge; // 0表示使用选中类别下全部合规Links
};
```

默认规则：

```text
protocolPreference = GetProtocolByEngine(engine)   // 沿用现有有序列表
endpointPreference = [DEVICE, HOST]
oneLinkPerSrcDie   = true
minLinksPerEdge    = 1
maxLinksPerEdge    = 0
```

`protocolPreference`必须是`std::vector`而不是`std::set`：集合无法表达"UBC_CTP优先于PCIE"，会导致同一条边
同时建出两种协议的Channel，Channel与Notify数量与迁移前不一致。

MultiJetty等需要多链路的路径通过`oneLinkPerSrcDie = false`加`maxLinksPerEdge`表达。engine或明确资源策略可以
收窄协议、改变Endpoint顺序或限制Link数。算法名称不直接参与Resolver判断。

### 4.3 EdgeLinkBinding

```cpp
struct EdgeLinkBinding {
    uint32_t peerRank;
    uint32_t planeIndex;
    uint32_t levelIndex;
    std::vector<HcclChannelDesc> channels;
};
```

`levelIndex`只对应当前`topoInfo.physicalLevels`，不作为跨通信域稳定ID。

与`ChannelRequirement`同理，Binding中也不带`algLevel`：Resolver不感知算法层，输出直接供当前Template
填充自己的`channels[0]`。

### 4.4 Resolver接口

```cpp
HcclResult ResolveLinkBindings(
    HcclComm comm,
    const TopoInfoWithNetLayerDetails &topoInfo,
    const ChannelRequirementScope &scope,
    const std::vector<ChannelRequirement> &requirements,
    const LinkSelectionPolicy &policy,
    std::vector<EdgeLinkBinding> &bindings);
```

Resolver不修改`topoInfo`、Hierarchy或Requirement。

`topoInfo.physicalLevels`为空（Topo解析降级）时返回`HCCL_E_NOT_SUPPORT`，调用方回退到旧建链路径。

## 5. Template与CalcRes接入

### 5.1 单一需求来源

当前Template的`CalcRes`通常通过`CalcChannelRequestMesh1D`、`CalcChannelRequestNhr`等函数直接构造Channel描述。
迁移后流程为：

```text
Template根据Hierarchy和算法步序
    -> CalcChannelRequirements
    -> LinkBindingResolver
    -> HcclChannelDesc
    -> 合并到AlgResourceRequest
```

`CalcChannelRequirements`必须复用Template执行阶段的peer计算工具，不能另写一套通信步序。

### 5.2 Binding到AlgResourceRequest的映射

**分组保持现状**：每个Template把自己的`EdgeLinkBinding`展开成`HcclChannelDesc`后仍然只填
`AlgResourceRequest::channels[0]`，外层下标继续由Executor在合并各Template结果时分配。本次改造不触碰
Executor的合并逻辑。

**去重键必须扩展**：`(peerRank, localEndpoint.protocol, localEndpoint.commAddr, localEndpoint.loc,
channelProtocol)`。仅按`peerRank`去重是不够的——同一peer在不同平面上可能绑定到不同Level、不同协议的Link。

现有`CompReqChannelWithExistChannel`（增量建链路径）只检查`channels[0]`且仅按`remoteRank`去重，会把新需求
误判为已存在，必须按上述去重键改造。"只处理`channels[0]`"这一点保持不变：其调用点之后紧接着就是
`resRequest.channels[0].size() == 0`判断，扩展到多层需要连带改造`IncrementalCreateChannel`，超出本次范围。
该函数列入实施步骤第8项。

### 5.3 分阶段迁移

1. 为现有`CalcChannelRequest*`增加内部Requirement生成路径；
2. 先让新路径与旧Channel描述做Golden对比（包括Channel数量、协议和顺序）；
3. 接入Resolver后删除该函数中的RankGraph和固定NetLayer选择；
4. 保留现有Template `CalcRes`签名；
5. 最终删除分散在Executor、Template和Channel中的特殊选链分支。

### 5.4 不同Template的需求差异

- Mesh可以需要与rankList中的多个peer建链；
- NHR只声明其通信步序实际访问的peer；
- 声明了并行范围的算法（如UBX mesh+clos）对不同平面分别生成Requirement，由`planeIndex`区分；
- 同一peer在多个平面上重复出现时，资源是否复用由5.2的去重键决定；
- MultiJetty通过`oneLinkPerSrcDie = false`与`maxLinksPerEdge`表达，不通过特殊TopoMatch表达。

## 6. 候选PhysicalLevel

对每个Requirement，按`physicalLevels[0..n)`顺序检查Level。

Level成为候选必须满足：

1. `localRanks`包含当前rank和`peerRank`；
2. 至少一个Source提供可查询的实际NetLayer；
3. `HcclRankGraphGetLinks`返回非空Link；
4. Link协议出现在`protocolPreference`中；
5. Link的本端Endpoint位置出现在`endpointPreference`中；
6. TopoInstance Source的Link**本端**Endpoint与`PhysicalSourceInfo::endpoints`匹配（见7.2）；
7. 按第8章完成筛选后，合规Link数量满足`minLinksPerEdge`。

rank范围只用于候选过滤，不能代替`GetLinks`。

## 7. Link查询与归属

### 7.1 查询规则

Resolver是建链阶段唯一允许调用`HcclRankGraphGetLinks`的组件：

```cpp
HcclRankGraphGetLinks(
    comm,
    source.ref.netLayer,
    srcRank,   // direction决定：SEND/BIDIRECTIONAL为myRank，RECV为peerRank
    dstRank,
    &links,
    &linkNum);
```

同一Level存在多个Source时，按实际NetLayer分别查询并合并结果；相同NetLayer只查询一次。

### 7.2 Source归属

- **NetInstance Source**：Link来自对应NetLayer，且rank范围和engine约束满足；
- **TopoInstance Source**：除NetLayer一致外，Link的**本端**Endpoint必须与该Source的Endpoint快照匹配。

只能按本端Endpoint判断归属。`HcclRankGraphGetEndpointDesc(comm, layer, topoInstId, ...)`返回的是**当前rank**
在该TopoInstance上的Endpoint，无法枚举对端rank的Endpoint集合，因此不存在"dst Endpoint与本地快照匹配"
这种可实现的规则。

同一NetLayer内可能存在多个相同协议的TopoInstance，因此也不能只按`CommProtocol`判断归属。

若本端Endpoint快照为空或不足以区分同一NetLayer上的多个TopoInstance，该Source不参与归属判断：

- 该Level仍存在NetInstance Source时，按NetInstance规则继续；
- 否则该Level不合格，继续检查更高Level；全部不合格时返回不支持，不做猜测。

对端信息确有必要时，只能通过`HcclRankGraphGetEndpointInfo(comm, peerRank, &link.dstEndpointDesc, ...)`
查询已知Endpoint的属性，不能用于枚举或发现。

### 7.3 Link去重键

使用以下字段构造稳定去重和排序键：

```text
levelIndex
source.netLayer
srcEndpoint.protocol / location / address
dstEndpoint.protocol / location / address
linkProtocol
hop
```

dst Endpoint可以进入比较键——它直接来自`CommLink`本身，与7.2禁止的"用本地快照校验dst归属"是两件事。

若HCOMM提供稳定Link ID，后续可将其加入比较键，但不作为首期前置条件。

## 8. 选择规则

### 8.1 Level优先级

选择第一个存在合规Link且满足最小数量要求的PhysicalLevel。低Level优先级高于协议偏好：

```text
低Level PCIe可用，高Level UBCTP也可用 -> 选择低Level PCIe
```

低Level不存在合规Link时才检查更高Level。

### 8.2 Endpoint位置优先级

确定Level后，按`endpointPreference`选择第一个存在合规Link的Endpoint位置。默认：

```text
Device > Host
```

同一Level存在Device Link时不使用Host Link；Device不可用时Host作为兜底。

### 8.3 协议优先级

确定Level和Endpoint位置后，按`protocolPreference`顺序选择**第一个存在Link的协议**，只使用该协议的Link。

这是2.1.1现有语义的直接保留。同一条边不得混合多种协议：

```text
同一Level同时存在 UBC_CTP 与 PCIE 的Device Link
    -> 只使用 UBC_CTP
```

### 8.4 Link数量

确定Level、Endpoint位置和协议后：

1. `oneLinkPerSrcDie == true`时，按本端Endpoint的`ENDPOINT_ATTR_DIE_ID`去重，每个die保留一条；
2. 按7.3的比较键排序、去重；
3. `maxLinksPerEdge == 0`时保留全部剩余Link；
4. 非零上限时保留排序后的前N条；
5. 结果少于`minLinksPerEdge`时，回到8.3尝试下一个协议；协议用尽后回到8.2尝试下一个Endpoint位置；
   仍不满足则该Level不合格；
6. 选中Level后不从其他Level补Link。

同一rank边不得混合多个PhysicalLevel。混合建链只发生在不同rank边之间。

### 8.5 伪代码

```cpp
for (uint32_t level = 0; level < topoInfo.physicalLevels.size(); ++level) {
    if (!Contains(topoInfo.physicalLevels[level], myRank, req.peerRank)) {
        continue;
    }

    // 按Source分别查询并完成归属过滤，相同netLayer只查一次
    auto links = QueryAndAttributeLinks(comm, topoInfo.physicalLevels[level], req);

    for (auto location : policy.endpointPreference) {
        auto byLoc = FilterBySrcLocation(links, location);
        for (auto protocol : policy.protocolPreference) {
            auto selected = FilterByProtocol(byLoc, protocol);
            if (selected.empty()) {
                continue;
            }
            selected = DedupBySrcDieIfNeeded(comm, selected, policy);
            selected = SortDedupAndLimit(selected, policy.maxLinksPerEdge);
            if (selected.size() >= policy.minLinksPerEdge) {
                return BuildBinding(level, req, selected);
            }
        }
    }
}
return HCCL_E_NOT_SUPPORT;
```

## 9. HostDPU

### 9.1 建链阶段

HostDPU是Link绑定结果，不是TopoMatch模式：

1. 低Level Device Link可用时优先Device；
2. 同一合并Level同时存在Device和Host Source时优先Device；
3. 当前Level无合规Link时检查更高Level；
4. 首个可用类别为Host时使用HostDPU；
5. HostDPU不得改变`infos.size()`、平面数或rank分组。

```text
全覆盖Device Source + 全覆盖Host Source
    -> Topo解析合并为一个Level
    -> Resolver选择Device

内层Device范围 + 外层Host范围
    -> 内层peer选择Device
    -> 外层peer选择Host
```

### 9.2 选择阶段仍然需要HostDPU

HostDPU判定当前不只用于建链，也是**算法选择输入**：`AutoSelectorBase::Select`据此切到`SelectDPUAlgo`
并改写`opParam.engine`；`send_op`/`recv_op`用`IsHostDpu`选择DPU Executor。这一职责本次不取消，但要收敛
数据来源并保证与建链结果一致：

1. Selector读取`physicalLevels`最外层Source的`endpoints`判断Device/Host能力，不再重复调用
   `HcclRankGraphGetTopoInstsByLayer`/`GetEndpointDesc`（即现有`CheckHostDPUOnly`的内部实现改为读取
   已解析的Topo数据，对外签名和语义不变）；
2. 需要不同算法层数的场景（现有`TopoMatchMultilevel`在HostDPU且物理三层时降为二层），由**Selector选择
   声明`{LEVEL_2}`的Executor**完成，而不是由TopoMatch动态降层；
3. 一致性约束：Selector判定为HostDPU时，Resolver在最外层必须实际选中Host Endpoint。二者不一致时
   返回`HCCL_E_NOT_SUPPORT`并打印两侧判定依据，不允许静默继续；
4. `physicalLevels`为空时，`CheckHostDPUOnly`回退到现有实现。

## 10. 示例

### 10.1 全UBCTP

```text
L0: 4 ranks, UBCTP
L1: 16 ranks, UBCTP
L2: 32 ranks, UBCTP
```

- `0 -> 4`：L0无法承载时选择L1；
- `0 -> 16`：选择L2；
- 不因L2协议相同而跳过L1。

### 10.2 PCIe-SW与Host

```text
L0: 4 ranks, UBCTP Device
L1: 16 ranks, PCIe Device
L2: 32 ranks, UBCTP Host
```

- `0 -> 4`：L1存在PCIe Link时选择L1；
- `0 -> 16`：L1不能承载，选择L2 Host；
- 同一算法层允许两个peer分别选择L1和L2；
- Hierarchy保持不变。

### 10.3 同Level多协议

```text
L1: 16 ranks，同时存在 UBC_CTP 与 PCIE 的Device Link
protocolPreference = [UBC_CTP, UBC_TP, PCIE, ...]
```

只建UBC_CTP Channel。若把`protocolPreference`退化成无序集合并保留全部合规Link，这条边会多出一份PCIE
Channel与对应Notify，属于回归。

### 10.4 `0 -> 7`负例

不能因为L1标记为"16 ranks、PCIe"就推断`0 -> 7`一定存在PCIe Link。

```text
L1 GetLinks(0,7)无归属该Source的PCIe Link
    -> L1不合格
    -> 继续检查L2
```

所有Level都没有合规Link时返回不支持，不能伪造Channel。

## 11. HCOMM与架构边界

### 11.1 接口

建链阶段允许调用：

```cpp
HcclRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
    CommLink **links, uint32_t *linkNum);

// 仅用于查询"已选中Link的Endpoint"的属性，例如die去重所需的 ENDPOINT_ATTR_DIE_ID
HcclRankGraphGetEndpointInfo(
    HcclComm comm, uint32_t rankId, const EndpointDesc *endpointDesc,
    EndpointAttr endpointAttr, uint32_t infoLen, void *info);
```

TopoInstance的Endpoint集合由Topo解析阶段保存，Resolver不重新调用`GetEndpointDesc`做拓扑发现。
`GetEndpointInfo`只允许作用于已经从`CommLink`中拿到的Endpoint描述。

### 11.2 调用方式

- TopoInstance与Endpoint系列经`src/common/hcomm_dlsym/hccl_rank_graph_dl.h`弱符号调用；
- `HcclRankGraphGetLinks`沿用现有直接调用方式，本次不改变；
- 不包含HCOMM私有头，不要求HCOMM反向依赖HCCL；
- 实验Endpoint接口不可用、无法判断TopoInstance归属时按7.2处理，不猜测。

### 11.3 调用边界

目标状态下：

- Topo解析组件调用拓扑与Endpoint查询；
- LinkBindingResolver是**建链选路**阶段唯一的`GetLinks`调用方；
- TopoMatch不调用任何RankGraph接口；
- Executor与Template不再为选路调用RankGraph。

以下现有调用**不在本次收敛范围**，需在文档中显式排除，避免实施时误删：

- Template为数据面切分查询die属性（`ccu_alg_template_base.cc`、`ccu_temp_*_mesh2die.cc`中的
  `GetEndpointInfo`）；
- Selector为算法可选性做的连通性探测（`send_auto_selector.cc`、`recv_auto_selector.cc`中的`GetLinks`）；
- `InitRankInfo`中的`GetPairLinkCounter`等既有拓扑推导。

这些属于独立议题，若后续要收敛需另立需求。

## 12. 失败语义

| 失败场景 | 返回值 |
|----------|--------|
| Requirement参数非法、peer越界、`planeIndex`越界 | `HCCL_E_PARA` |
| Topo Source或Endpoint数据非法 | `HCCL_E_INTERNAL` |
| `physicalLevels`为空 | `HCCL_E_NOT_SUPPORT`（调用方回退旧路径） |
| 所有Level均无合规Link | `HCCL_E_NOT_SUPPORT` |
| 合规Link数小于最小要求 | `HCCL_E_NOT_SUPPORT`或资源错误 |
| Selector判定HostDPU但Resolver选中Device | `HCCL_E_NOT_SUPPORT` |
| Channel创建失败 | 保留原始资源错误 |

Bindings在临时对象中构造，所有Requirement成功后再提交。任一边失败时不得返回部分Binding。

## 13. 实施步骤

1. 新增`ChannelRequirement`、`LinkSelectionPolicy`和`EdgeLinkBinding`；
2. 实现Link查询、Source归属（本端Endpoint匹配）、die去重、稳定排序和去重工具；
3. 实现`LinkBindingResolver`，含Level/位置/协议三级优先与回退；
4. 为现有`CalcChannelRequest*`接入Requirement生成路径并做Golden对比；
5. 迁移Mesh和NHR基础路径；
6. 迁移MultiJetty、PCIe-SW、UBX（含并行范围）和HostDPU路径；
7. 删除TopoMatch和Executor中的选链逻辑；
8. 按5.2改造`CompReqChannelWithExistChannel`的分组与去重键；
9. 删除`channel.cc`中的固定NetLayer和`links[0]`默认规则；
10. 按9.2改造`CheckHostDPUOnly`的数据来源并补充一致性校验；
11. 完成双端一致性、资源上限和端到端测试。

## 14. 测试与验收

### 14.1 Level与Endpoint选择

1. 多个Level都有Link时选择最低Level；
2. 低Level无Link时回退到更高Level；
3. 低Level协议不支持时回退；
4. 同一Level存在Device和Host时选择Device；
5. Device不可用但Host可用时选择Host；
6. 选中Level后不混入其他Level的Link。

### 14.2 协议与Link数量

1. 同一Level同时存在UBC_CTP与PCIE时只建UBC_CTP Channel（协议顺序回归用例）；
2. 首选协议Link数不足`minLinksPerEdge`时按顺序回退到下一协议；
3. `oneLinkPerSrcDie = true`时每个本端die只保留一条Link，Channel数量与迁移前一致；
4. `oneLinkPerSrcDie = false`且`maxLinksPerEdge = 0`时保留全部合规Link（MultiJetty）；
5. 非零上限按稳定顺序裁剪；
6. 完全重复Link被去重；
7. Channel和Notify资源数量与选择结果一致。

### 14.3 Link归属

1. 同一NetLayer多个TopoInstance时按本端Endpoint正确归属；
2. 相同协议但本端Endpoint不同的Link不误归属；
3. `0 -> 7`无实际PCIe Link时不创建PCIe Channel；
4. 本端Endpoint快照缺失且该Level只有TopoInstance Source时返回不支持；
5. 本端Endpoint快照缺失但该Level存在NetInstance Source时按NetInstance规则成功绑定。

### 14.4 混合建链与一致性

1. 同一算法层不同peer分别绑定不同Level；
2. 同一peer在同一`planeIndex`上不混用多个Level或多种协议；
3. 同一peer在不同平面上绑定不同Level时，资源按5.2去重键正确区分，不被误合并；
4. src/dst两端选择等价Level和Link集合；
5. Requirement顺序变化不影响Binding结果；
6. Device/Host变化不回写或改变Hierarchy；
7. Selector的HostDPU判定与Resolver最终选择一致；不一致时报错可定位；
8. `physicalLevels`为空时回退旧建链路径，行为与重构前一致。

### 14.5 验收标准

1. Template通过统一Requirement表达Channel需求；
2. LinkBindingResolver是建链选路阶段唯一`GetLinks`调用方（11.3列出的例外除外）；
3. Link选择严格遵循Level、Endpoint位置、协议顺序和数量策略；
4. 每个rank对均校验实际Link存在性；
5. TopoInstance Link归属只依赖本端Endpoint，不只依赖协议；
6. 同一算法层支持不同peer绑定不同Level；
7. HostDPU不改变Hierarchy，且选择阶段与建链阶段判定一致；
8. 不存在固定NetLayer或默认`links[0]`规则；
9. Channel去重键覆盖协议与Endpoint，不再只按`remoteRank`；
10. HCCL与HCOMM的调用方式保持不变，未引入私有依赖；
11. 相关UT、ST和端到端资源测试通过。

## 15. 风险与开放问题

### 15.1 风险

- 协议顺序或die去重遗漏导致Channel/Notify数量变化：以2.1为基线建立数量对比用例；
- src/dst排序不一致：统一比较键并执行双端测试；
- Template重复实现peer步序：Requirement生成必须复用执行阶段工具；
- 本端Endpoint快照不足以判断Link归属：按7.2降级或返回不支持，不按协议猜测；
- Selector与Resolver的HostDPU判定分裂：增加一致性校验并在失败时输出两侧依据；
- 分阶段迁移时新旧Channel路径并存：增加Golden对比并按Template逐个切换。

### 15.2 开放问题

1. 各engine的`protocolPreference`、`oneLinkPerSrcDie`、最小和最大Link数默认值是否需要按算法细分；
2. 同一peer在多个平面上出现时的资源复用策略（当前按5.2去重键区分，是否需要更激进的合并）；
3. HCOMM是否提供稳定Link ID以简化双端排序；
4. die属性等Endpoint扩展属性是否值得在Topo解析阶段预取，以减少建链期`GetEndpointInfo`调用次数。

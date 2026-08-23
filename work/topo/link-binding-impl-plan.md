# 物理建链统一重构 — 编码实现方案

- 文档状态：草案
- 更新日期：2026-08-11
- 对应设计：[物理建链统一重构](./link-binding-refactor-design.md)
- 前置实现：[Topo解析实现方案](./topo-parse-impl-plan.md)、[TopoMatch实现方案](./topo-match-impl-plan.md)
- 适用仓库：`cann/hccl`

---

## 1. 范围

实现设计文档第4-9章：`ChannelRequirement`、`LinkSelectionPolicy`、`EdgeLinkBinding`、
`LinkBindingResolver`，以及`channel.cc`中`CalcChannelRequest*`系列的分批迁移。

三份实现方案里这一份**风险最高**：它直接决定实际创建多少Channel、多少Notify，任何一处顺序或过滤规则的
偏差都会变成资源数量差异或建链失败。因此本方案的核心不是"怎么写新代码"，而是**怎么保证新旧路径等价**：
第5.1节的双跑对比开关是整个迁移的安全网，建议在写Resolver之前先把它做出来。

## 2. 实施前已确认的代码事实

| # | 事实 | 位置 | 影响 |
|---|------|------|------|
| L1 | `AlgResourceRequest.channels`的外层下标由**Executor在合并时**分配；每个Template只填自己的`channels[0]` | `ins_v2_scatter_sequence_executor.cc:65-81` | Template内部无从得知自己的`algLevel`，设计文档§4.1需要调整（见4.1） |
| L2 | Template构造时收到整个`infos[level]`，内部用`subcommInfo[COMM_LEVEL0]`取平面0 | `channel.cc:382`、`ins_v2_scatter_sequence_executor.cc:61` | `planeIndex`就是`subcommInfo`的下标，Template入参不用改 |
| L3 | 协议偏好是**有序**的，`ProcessLinkForProtocol`在第一个有Link的协议处`break` | `channel.cc:225-268`、`channel.cc:297-327` | `LinkSelectionPolicy`必须用`vector`；选择顺序必须协议内层、位置外层 |
| L4 | `ProcessLinkForProtocol`用`ENDPOINT_ATTR_DIE_ID`对本端Endpoint去重（`seenDie`） | `channel.cc:303-321` | `oneLinkPerSrcDie`必须实现，否则Channel数量翻倍 |
| L5 | `HcclRankGraphGetLinks`返回HCOMM管理的裸指针，现有代码立即复制成`std::vector<CommLink>` | `channel.cc:406` | Resolver同样必须立即复制，不得跨调用持有 |
| L6 | `CompReqChannelWithExistChannel`只处理`channels[0]`、只按`remoteRank`去重，用于增量建链路径 | `op_common.cc:1124-1139`、调用点`op_common.cc:1470` | 需按新去重键改造，但影响面局限在增量建链 |
| L7 | `channel.h`导出16个`CalcChannelRequest*`/`Process*`函数 | `channel.h:21-81` | 迁移工作量主体；必须逐个而不是一次性替换 |
| L8 | `GetEndpointInfo`是弱符号，未加载时返回`(HcclResult)(-1)` | `hccl_rank_graph_dl.cc:29-31` | die去重前必须查`HcommIsSupportHcclRankGraphGetEndpointInfo()`；不支持时退化为不去重 |
| L9 | `CalcChannelRequestMesh1D`按netLayer升序遍历、找到即`break` | `channel.cc:395-414` | 与设计文档§8.1"低Level优先"语义一致，迁移后行为可等价 |
| L10 | `CheckNetLayerExists`带`linkRequired`语义：层不存在时可警告跳过也可报错 | `channel.cc:475-497` | 迁移`CalcChannelRequestMesh1DByLevel`时必须保留这一区分 |
| L11 | ST的AICPU目标**不**编译`channel.cc` | `.../aicpu/CMakeLists.txt` | Resolver是Host-only，新文件不进AICPU源表 |

## 3. 交付物清单

### 3.1 新增文件

| 文件 | 内容 | 依赖comm |
|------|------|----------|
| `src/ops/op_common/executor/channel/link_selection_policy.h` | `ChannelDirection`、`ChannelRequirement`、`LinkSelectionPolicy`、`EdgeLinkBinding` | 否 |
| `src/ops/op_common/executor/channel/link_selection_policy.cc` | `GetDefaultPolicy(engine)`（复用`GetProtocolByEngine`） | 否 |
| `src/ops/op_common/executor/channel/link_filter.h/.cc` | **纯函数**：协议/位置过滤、稳定比较键、排序去重、数量裁剪 | 否 |
| `src/ops/op_common/executor/channel/link_binding_resolver.h/.cc` | 查询、Source归属、Level遍历、Binding组装 | 是 |
| `src/ops/op_common/executor/channel/binding_compare.h/.cc` | 新旧路径双跑对比开关（迁移期临时件） | 是 |
| `test/ut/link_filter/CMakeLists.txt` + `link_filter_test.cc` | 纯函数UT | — |

`link_filter`从Resolver里单独拆出来的理由与前两份方案一致：选择规则（协议顺序、die去重、裁剪）是
最容易出错、也最需要密集测试的部分，把它做成"输入一组`CommLink`、输出一组`HcclChannelDesc`"的纯函数，
就可以在UT里穷举各种组合，不需要模拟器。

### 3.2 修改文件

| 文件 | 修改 |
|------|------|
| `src/ops/op_common/executor/channel/channel.cc` | `CalcChannelRequest*`逐个接入Resolver（分批） |
| `src/ops/op_common/executor/channel/CMakeLists.txt` | 追加新`.cc` |
| `src/ops/op_common/op_common.cc` | `CompReqChannelWithExistChannel`去重键；`CheckHostDPUOnly`数据来源 |
| `src/common/hcomm_dlsym/hccl_rank_graph_dl.h` | `DECL_SUPPORT_FLAG`（若Topo解析方案的PR3尚未合入） |
| `test/ut/CMakeLists.txt` | `add_subdirectory(link_filter)` |

## 4. 详细设计

### 4.1 需求结构（对设计文档的一处修正）

L1是本方案发现的最重要的事实：**Template不知道自己属于哪个算法层**。`AlgResourceRequest.channels`的外层
下标是Executor在合并两个Template的`AlgResourceRequest`时决定的，Template内部永远只填`channels[0]`。

因此设计文档§4.1把`algLevel`放进`ChannelRequirement`是落不了地的。修正方案：

```cpp
enum class ChannelDirection : u32 { SEND, RECV, BIDIRECTIONAL };

// Template只声明"要访问谁"
struct ChannelRequirement {
    u32 peerRank = INVALID_VALUE_RANKID;
    ChannelDirection direction = ChannelDirection::BIDIRECTIONAL;
};

// 调用上下文由Template的CalcRes提供，不进每条Requirement
struct ChannelRequirementScope {
    u32 srcRank = INVALID_VALUE_RANKID;
    u32 planeIndex = 0;                 // subcommInfo的下标，见L2
    const std::vector<u32>* scopeRanks = nullptr;  // 该平面的rankList，用于候选过滤前的自检
};

struct EdgeLinkBinding {
    u32 peerRank = INVALID_VALUE_RANKID;
    u32 planeIndex = 0;
    u32 levelIndex = INVALID_UINT;      // 对应topoInfo.physicalLevels
    std::vector<HcclChannelDesc> channels;
};
```

`algLevel`从`ChannelRequirement`和`EdgeLinkBinding`中去掉：它是Executor合并阶段的概念，
Resolver不需要、也无法感知。**Executor的合并逻辑因此完全不用改**，这同时消除了设计文档§5.2
"按`(algLevel, planeIndex)`分组"带来的Executor改造面。

> 设计文档已同步：§4.1删除`ChannelRequirement::algLevel`并引入`ChannelRequirementScope`，§4.3的
> `EdgeLinkBinding`同步删除`algLevel`，§4.4的Resolver签名增加`scope`入参，§5.2的分组规则改为
> "Template维持只填`channels[0]`，外层下标仍由Executor分配"，只保留去重键部分。

### 4.2 选择策略（`link_selection_policy.h/.cc`）

```cpp
struct LinkSelectionPolicy {
    std::vector<CommProtocol> protocolPreference;   // 有序，L3
    std::vector<EndpointLocType> endpointPreference;
    bool oneLinkPerSrcDie = true;                   // L4
    u32 minLinksPerEdge = 1;
    u32 maxLinksPerEdge = 0;                        // 0 = 全部
};

HcclResult GetDefaultPolicy(const OpParam& param, LinkSelectionPolicy& policy);
```

`GetDefaultPolicy`直接调用现有`GetProtocolByEngine(param, policy.protocolPreference)`填协议偏好，
其余字段取默认值。这保证了"默认策略 == 现有行为"，是等价迁移的前提，也避免在两处维护engine到协议的映射。

### 4.3 选择规则（`link_filter.h/.cc`，纯函数）

```cpp
struct LinkCandidate {
    CommLink link;      // 已从HCOMM裸指针复制（L5）
    u32 netLayer;
    u32 levelIndex;
};

struct LinkCompareKey {                 // 设计文档§7.3
    u32 levelIndex; u32 netLayer;
    CommProtocol srcProtocol; EndpointLocType srcLoc; u64 srcAddr;
    CommProtocol dstProtocol; EndpointLocType dstLoc; u64 dstAddr;
    CommProtocol linkProtocol; u32 hop;
    bool operator<(const LinkCompareKey& rhs) const;   // 逐字段字典序
    bool operator==(const LinkCompareKey& rhs) const;
};

// 核心：在已经确定Level的一组候选里，按位置->协议->die->数量选出最终Link
HcclResult SelectLinksForEdge(
    const std::vector<LinkCandidate>& candidates,
    const LinkSelectionPolicy& policy,
    const std::function<bool(const EndpointDesc&, u32& dieId)>& dieQuery,  // 注入，便于UT
    std::vector<LinkCandidate>& selected);
```

`SelectLinksForEdge`的实现严格按设计文档§8.2-8.4的顺序：

```text
for location in policy.endpointPreference:
    byLoc = [c for c in candidates if c.link.srcEndpointDesc.loc.locType == location]
    if byLoc.empty(): continue
    for protocol in policy.protocolPreference:
        byProto = [c for c in byLoc if c.link.linkAttr.linkProtocol == protocol]
        if byProto.empty(): continue                  // L3：只取第一个有Link的协议
        sort(byProto) by LinkCompareKey
        unique(byProto) by LinkCompareKey
        if policy.oneLinkPerSrcDie:
            byProto = 每个dieId保留第一条            // L4；dieQuery失败的条目一律保留
        if policy.maxLinksPerEdge != 0:
            byProto.resize(min(size, maxLinksPerEdge))
        if byProto.size() >= policy.minLinksPerEdge:
            selected = byProto; return SUCCESS
        // 数量不足 -> 继续下一个协议（设计文档§8.4.5）
return HCCL_E_NOT_SUPPORT
```

两个必须注意的细节：

1. **die去重要在排序去重之后**。现有`ProcessLinkForProtocol`是边遍历边`seenDie.insert`，保留的是
   HCOMM返回顺序里的第一条；新实现先按`LinkCompareKey`排序再去重，保留的是**排序后**的第一条。
   这是一处有意的行为变化——它换来的是src/dst两端一致性（设计文档§14.4.4）。必须在双跑对比里
   单独确认这一项的差异是"顺序不同但集合相同"，而不是"选中的Link不同"；
2. `dieQuery`以`std::function`注入，Resolver传入真实的`GetEndpointInfo`封装，UT传入桩。
   这样`link_filter`保持零外部依赖（L8的support flag判断放在Resolver侧的封装里，
   不支持时`dieQuery`直接返回false，等价于不去重）。

### 4.4 Resolver（`link_binding_resolver.h/.cc`）

```cpp
HcclResult ResolveLinkBindings(
    HcclComm comm,
    const TopoInfoWithNetLayerDetails& topoInfo,
    const ChannelRequirementScope& scope,
    const std::vector<ChannelRequirement>& requirements,
    const LinkSelectionPolicy& policy,
    std::vector<EdgeLinkBinding>& bindings);
```

主循环：

```text
topoInfo.physicalLevels.empty() -> HCCL_E_NOT_SUPPORT（调用方回退旧路径）

tmpBindings.clear()
for req in requirements:
    bound = false
    for levelIdx in [0, physicalLevels.size()):
        level = physicalLevels[levelIdx]
        if !Contains(level.localRanks, scope.srcRank) || !Contains(level.localRanks, req.peerRank):
            continue
        candidates = QueryAndAttribute(comm, level, levelIdx, scope.srcRank, req)
        if candidates.empty(): continue
        if SelectLinksForEdge(candidates, policy, dieQuery, selected) == SUCCESS:
            tmpBindings.push_back(BuildBinding(levelIdx, scope.planeIndex, req, selected))
            bound = true; break
    if !bound: return HCCL_E_NOT_SUPPORT      // 任一边失败即整体失败，不返回部分结果
bindings = std::move(tmpBindings)
```

**`QueryAndAttribute`**（设计文档§7.1-7.2）：

```text
queriedLayers.clear()
for source in level.sources:
    if source.ref.netLayer ∈ queriedLayers: continue    // 相同NetLayer只查一次
    queriedLayers.insert(source.ref.netLayer)

    (srcRank, dstRank) = (req.direction == RECV)
                       ? (req.peerRank, scope.srcRank)
                       : (scope.srcRank, req.peerRank)
    GetLinks(comm, source.ref.netLayer, srcRank, dstRank, &raw, &num)
    links = vector<CommLink>(raw, raw + num)             // L5：立即复制

    for link in links:
        if !AttributeToSource(link, level, source.ref.netLayer): continue
        candidates.push_back({link, source.ref.netLayer, levelIdx})
```

**`AttributeToSource`**（本端Endpoint归属，设计文档§7.2）：

```text
该netLayer上的Source分两类：
  - 存在NET_INSTANCE Source     -> Link无条件归属该Level（rank范围已在外层校验）
  - 只有TOPO_INSTANCE Source    -> link.srcEndpointDesc 必须匹配某个 source.endpoints[i]
                                   匹配字段：protocol + loc.locType + commAddr
                                   若所有TopoInstance Source的endpoints都为空
                                   -> 该Level不合格（设计文档§7.2的降级分支）
绝不使用 link.dstEndpointDesc 做归属判断（本地快照拿不到对端Endpoint集合）
```

`AttributeToSource`本身也是纯函数（输入是`CommLink`与`PhysicalLevelInfo`），放进`link_filter.cc`一起测。

### 4.5 双跑对比开关（`binding_compare.h/.cc`）

迁移期的安全网，建议**先于Resolver实现**：

```cpp
// 环境变量 HCCL_LINK_BINDING_COMPARE=1 时启用
bool IsLinkBindingCompareEnabled();

// 比较两组Channel描述；差异打ERROR日志（含逐条明细），返回是否一致
bool CompareChannelDescs(
    const std::vector<HcclChannelDesc>& legacy,
    const std::vector<HcclChannelDesc>& resolved,
    const std::string& tag);
```

接入方式（以`CalcChannelRequestMesh1D`为例）：

```cpp
HcclResult CalcChannelRequestMesh1D(...)
{
    if (!UseResolver(topoInfo)) {                       // physicalLevels为空 -> 直接旧路径
        return LegacyCalcChannelRequestMesh1D(...);
    }
    std::vector<HcclChannelDesc> resolved;
    HcclResult ret = ResolverPath(..., resolved);

    if (IsLinkBindingCompareEnabled()) {
        std::vector<HcclChannelDesc> legacy;
        (void)LegacyCalcChannelRequestMesh1D(..., legacy);
        (void)CompareChannelDescs(legacy, resolved, "Mesh1D");
    }
    channels = std::move(resolved);
    return ret;
}
```

比较维度：数量、`remoteRank`集合、每个`remoteRank`下的`(channelProtocol, localEndpoint, remoteEndpoint)`
集合。**顺序不参与比较**（新实现按`LinkCompareKey`排序，与HCOMM返回顺序不同是预期的，见4.3.1）。

对比开关在ST全量回归里打开跑一轮，确认无差异后再逐个算法切换默认路径；全部迁移完成后（PR7）删除
`binding_compare.*`与`Legacy*`函数。

### 4.6 HostDPU（设计文档§9.2）

分两步，且**第二步不与Resolver迁移同一个PR**：

1. `CheckHostDPUOnly`内部改为优先读`topoInfo.physicalLevels`最外层Source的`endpoints`判断
   Device/Host，`physicalLevels`为空时回退现有实现。函数签名与语义不变，因此`AutoSelectorBase`、
   `send_op`、`recv_op`的调用点全部不动；
2. 一致性校验：Resolver完成绑定后，若Selector此前判定为HostDPU（通过`param.engine == COMM_ENGINE_CPU`
   与`opExecuteConfig == HOSTCPU`识别）而最外层实际选中了Device Endpoint，返回`HCCL_E_NOT_SUPPORT`
   并打印两侧依据。

第1步是纯粹的数据来源替换，可以独立验证；第2步会引入新的失败路径，需要单独观察。

### 4.7 增量建链去重（L6）

`CompReqChannelWithExistChannel`改为按设计文档§5.2的去重键：

```cpp
struct ChannelIdentity {
    u32 remoteRank;
    CommProtocol localProtocol;
    EndpointLocType localLoc;
    u64 localAddr;
    CommProtocol channelProtocol;
    bool operator<(const ChannelIdentity&) const;
};
```

保持"只处理`channels[0]`"不变——该函数服务的是增量建链路径，调用点
（`op_common.cc:1470`）之后紧接着就是`resRequest.channels[0].size() == 0`的判断，扩展到多层需要
连带改造`IncrementalCreateChannel`，超出本次范围。在函数注释里写明这一约束。

## 5. 测试方案

### 5.1 UT（`test/ut/link_filter/`）

被测：`link_filter.cc`（`SelectLinksForEdge`、`LinkCompareKey`、`AttributeToSource`）。
夹具：手工构造`std::vector<LinkCandidate>`，`dieQuery`用lambda桩。

| 组 | 用例 | 断言 |
|----|------|------|
| 协议顺序 | `PicksFirstAvailableProtocol` | 同时有UBC_CTP与PCIE时只返回UBC_CTP（对应设计文档§10.3） |
| | `FallsBackWhenFirstProtocolAbsent` | 无UBC_CTP时选UBC_TP |
| | `FallsBackWhenFirstProtocolBelowMin` | 首选协议只有1条但`minLinksPerEdge=2`时继续下一协议 |
| 位置优先 | `DevicePreferredOverHost` | 同Level有Device与Host时选Device |
| | `HostFallback` | 只有Host时选Host |
| die去重 | `OneLinkPerDie` | 4条Link分属2个die时返回2条 |
| | `DieQueryFailureKeepsAll` | `dieQuery`返回false时不去重 |
| | `DedupDisabled` | `oneLinkPerSrcDie=false`时返回全部 |
| 数量 | `MaxLinksTruncates` | `maxLinksPerEdge=2`时按排序后前2条 |
| | `ExactDuplicatesRemoved` | 完全相同的Link去重 |
| | `BelowMinReturnsNotSupport` | 所有协议都不满足最小数量 |
| 排序稳定性 | `OrderIndependent` | 打乱输入顺序，输出逐字段相同 |
| | `SrcDstSymmetry` | 构造对称的两端候选，选出的`LinkCompareKey`集合相同 |
| 归属 | `TopoInstanceMatchesBySrcEndpoint` | 本端Endpoint匹配才归属 |
| | `SameProtocolDifferentEndpointNotAttributed` | 协议相同但本端Endpoint不同 -> 不归属 |
| | `NeverUsesDstEndpoint` | 只改dst Endpoint不影响归属结果 |
| | `EmptySnapshotWithNetInstanceStillAttributes` | 该Level有NetInstance Source时仍归属 |
| | `EmptySnapshotTopoOnlyRejects` | 只有TopoInstance且快照为空 -> 不合格 |

`NeverUsesDstEndpoint`是针对设计文档原始版本那条不可实现规则的**回归护栏**，必须有。

### 5.2 ST

1. **双跑对比全量回归**：`HCCL_LINK_BINDING_COMPARE=1`跑完整ST套件，断言零差异；
2. **Level选择**：设计文档§14.1的6条，构造多Level拓扑逐条验证；
3. **`0->7`负例**：L1标为PCIe但实际无Link时不创建PCIe Channel，回退到L2；
4. **混合建链**：同一算法层两个peer分别绑定L1与L2；
5. **资源数量**：Channel与Notify数量与迁移前逐拓扑对比（这是最容易回归的一项，建议做成基线表）；
6. **HostDPU**：Selector判定与Resolver选择一致；构造不一致场景断言返回`HCCL_E_NOT_SUPPORT`；
7. **降级回退**：`physicalLevels`为空时走旧路径，结果与重构前一致；
8. **增量建链**：同一peer不同协议的Channel不被`CompReqChannelWithExistChannel`误判为已存在。

### 5.3 验收对照

| 设计文档§14.5验收项 | 对应验证 |
|---------------------|----------|
| 1 统一Requirement表达 | 代码走查（`CalcChannelRequest*`内不再直接构造`HcclChannelDesc`） |
| 2 Resolver是唯一`GetLinks`调用方 | `channel.cc`中grep`GetLinks`应只剩Legacy函数；PR7后为空 |
| 3 遵循Level/位置/协议/数量策略 | UT全量 + ST-2 |
| 4 每个rank对校验实际Link | ST-3 |
| 5 归属不只依赖协议 | UT归属组 |
| 6 同层不同peer可绑不同Level | ST-4 |
| 7 HostDPU不改Hierarchy | ST-6 + TopoMatch侧用例 |
| 8 无固定NetLayer/`links[0]` | PR7后代码走查 |
| 9 去重键覆盖协议与Endpoint | ST-8 |
| 10 调用方式不变 | 编译 + 走查 |
| 11 UT/ST/端到端通过 | 全部 |

## 6. 提交拆分

| PR | 内容 | 风险 | 验证 |
|----|------|------|------|
| PR1 | `binding_compare.*`双跑对比设施 | 零（默认关闭） | 手工开启跑一次ST |
| PR2 | `link_selection_policy.*` + `link_filter.*` + UT | 零（无调用方） | `link_filter_test`全量 |
| PR3 | `link_binding_resolver.*` | 零（无调用方） | 编译 |
| PR4 | `CalcChannelRequestMesh1D`接入 + 对比 | 中 | ST-1全量对比 + ST-5 |
| PR5 | `CalcChannelRequestNhr`系列接入 | 中 | ST-1/5 |
| PR6 | MultiJetty / PCIe-SW / UBX / Mesh2D 接入 | 高 | ST-1/2/3/4/5 |
| PR7 | `CompReqChannelWithExistChannel`去重键 | 中 | ST-8 |
| PR8 | `CheckHostDPUOnly`数据来源 + 一致性校验 | 中 | ST-6 |
| PR9 | 删除Legacy函数、`binding_compare.*`、固定NetLayer与`links[0]`分支 | 中 | 全量ST |

PR1先行是这份方案与前两份最大的不同：没有对比设施就开始迁移，等于在没有基线的情况下改动资源分配逻辑。

## 7. 风险与回滚

| 风险 | 触发条件 | 缓解 | 回滚 |
|------|----------|------|------|
| Channel/Notify数量变化 | 协议顺序或die去重实现偏差 | PR1对比设施 + ST-5基线表 + UT协议/die组 | 单独revert对应接入PR |
| die去重顺序变化选中不同Link | 排序后去重 vs 遍历顺序去重（4.3.1） | 对比时区分"集合相同顺序不同"与"集合不同"；后者阻塞合入 | — |
| 部分边失败导致整个算子不可用 | 某个peer在所有Level都无合规Link | 失败日志带peerRank、已尝试的Level与每层被拒原因 | 该算法回退旧路径 |
| Legacy与新路径长期并存 | PR9迟迟不做 | Legacy函数加`TODO(migration)`；CI检查其数量不增 | — |
| `GetEndpointInfo`不可用导致die去重失效 | 低版本HCOMM | L8：support flag为false时不去重，并打一次WARNING | — |
| 增量建链误判 | 去重键改造不彻底 | ST-8 | revert PR7 |

回滚粒度：PR4-PR6每个只影响一类Template，`UseResolver()`里加一个按算法名/engine的白名单开关，
可以在不revert代码的情况下把单个算法切回旧路径——建议实现这个开关，它比revert更快。

## 8. Code Review检查清单

- [ ] `LinkSelectionPolicy::protocolPreference`是`std::vector`而不是`std::set`
- [ ] `SelectLinksForEdge`在第一个有Link的协议处停止，不跨协议合并
- [ ] die去重存在，且`dieQuery`失败时保留全部而不是丢弃
- [ ] `GetLinks`返回的裸指针在函数内立即复制成`std::vector<CommLink>`
- [ ] 归属判断只用`link.srcEndpointDesc`，全文件grep`dstEndpointDesc`只出现在比较键构造里
- [ ] `ChannelRequirement`不含`algLevel`（4.1的修正）
- [ ] Resolver在`physicalLevels`为空时返回`HCCL_E_NOT_SUPPORT`，调用方有回退分支
- [ ] 任一边失败时不返回部分Binding（`tmpBindings`在全部成功后才赋值）
- [ ] 所有实验接口调用前检查`HcommIsSupportXxx()`
- [ ] `CheckNetLayerExists`的`linkRequired`语义在迁移后保留（L10）
- [ ] 每个Legacy函数带`TODO(migration)`注释
- [ ] 新`.cc`未加入ST的AICPU源表（L11）
- [ ] 失败日志包含peerRank与逐Level拒绝原因，可定位

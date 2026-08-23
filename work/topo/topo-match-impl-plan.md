# TopoMatch统一重构 — 编码实现方案

- 文档状态：草案
- 更新日期：2026-08-12（按代码核对结果修订）
- 对应设计：[TopoMatch统一重构](./topo-match-refactor-design.md)
- 前置实现：[Topo解析实现方案](./topo-parse-impl-plan.md)（`physicalLevels`必须已可用）
- 适用仓库：`cann/hccl`

---

## 1. 范围

实现设计文档第4-8章：`AlgTopoRequest`输入契约、`SelectDim`维度选择、Level1/2/3三个主流程、
`AlgHierarchyView`与`ValidateAlgHierarchy`，以及Executor的分步迁移。

**本方案最重要的结构性特点**：新TopoMatch的全部逻辑都是**纯函数**——输入是一个可以在测试里手工构造的
`TopoInfoWithNetLayerDetails`加一个`AlgTopoRequest`，输出是`AlgHierarchyInfoForAllLevel`，全程不碰
`HcclComm`、不调RankGraph。因此设计文档§13的全部用例都可以在UT里离线跑完，不需要ST模拟器。
这是相对现有Matcher（每个都要真comm）的最大收益，实施时应优先兑现。

## 2. 实施前已确认的代码事实

> **行号说明**：本表行号锚定于`2ef7d69a [build] format all C/C++ files`之前的代码，多数引用会向后
> 偏移20-50行。实施前请按符号名重新定位。

| # | 事实 | 位置 | 影响 |
|---|------|------|------|
| G1 | `InsCollAlgBase::CalcAlgHierarchyInfo`是**纯虚**（`= 0`） | `executor_v2_base.h:35-37` | 设计文档§10.1写的"基类默认实现"落不了地，需要改用protected非虚辅助函数（见4.5） |
| G2 | 每个Executor的`CalcAlgHierarchyInfo`实现体都是`AlgTopoMatch topoMatch; topoMatch.MatchTopo(...)`三行 | 如`ins_v2_barrier_sole_executor.cc:19-25` | 迁移是机械替换，风险可控 |
| G3 | 注册宏有7个变体都把`AlgTopoMatch`作为模板参数；已存在`REGISTER_EXECUTOR_IMPL_NO_TOPOMATCH`先例 | `coll_alg_v2_exec_registry.h:43-127` | 删模板参数有现成范式，但要改所有注册点 |
| G4 | `TopoMatchBase::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails*, AlgHierarchyInfoForAllLevel&)` | `topo_match_base.h:62-63` | 薄适配器保持该签名，内部转调新入口 |
| G5 | Template构造时收到的是整个`infos[level]`（含全部平面），内部用`subcommInfo[COMM_LEVEL0]`取平面0 | `ins_v2_scatter_sequence_executor.cc:61-63`、`channel.cc:382` | 保留并行范围语义后，Template的入参类型完全不用改 |
| G6 | `ins_v2_scatter_sequence_executor.cc:54-55`把`infos[level].size()`当rankSize用（恒为1，是bug）；同文件:123-124用的才是`infos[level][0].size()` | 同上 | 迁移时顺手修正，并作为回归用例 |
| G7 | ST的AICPU目标编译`topo.cc`与`topo_match_1d/_base/_multilevel/_ubx/_pcie_mix.cc`（**白名单，不含**`_concurrent`、`_ubx_1d`、`_3_level`、`_squeeze_2d`）；现有Matcher整体用`#ifndef AICPU_COMPILE`包裹，AICPU下返回SUCCESS且Hierarchy为空 | `.../aicpu/CMakeLists.txt:189-194` | 新Matcher是纯计算、不调RankGraph，**可以且应该**在AICPU下正常编译，不加`#ifndef`包裹；但AICPU源表是白名单，新文件必须显式追加 |
| G8 | `CalcAlgHierarchyInfo`的外部调用点**共3处**：`calc_resource_graph_mode.cc:409`、`op_common.cc:765`（`GeReuseResource`）、`op_common.cc:1194`（`HcclGetAlgRes`） | 同左 | 分派逻辑改在基类辅助函数里，3个调用点都不动。但`HcclGetAlgRes`在`TryReuseResource`命中时**直接返回、根本不重算Hierarchy**（`op_common.cc:1185`），因此迁移必须保证新路径与已缓存ctx中的Hierarchy一致，否则同一通信域内先后两个算子会拿到不同分组 |
| G9 | `AlgHierarchyInfoForAllLevel`经`AlgResourceCtxSerializable`序列化，`infos`是`vector<vector<vector<u32>>>`，走`BinaryStream`的vector递归重载 | `alg_param.h:486,521` | 保留三维结构则序列化零改动；这是"保留并行范围"相对"压成二维"的额外收益 |

## 3. 交付物清单

### 3.1 新增文件

| 文件 | 内容 | 依赖comm |
|------|------|----------|
| `src/ops/op_common/topo/alg_topo_request.h` | `AlgLevelNum`、`AlgPlaneSource`、`AlgLevelSpec`、`AlgTopoRequest` | 否 |
| `src/ops/op_common/topo/alg_hierarchy_view.h` | `AlgHierarchyView`、`ValidateAlgHierarchy`声明 | 否 |
| `src/ops/op_common/topo/alg_hierarchy_view.cc` | 上述实现 | 否 |
| `src/ops/op_common/topo/topo_match_unified.h` | `MatchTopo`、`SelectDim`声明 | 否 |
| `src/ops/op_common/topo/topo_match_unified.cc` | Level1/2/3与公共规则实现 | 否 |
| `test/ut/topo_match_unified/CMakeLists.txt` | UT构建 | — |
| `test/ut/topo_match_unified/topo_match_unified_test.cc` | 设计文档§13.1-13.3全部用例 | — |
| `test/ut/topo_match_unified/alg_hierarchy_view_test.cc` | View与Validate用例 | — |

`AlgTopoRequest`放在独立头而不是`alg_param.h`：它不参与序列化，只是Executor的静态声明，不需要被
Device侧看到，也不该让`alg_param.h`继续膨胀。

### 3.2 修改文件

| 文件 | 修改 |
|------|------|
| `src/ops/op_common/executor/executor_v2_base.h/.cc` | 新增`GetAlgTopoRequest()`虚函数与`ResolveAlgHierarchy()`protected辅助 |
| `src/ops/op_common/topo/CMakeLists.txt` | 追加两个新`.cc`到**无条件**`src_list`（文件开头那个块）。注意特殊Matcher都在`if(NOT HCCL_CANN_COMPAT_850)`分支下，新文件不走能力宏、不放进该分支 |
| `test/st/algorithm/utils/src/aicpu/CMakeLists.txt` | 追加两个新`.cc`（白名单，必须显式列出，见G7） |
| 各`topo_match_*.cc` | 逐个改为薄适配器（分批） |
| 各Executor`.cc/.h` | 实现`GetAlgTopoRequest()`，`CalcAlgHierarchyInfo`改为调用基类辅助（分批） |
| `test/ut/CMakeLists.txt` | `add_subdirectory(topo_match_unified)` |

## 4. 详细设计

### 4.1 输入契约（`alg_topo_request.h`）

```cpp
namespace ops_hccl {

enum class AlgLevelNum : u32 { INVALID = 0, LEVEL_1 = 1, LEVEL_2 = 2, LEVEL_3 = 3 };

enum class AlgPlaneSource : u32 {
    PRIMARY = 0,           // 与该层主rankList相同
    NEXT_PHYSICAL_RANGE,   // 该层主范围之上的下一个PhysicalLevel
};

struct AlgLevelSpec {
    std::vector<AlgPlaneSource> planes;  // 空或{PRIMARY}表示单平面
};

struct AlgTopoRequest {
    AlgLevelNum algLevel = AlgLevelNum::INVALID;
    u32 baseLevelIdx = 0;                // 精确下标，不向上扫描（设计§4.1）
    std::vector<AlgLevelSpec> levels;    // 空表示所有层单平面

    bool IsValid() const;                 // algLevel != INVALID 且 planes[0] == PRIMARY
    u32 PlaneNumOf(u32 level) const;      // levels未覆盖该层时返回1
    AlgPlaneSource PlaneSourceOf(u32 level, u32 plane) const;
};

} // namespace ops_hccl
```

`PlaneNumOf`/`PlaneSourceOf`把"未声明即单平面"的默认逻辑收在一处，避免每个使用点重复判空。

### 4.2 公共维度选择（`topo_match_unified.cc`）

```cpp
HcclResult SelectDim(
    const std::vector<u32>& rangeRanks,  // 已升序去重
    u32 stride, u32 dimSize, u32 myRank,
    std::vector<u32>& out);
```

实现：

```text
1. candidates.clear()
   for r in rangeRanks:  if (r % stride == myRank % stride) candidates.push_back(r)
   （rangeRanks已升序 -> candidates天然升序，不需再排序）
2. it = lower_bound(candidates, myRank)；myRank不在其中 -> HCCL_E_NOT_SUPPORT
3. dimSize == 0 || candidates.size() % dimSize != 0 -> HCCL_E_NOT_SUPPORT
4. idx   = it - candidates.begin()
   begin = (idx / dimSize) * dimSize
   out.assign(candidates.begin() + begin, candidates.begin() + begin + dimSize)
```

三个不变量在函数内自查：`out.size() == dimSize`、`out`含`myRank`、`out ⊆ rangeRanks`。

`stride == 1 && dimSize == rangeRanks.size()`时结果就是`rangeRanks`本身，这是最常见的Level0路径；
实现里不需要为它写特例分支，走通用逻辑即可，UT单独覆盖该退化路径。

**`SelectDim`本身对非连续`rangeRanks`是正确的**（如Level0直接使用`{0,2,4,6}`），受限的是**用
`stride`构造上层补集**这一步——它要求各Level0分组按`rank % d0`划分等价类，即首期只支持连续块布局
（设计§5.3）。交错布局会在`ValidateAlgHierarchy`的正交性检查处失败，返回`HCCL_E_NOT_SUPPORT`。
不要在`SelectDim`里加连续性判断：单独看一个范围无法判定，只有正交性检查能发现。

### 4.3 三个主流程

统一骨架：

```cpp
HcclResult MatchTopo(
    const TopoInfoWithNetLayerDetails& topoInfo, const AlgTopoRequest& request,
    AlgHierarchyInfoForAllLevel& hierarchy)
{
    CHK_PRT_RET(!request.IsValid(), ..., HCCL_E_PARA);
    CHK_PRT_RET(topoInfo.userRankSize == 0, ..., HCCL_E_PARA);
    CHK_PRT_RET(topoInfo.physicalLevels.empty(), ..., HCCL_E_NOT_SUPPORT);

    AlgHierarchyInfoForAllLevel tmp;              // 临时对象，全部校验通过后再赋值
    switch (request.algLevel) {
        case LEVEL_1: CHK_RET(MatchLevel1(topoInfo, request, tmp)); break;
        case LEVEL_2: CHK_RET(MatchLevel2(topoInfo, request, tmp)); break;
        case LEVEL_3: CHK_RET(MatchLevel3(topoInfo, request, tmp)); break;
        default: return HCCL_E_PARA;
    }
    CHK_RET(BuildPlanes(topoInfo, request, tmp));   // 统一补齐所有层的并行范围
    CHK_RET(ValidateAlgHierarchy(tmp, request, topoInfo.userRankSize, topoInfo.userRank));
    hierarchy = std::move(tmp);
    return HCCL_SUCCESS;
}
```

三个`MatchLevelN`只负责填每层的`infos[level][0]`（主rankList）并记录每层的**物理主范围下标**，
并行范围统一由`BuildPlanes`补齐。这样"平面"逻辑只有一份实现，不会在三个流程里各写一遍。

内部上下文：

```cpp
struct MatchContext {
    u32 primaryLevelIdx[HCCL_LOGIC_TOPO_LEVEL_NUM]; // 该算法层对应的physicalLevels下标，无则INVALID_UINT
};
```

**共用子过程**：

```cpp
// 取 physicalLevels[baseLevelIdx] 作为B0，精确不扫描；越界或退化返回 NOT_SUPPORT
HcclResult SelectBaseLevel(const topoInfo, u32 baseLevelIdx, u32& levelIdx, u32& d0);
// partitionGcd > 0 ? partitionGcd : localRanks.size()
u32 DimOfLevel(const PhysicalLevelInfo& level);
// partitionGcd == 0 时的交叉校验（设计§5.2）
HcclResult ValidateLocalDim(const topoInfo, u32 levelIdx, u32 d0);
// partitionGcd > 0 时要求 partitionUniform（设计§8.0），仅Level3调用
HcclResult ValidateUniform(const PhysicalLevelInfo& level);
// 找 localRanks.size() == userRankSize 的Level；不存在返回false
bool FindFullCoverLevel(const topoInfo, u32& levelIdx);
```

`SelectBaseLevel`**不向上扫描**（设计§4.1）：

```text
baseLevelIdx >= physicalLevels.size()      -> HCCL_E_NOT_SUPPORT
d0 = DimOfLevel(physicalLevels[baseLevelIdx])
partitionGcd == 0 -> ValidateLocalDim(...)  // 设计§5.2，失败即 NOT_SUPPORT
d0 <= 1 || rankSize % d0 != 0              -> HCCL_E_NOT_SUPPORT
levelIdx = baseLevelIdx
```

写成循环"找第一个非退化"是**错的**：拓扑结构变化会让下标含义偏移，扫描会把偏移变成静默的分组变化，
而精确选取会把它暴露成明确的`NOT_SUPPORT`。这也是Squeeze2D与Sequence2防塌缩保证的前提。

`ValidateLocalDim`实现设计§5.2：

```text
rankSize % d0 != 0 -> NOT_SUPPORT
for L in physicalLevels:
    若 L.partitionGcd > 0 且 L.localRanks ⊇ physicalLevels[levelIdx].localRanks:
        L.partitionGcd % d0 != 0 -> NOT_SUPPORT
```

**Level1**（按设计§6.2，与现有`TopoMatch1D`/`TopoMatchConcurrent`一致）：

```text
infos[0][0] = {0, 1, ..., userRankSize - 1}      // 直接构造，不查找全覆盖Level
primaryLevelIdx[0] = INVALID_UINT                // 最外层，只能声明PRIMARY平面
```

> **不要**在Level1里调`FindFullCoverLevel`并在失败时返回`NOT_SUPPORT`。现有`TopoMatch1D`和
> `TopoMatchConcurrent`都是无条件构造`{0..rankSize-1}`、不查询任何物理范围；要求存在全覆盖Level
> 会让"各层`netInstNum`都大于1"的通信域上全部Sole/Concurrent算法失败，属于回归。
>
> 相应地，这两个Matcher承担的**准入拦截**（topoType必须是CLOS/CUSTOM、Concurrent额外要求
> `topoLevelNums <= 2`）不进`MatchTopo`，改由Selector承担——迁移Sole/Concurrent之前必须先确认
> Selector已覆盖，见设计§6.3，这是PR4的准入条件。

**Level2**：

```text
SelectBaseLevel(baseLevelIdx) -> (b0, d0)
d1 = rankSize / d0
校验 d0 > 1 && d1 > 1 && d0 * d1 == rankSize
FindFullCoverLevel -> full；不存在 -> HCCL_E_NOT_SUPPORT
infos[0][0] = SelectDim(physicalLevels[b0].localRanks, 1,  d0)
infos[1][0] = SelectDim(physicalLevels[full].localRanks, d0, d1)
primaryLevelIdx = {b0, INVALID_UINT}
```

Level2**不做**非对称拒绝：现有实现对两层拓扑是支持非对称的（GCD子组切分路径）。

**Level3**：

```text
SelectBaseLevel(baseLevelIdx) -> (b0, d0)
ValidateUniform(physicalLevels[b0])                       // 设计§8.0
b1 = b0之后第一个满足：localRanks严格包含b0.localRanks 且 能形成 d1 > 1 的Level
    partitionGcd > 0 : 要求 partitionGcd % d0 == 0, d1 = partitionGcd / d0
    partitionGcd == 0: 要求 localRanks.size() % d0 == 0, d1 = localRanks.size() / d0
                       并调 ValidateLocalDim
ValidateUniform(physicalLevels[b1])                       // 设计§8.0
d2 = rankSize / (d0 * d1)
校验 d0 > 1 && d1 > 1 && d2 > 1 && d0*d1*d2 == rankSize
FindFullCoverLevel -> full；不存在 -> HCCL_E_NOT_SUPPORT
infos[0][0] = SelectDim(b0.localRanks,   1,       d0)
infos[1][0] = SelectDim(b1.localRanks,   d0,      d1)
infos[2][0] = SelectDim(full.localRanks, d0 * d1, d2)
primaryLevelIdx = {b0, b1, INVALID_UINT}
```

```cpp
HcclResult ValidateUniform(const PhysicalLevelInfo& level)
{
    // partitionGcd == 0 表示无全局统计，不做该判定（与旧实现口径一致：
    // 旧实现的 isSymmetric/layer1Symmetric 也只基于 instSizeList）
    if (level.partitionGcd > 0 && !level.partitionUniform) {
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}
```

`ValidateUniform`复刻现有`TopoMatchMultilevel`的三层非对称拒绝
（`topoLevelNums >= 3 && (!isSymmetric || !layer1Symmetric) -> HCCL_E_NOT_SUPPORT`）。
**必须用`partitionUniform`而不是`partitionGcd`**：`[8,8,8,8]`与`[8,24]`的GCD都是8，仅凭GCD区分不了
对称与非对称，漏掉这一位会让重构后的Level3接受旧实现明确拒绝的拓扑。

`b1`的扫描与`baseLevelIdx`的"精确不扫描"不冲突：`baseLevelIdx`是算法声明的锚点，`b1`是由整除关系
推导的，扫描每一步都带可验证约束（严格包含 + 整除 + 非退化 + 对称）。

设计文档§8.2的"每个Instance大小可被d0整除"已被`partitionGcd % d0 == 0`等价替代，实现里不需要遍历
原始统计列表——这是`partitionGcd`设计的直接收益。

**BuildPlanes**：

```text
for level in [0, algLevel):
    planeNum = request.PlaneNumOf(level)
    infos[level].resize(planeNum)   // infos[level][0] 已填
    for k in [1, planeNum):
        switch (request.PlaneSourceOf(level, k)):
            PRIMARY:
                infos[level][k] = infos[level][0]
            NEXT_PHYSICAL_RANGE:
                primaryLevelIdx[level] == INVALID_UINT -> HCCL_E_PARA（最外层不允许）
                next = primaryLevelIdx[level] + 1
                next >= physicalLevels.size() -> HCCL_E_NOT_SUPPORT
                infos[level][k] = physicalLevels[next].localRanks
```

`NEXT_PHYSICAL_RANGE`用`HCCL_E_PARA`区分"声明本身非法"（最外层）与`HCCL_E_NOT_SUPPORT`区分
"声明合法但拓扑不满足"（没有更高范围），便于定位问题出在算法声明还是运行环境。

### 4.4 View与校验（`alg_hierarchy_view.h/.cc`）

```cpp
class AlgHierarchyView {
public:
    explicit AlgHierarchyView(const AlgHierarchyInfoForAllLevel& hierarchy) : hierarchy_(hierarchy) {}
    u32 LevelNum() const;
    u32 PlaneNum(u32 level) const;                              // 越界返回0
    u32 LevelRankSize(u32 level) const;                         // 越界返回0
    const std::vector<u32>& Plane(u32 level, u32 plane = 0) const; // 越界返回静态空vector
private:
    static const std::vector<u32>& EmptyRanks();                // 函数内static，避免静态初始化顺序问题
    const AlgHierarchyInfoForAllLevel& hierarchy_;
};
```

`ValidateAlgHierarchy`逐条实现设计文档§4.5的10个不变量。实现要点：

- 不变量7（不同算法层主rankList交集仅为myRank）：两两`std::set_intersection`，层数≤3，代价可忽略。
  **失败时返回`HCCL_E_NOT_SUPPORT`而不是`HCCL_E_INTERNAL`**——交错布局的Level0是合法但首期不支持的
  拓扑，取模补集与Level0相交是设计§5.3约束的预期表现，不是本模块的bug。归入`INTERNAL`还会绕过4.5的
  迁移期回退（回退只认`NOT_SUPPORT`）。日志需带上层号、`d0`/`d1`与两层rankList；
- 不变量8（平面约束）：`std::includes`判断相等或严格包含；
- 不变量10（全rank一致性）无法在单rank内验证，由UT的"遍历全部rank"用例覆盖，不放进运行期校验。

View不做任何校验，也不缓存；调用方在使用前必须已经过`ValidateAlgHierarchy`，或自行判断
`LevelRankSize(level) != 0`。头文件注释里写明这一契约。

### 4.5 Executor接入（`executor_v2_base.h/.cc`）

因为`CalcAlgHierarchyInfo`是纯虚（G1），不能靠"基类默认实现"做分派。改为：

```cpp
class InsCollAlgBase {
public:
    // 未迁移的Executor不重写，返回 algLevel = INVALID
    virtual AlgTopoRequest GetAlgTopoRequest() const { return AlgTopoRequest{}; }

    virtual HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) = 0;   // 保持纯虚

protected:
    // 统一分派：迁移后的Executor在CalcAlgHierarchyInfo里调用它
    HcclResult ResolveAlgHierarchy(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
        AlgHierarchyInfoForAllLevel& hierarchy,
        const std::function<HcclResult()>& legacyFallback) const;
};
```

`ResolveAlgHierarchy`实现设计文档§10.1的四条分派规则：

```text
1. request.algLevel == INVALID              -> legacyFallback()
2. topoInfo->physicalLevels.empty()         -> WARNING + legacyFallback()
3. ret = MatchTopo(*topoInfo, request, hierarchy)
4. ret == HCCL_E_NOT_SUPPORT                -> WARNING + legacyFallback()   // 迁移期
   其他非SUCCESS                             -> 直接返回，不回退
```

第4条是**迁移期的必需项**。`physicalLevels`非空不代表新规则一定能匹配：layer0 `topoInstNum == 0`
（旧实现给`layer0Size = 1`继续跑，新实现`d0 > 1`失败）、交错布局Level0、部分连通拓扑，这些今天都是
能跑的。没有第4条，每个迁移PR的爆炸半径就是"这类算法在某些拓扑上直接不可用"。

回退时的WARNING必须带上算法名、`algLevel`、`baseLevelIdx`与`MatchTopo`的失败点，它同时是统计降级面
的唯一手段——PR8删除第4条之前，必须先用这条日志确认现网无命中。

`HCCL_E_INTERNAL`与`HCCL_E_PARA`**不回退**：前者是TopoMatch自身的bug，后者是Executor声明写错了，
回退只会把它们藏起来。

保持纯虚的理由：改成非纯虚会让漏实现的Executor静默走到基类默认行为；保持纯虚则每个Executor都必须
显式写出自己的选择，迁移进度在代码里一目了然。

迁移后的Executor实现体（以Sequence2为例）：

```cpp
template <typename AlgTopoMatch, typename T0, typename T1>
AlgTopoRequest InsV2ScatterSequenceExecutor<AlgTopoMatch, T0, T1>::GetAlgTopoRequest() const
{
    AlgTopoRequest req;
    req.algLevel = AlgLevelNum::LEVEL_2;
    return req;
}

template <typename AlgTopoMatch, typename T0, typename T1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, T0, T1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& hierarchy)
{
    return ResolveAlgHierarchy(comm, topoInfo, hierarchy, [&]() {
        AlgTopoMatch topoMatch;
        return topoMatch.MatchTopo(comm, topoInfo, hierarchy);
    });
}
```

模板参数`AlgTopoMatch`在这一阶段仍然保留，只作为回退路径使用；等某个Executor的全部目标拓扑都验证通过后，
再按设计文档§10.2删除该参数与对应注册宏形参。

典型声明速查：

| Executor类别 | `GetAlgTopoRequest()` |
|--------------|------------------------|
| Sole | `{LEVEL_1}` |
| Concurrent（全域双平面） | `{LEVEL_1, levels[0].planes = {PRIMARY, PRIMARY}}` |
| Concurrent（mesh+clos，原`TopoMatchUBX`） | `{LEVEL_2, levels[0].planes = {PRIMARY, NEXT_PHYSICAL_RANGE}}` |
| Parallel / Sequence2 | `{LEVEL_2}` |
| AllGatherParallel（原`TopoMatchSqueeze2D`） | `{LEVEL_2, baseLevelIdx = 1}` |
| Sequence3 / OmniPipe | `{LEVEL_3}` |
| HostDPU多级 | `{LEVEL_2}`，无论`topoLevelNums`是2还是3（设计§9.3） |

声明`baseLevelIdx != 0`的Executor必须在实现处加注释说明其拓扑前提（设计§4.1）：下标是拓扑相关的
稠密序数，Squeeze2D之所以能用`1`，是因为它的Selector把拓扑限定在了layer0/1/2的`topoInstNum`均为1
的纯Mesh1D形态。

### 4.6 薄适配器

旧Matcher类保留`MatchTopo(comm, topoInfo, hierarchy)`签名（G4），内部改为：

```cpp
HcclResult TopoMatchMultilevel::MatchTopo(
    const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& hierarchy)
{
    // 必须完整复刻旧实现的层数推导，包括 HostDPU 降级
    bool hostDPUOnly = false;
    bool needDowngrade = (CheckHostDPUOnly(comm, topoInfo, hostDPUOnly) == HCCL_SUCCESS) && hostDPUOnly
                         && topoInfo->topoLevelNums == COMM_LAYER_SIZE_3;

    AlgTopoRequest req;
    req.algLevel = (topoInfo->topoLevelNums == COMM_LAYER_SIZE_3 && !needDowngrade)
                       ? AlgLevelNum::LEVEL_3 : AlgLevelNum::LEVEL_2;
    return MatchTopo(*topoInfo, req, hierarchy);
}
```

> **`needDowngrade`不能省。**旧实现的层数是
> `commLayerSize = (topoLevelNums == 3 && !needDowngrade) ? 3 : 2`，写成
> `topoLevelNums >= 3 ? LEVEL_3 : LEVEL_2`会让HostDPU三层拓扑的`infos.size()`从2变成3，
> Executor的`CalcRes`、channel数量与Template构造全部错位。而"薄适配器与旧实现逐字段等价"（ST-1）
> 是整个迁移的安全网，这一条破了后续所有迁移PR的验证前提都不成立。
>
> 另注意是`== COMM_LAYER_SIZE_3`而不是`>= 3`：旧实现在入口处已用
> `topoLevelNums > COMM_LAYER_SIZE_3`拦截，适配器不要放宽这个边界。
>
> 薄适配器里根据`topoLevelNums`推导`algLevel`是**迁移期的临时妥协**，与设计文档§10.3"物理层数不
> 改变算法层数"相冲突。正式方案是由Selector区分HostDPU并选择声明`{LEVEL_2}`的Executor（设计§9.3）。
> 因此薄适配器只允许作为过渡态存在，必须在对应Executor实现了`GetAlgTopoRequest()`之后立刻删除。
> 实施时给每个薄适配器加`// TODO(migration): remove after <Executor>`注释，并在最后一个PR里统一清理。

mesh2d分支（`topo_match_multilevel.cc:65-99`）不迁移：该拓扑的`physicalLevels`在解析阶段就是空的，
分派规则2会自动回退到旧实现，薄适配器需要保留原有代码路径而不是删除。

## 5. 测试方案

### 5.1 UT（`test/ut/topo_match_unified/`）

复刻`test/ut/reduce_scatter_birs/CMakeLists.txt`。编译`topo_match_unified.cc`与`alg_hierarchy_view.cc`，
不链接RankGraph与`hccl_comm`。

测试夹具：一个`MakeTopoInfo(userRank, userRankSize, {{ranks, gcd}, ...})`辅助函数直接构造
`TopoInfoWithNetLayerDetails::physicalLevels`，无需任何桩。

**`topo_match_unified_test.cc`**（对应设计文档§13.1-13.3）：

| 组 | 用例 | 断言 |
|----|------|------|
| SelectDim | `DegenerateWholeRange` | `stride=1, dimSize=size`时返回整个范围 |
| | `SplitBySubBlock` | `[0..7]`、`stride=1`、`dimSize=4`，rank5得`{4,5,6,7}` |
| | `StrideFilter` | 全域12、`stride=4`、`dimSize=3`，rank5得`{1,5,9}` |
| | `NonContiguousMembers` | 范围`{0,2,4,6}`、`dimSize=4`，返回该范围本身（`SelectDim`本身不受连续性约束） |
| | `NotDivisibleReturnsNotSupport` | `candidates.size() % dimSize != 0` |
| Level1 | `FullDomainWithoutFullCoverLevel` | `physicalLevels`最大范围**小于**rankSize时，仍返回`{0..rankSize-1}`且SUCCESS（守护§6.2，防止回归成`NOT_SUPPORT`） |
| | `MultiPhysicalLevelUnchanged` | `[8,8,8,8]->[32]`，仍为单层全rank |
| | `DualPlaneDuplicates` | `planes={PRIMARY,PRIMARY}`，`infos[0].size()==2`且内容相同 |
| | `NextPhysicalRangeOnLevel1IsPara` | Level1唯一层是最外层，声明`NEXT_PHYSICAL_RANGE`返回`HCCL_E_PARA` |
| Level2 | `Gcd8x4` / `Gcd4x7` / `Gcd4x3` | 对应设计文档§13.2前三行 |
| | `Gcd1ReturnsNotSupport` | `[8,9]` |
| | `OnlyFullCoverReturnsNotSupport` | 只有`[32]` |
| | `NoFullCoverLevelReturnsNotSupport` | Level2的最外层维度需要全覆盖范围，不存在时`NOT_SUPPORT` |
| | `BaseLevelIdx0Gives8x4` / `BaseLevelIdx1Gives16x2` | 同一物理视图两种声明（防塌缩） |
| | `BaseLevelIdxOutOfRange` | 返回`HCCL_E_NOT_SUPPORT` |
| | `BaseLevelIdxDegenerateDoesNotScanUp` | `physicalLevels[1]`退化、`physicalLevels[2]`可用时，声明`baseLevelIdx=1`必须返回`NOT_SUPPORT`，**不得**自动改用`[2]`（守护"精确不扫描"） |
| | `InterleavedLevel0ReturnsNotSupport` | Level0为`{0,2,4,6}`、rankSize=8：返回`HCCL_E_NOT_SUPPORT`（不正交），**断言不是`HCCL_E_INTERNAL`** |
| | `ContiguousBlockLevel0Ok` | Level0为`{0,1,2,3}`：正常匹配（与上一条配对，证明拒绝的是布局不是非连续本身） |
| | `LocalDimCrossCheckRejects` | Level0 `partitionGcd==0`、size=6，上层`partitionGcd==8`：`NOT_SUPPORT`（设计§5.2） |
| | `LocalDimCrossCheckAccepts` | 同上但size=8：匹配成功（与上一条配对） |
| | `AsymmetricLevel2IsSupported` | `[8,8,8,4]`两层：`4 x 7`匹配成功（Level2不做非对称拒绝） |
| Level3 | `Nested8x2x2` / `Local4x2x2` | 设计文档§13.3前两行；后者各层`partitionGcd==0`，不触发对称判定 |
| | `MissingMiddleRangeNotSupport` | `[8,8,8,8]->[32]` |
| | `AsymmetricB1ReturnsNotSupport` | B1 `partitionGcd==8`但`partitionUniform==false`（如`[8,24]`）：`NOT_SUPPORT` |
| | `SymmetricSameGcdIsSupported` | B1 `partitionGcd==8`且`partitionUniform==true`（如`[8,8,8,8]`）：匹配成功。**与上一条配对**——两者GCD相同，只有`partitionUniform`不同，缺了这对用例就证明不了该字段真的生效 |
| | `DegenerateD1NotSupport` | `[8,8,8,4]->[16,12]->[28]` |
| 平面 | `NextPhysicalRangeGivesClos` | `infos[0][0]`为mesh、`infos[0][1]`为clos，且mesh ⊂ clos |
| | `NextPhysicalRangeNoHigherRange` | `HCCL_E_NOT_SUPPORT` |
| | `NextPhysicalRangeOnOuterLevel` | `HCCL_E_PARA` |
| 全rank | `AllRankConsistency` | 对rankSize内每个rank跑一遍，校验设计文档§13.4的1-4条 |
| | `DuplicateFullCoverSourcesUnchanged` | Device/Host双来源合并后Hierarchy不变 |
| | `InputOrderIndependent` | 打乱`physicalLevels`内Source顺序，输出不变 |

`AllRankConsistency`是这批用例里价值最高的一个：它把"同一group内各rank看到的rankList完全相同"这个
过去只能靠集群跑出来的性质变成了一个离线断言。

**`alg_hierarchy_view_test.cc`**：

| 用例 | 断言 |
|------|------|
| `ViewDoesNotCopy` | `&view.Plane(0,0) == &hierarchy.infos[0][0]` |
| `OutOfRangeReturnsEmpty` | 越界层/平面返回空vector且不崩溃 |
| `ValidateRejectsWrongLevelNum` | `infos.size() != algLevel` |
| `ValidateRejectsWrongPlaneNum` | 与声明不符 |
| `ValidateRejectsMissingMyRank` | 某层不含myRank |
| `ValidateRejectsProductMismatch` | `d0*d1 != rankSize` |
| `ValidateRejectsNonOrthogonal` | 两层交集不止myRank |
| `ValidateRejectsOverlapPlanes` | 平面互相重叠但不包含 |

### 5.2 ST

UT覆盖了算法本身，ST只需覆盖**集成面**：

1. **薄适配器等价**：对每个仍在用旧Matcher的算法，跑现有ST用例，断言Hierarchy与重构前逐字段一致。
   **必须单列一条HostDPU + 物理三层的用例**，断言`infos.size() == 2`——这是薄适配器最容易写错、
   且错了之后影响面最大的一处（4.6）；
2. **降级回退**：分两条——(a)`physicalLevels`为空，断言走旧实现且结果与重构前一致；
   (b)`physicalLevels`非空但新规则返回`NOT_SUPPORT`（如layer0 `topoInstNum == 0`），断言同样回退旧
   实现且结果与重构前一致（4.5第4条）；
3. **UBX对比**（设计文档§9.2要求的专项）：同一拓扑下分别用旧`TopoMatchUBX`与新
   `{LEVEL_2, planes={PRIMARY, NEXT_PHYSICAL_RANGE}}`跑，输出逐rank对比并记录差异清单，
   **差异被评审确认后**才允许切换该算法；
4. **Squeeze不塌缩**：AllGatherParallel与Sequence2在同一拓扑下Hierarchy不同；
5. **序列化不变**：`AlgHierarchyInfoForAllLevel`往返一致（G9，理论上零改动，加一条守护用例）。

### 5.3 验收对照

| 设计文档§13.6验收项 | 对应验证 |
|---------------------|----------|
| 1 只读`localRanks`/`partitionGcd` | 代码走查 + UT不提供其他字段也能通过 |
| 2 只有三个主实现 | 代码走查 |
| 3 无RankGraph调用 | `topo_match_unified.cc`中grep`HcclRankGraphGet`为空 |
| 4 直接输出原结构 | UT + ST-5 |
| 5 显式声明层数与平面数 | UT`BaseLevelSkip*`、`DualPlane*` |
| 6 Mesh/NHR与协议不改变Hierarchy | UT`DuplicateFullCoverSources*` + ST-1 |
| 7 并行范围只来自声明 | UT平面组 |
| 8 算法名可用且Squeeze语义未丢 | ST-4 |
| 9 UT/ST/序列化通过 | 全部 |

## 6. 提交拆分

| PR | 内容 | 风险 | 验证 |
|----|------|------|------|
| PR1 | `alg_topo_request.h` + `alg_hierarchy_view.*` + Validate + UT | 零（无调用方） | `alg_hierarchy_view_test` |
| PR2 | `topo_match_unified.*`（SelectDim + Level1/2/3 + BuildPlanes）+ UT | 零（无调用方） | `topo_match_unified_test`全量 |
| PR3 | `InsCollAlgBase::GetAlgTopoRequest` + `ResolveAlgHierarchy`（含NOT_SUPPORT回退） | 零（默认INVALID，全部走回退） | 全量ST回归 |
| PR3.5 | 补齐Sole/Concurrent的Selector准入条件（设计§6.3） | 低 | 全量ST回归 |
| PR4 | 迁移Sole与Concurrent（Level1，含双平面）。**前置：PR3.5已合入** | 低 | ST-1/2 |
| PR5 | 迁移Parallel/Sequence2/PcieMix（Level2）+ 修G6的bug | 中 | ST-1/2/4 |
| PR6 | 迁移Sequence3/OmniPipe（Level3，含`ValidateUniform`） | 中 | ST-1/2 |
| PR6.5 | Selector区分HostDPU并迁移HostDPU多级算法到`{LEVEL_2}`（设计§9.3） | 中 | ST-1 的HostDPU用例 |
| PR7 | UBX双平面迁移（需PR前置的对比结论） | 高 | ST-3 |
| PR8 | 删除模板参数、注册宏形参、旧Matcher类（mesh2d除外）与4.5第4条的NOT_SUPPORT回退 | 中（改动面大但机械） | 全量ST + 编译；删回退前先用降级日志确认现网无命中 |

PR1-PR3合入后系统行为**完全不变**，新代码已经带着全量UT进主干，后续每个迁移PR只影响一类算法，
出问题时revert粒度是"一类算法"而不是"整个重构"。

## 7. 风险与回滚

| 风险 | 触发条件 | 缓解 | 回滚 |
|------|----------|------|------|
| **薄适配器丢失HostDPU降级** | 按`topoLevelNums`推导层数、漏掉`needDowngrade` | 4.6的实现 + ST-1的HostDPU三层专项用例（断言`infos.size()==2`） | 单独revert该适配器 |
| **Level1回归成要求全覆盖Level** | 照搬Level2/3的`FindFullCoverLevel` | 4.3的显式说明 + UT`FullDomainWithoutFullCoverLevel` | — |
| **Sole/Concurrent失去准入拦截** | 先迁移后补Selector | PR3.5作为PR4的前置；设计§6.3 | revert PR4 |
| `baseLevelIdx`被实现成向上扫描 | 复用"找第一个非退化"的写法 | 4.3的`SelectBaseLevel` + UT`BaseLevelIdxDegenerateDoesNotScanUp` | — |
| Level3静默接受非对称拓扑 | 漏掉`ValidateUniform`，或误用`partitionGcd`判对称 | UT`AsymmetricB1ReturnsNotSupport`/`SymmetricSameGcdIsSupported`配对用例 | — |
| 不正交返回`INTERNAL`导致回退失效 | 错误码分类写错 | 4.4的说明 + UT断言错误码具体值 | — |
| 薄适配器沉淀成新的隐式规则 | PR8迟迟不做 | 每个适配器加`TODO(migration)`并在PR8统一清理；CI加一条"薄适配器数量不增"的检查 | — |
| UBX分组语义改变 | `infos[1]`由链路过滤改为取模构造 | ST-3逐rank对比，未确认前不切换 | 单独revert PR7 |
| 取消`GetLinks`过滤后失败点后移 | 部分连通拓扑 | 设计文档§13.4.8用例；错误码固定为`HCCL_E_NOT_SUPPORT`并带上层号与维度日志 | — |
| 删模板参数改动面过大 | PR8一次性改所有注册点 | 按算子目录拆成多个commit，每个commit独立可编译 | 按commit revert |
| G6的bug修复改变现有行为 | `infos[0].size()`→`infos[0][0].size()` | 该值原本恒为1，修复后变成真实rankSize，属于**行为修正**；单独一个commit并附ST对比 | 单独revert该commit |

关于G6需要特别说明：`ins_v2_scatter_sequence_executor.cc:54-55`修复后`rankSizeLevel0_`会从1变成真实值。
如果下游有代码在无意中依赖了这个1，修复会暴露问题。因此该修复必须**单独成commit**、单独跑ST，
不要和迁移混在一起提交。

## 8. Code Review检查清单

- [ ] `topo_match_unified.cc`中不出现`HcclRankGraphGet`、`HcclComm`、`GetLinks`
- [ ] `MatchTopo`在临时对象上构建，校验通过后才赋值给出参
- [ ] `physicalLevels`为空时返回`HCCL_E_NOT_SUPPORT`
- [ ] **`MatchLevel1`直接构造`{0..rankSize-1}`，没有调用`FindFullCoverLevel`**
- [ ] `SelectBaseLevel`按`baseLevelIdx`精确取值，**没有**"找第一个非退化"的循环
- [ ] `partitionGcd == 0`取维度时调用了`ValidateLocalDim`（设计§5.2交叉校验）
- [ ] Level3对B0与B1都调用了`ValidateUniform`，且判据是`partitionUniform`而不是`partitionGcd`
- [ ] Level2**没有**调用`ValidateUniform`（两层拓扑现有实现支持非对称）
- [ ] 不变量7（正交性）失败返回`HCCL_E_NOT_SUPPORT`，不是`HCCL_E_INTERNAL`
- [ ] `SelectDim`的成员全部取自`rangeRanks`，没有`(myRank / d) * d + i`形式的构造式
- [ ] 并行范围只在`BuildPlanes`一处生成，三个`MatchLevelN`里没有重复实现
- [ ] `NEXT_PHYSICAL_RANGE`用在最外层时返回`HCCL_E_PARA`而不是`NOT_SUPPORT`
- [ ] `ResolveAlgHierarchy`对`NOT_SUPPORT`回退旧实现，对`INTERNAL`/`PARA`不回退
- [ ] **薄适配器复刻了`needDowngrade`**，HostDPU三层拓扑产出2层Hierarchy
- [ ] 薄适配器用`== COMM_LAYER_SIZE_3`而不是`>= 3`
- [ ] `AlgHierarchyView`的越界返回是函数内`static`空vector，不是全局对象
- [ ] `CalcAlgHierarchyInfo`保持纯虚，未被改成带默认实现的虚函数
- [ ] 每个薄适配器都带`TODO(migration)`注释
- [ ] 声明`baseLevelIdx != 0`的Executor带有拓扑前提注释
- [ ] 新`.cc`加入`topo/CMakeLists.txt`的**无条件**`src_list`（不在`if(NOT HCCL_CANN_COMPAT_850)`分支内），并显式加入ST的AICPU源表，且**未**用`#ifndef AICPU_COMPILE`包裹
- [ ] G6的bug修复是独立commit

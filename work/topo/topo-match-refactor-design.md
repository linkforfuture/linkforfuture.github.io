# TopoMatch统一重构需求设计

- 文档状态：草案
- 更新日期：2026-08-12（按代码核对结果修订）
- 适用仓库：`cann/hccl`
- 上游设计：[Topo信息解析与标准化](./topo-parse-refactor-design.md)
- 下游设计：[物理建链统一重构](./link-binding-refactor-design.md)

---

## 1. 概要

本文档只定义算法拓扑匹配：TopoMatch读取Topo解析阶段生成的`physicalLevels`，按Executor声明的`AlgTopoRequest`
生成`AlgHierarchyInfoForAllLevel`。

```text
TopoInfoWithNetLayerDetails.physicalLevels
    + Executor.GetAlgTopoRequest()
    -> MatchTopo(Level1/2/3)
    -> AlgHierarchyInfoForAllLevel
```

核心不变量：

```text
Hierarchy(myRank) = F(topoInfo.physicalLevels, algTopoRequest, myRank)
```

相同物理分区视图、相同`AlgTopoRequest`和相同`myRank`必须得到完全一致的Hierarchy。Executor名称、Mesh/NHR类型、
engine、协议、Endpoint和HostDPU不得参与Hierarchy计算。

`AlgTopoRequest`是显式的静态声明，不是运行期探测结果：它只包含算法层数、维度起点和每层并行范围数，
不包含任何协议或链路信息。

## 2. 背景与问题

### 2.1 物理拓扑与算法拓扑

物理拓扑描述可用通信范围与能力，算法拓扑描述Executor执行多少层以及当前rank在每层属于哪个子通信域。两者层数
不一一对应。

例如：

```text
物理范围：[8,8,8,8] -> [32]Device -> [32]Host
二级算法：8 x 4
```

两个全覆盖来源只代表不同建链能力，不能产生第三个算法维度。

### 2.2 当前问题

1. `TopoMatchMultilevel`根据物理层数动态返回两层或三层；
2. UBX、PCIe-Mix、Squeeze、Concurrent等特殊Matcher重复解释Hierarchy；
3. Mesh和NHR可能获得不同rank分组；
4. HostDPU检测会反向改变Hierarchy层数（`TopoMatchMultilevel`在HostDPU且物理三层时降为二层）；
5. `infos`第二维语义不统一：`TopoMatchConcurrent`放两份相同rankList，`TopoMatchUBX`放mesh与clos两个
   不同范围，其他Matcher只放一个；
6. 部分消费者错误地用`infos[level].size()`表示层大小（例如`ins_v2_scatter_sequence_executor.cc`的
   `CalcRes`），另一些用`infos[level][0].size()`；
7. TopoMatch直接查询RankGraph并按Link过滤rank；
8. Executor根据物理标志重新解释`infos`第二维。

### 2.3 第二维的真实语义

第5点必须在重构中被保留而不是消除。现网`infos[level]`的多个元素表达的是**同一算法层上并行的多个范围**：

- `TopoMatchConcurrent`：两个相同的全域rankList，供两个并发执行平面使用；
- `TopoMatchUBX`：`infos[0][0]`为Mesh1D范围、`infos[0][1]`为CLOS范围，被Concurrent/OmniPipe类Executor
  当作两个不同的子通信域，分别驱动Mesh模板与NHR MultiJetty模板。

因此本次重构统一的是**语义**（第二维只表示并行范围，且必须显式声明），而不是把它压成1。

## 3. 目标与非目标

### 3.1 目标

1. TopoMatch只读取`TopoInfoWithNetLayerDetails::physicalLevels`的`localRanks`、`partitionGcd`和
   `partitionUniform`（后者仅用于准入判定）；
2. TopoMatch收敛为Level1、Level2、Level3三个独立主流程；
3. Executor显式声明`AlgTopoRequest`（层数、维度起点、每层并行范围）；
4. 相同分区视图和相同`AlgTopoRequest`产生相同Hierarchy；
5. 二级算法不因物理范围超过两层而增加算法层；
6. 三级算法遇到退化维度时返回不支持；
7. TopoInstance与NetInstance同范围合并后只贡献一个算法范围；
8. 规则非对称NetInstance使用`partitionGcd`得到全rank一致维度；
9. 保留现有`AlgHierarchyInfoForAllLevel`名称、内容和序列化；
10. 并行范围由`AlgTopoRequest`显式声明，不再由Matcher类型隐式决定；
11. Mesh、NHR、PCIe-Mix不再对应独立分组语义。

### 3.2 非目标

1. 解析或标准化RankGraph；
2. 查询Link或选择Channel、协议、Endpoint；
3. 修改Mesh、NHR等Template的通信步序；
4. 调整算法候选、CostModel或排序；
5. 删除或合并Selector可见算法名；
6. 修改`AlgHierarchyInfoForAllLevel`数据结构；
7. 迁移依赖同层互不包含范围的场景（2D Mesh的x/y环）：这类拓扑在Topo解析阶段就会降级为空
   `physicalLevels`，继续由旧Matcher承载；
8. 首期支持缺少稳定逻辑坐标的任意非连续Instance布局。

## 4. 输入与输出契约

### 4.1 接口

```cpp
enum class AlgLevelNum : uint32_t {
    INVALID = 0,
    LEVEL_1 = 1,
    LEVEL_2 = 2,
    LEVEL_3 = 3,
};

// 同一算法层上并行范围的来源
enum class AlgPlaneSource : uint32_t {
    PRIMARY = 0,          // 与该层主rankList完全相同（并发副本）
    NEXT_PHYSICAL_RANGE,  // 该层主范围在physicalLevels中的下一个范围
};

struct AlgLevelSpec {
    // 空或{PRIMARY}表示单平面；planes[0]必须为PRIMARY
    std::vector<AlgPlaneSource> planes;
};

struct AlgTopoRequest {
    AlgLevelNum algLevel = AlgLevelNum::INVALID;
    // Level0维度所用的PhysicalLevel下标，精确指定，不向上扫描
    uint32_t baseLevelIdx = 0;
    // 空表示所有算法层均为单平面
    std::vector<AlgLevelSpec> levels;
};

HcclResult MatchTopo(
    const TopoInfoWithNetLayerDetails &topoInfo,
    const AlgTopoRequest &request,
    AlgHierarchyInfoForAllLevel &hierarchy);
```

TopoMatch只能读取：

- `topoInfo.userRank`；
- `topoInfo.userRankSize`；
- `topoInfo.physicalLevels[].localRanks`；
- `topoInfo.physicalLevels[].partitionGcd`；
- `topoInfo.physicalLevels[].partitionUniform`（仅用于Level3的非对称准入判定，不参与任何成员计算）。

TopoMatch不接收`HcclComm`、Executor对象、算法名或engine类型，也不读取PhysicalLevel的`sources`
（含Endpoint、协议和TopoType）。

#### `baseLevelIdx`的语义与前提

`baseLevelIdx`是保留Squeeze类算法语义的唯一手段：不引入它，Squeeze2D迁移后必然与Sequence2输出相同的
Hierarchy，两个算法名会塌缩成同一个实现。

它是**精确下标，不向上扫描**：若`physicalLevels[baseLevelIdx]`无法形成非退化维度，直接返回
`HCCL_E_NOT_SUPPORT`，不会自动改用更高的Level。这一点必须严格执行，理由见下。

**使用前提（必须写进每个声明`baseLevelIdx != 0`的Executor的注释）**：`physicalLevels`的下标是
**拓扑相关**的稠密序数，不是固定的层编号。同一个下标在不同拓扑上指向的物理范围可能完全不同。例如
MESH_1D_CLOS拓扑在netLayer 0上同时存在一个1DMESH TopoInstance和一个覆盖整个NetInstance的CLOS
TopoInstance，标准化后`physicalLevels[0]`是mesh范围、`physicalLevels[1]`是**layer 0的CLOS范围**，
而不是layer 1。任何新增的嵌套范围（2-die SIO对、PCIe-SW子组）都会让全部下标整体后移一位。

因此：

1. 声明`baseLevelIdx != 0`的Executor，其Selector条件必须限定到已验证的拓扑形态。现有Squeeze2D正是
   如此——它要求layer0/layer1/layer2的`topoInstNum`都为1，实际只跑在纯Mesh1D上，该形态下
   `physicalLevels[1]`确实对应layer 1；
2. "精确下标不扫描"是这套机制的安全阀：拓扑结构变化导致下标含义偏移时，结果是**明确的
   `HCCL_E_NOT_SUPPORT`**，而不是静默换用另一个物理范围。允许向上扫描会把偏移变成静默的分组变化，
   那是最难定位的一类故障；
3. 序数下标是首期的折中。用属性锚定（如"Source的netLayer等于k"）更稳，但需要TopoMatch读取
   `sources[].ref.netLayer`，与4.4的读取契约冲突。列入14.2开放问题。

### 4.2 沿用现有Hierarchy结构

现有结构保持不变：

```cpp
struct AlgHierarchyInfoForAllLevel {
    std::vector<std::vector<std::vector<u32>>> infos;
};
```

新TopoMatch统一解释为：

```text
infos[algorithmLevel][planeIndex][rankIndex]
```

- `infos[level].size()`等于该层声明的并行范围数，默认为1；
- `infos[level][0]`是该层的**主rankList**，唯一参与算法维度乘积校验；
- `infos[level][k] (k > 0)`是并行范围，供同层多个Template或多个执行平面使用，不产生额外算法维度。

第二维不再用于表达TopoInstance身份或物理平面细节；这些信息只存在于`physicalLevels`和建链阶段。

### 4.3 并行范围规则

`AlgLevelSpec::planes[k]`的解析：

| 取值 | 生成方式 | 约束 |
|------|----------|------|
| `PRIMARY` | 复制`infos[level][0]` | 无 |
| `NEXT_PHYSICAL_RANGE` | 取该层主范围在`physicalLevels`中下一个范围的`localRanks` | 该层必须有对应的物理主范围；不存在下一个范围时返回不支持 |

`NEXT_PHYSICAL_RANGE`只允许用于存在物理主范围的算法层：Level2/Level3的第0层，以及Level3的第1层。
最外层维度由取模关系构造，没有对应的单一物理范围，只能声明`PRIMARY`。

并行范围必须满足：

1. 每个平面非空且包含myRank；
2. 每个平面与`infos[level][0]`相等或严格包含它；
3. 平面之间不得互相重叠但互不包含；
4. 平面不参与`d0 * d1 [* d2] == rankSize`校验。

### 4.4 访问与校验

新增非持有型View，不复制或缓存Hierarchy：

```cpp
class AlgHierarchyView {
public:
    explicit AlgHierarchyView(const AlgHierarchyInfoForAllLevel &hierarchy);
    uint32_t LevelNum() const;
    uint32_t PlaneNum(uint32_t level) const;              // 越界返回0
    uint32_t LevelRankSize(uint32_t level) const;         // 即Plane(level, 0).size()，越界返回0
    const std::vector<u32> &Plane(uint32_t level, uint32_t plane = 0) const; // 越界返回静态空vector
private:
    const AlgHierarchyInfoForAllLevel &hierarchy_;
};
```

View是只读访问器，越界访问返回空值而不是错误码——引用返回类型无法承载`HcclResult`。合法性判定集中在
唯一的校验入口：

```cpp
HcclResult ValidateAlgHierarchy(
    const AlgHierarchyInfoForAllLevel &hierarchy,
    const AlgTopoRequest &request,
    uint32_t rankSize,
    uint32_t myRank);
```

调用方在使用View之前必须已通过`ValidateAlgHierarchy`，或自行判断`LevelRankSize() != 0`。

### 4.5 输出不变量

1. `infos.size() == static_cast<uint32_t>(request.algLevel)`；
2. `infos[level].size()`等于该层声明的平面数（未声明时为1）；
3. 每个`infos[level][plane]`非空且包含myRank，且只包含一次；
4. rank ID无重复、无越界；
5. Level2满足`|infos[0][0]| * |infos[1][0]| == rankSize`；
6. Level3满足`|infos[0][0]| * |infos[1][0]| * |infos[2][0]| == rankSize`；
7. 不同算法层主rankList的交集仅为myRank；
8. 并行范围满足4.3节的四条约束；
9. 不允许的退化维度返回`HCCL_E_NOT_SUPPORT`；
10. 对全部rank执行后，同一主rankList内各rank看到的该层主rankList完全相同。

## 5. 公共匹配规则

### 5.1 PhysicalLevel选择

`physicalLevels`已由Topo解析阶段按范围从小到大稠密排列。TopoMatch使用`physicalLevels[baseLevelIdx]`
作为Level0维度的来源，不使用固定P0/P1/P2/P3编号，也不根据PhysicalLevel数量推导算法层数。

`baseLevelIdx`越界或该Level无法形成非退化维度时返回`HCCL_E_NOT_SUPPORT`，**不向上扫描**（理由见4.1）。

`physicalLevels`为空（Topo解析降级）时直接返回`HCCL_E_NOT_SUPPORT`。

### 5.2 维度证据

PhysicalLevel有两种维度证据：

```text
partitionGcd > 0 ：使用全局分区统计的GCD
partitionGcd == 0：使用当前rank规则嵌套范围的localRanks.size()
```

缺少全局分区统计不等于无效范围。仅含TopoInstance的规则局部范围仍可参与匹配，但必须通过下面的交叉校验。

#### `partitionGcd == 0`的交叉校验（强制）

`partitionGcd == 0`时维度来自`localRanks.size()`，这是一个**纯本地量**。若各rank所在TopoInstance的
大小不一致，每个rank会算出不同的`d0`，而TopoMatch是纯函数、运行期无法发现这种不一致——症状是
**挂死而不是报错**，且现场离根因很远。

因此从`partitionGcd == 0`的Level取维度`d0`时，必须同时满足：

1. `userRankSize % d0 == 0`；
2. 若`physicalLevels`中存在任一`partitionGcd > 0`且其`localRanks`包含该Level的`localRanks`的Level `L`，
   则要求`L.partitionGcd % d0 == 0`。

任一条不满足返回`HCCL_E_NOT_SUPPORT`。

第2条是核心：它用**上层的全局分区统计**给下层的局部量背书。举例，若当前rank看到的mesh是6个rank，
而包含它的NetLayer的全局分区粒度是8，则`8 % 6 != 0`，说明"6"不可能是一个全rank一致的维度，
在TopoMatch阶段就拒绝，而不是让它走到运行期挂死。

可满足性已核对：UBX的mesh-in-clos场景中，mesh（gcd 0，size 8）的上层是layer 0的CLOS范围
（gcd 16），`16 % 8 == 0`通过；8.4的局部`4 -> 8 -> 16`场景中全覆盖Level的gcd为16，`16 % 4 == 0`通过。

### 5.3 维度成员选择

所有算法维度统一由一个规则生成，避免对rank ID连续性做隐式假设：

```text
SelectDim(range, stride, dimSize, myRank):
    1. candidates = { r ∈ range.localRanks | r % stride == myRank % stride }，升序排列
    2. 校验 myRank ∈ candidates 且 candidates.size() % dimSize == 0
    3. idx = candidates中myRank的下标
    4. 返回 candidates[(idx / dimSize) * dimSize .. (idx / dimSize) * dimSize + dimSize)
```

规则说明：

- 成员始终取自实际的`localRanks`，而不是由`(myRank / d) * d + i`公式凭空构造；
- `stride == 1`且`dimSize == range.localRanks.size()`时退化为"直接使用该范围的全部成员"，这是最常见的
  Level0情形；
- 非对称场景下`dimSize < candidates.size()`，规则按当前rank所在位置切出对应子块，与现有
  `TopoMatchMultilevel`的GCD子组切分行为一致；
- 第2步校验失败返回`HCCL_E_NOT_SUPPORT`。

#### 首期约束：Level0必须是rank ID连续块

**上层维度用`stride`按取模构造正交补，只在Level0的成员是rank ID连续块（更一般地：各Level0分组按
`rank % d0`划分等价类）时成立。**这是首期的显式约束，不是实现瑕疵。

反例：`rankSize = 8`，Level0成员为`{0,2,4,6}`（交错布局，`d0 = 4`，`d1 = 2`）：

```text
G0 = {0, 2, 4, 6}
G1 = SelectDim(全域, stride = 4, dimSize = 2) = {r : r % 4 == 0} = {0, 4}
G0 ∩ G1 = {0, 4}     ← 违反4.5不变量7（不同算法层主rankList的交集仅为myRank）
```

正确的补集应该是`{0, 1}`（各分组内相同位置的成员），但当前rank只看得见自己那一组的成员，
**从原理上无法推断兄弟分组的布局**——这需要HCOMM提供全局成员查询，属于Topo解析设计§2.2的非目标。

因此：

1. 交错布局在首期返回`HCCL_E_NOT_SUPPORT`（由不变量7兜住，见11.2的错误码规定）；
2. 这**不是相对现有实现的回归**：`TopoMatchMultilevel::TopoForLayer1`用的是同一条取模规则
   （`rankId % layer0Size != myRank % layer0Size`则跳过），`TopoMatchUBX::TopoForLayer3`、
   `TopoMatchSqueeze2D::TopoForLayer1`同理。新实现只是把隐式假设写成了显式约束；
3. `SelectDim`本身对非连续`localRanks`是正确的（如Level0直接使用`{0,2,4,6}`全体），受限的是
   **用取模构造上层补集**这一步。因此Level1（单层，无补集）不受本约束影响。

### 5.4 逻辑顺序约束

首期使用通信域rank ID作为稳定逻辑位置。生成的rankList必须被对应的物理范围包含。校验失败返回不支持，
不得调用`GetLinks`过滤rank，也不得返回缩短的rankList。

这与现有实现有一处明确的行为差异：`TopoMatchMultilevel`等旧Matcher会用`GetLinks`剔除无链路的peer，
从而返回更短的rankList。新实现把"是否真的有链路"完全交给建链阶段判断，因此在部分连通拓扑上，
失败点从TopoMatch后移到LinkBindingResolver，错误码由`HCCL_E_NOT_SUPPORT`统一表达。该差异必须在
迁移期用例中显式覆盖（见13.4）。

任意非连续成员布局若无法由rank ID和现有全局统计得到稳定坐标，不纳入首期支持。

## 6. Level1匹配

### 6.1 适用算法

- Sole；
- Concurrent的拓扑输入；
- 其他单阶段全通信域算法。

### 6.2 规则

主rankList为整个通信域：

```text
infos[0][0] = {0, 1, ..., rankSize - 1}
```

**Level1不查找全覆盖PhysicalLevel，也不要求它存在。**主rankList由`userRankSize`直接构造，
这与现有`TopoMatch1D`（`topo_match_1d.cc`）和`TopoMatchConcurrent`（`topo_match_concurrent.cc`）
的行为完全一致——两者都是无条件构造`{0, ..., rankSize-1}`，不查询任何物理范围。

要求存在全覆盖Level会引入回归：只要没有任何一层NetInstance覆盖全域（各层`netInstNum`都大于1），
全部Sole/Concurrent算法会失败，而这些算法今天是能跑的。

Level1不检查物理层数、协议、Endpoint或全rank直连性，也不受5.3"Level0必须是连续块"约束（单层无补集）。

### 6.3 准入约束

现有`TopoMatch1D`与`TopoMatchConcurrent`除了构造rankList之外，还承担了**准入拦截**职责，
这部分职责在重构后必须保留，但**不放进`MatchTopo`**：

| 现有Matcher | 现有约束 | 位置 |
|-------------|----------|------|
| `TopoMatch1D` | 每个netLayer的`topoType`必须是`COMM_TOPO_CLOS`或`COMM_TOPO_CUSTOM`，否则`HCCL_E_PARA` | `topo_match_1d.cc` |
| `TopoMatchConcurrent` | 同上，额外要求`topoLevelNums <= 2` | `topo_match_concurrent.cc` |
| 两者 | `shouldGoOutPlace(deviceType)`、`userRankSize != 0` | 同上 |

处理原则——**准入判定与分组计算分离**：

1. 分组计算（`MatchTopo`）保持纯函数，不读`sources`、不查RankGraph，因此协议与TopoType不可能影响
   Hierarchy，4.4的读取契约和13.6的验收项1都不被破坏；
2. 准入判定移入**Selector**。Selector本就持有完整`topoInfo`，且按Topo解析设计§4.4允许读取
   `sources[].topoType`与`sources[].endpoints`；
3. **迁移前置动作**：迁移Sole/Concurrent之前，必须先确认对应Selector已经排除了非CLOS/CUSTOM拓扑，
   以及（对Concurrent）`topoLevelNums > 2`的情况。若未排除，先在Selector补齐再迁移。这是PR4的
   准入条件，不是可选项——漏掉会让Sole类算法被选到它跑不了的拓扑上，而新Level1不再拦截。

声明多平面时按4.3节生成。典型用法：

```text
planes = {PRIMARY, PRIMARY}
    -> infos[0] = {全域rankList, 全域rankList}
    -> 对应现有 TopoMatchConcurrent
```

TopoMatch按声明生成平面，Executor不需要自行复制rankList；未声明多平面时也不会产生第二份数据。

## 7. Level2匹配

### 7.1 维度选择

`B0 = physicalLevels[baseLevelIdx]`，精确选取，不向上扫描（见4.1、5.1）。

```text
B0.partitionGcd > 0 ：d0 = B0.partitionGcd
B0.partitionGcd == 0：d0 = B0.localRanks.size()，并按5.2做交叉校验
d1 = rankSize / d0
```

要求：

```text
baseLevelIdx < physicalLevels.size()
d0 > 1
d1 > 1
d0 * d1 == rankSize
```

任一条不满足返回`HCCL_E_NOT_SUPPORT`。

示例（`baseLevelIdx = 0`）：

| 最低范围统计 | 结果 |
|--------------|------|
| `[4,4,4,4,4,4,4,4]` | `4 x 8` |
| `[8,8,8,8]` | `8 x 4` |
| `[8,8,8,4]` | `4 x 7` |
| `[8,4]` | `4 x 3` |
| `[8,9]` | 不匹配 |

唯一物理范围已经覆盖通信域时，`d1`退化为1。首期返回不支持，不在TopoMatch中引入与物理拓扑无关的平方根
因子拆分策略。

### 7.2 rank分组

```text
allRanksLevel = physicalLevels中 localRanks.size() == rankSize 的范围

G0 = SelectDim(B0,             stride = 1,  dimSize = d0, myRank)
G1 = SelectDim(allRanksLevel,  stride = d0, dimSize = d1, myRank)
```

`physicalLevels`中不存在全覆盖范围时（当前rank看不到完整通信域），Level2与Level3均返回
`HCCL_E_NOT_SUPPORT`，不用`[0, rankSize)`凭空补齐。

输出：

```text
infos[0][0] = G0
infos[1][0] = G1
```

必须验证：

- `G0`被`B0.localRanks`包含；
- `G0`和`G1`均包含myRank；
- `G0 ∩ G1 == {myRank}`；
- `|G0| * |G1| == rankSize`。

高于`B0`的其他物理范围不增加算法层，只供并行范围声明和后续建链使用。

### 7.3 维度起点示例

同一物理视图`[8,8,8,8] -> [16,16] -> [32]`：

| 请求 | 结果 | 对应现有算法 |
|------|------|--------------|
| `{LEVEL_2, baseLevelIdx = 0}` | `8 x 4` | Sequence2 / Parallel |
| `{LEVEL_2, baseLevelIdx = 1}` | `16 x 2` | Squeeze2D |

两者物理视图完全相同，仅因显式声明不同而产生不同Hierarchy，核心不变量依然成立。

## 8. Level3匹配

### 8.0 非对称拒绝（与现有实现保持一致）

现有`TopoMatchMultilevel`对三层拓扑显式拒绝任何非对称场景：

```cpp
// topo_match_multilevel.cc
if (topoLevelNums >= COMM_LAYER_SIZE_3 && (!isSymmetric || !layer1Symmetric)) {
    return HCCL_E_NOT_SUPPORT;
}
```

其中`isSymmetric`/`layer1Symmetric`是layer0/layer1的`instSizeList`元素是否全等。

**重构后Level3保持同一约束**：`B0`与`B1`若有全局分区统计，则其统计必须全等。判据是Topo解析阶段
生成的`partitionUniform`：

```text
对 B0 和 B1 分别要求：
    partitionGcd == 0  -> 无全局统计，不做该判定（与旧实现一致：旧实现也只基于instSizeList）
    partitionGcd >  0  -> 要求 partitionUniform == true，否则 HCCL_E_NOT_SUPPORT
```

**必须用`partitionUniform`而不是`partitionGcd`**：`[8,8,8,8]`与`[8,24]`的GCD都是8，仅凭GCD无法区分
对称与非对称。缺了这一位，重构后的Level3会接受旧实现明确拒绝的拓扑，属于未经评审的行为扩大。

Level2不受本节约束——现有实现对两层拓扑是支持非对称的（走GCD子组切分路径）。

### 8.1 Level0维度

`B0 = physicalLevels[baseLevelIdx]`，`d0`按Level2规则计算（含5.2的交叉校验）。

### 8.2 中间范围

从`B0`之后选择第一个严格包含`B0.localRanks`、且能够形成非退化中间维度的PhysicalLevel `B1`：

```text
B1.partitionGcd > 0 ：
    要求 B1.partitionGcd % d0 == 0
    d1 = B1.partitionGcd / d0

B1.partitionGcd == 0：
    要求 B1.localRanks.size() % d0 == 0
    d1 = B1.localRanks.size() / d0

d2 = rankSize / (d0 * d1)
```

`partitionGcd`已经是该Level全部Source各Instance大小的最大公约数，因此"每个Instance大小可被`d0`整除"
等价于"`partitionGcd`可被`d0`整除"，无需再遍历原始统计列表。

要求：

```text
d0 > 1
d1 > 1
d2 > 1
d0 * d1 * d2 == rankSize
B0、B1 均通过8.0的非对称拒绝
```

`B1`的选择是从`B0`之后向上扫描第一个合规范围，这与`baseLevelIdx`的"精确不扫描"规则不冲突：
`baseLevelIdx`是**算法声明**的锚点，必须精确；`B1`是由`B0`和整除关系**推导**出来的，扫描的每一步都
带有可验证的约束（严格包含 + 整除 + 非退化 + 对称），不存在静默换层的风险。

不存在合规`B1`时返回不支持。重复全覆盖来源已经在Topo解析阶段合并，不能人为构造第三个算法维度。

### 8.3 rank分组

```text
G0 = SelectDim(B0,            stride = 1,       dimSize = d0, myRank)
G1 = SelectDim(B1,            stride = d0,      dimSize = d1, myRank)
G2 = SelectDim(allRanksLevel, stride = d0 * d1, dimSize = d2, myRank)
```

输出：

```text
infos[0][0] = G0
infos[1][0] = G1
infos[2][0] = G2
```

必须验证`G0`被`B0.localRanks`包含、`G1`被`B1.localRanks`包含，并满足统一输出不变量。

### 8.4 示例

```text
[8,8,8,8] -> [16,16] -> [32]  => 8 x 2 x 2
[8,8,8,8] -> [32]             => 不匹配（无中间范围）
[8,8,8,4] -> [16,12] -> [28]  => 非对称，按8.0拒绝（即使不看8.0，d1也退化为1）
[8,8,8,8] -> [16,16,32] 之类各Instance不等的中间层 => 非对称，按8.0拒绝
局部4 -> 8 -> 16               => 4 x 2 x 2（各层partitionGcd为0时不做对称判定）
```

局部`4 -> 8 -> 16`时：

```text
rank0: G0=[0,1,2,3], G1=[0,4], G2=[0,8]
rank4: G0=[4,5,6,7], G1=[0,4], G2=[4,12]
```

## 9. Mesh、NHR与特殊场景

### 9.1 Mesh与NHR

Mesh和NHR只描述Template通信步序，不改变rank分组。相同`physicalLevels`和`AlgTopoRequest`必须得到相同
Hierarchy。

具体邻居、通信轮数和Channel数量由Template决定，并交给建链需求处理。

### 9.2 特殊Matcher迁移

| 现有Matcher | 目标`AlgTopoRequest` |
|-------------|----------------------|
| `TopoMatch1D` | `{LEVEL_1}` |
| `TopoMatchUBX1d` | `{LEVEL_1}` |
| `TopoMatchConcurrent` | `{LEVEL_1, levels[0].planes = {PRIMARY, PRIMARY}}` |
| `TopoMatchUBX` | `{LEVEL_2, levels[0].planes = {PRIMARY, NEXT_PHYSICAL_RANGE}}` |
| `TopoMatchMultilevel`（Mesh1D分支） | 按Executor声明进入`{LEVEL_2}`或`{LEVEL_3}` |
| `TopoMatchMultilevel`（HostDPU分支） | 恒为`{LEVEL_2}`，见9.3 |
| `TopoMatchMultilevel`（mesh2d分支） | 不迁移，见3.2非目标7 |
| `TopoMatch3Level` | `{LEVEL_3}` |
| `TopoMatchPcieMix` | `{LEVEL_2}`，协议差异移入建链 |
| `TopoMatchSqueeze2D` | `{LEVEL_2, baseLevelIdx = 1}` |

`TopoMatchUBX`的映射需要专项验证：其现有`infos[1]`来自netLayer 3并按链路存在性过滤，新实现改由取模关系
构造。迁移前必须先补齐13.3的UBX用例并与旧输出做逐rank对比，确认差异可接受后再切换。

迁移期旧类先作为调用统一入口的薄适配器，不保留旧分组逻辑。mesh2d分支所在的通信域在Topo解析阶段
`physicalLevels`为空，薄适配器需保留回退到旧实现的分支。

### 9.3 HostDPU

现有`TopoMatchMultilevel`会在HostDPU场景反向改变算法层数（背景与问题2.2第4点）：

```cpp
// topo_match_multilevel.cc
bool needDowngrade = hostDPUOnly && topoInfo->topoLevelNums == COMM_LAYER_SIZE_3;
uint32_t commLayerSize = (topoInfo->topoLevelNums == COMM_LAYER_SIZE_3 && !needDowngrade)
                             ? COMM_LAYER_SIZE_3 : COMM_LAYER_SIZE_2;
```

即：HostDPU + 物理三层 → 产出**二级**Hierarchy。

**重构后的规则**：涉及HostDPU的多级算法一律声明`{LEVEL_2}`，即使`topoLevelNums`为3也生成二级
Hierarchy。这与现有行为一致，同时消除了2.2第4点描述的"运行期探测反向改写Hierarchy形状"——
层数变成算法的静态属性，而不是`CheckHostDPUOnly`的返回值。

两个直接后果，实施时必须落实：

1. **Selector必须先于Executor区分HostDPU**。既然层数是静态声明，HostDPU与非HostDPU就是两个不同的
   Executor声明，选择权归Selector。这正是10.3所要求的处理方式："如果某个场景确实需要不同的层数，
   正确做法是由Selector选择另一个Executor"；
2. **迁移期的薄适配器必须完整复刻`needDowngrade`**。薄适配器的唯一价值是与旧实现逐字段等价，
   若只按`topoLevelNums`推导层数、丢掉HostDPU降级，HostDPU三层拓扑的`infos.size()`会从2变成3，
   Executor的`CalcRes`、channel数量与Template构造全部错位——而"薄适配器等价"是整个迁移的安全网，
   这一条破了后续所有迁移PR的验证前提都不成立。

## 10. Executor与注册机制

### 10.1 保留现有入口签名

`InsCollAlgBase::CalcAlgHierarchyInfo`现有签名保留。新增`GetAlgTopoRequest()`并提供默认无效值，便于逐步迁移：

```cpp
class InsCollAlgBase {
public:
    virtual AlgTopoRequest GetAlgTopoRequest() const { return AlgTopoRequest{}; } // algLevel = INVALID

    virtual HcclResult CalcAlgHierarchyInfo(
        HcclComm comm,
        TopoInfoWithNetLayerDetails *topoInfo,
        AlgHierarchyInfoForAllLevel &hierarchy);
};
```

基类默认实现的分派规则：

1. `GetAlgTopoRequest().algLevel == INVALID`：走旧Matcher路径（未迁移的Executor）；
2. 否则若`topoInfo->physicalLevels`为空：走旧Matcher路径并打WARNING（Topo解析降级场景）；
3. 否则调用`MatchTopo`；
4. **迁移期**：`MatchTopo`返回`HCCL_E_NOT_SUPPORT`时，回退旧Matcher路径并打WARNING。

第4条是迁移期的必需项而不是保险措施。新规则在若干场景上比旧实现严格，`physicalLevels`非空不等于新
规则一定能匹配：

| 场景 | 旧`TopoMatchMultilevel` | 新Level2 |
|------|-------------------------|----------|
| layer0 `topoInstNum == 0` | `infos[0] = {{myRank}}`、`layer0Size = 1`，继续执行 | `d0 > 1`校验失败 |
| Level0成员交错布局 | 同样按取模构造，但不校验正交性 | 不变量7失败（见5.3） |
| 部分连通（旧实现用`GetLinks`剔除peer） | 返回变短的rankList | 完整rankList，失败点后移到建链 |

没有第4条，这些场景会从"能跑"直接变成"算子失败"，而每个迁移PR的爆炸半径就是"这类算法在某些拓扑上
不可用"。有了第4条，新规则匹配不上时行为与重构前完全一致，WARNING日志同时提供了统计降级面的手段。

第4条在最后一个清理PR（实施步骤11）中随旧Matcher一起删除，删除前必须先用降级日志统计确认现网无命中。

`HcclComm`仅为保持调用和override签名而保留；基类新实现不把它传入`MatchTopo`。

典型映射：

```text
Sole                    -> {LEVEL_1}
Concurrent（全域双平面） -> {LEVEL_1, planes = {PRIMARY, PRIMARY}}
Concurrent（mesh+clos）  -> {LEVEL_2, planes = {PRIMARY, NEXT_PHYSICAL_RANGE}}
Parallel / Sequence2    -> {LEVEL_2}
Squeeze2D               -> {LEVEL_2, baseLevelIdx = 1}
Sequence3 / OmniPipe    -> {LEVEL_3}
HostDPU 多级            -> {LEVEL_2}（无论物理层数，见9.3）
```

### 10.2 模板参数分步迁移

1. 旧TopoMatch先成为Level1/2/3薄适配器，Executor模板声明暂时不变；
2. Executor完成迁移后删除`AlgTopoMatch`模板参数；
3. 注册宏在第二步同步删除TopoMatch参数；
4. Selector可见算法名全部保留；
5. 不以注册数量下降作为本需求目标。

### 10.3 禁止重新解释Hierarchy

Executor不得根据以下字段改变`infos`层数或平面数：

- `level0Topo`；
- `level0PcieMix`；
- `topoLevelNums`；
- `level2Uboe`、`level2Ubg`；
- HostDPU判定结果。

层数和平面数只能来自`AlgTopoRequest`。如果某个场景确实需要不同的层数或平面数，正确做法是由Selector
选择另一个Executor，而不是让Executor在运行期改写Hierarchy形状。HostDPU是这条规则的第一个实际案例，
处理方式见9.3。

迁移后的Executor通过`AlgHierarchyView`读取`Plane(level, planeIndex)`。

## 11. 接口边界与失败语义

### 11.1 RankGraph边界

TopoMatch、Executor的Hierarchy计算和Hierarchy View不得调用任何`HcclRankGraphGet*`接口。

### 11.2 失败语义

| 失败场景 | 返回值 |
|----------|--------|
| `algLevel`非法、`planes[0] != PRIMARY`、在最外层声明`NEXT_PHYSICAL_RANGE`、rankSize为0 | `HCCL_E_PARA` |
| `physicalLevels`内容非法或myRank缺失 | `HCCL_E_INTERNAL` |
| `physicalLevels`为空 | `HCCL_E_NOT_SUPPORT` |
| `baseLevelIdx`越界或该Level退化 | `HCCL_E_NOT_SUPPORT` |
| 合法Topo无法构造目标算法层数或平面 | `HCCL_E_NOT_SUPPORT` |
| Level3遇到非对称分区（8.0） | `HCCL_E_NOT_SUPPORT` |
| `partitionGcd == 0`的维度未通过5.2交叉校验 | `HCCL_E_NOT_SUPPORT` |
| **分组不正交（不变量7失败）** | `HCCL_E_NOT_SUPPORT` |
| 分组重复、越界、层数或平面数与声明不符 | `HCCL_E_INTERNAL` |

错误码的划分标准是**责任归属**：

- `HCCL_E_PARA`：`AlgTopoRequest`的声明本身非法，Executor代码有问题；
- `HCCL_E_NOT_SUPPORT`：声明合法、拓扑合法，但二者组合不在首期支持范围内，属于运行环境问题；
- `HCCL_E_INTERNAL`：TopoMatch自身产出了自相矛盾的结果，是本模块的bug。

**不正交必须归入`NOT_SUPPORT`而不是`INTERNAL`**：5.3已经说明，交错布局的Level0是一种合法但首期不支持的
拓扑，取模构造的补集与Level0相交是这一约束的**预期表现**，不是代码缺陷。归入`INTERNAL`会同时造成两个
后果——误导排查方向，以及绕过10.1第4条的迁移期回退（回退只认`NOT_SUPPORT`）。

Hierarchy在临时对象中构建，全部校验通过后再赋值；失败时输出保持为空。

## 12. 实施步骤

1. 新增`AlgLevelNum`、`AlgPlaneSource`、`AlgTopoRequest`、`AlgHierarchyView`和统一校验函数；
2. 实现`SelectDim`与公共维度校验（含5.2交叉校验、5.3连续块约束）；
3. 补齐Sole/Concurrent的Selector准入条件（6.3），确认后再实现Level1并迁移Sole、Concurrent；
4. 实现Level2并迁移Parallel、Sequence2、Squeeze2D（`baseLevelIdx = 1`）、PcieMix、HostDPU多级（9.3）；
5. 实现Level3（含8.0非对称拒绝）并迁移Sequence3、OmniPipe；
6. 补齐UBX专项对比用例后迁移UBX的mesh+clos双平面；
7. 将特殊Matcher改为薄适配器，保留`physicalLevels`为空时的旧实现回退；
8. 修正`infos[level].size()`被误当作层大小的消费点；
9. 逐个删除Executor对旧Hierarchy形状的条件解释；
10. 删除已迁移路径中的`AlgTopoMatch`模板参数；
11. 最后删除旧Matcher类和构建项（mesh2d分支除外）。

## 13. 测试与验收

### 13.1 Level1

| 物理范围 / 请求 | 预期 |
|-----------------|------|
| `[32]`，单平面 | 单层全部rank，`infos[0].size() == 1` |
| `[8,8,8,8] -> [32]`，单平面 | 单层全部rank |
| Device全覆盖加Host全覆盖 | Hierarchy不变 |
| `planes = {PRIMARY, PRIMARY}` | `infos[0].size() == 2`且两份内容相同 |

### 13.2 Level2

| 物理范围 / 请求 | 预期 |
|-----------------|------|
| `[8,8,8,8] -> [32]` | `8 x 4` |
| `[8,8,8,4] -> [28]` | `4 x 7` |
| `[8,4] -> [12]` | `4 x 3` |
| `[8,9] -> [17]` | 不匹配 |
| 只有全覆盖`[32]` | 不匹配 |
| `[8,8,8,8] -> [16,16] -> [32]`，`baseLevelIdx = 0` | `8 x 4` |
| `[8,8,8,8] -> [16,16] -> [32]`，`baseLevelIdx = 1` | `16 x 2` |
| `baseLevelIdx`越界 | `HCCL_E_NOT_SUPPORT` |
| `baseLevelIdx`指向的Level退化 | `HCCL_E_NOT_SUPPORT`，且**不**自动改用更高Level（守护"精确不扫描"） |
| 交错布局的Level0（如`{0,2,4,6}`，rankSize=8） | `HCCL_E_NOT_SUPPORT`（不变量7失败，见5.3）；**不是**`HCCL_E_INTERNAL` |
| 非连续但块状的Level0（如`{0,1,2,3}`在`{0..3,8..11}`范围内） | 正常匹配，成员取自实际`localRanks` |
| `partitionGcd == 0`且上层统计不整除（mesh=6、上层gcd=8） | `HCCL_E_NOT_SUPPORT`（5.2交叉校验） |

### 13.3 Level3与并行范围

| 物理范围 / 请求 | 预期 |
|-----------------|------|
| `[8,8,8,8] -> [16,16] -> [32]` | `8 x 2 x 2` |
| 局部`4 -> 8 -> 16`（各层`partitionGcd == 0`） | `4 x 2 x 2`，不做对称判定 |
| `[8,8,8,8] -> [32]` | 不匹配（无中间范围） |
| `[8,8,8,4] -> [16,12] -> [28]` | 不匹配（8.0非对称拒绝） |
| B0对称但B1非对称（`partitionUniform == false`） | 不匹配；**GCD相同的对称拓扑必须匹配成功**，两条配对验证`partitionUniform`确实生效 |
| 重复Device/Host全覆盖来源 | Hierarchy不变 |
| HostDPU + 物理三层，声明`{LEVEL_2}` | 产出**二级**Hierarchy，与重构前`needDowngrade`路径逐字段一致（9.3） |
| UBX mesh+clos：`{LEVEL_2, planes={PRIMARY, NEXT_PHYSICAL_RANGE}}` | `infos[0][0]`为mesh范围、`infos[0][1]`为clos范围，且mesh ⊂ clos |
| `NEXT_PHYSICAL_RANGE`但无更高范围 | 不匹配 |
| 在最外层声明`NEXT_PHYSICAL_RANGE` | `HCCL_E_PARA` |

### 13.4 全rank不变量与迁移对比

1. 若B属于A的某层主rankList，则A和B看到的该层主rankList完全相同；
2. 每个rank在每层恰好属于一个主rankList；
3. 算法维度乘积等于rankSize；
4. 不同算法层主rankList的交集仅为当前rank；
5. Mesh和NHR得到相同Hierarchy；
6. 协议和Endpoint变化不改变Hierarchy；
7. RankGraph原始返回顺序变化不改变Hierarchy；
8. **部分连通对比用例**：构造旧Matcher会用`GetLinks`剔除peer的拓扑，确认新实现要么产出完整rankList
   并由建链阶段成功绑定，要么在TopoMatch阶段明确返回不支持，不出现静默变短的rankList。

### 13.5 兼容性测试

1. `AlgHierarchyInfoForAllLevel`定义和序列化格式不变；
2. 单平面声明下`infos[level].size()`恒为1；多平面声明下等于声明值；
3. `AlgHierarchyView`不复制数据，越界访问返回空值且不崩溃；
4. 旧Matcher薄适配器与统一入口输出一致；
5. `physicalLevels`为空时薄适配器回退到旧实现，行为与重构前一致；
6. Selector可见算法名在迁移前后保持可注册、可选择；
7. Squeeze2D与Sequence2在同一物理视图下输出不同Hierarchy（防止算法塌缩回归）。

### 13.6 验收标准

1. TopoMatch只读取`physicalLevels`的`localRanks`、`partitionGcd`和`partitionUniform`；
2. 只保留Level1、Level2、Level3三个主实现；
3. 新实现中不存在RankGraph调用；
4. 新实现直接输出原有`AlgHierarchyInfoForAllLevel`；
5. Executor显式声明`AlgTopoRequest`，物理层数不再改变算法层数与平面数；
6. Mesh/NHR和协议差异不改变Hierarchy；
7. 并行范围只来自显式声明，且满足4.3节约束；
8. 现有算法注册名保持可用，且Squeeze类算法语义未丢失；
9. 相关UT、ST以及Host/Device序列化测试通过。

## 14. 风险与开放问题

### 14.1 风险

- 旧Executor依赖`infos`第二维特殊语义：第二维语义被保留并显式化，逐个Executor改为按声明读取；
- UBX的`infos[1]`由链路过滤改为取模构造：先做逐rank对比再切换，必要时该场景延后迁移；
- Squeeze和PCIe-Mix旧分层影响算法性能：`baseLevelIdx`保证分组不变，仍需建立性能基线确认无回退；
- 取消`GetLinks`过滤导致失败点后移：通过13.4.8用例覆盖，并确保错误码可定位；
- 当前rank局部TopoInstance视图不一致：5.2的交叉校验能拦住能被全局统计证伪的部分，其余靠全rank测试；
  完全没有上层统计可比对时仍然只能依赖测试，这是`partitionGcd == 0`路径的固有限制；
- `baseLevelIdx`的下标含义随拓扑漂移：靠"精确不扫描 + Selector限定拓扑形态"把漂移暴露成
  `NOT_SUPPORT`而不是静默换层，见4.1；
- HostDPU层数由运行期探测变为静态声明：需要Selector先行区分，见9.3；
- 薄适配器丢失`needDowngrade`：见9.3第2点，这是薄适配器最容易写错的一处；
- 一次性删除模板参数导致修改面过大：先保留薄适配器，再分批清理。

### 14.2 开放问题

1. `rankSize == 1`时Level1、Level2和Level3的退化策略；
2. 任意非连续Instance布局是否需要HCOMM提供稳定逻辑坐标（5.3的连续块约束能否放开取决于此）；
3. 同层互不包含范围（2D Mesh）是否需要引入第三种`AlgPlaneSource`，还是长期由旧Matcher承载；
4. 旧Squeeze与PCIe-Mix是否需要独立性能门限；
5. `baseLevelIdx`能否改为属性锚定（如"Source的netLayer等于k"）。属性锚定不随拓扑漂移，但需要
   TopoMatch读取`sources[].ref.netLayer`，与4.4的读取契约冲突。若要采纳，需要先决定是把`netLayer`
   提升为PhysicalLevel的一级字段（不属于`sources`），还是放宽读取契约。

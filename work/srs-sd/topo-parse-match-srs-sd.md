# 1. Topo解析标准化与算法拓扑匹配（SRS）

- 文档状态：草案
- 更新日期：2026-08-17（Topo Parse按已落地实现修订：ranktable层级与topo层级合一、新增链路属性与整机形态）
- 适用仓库：`cann/hccl`
- 面向读者：测试
- 详细设计参考：[Topo解析标准化](./topo-parse-refactor-design.md)、[TopoMatch统一重构](./topo-match-refactor-design.md)

## 1.1 介绍

HCCL在选算法和建链之前，需要先回答两个问题：

1. **这个通信域的物理拓扑长什么样**——哪些卡在一台机器里、哪些在一个超节点里、彼此是Mesh直连还是走CLOS交换；
2. **某个算法要把这些卡分成几层、每层跟谁通信**——比如二级算法要先在机内8卡做一轮、再在机间4个节点做一轮。

现状是这两件事混在一起做的，且每种拓扑形态各有一个Matcher（Multilevel、UBX、PcieMix、Squeeze、Concurrent……），
带来四类问题：

| 问题 | 现象 |
|------|------|
| 层数不稳定 | 算法层数由物理层数决定，同一个算法在不同拓扑上可能跑出二级或三级 |
| 反向改写 | HostDPU这种运行期探测结果会反过来把三级Hierarchy降成二级 |
| 语义不统一 | `infos`第二维在不同Matcher里含义不同，消费侧各自解释，有的把它当层大小用 |
| 职责越界 | TopoMatch自己去查RankGraph、按链路存在性过滤rank，物理信息与算法分组耦合 |

本需求把这条链拆成**两个职责单一的阶段**：

- **Topo Parse（本期已实现）**：把HCOMM RankGraph里的物理信息提取成一份标准化、可序列化的
  "物理范围链" `physicalLevels`。**topo文件描述server内层级、ranktable描述server间层级，二者在这里合一**：
  同一物理层级只出现一次，既带全局分区又带链路属性（互联形态、Device/Host位置、协议集合、物理端口数）。
  它只做信息提取，不做任何算法决策。同时提取本卡整机形态 `deviceFormFactor` 供性能建模使用。
- **Topo Match（本期设计，分批实施）**：只读 `physicalLevels` + Executor的静态声明，产出算法分层
  `AlgHierarchyInfoForAllLevel`。收敛为Level1/Level2/Level3三个主流程，不再有按拓扑形态分叉的Matcher。

核心不变量（测试可直接作为判据）：

```text
Hierarchy(myRank) = F(physicalLevels, algTopoRequest, myRank)
```

即：物理视图相同、算法声明相同、rank相同 → 分层结果必须逐字节相同。算法名、Mesh/NHR类型、engine、
通信协议、Endpoint、HostDPU **一律不得**影响分层结果。

## 1.2 输入

**Topo Parse的输入**：HCOMM RankGraph，经现有 `HcclRankGraphGet*` 接口获取

| 输入项 | 含义 |
|--------|------|
| netLayer列表 | 当前rank参与的网络层级（机内/机间/超节点间……） |
| 每层的NetInstance分区 | 该层把整个通信域切成了几块、每块多大 |
| 每层当前rank所在块的rank列表 | 当前rank在这一层能直接通信的rank集合 |
| 每层的TopoInstance | 当前rank所在块内部的实际互联形态（1DMESH / CLOS） |
| Endpoint描述 | 当前rank在各TopoInstance上的通信端点（协议 + 地址 + 位置） |
| Endpoint端口数 | 每个通信接口的物理端口数（`ENDPOINT_ATTR_BW_COEFF`，即portGroupSize） |

另有一路与RankGraph无关的输入：**ACL的整机形态属性**（`ACL_DEV_ATTR_DEVICE_FORM_FACTOR`），
用于判断本卡是否POD机型（POD的交换机层为2:1收敛，性能建模需要据此修正）。

**Topo Match的输入**：

| 输入项 | 来源 |
|--------|------|
| `physicalLevels` | Topo Parse的输出（经序列化下发/缓存往返后仍然有效） |
| `AlgTopoRequest` | Executor的**静态声明**：算法层数、Level0取哪一层物理范围、每层要几个并行范围 |
| `userRank` / `userRankSize` | 现有Topo字段 |

`AlgTopoRequest` 是写死在Executor里的常量声明，不是运行期探测结果，这是"层数稳定"的根本保障。

## 1.3 处理

**Topo Parse要做的事（5件）**：

1. **收集**：遍历当前rank参与的每个netLayer，取本层NetInstance（给全局分区）与含当前rank的
   TopoInstance（给链路属性）；
2. **合一**：rank集合与NetInstance相同的TopoInstance与之合并成一个Level；更细的TopoInstance单独成级；
   该层没有TopoInstance时，NetInstance单独成级并置 `hasTopoInst = false`；
3. **快照与提炼**：把HCOMM返回的裸指针数据立即复制成HCCL自己管理的容器；从Endpoint提炼出该级的
   Device/Host位置与协议集合；按**iface**（不是按endpoint）统计物理端口数；
4. **规范化**：rank列表升序去重、分区大小与端口数降序、协议集合去重升序、Endpoint按稳定键排序——
   消除HCOMM哈希遍历序带来的跨rank/跨进程差异；
5. **定序**：按 `(块大小升序, view LOCAL在前, 互联紧密度)` 三键排序，形成一条从小到大的范围链；
6. **自校验**：第6章不变量逐条检查，任一条不过 → **整体降级**为空视图。

**Topo Match要做的事（4件）**：

1. **按声明选层数**：算法层数 = `AlgTopoRequest.algLevel`，取值仅 Level1 / Level2 / Level3；
2. **算维度**：从 `physicalLevels[baseLevelIdx]` 得到最内层维度 `d0`，其余维度按整除关系推导；
3. **切成员**：统一用 `SelectDim(range, stride, dimSize, myRank)` 规则，从真实rank列表里切出当前rank所属的组；
4. **校验**：维度乘积等于rankSize、各层rank组交集只有myRank、每组都含myRank且不重复；不满足返回 `HCCL_E_NOT_SUPPORT`。

两阶段共同的处理原则：

- **失败不致命**。Parse失败 → `physicalLevels` 为空，全部现有Topo字段照常输出，老路径完全不受影响；
  Match失败 → 迁移期回退旧Matcher并打WARNING，行为与重构前一致。
- **不查链路**。Match阶段不再调用 `GetLinks` 过滤rank，"有没有链路"完全交给建链阶段判断。

## 1.4 输出

**Topo Parse的输出**：`TopoInfoWithNetLayerDetails` 尾部新增成员

```cpp
DeviceFormFactor deviceFormFactor;               // 本卡整机形态：POD/A_K/A_X/PCIE_CARD/UNKNOWN
std::vector<PhysicalLevelInfo> physicalLevels;   // 按块大小从小到大排列
```

每个Level包含9项：

| 字段 | 含义 | 是否跨rank一致 |
|------|------|----------------|
| `localRanks` | 当前rank在这一级能看到的rank集合 | **否**，本地量 |
| `view` | `LOCAL`=只知道自己这一块；`GLOBAL`=知道这一层的完整划分 | 是 |
| `partitionSizes` | 这一层被切成的全部块大小，降序。`LOCAL`级恒为空 | **是**，逐字节相同 |
| `ref` | 这一级来自哪个netLayer / TopoInstance | 是 |
| `hasTopoInst` | 有无TopoInstance支撑。`false`时下面5项全部无意义 | 是（由机型契约保证） |
| `topoType` | 互联形态：`1DMESH` / `CLOS` | 是（同上） |
| `locType` | 链路落在 `DEVICE` 还是 `HOST` | 否 |
| `protocols` | 该级出现的协议**集合**，去重升序（如 `{ub_ctp, ub_mem}`） | 否 |
| `portNums` | 本卡在该级各条物理链路的端口数，降序（如 `{8}` 或 `{6,2}`） | **否**，同机不同die都可能不同 |

示例（2机×8卡，rank 0视角）：

```text
[0] {0..7}  GLOBAL {8,8}  1DMESH, device, {hccs}, portNums={1}   L0
[1] {0..15} GLOBAL {16}   CLOS,   device, {roce}, portNums={1}   L1
```

**每个物理层级只出现一次。** 早期方案对同一rank集合会同时产出一个 `LOCAL` 级和一个 `GLOBAL` 级
（上例为4级），现已合一。测试若沿用旧的级数期望值会全部失败，这是**预期的行为变更**。

**注意 `partitionSizes` 是全通信域的划分，不是"我在哪一块"**。`[16,4]` 这种非对称拓扑上，
4卡块里的rank看到的也是 `{16,4}`，这使它成为跨rank一致性校验唯一可用的锚点。

**两条派生结论由消费侧自己算，不在本结构里**：

- **是否需要使用host网卡** = 最高一级的 `locType == HOST`；
- **是否POD机型** = `IsPodForm()`，即 `deviceFormFactor == POD`。
  注意 `POD` 的枚举值就是 **0**，与"未取到"极易混淆，判定必须走 `IsPodForm()`。

**Topo Match的输出**：沿用现有结构，不改定义、不改序列化

```cpp
struct AlgHierarchyInfoForAllLevel {
    std::vector<std::vector<std::vector<u32>>> infos;   // infos[算法层][并行范围][rank]
};
```

三个Level的输出如下：

| | 算法层数 | 输出内容 | 典型算法 |
|---|---|---|---|
| **Level1** | 1 | `infos[0][0]` = 整个通信域 `{0..rankSize-1}` | Sole、Concurrent |
| **Level2** | 2 | `infos[0][0]` = 内层组（d0个rank）<br>`infos[1][0]` = 外层组（d1个rank）<br>满足 `d0 × d1 == rankSize` | Parallel、Sequence2、Squeeze2D、PcieMix、HostDPU多级 |
| **Level3** | 3 | `infos[0][0]` / `infos[1][0]` / `infos[2][0]`<br>满足 `d0 × d1 × d2 == rankSize` | Sequence3、OmniPipe |

Level2具体例子（`[8,8,8,8] -> [32]`，rank 0）：

```text
infos[0][0] = {0,1,2,3,4,5,6,7}    机内8卡
infos[1][0] = {0,8,16,24}          机间4节点
```

Level3具体例子（`[8,8,8,8] -> [16,16] -> [32]`，rank 0）：

```text
infos[0][0] = {0..7}       机内8卡
infos[1][0] = {0,8}        超节点内2机
infos[2][0] = {0,16}       超节点间2组
```

第二维（并行范围）**只在Executor显式声明时才有多个**，默认恒为1：

| 声明 | 输出 |
|------|------|
| 未声明 | `infos[level].size() == 1` |
| `{PRIMARY, PRIMARY}` | 两份**完全相同**的rankList（Concurrent双执行平面） |
| `{PRIMARY, NEXT_PHYSICAL_RANGE}` | `infos[0][0]`=mesh范围、`infos[0][1]`=包含它的clos范围（UBX） |

并行范围**不参与**维度乘积校验。

## 1.5 约束分析

| 支持的算子名称 | 全部走新流程的集合通信算子（AllReduce/AllGather/ReduceScatter/Broadcast/Reduce/Scatter/AlltoAll系列）。Topo解析与匹配是公共框架能力，不针对单个算子 |
| ------------------ | ---- |
| 支持的算法名称 | Level1：Sole、Concurrent；Level2：Parallel、Sequence2、Squeeze2D、PcieMix、HostDPU多级；Level3：Sequence3、OmniPipe。未迁移算法继续走旧Matcher |
| 支持的芯片类型 | **仅910_95**（`shouldGoOutPlace` 限定 `DEV_TYPE_950` / `DEV_TYPE_960`）。其余芯片在算子入口就走老流程，`CalcTopoShape` 不执行本特性 |
| 支持的展开模式 | AICPU、AIV、CCU 均适用（本特性在Host侧计算，结果随topoInfo序列化下发Device） |
| 支持的拓扑形态 | 规则嵌套的范围链：MESH_1D、MESH_1D_CLOS、CLOS、多层超节点、UBX/PC16（同层MESH+CLOS并存）、非对称NetInstance（如`[16,4]`）。**不支持**：同层互相重叠但互不包含的范围（2D Mesh的x/y环）、Level0成员交错布局（如`{0,2,4,6}`） |
| 支持的调用类型 | 单算子模式、图模式均支持 |
| 支持的数据类型 | 不涉及（拓扑阶段与数据类型无关） |
| 支持的数据量 | 不涉及 |
| 是否支持绕路 | 不涉及（本阶段不选链路、不选协议） |
| 是否支持确定性计算 | 不涉及 |

补充约束：

1. **不改变任何现有Topo字段**的名称、顺序和取值，`physicalLevels` 与 `deviceFormFactor`
   只追加在结构体和序列化流的尾部；
2. **Level总数上限32**（`PHYSICAL_LEVEL_NUM_LIMIT`），反序列化超限时丢弃整个physicalLevels段；
3. 首期Level0必须是"按 `rank % d0` 划分等价类"的布局，否则Level2/Level3返回不支持；
4. Level3遇到非对称分区拒绝（与现有 `TopoMatchMultilevel` 行为一致），Level2支持非对称。

---

# 2. Topo解析标准化与算法拓扑匹配（SD）

## 2.1 功能描述

本特性由两个独立模块组成，接口边界是 `physicalLevels` 这一个数据结构。

### 模块一：PhysicalLevel解析标准化

| 子功能 | 说明 |
|--------|------|
| 逐层合一 `BuildLayerCandidates` | 每个netLayer：取NetInstance（分区）与含当前rank的TopoInstance（链路属性），按rank集合是否相同决定合并还是各自成级 |
| 链路属性提炼 | 从Endpoint快照得出 `locType` / `protocols`；按 **iface** 统计 `portNums` |
| 整机形态 `CalcDeviceFormFactor` | 查ACL得出 `deviceFormFactor`，取不到即 `UNKNOWN` |
| 标准化 `NormalizePhysicalLevels` | 三键排序，形成范围链 |
| 校验 `ValidatePhysicalLevels` | 第6章不变量自检，作为构造期自检使用 |
| 序列化 | 随 `TopoInfoWithNetLayerDetails::Serialize/DeSerialize` 一起走Host缓存往返与Host→Device下发 |

**合一规则**（三选一，测试可直接作为判据）：

| 情形 | 产出 | `view` | `partitionSizes` | `hasTopoInst` |
|------|------|--------|------------------|---------------|
| TopoInstance与本层NetInstance **rank集合相同** | 合并成一个Level | `GLOBAL` | 该层完整分区 | `true` |
| TopoInstance **比NetInstance更细** | 单独成级 | `LOCAL` | 空 | `true` |
| 该层**没有**TopoInstance | NetInstance单独成级 | `GLOBAL` | 该层完整分区 | `false` |

除netLayer 0外每层只有一个TopoInstance，合并是确定的。netLayer 0在特殊机型上可能挂多个
（典型是同范围的一个Mesh一个CLOS），此时各出一级、**两级都带该层的分区**，由 `topoType` 区分。

**范围链上相邻两级的rank集合仍允许相等**（netLayer 0的Mesh与CLOS就是），测试不要按"严格递增"判断。

**`portNums` 按iface统计而不是按endpoint**：一个iface跑N种协议就有N个Endpoint，但它们是同一条物理
链路、端口数只算一次。`{ub_ctp, ub_mem}` 的8口iface应得 `portNums == {8}` 而不是 `{8,8}`。

### 模块二：TopoMatch三级匹配

| 子功能 | 说明 |
|--------|------|
| Level1 | 单层全通信域。**不查找也不要求存在全覆盖PhysicalLevel**，直接由 `userRankSize` 构造 |
| Level2 | 两层。`d0` 来自 `physicalLevels[baseLevelIdx]`，`d1 = rankSize / d0` |
| Level3 | 三层。在 `B0` 之上向上扫描第一个严格包含它且能形成非退化维度的范围作为 `B1` |
| 并行范围 | 按 `AlgLevelSpec::planes` 声明生成，`PRIMARY`（复制主组）或 `NEXT_PHYSICAL_RANGE`（取下一个物理范围） |
| 统一校验 `ValidateAlgHierarchy` | 输出的10条不变量集中校验 |
| 只读访问 `AlgHierarchyView` | 非持有型访问器，越界返回空值而非崩溃 |

**`baseLevelIdx` 是精确下标，不向上扫描**。指向的Level无法形成非退化维度时直接返回 `HCCL_E_NOT_SUPPORT`。
这是刻意的安全阀：拓扑结构变化导致下标含义偏移时，结果是**明确的不支持**而不是静默换用另一个物理范围。
它同时是保住Squeeze2D语义的唯一手段——不引入它，Squeeze2D与Sequence2会输出相同Hierarchy，两个算法名塌缩。

## 2.2 流程描述

```text
                 ┌──────────────────── Host侧，每个通信域一次 ────────────────────┐
HCOMM RankGraph  │                                                              │
       │         │  CalcTopoShape                                               │
       └────────>│    ├─ 现有9个步骤（ExtractNetLayerDetails … CalcLevel2Ubg）   │
                 │    │   全部保持不变、顺序不变                                  │
                 │    ├─ CalcDeviceFormFactor           ← 新增（查ACL整机形态）    │
                 │    ├─ BuildLayerCandidates           ← 新增（逐层合一）        │
                 │    ├─ NormalizePhysicalLevels        ← 新增（三键排序）        │
                 │    └─ ValidatePhysicalLevels         ← 新增（不变量自检）       │
                 │              │                                               │
                 │              ├─ 通过 → physicalLevels 非空                    │
                 │              └─ 失败 → physicalLevels 为空（整体降级）          │
                 └──────────────┬───────────────────────────────────────────────┘
                                │ Serialize → EngineCtx缓存 / 下发Device
                                ▼
                 ┌──────────────────── 每个算子一次 ─────────────────────────────┐
                 │  InsCollAlgBase::CalcAlgHierarchyInfo                        │
                 │    ├─ GetAlgTopoRequest() == INVALID  → 旧Matcher            │
                 │    ├─ physicalLevels 为空             → 旧Matcher + WARNING   │
                 │    ├─ MatchTopo(topoInfo, request, hierarchy)                │
                 │    │     ├─ Level1 / Level2 / Level3                         │
                 │    │     └─ ValidateAlgHierarchy                             │
                 │    └─ 返回 NOT_SUPPORT → 旧Matcher + WARNING（迁移期兜底）      │
                 └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    AlgHierarchyInfoForAllLevel → Executor → Template → 建链
```

**三条降级路径必须全部覆盖到**（第2、3条是迁移期的安全网，是灰度期最可能命中的分支）：

| 分支 | 触发条件 | 表现 |
|------|----------|------|
| Parse整体降级 | RankGraph查询失败、拓扑不成范围链、不变量不过 | `physicalLevels` 为空，算子照常执行（走旧Matcher） |
| Parse局部降级 | Endpoint取不到 | 该级 `endpoints`/`locType`/`protocols`/`portNums` 为空，`hasTopoInst` 仍为 `true` |
| Parse局部降级 | 端口数任一条查不到 | 该级 `portNums` **整体**为空（全有或全无，不留残缺数组） |
| Parse局部降级 | 该层无TopoInstance | 该层出一个 `hasTopoInst == false` 的级，仍带分区。**这是正常形态不是错误** |
| Parse局部降级 | ACL查不到整机形态 | `deviceFormFactor == UNKNOWN`，`physicalLevels` 不受影响 |
| Match迁移期回退 | `MatchTopo` 返回 `HCCL_E_NOT_SUPPORT` | 回退旧Matcher，打WARNING，结果与重构前一致 |

## 2.3 数据描述

### PhysicalLevelInfo

```cpp
enum class PhysicalLevelView : uint32_t { LOCAL = 0, GLOBAL = 1 };

struct PhysicalSourceRef { uint32_t netLayer; uint32_t topoInstId; };

struct PhysicalLevelInfo {
    std::vector<uint32_t> localRanks;      // 升序去重
    PhysicalLevelView     view;
    std::vector<uint32_t> partitionSizes;  // GLOBAL：该层全部块大小，降序；LOCAL：空
    PhysicalSourceRef     ref;

    bool                       hasTopoInst;  // false 时以下5项全部无意义
    CommTopo                   topoType;     // 1DMESH / CLOS
    EndpointLocType            locType;      // DEVICE / HOST
    std::vector<CommProtocol>  protocols;    // 去重升序
    std::vector<uint32_t>      portNums;     // 按iface去重，降序
    std::vector<EndpointDesc>  endpoints;    // 按稳定键排序
};

// 取值与 acl_rt.h 的 ACL_DEVICE_FORM_FACTOR_* 逐一对应
enum class DeviceFormFactor : uint32_t {
    POD = 0, A_K = 1, A_X = 2, PCIE_CARD = 3, UNKNOWN = INVALID_UINT,
};
```

**没有 `PhysicalSourceType` / `PhysicalSourceInfo`。** 合一之后一个Level可能同时对应一个NetInstance
和一个TopoInstance，"来源类别"这个判别式不再成立；`ref` 同时持有两者身份，链路属性平铺在Level上。
早期版本的 `source.xxx` 访问路径全部变成 `level.xxx`。

**`view` 的含义也随之变化**：从"来自哪一类接口"变成"知不知道全局分区"，与 `partitionSizes.empty()`
严格等价。

**不存储任何派生量**。早期版本曾存 `partitionGcd` 和 `partitionUniform`，已删除，原因两条：
职责上非对称的判断归TopoMatch，解析侧只做信息提取；一致性上存储的派生值可能与来源不一致且本地无法察觉。
消费侧从 `partitionSizes` 一行即可重算（`uniform` = 全部元素相等，`gcd` = 一次欧几里得归约）。

只存派生量也不够：`[16,4]` 与 `[12,8]` 的GCD都是4、都非全等、rank数都是20，仅凭派生字段无法区分。

同理**不存 `bool isPod`**（存 `deviceFormFactor` 原始形态，判定走 `IsPodForm()`），也**不存
"是否需要host网卡"**（由最高一级的 `locType` 推导）。

**`DeviceFormFactor::UNKNOWN` 必须是 `INVALID_UINT` 而不是 0**：`ACL_DEVICE_FORM_FACTOR_POD` 就是 0，
用 0 当"未知"会让每一次取值失败都被读成"这是POD机型"，而POD恰好是消费侧要走2:1收敛特殊分支的那个值
——**误判方向正好是最坏的**。

### AlgTopoRequest

```cpp
enum class AlgLevelNum   : uint32_t { INVALID = 0, LEVEL_1, LEVEL_2, LEVEL_3 };
enum class AlgPlaneSource: uint32_t { PRIMARY = 0, NEXT_PHYSICAL_RANGE };

struct AlgTopoRequest {
    AlgLevelNum algLevel = AlgLevelNum::INVALID;
    uint32_t    baseLevelIdx = 0;            // Level0取哪个物理范围，精确下标
    std::vector<AlgLevelSpec> levels;        // 每层的并行范围声明，空表示全部单平面
};
```

### 序列化

- 现有全部字段的编码顺序和方式**完全不变**，新字段只追加在尾部；
- 新增成员 `u32 physicalLevelNum`（与现有 `topoInstDetailsOfLayerSize` 同一约定），
  超过 `PHYSICAL_LEVEL_NUM_LIMIT`(32) 时丢弃整段；
- 每个Level按 `localRanks / view / partitionSizes / ref.netLayer / ref.topoInstId / hasTopoInst /
  topoType / locType / protocols / portNums / endpoints` **共11项**逐字段编码，不做结构体裸拷贝；
- `deviceFormFactor` 编码在**最尾部**（在 `physicalLevels` 之后）。因此上面那条"超限即丢弃整段"的
  提前返回会连带跳过它、使其停在 `UNKNOWN`——纯位置流的必然结果，不是bug；
- 不引入magic、版本号、能力协商——Host与Device二进制成套发布，不存在版本错配场景；
- `AlgResourceCtxSerializable` 无需改动，`topoInfoSeqSize` 随之变化，尾部定位逻辑仍成立。

## 2.4 依赖性描述

| 依赖 | 说明 | 风险 |
|------|------|------|
| HCOMM RankGraph 接口 | `GetLayers` / `GetRanksByLayer` / `GetInstSizeListByLayer` / `GetTopoInstsByLayer` / `GetRanksByTopoInst` / `GetTopoType` / `GetEndpointNum` / `GetEndpointDesc` / `GetEndpointInfo` | 接口必定存在；返回失败按第9章降级处理 |
| ACL 设备号转换 | `aclrtGetDevice`(userDevId) → `aclrtGetLogicDevIdByUserDevId`(logicDevId) | 三套设备号（phy/user/logic）不可混用；`aclrtGetDeviceInfo`要的是logicDevId。另注意 `aclrtGetLogicDevIdByPhyDevId` 名不副实、实际返回userDevId（HCOMM `hal.c` 已注明） |
| ACL `aclrtGetDeviceInfo` | `ACL_DEV_ATTR_DEVICE_FORM_FACTOR`(409) | **HCOMM全仓无该属性的使用者**，其形态判断一律走 `ACL_DEV_ATTR_MAINBOARD_ID`(407) 取 `bit[7:5]`。二者读同一段硬件位域、取值逐一相等，但409在目标驱动版本上是否打通需上板确认（看 `[Topo][CalcDeviceFormFactor]` 的INFO日志） |
| `CalcTopoShape` 现有9个步骤 | 新增步骤追加在其**末尾**，复用已提取的 `netLayerDetails` / `topoInstDetailsOfLayer` | 现有字段提取逻辑零改动 |
| `TopoInfoWithNetLayerDetails` 序列化 | Host缓存往返（EngineCtx）+ Host→Device下发 | 两条路径都必须验证physicalLevels往返一致 |
| Executor `GetAlgTopoRequest()` | 基类提供默认 `INVALID`，未迁移Executor自动走旧路径 | 分批迁移，每批独立验证 |
| Selector 准入条件 | Level1不再拦截非CLOS/CUSTOM拓扑，该职责移到Selector | **迁移前置动作**：迁移Sole/Concurrent前必须先确认Selector已排除，否则算法会被选到跑不了的拓扑上 |

**HCOMM返回顺序不可依赖**：`GetInstSizeListByLayer` 与 `GetEndpointDesc` 内部都是 `unordered_map` 遍历，
输出是哈希序。规范化排序（分区降序、Endpoint按稳定键）就是为了消除这一点。测试如需构造用例，
可通过打乱输入顺序验证输出不变。

## 2.5 接口描述

### 接口一：PhysicalLevel构建（逐层合一）

| 函数原型 | `HcclResult BuildLayerCandidates(HcclComm comm, const TopoInfoWithNetLayerDetails *topoInfo, u32 layer, std::vector<PhysicalLevelInfo> &candidates)` |
| ---------- | ---- |
| 函数功能 | 取该netLayer的NetInstance（分区）与含当前rank的TopoInstance（链路属性），按rank集合是否相同决定合并还是各自成级；提炼 `locType`/`protocols`，按iface统计 `portNums` |
| 输入说明 | `comm`：通信域句柄；`topoInfo`：已完成现有9步提取的Topo对象；`layer`：netLayer编号，必须来自 `netLayerDetails.netLayers` |
| 输出说明 | `candidates`：追加该层产出的Level（1个或多个），未排序，仅在 `CalcTopoShape` 期间存在 |
| 返回值说明 | `HCCL_SUCCESS`：构建成功；其他：整体降级，调用方清空 `physicalLevels` 后仍返回 `HCCL_SUCCESS` |

### 接口一b：整机形态

| 函数原型 | `HcclResult CalcDeviceFormFactor(TopoInfoWithNetLayerDetails *topoInfo)` |
| ---------- | ---- |
| 函数功能 | 查ACL得出本卡整机形态并写入 `topoInfo->deviceFormFactor`。不依赖 `HcclComm`，只查本设备。<br>内部链路为 `aclrtGetDevice`(得userDevId) → `aclrtGetLogicDevIdByUserDevId`(转logicDevId) → `aclrtGetDeviceInfo`。**中间那步不能省**：`aclrtGetDeviceInfo` 要的是 logicDevId，而默认部署下 userDevId 与之恰好相等，漏转在大多数环境上测不出来，只在配了 `ASCEND_RT_VISIBLE_DEVICES` 或容器挂载部分设备时静默读到另一张卡 |
| 输入说明 | `topoInfo`：待填充的Topo对象 |
| 输出说明 | `deviceFormFactor`：`POD`/`A_K`/`A_X`/`PCIE_CARD`；未识别或查询失败时为 `UNKNOWN` |
| 返回值说明 | **恒为 `HCCL_SUCCESS`**。该字段是纯附加信息，取不到不应让通信域起不来 |

### 接口二：标准化

| 函数原型 | `HcclResult NormalizePhysicalLevels(PhysicalLevelCandidates &candidates, u32 userRank, std::vector<PhysicalLevelInfo> &levels)` |
| ---------- | ---- |
| 函数功能 | 对候选做rank集合规范化，按 `(块大小, view, topoType)` 三键排序，形成范围链 |
| 输入说明 | `candidates`：候选数组；`userRank`：当前rank |
| 输出说明 | `levels`：按块大小从小到大排列的PhysicalLevel |
| 返回值说明 | `HCCL_SUCCESS`：成功；其他：整体降级，`levels` 保持为空 |

### 接口三：校验

| 函数原型 | `HcclResult ValidatePhysicalLevels(const std::vector<PhysicalLevelInfo> &levels, u32 userRank, u32 userRankSize)` |
| ---------- | ---- |
| 函数功能 | 逐条检查设计文档第6章的不变量，作为构造期自检 |
| 输入说明 | `levels`：标准化结果；`userRank` / `userRankSize`：当前rank与通信域规模 |
| 输出说明 | 无（纯校验） |
| 返回值说明 | `HCCL_SUCCESS`：全部通过；`HCCL_E_NOT_SUPPORT`：存在不满足项，调用方整体降级 |

### 接口四：算法拓扑匹配

| 函数原型 | `HcclResult MatchTopo(const TopoInfoWithNetLayerDetails &topoInfo, const AlgTopoRequest &request, AlgHierarchyInfoForAllLevel &hierarchy)` |
| ---------- | ---- |
| 函数功能 | 按算法声明的层数与维度起点，从物理范围链切出当前rank在每个算法层的rank分组 |
| 输入说明 | `topoInfo`：只读 `userRank` / `userRankSize` / `physicalLevels[].localRanks` / `view` / `partitionSizes`；`request`：Executor静态声明 |
| 输出说明 | `hierarchy.infos[算法层][并行范围][rank]`，全部校验通过后才赋值，失败时保持为空 |
| 返回值说明 | `HCCL_SUCCESS`；`HCCL_E_PARA`（声明本身非法，Executor代码问题）；`HCCL_E_NOT_SUPPORT`（声明与拓扑组合不在首期范围）；`HCCL_E_INTERNAL`（自相矛盾的输出，本模块bug） |

`MatchTopo` **不接收** `HcclComm`、Executor对象、算法名、engine类型，也不读取 `ref` 与任何链路属性。
因此协议、Endpoint、TopoType、端口数、Device/Host位置和整机形态在原理上都不可能影响分层结果
——这是可以静态检查的验收项。

错误码的划分标准是**责任归属**，测试定位问题时可直接据此分流：

| 错误码 | 责任方 | 典型场景 |
|--------|--------|----------|
| `HCCL_E_PARA` | Executor声明有问题 | `algLevel` 非法、`planes[0] != PRIMARY`、在最外层声明 `NEXT_PHYSICAL_RANGE` |
| `HCCL_E_NOT_SUPPORT` | 运行环境不在首期支持范围 | `physicalLevels` 为空、`baseLevelIdx` 越界或退化、维度退化、Level3非对称、分组不正交 |
| `HCCL_E_INTERNAL` | TopoMatch自身bug | `physicalLevels` 内容非法、分组重复越界、层数与声明不符 |

**分组不正交必须是 `NOT_SUPPORT` 而不是 `INTERNAL`**：交错布局的Level0是合法但首期不支持的拓扑，
归为 `INTERNAL` 会误导排查方向，并绕过迁移期回退（回退只认 `NOT_SUPPORT`）。

## 2.6 约束分析

同 1.5 节。此处补充实施层面的约束：

| 约束项 | 内容 |
|--------|------|
| 现有结构 | `AlgHierarchyInfoForAllLevel` 定义、内容、序列化格式**不变**；`TopoInfo` / `NetLayerDetails` / `TopoInstDetails` 名称与顺序不变 |
| **级数期望值变更** | 合一之后每个物理层级只出现一次：常规2机×8卡从4级降为**2级**、超节点32卡从6级降为**3级**、UBX单机16卡从3级降为**2级**。**全部按旧级数写的断言都要更新**，这是预期的行为变更而不是回归 |
| 算法名 | Selector可见的算法名全部保持可注册、可选择，迁移前后不增不减 |
| 分批迁移 | 每批迁移后旧Matcher先作为薄适配器保留，`physicalLevels` 为空时回退旧实现 |
| 行为差异（需专项验证） | ① 取消 `GetLinks` 过滤 → 部分连通拓扑上失败点从TopoMatch后移到建链；② UBX的 `infos[1]` 从链路过滤改为取模构造；③ HostDPU层数由运行期探测改为静态声明 |
| 不迁移项 | 2D Mesh（同层x/y环）：Parse阶段就降级为空 `physicalLevels`，继续由旧Matcher承载 |

## 2.7 DFX设计

Parse阶段的日志前缀统一为 `[PhysicalLevel][Build]` / `[PhysicalLevel][Normalize]` / `[PhysicalLevel][Validate]`，
Match阶段为 `[TopoMatch]`。测试定位时可直接按前缀过滤。

| 校验内容 | 级别 | 搜索内容 |
| -------- | ---- | -------- |
| RankGraph查询失败（GetRanksByLayer / GetTopoInstsByLayer / GetRanksByTopoInst） | WARNING | `[PhysicalLevel][Build] get ranks by layer` / `get topo insts of layer` |
| `GetTopoType` 失败或返回预期外类型（排序第3键失去依据） | WARNING | `[PhysicalLevel][Build] get topo type of inst` |
| Endpoint取不到（局部降级，该级endpoints为空） | WARNING | `[PhysicalLevel][Build] get endpoint` |
| 端口数查询失败或取到0/超限（该级portNums整体清空） | WARNING | `[PhysicalLevel][Build] get port num failed` / `implausible port num` |
| 同一级各endpoint的locType不一致（置RESERVED） | WARNING | `[PhysicalLevel][Build] mixed endpoint locType` |
| 该netLayer无TopoInstance（正常形态，非错误） | DEBUG | `[PhysicalLevel][Build] has no topo instance` |
| 该层无同范围TopoInstance，只出带分区的级 | DEBUG | `[PhysicalLevel][Build] has no same-range topo instance` |
| 整机形态取不到（停在UNKNOWN） | WARNING | `[Topo][CalcDeviceFormFactor] get device form factor failed` |
| 整机形态取到未识别值（停在UNKNOWN，**不得落POD**） | WARNING | `[Topo][CalcDeviceFormFactor] unknown device form factor` |
| 整机形态取到已知值（**上板验收看这条**） | INFO | `[Topo][CalcDeviceFormFactor] userDevId[..] logicDevId[..] formFactor[..] isPod[..]`<br>两个id不等即说明配了VISIBLE_DEVICES之类，此时更要确认形态取自logicDevId |
| userDevId→logicDevId 转换失败 | WARNING | `[Topo][CalcDeviceFormFactor] get logic dev id by user dev id` |
| 无有效候选 / 标准化失败 → 整体降级 | WARNING | `[PhysicalLevel][Normalize] no valid candidate` |
| 标准化成功，输出Level数 | INFO | `[PhysicalLevel][Normalize] got` |
| rank列表为空或非升序 | WARNING | `[PhysicalLevel][Validate] rank list is empty or not ascending` |
| 该级不含myRank | WARNING | `[PhysicalLevel][Validate] does not contain myRank` |
| LOCAL级带了分区列表 / GLOBAL级带了端口数 | WARNING | `[PhysicalLevel][Validate] is LOCAL but` / `is GLOBAL but` |
| `hasTopoInst==false` 却带了链路属性 | WARNING | `[PhysicalLevel][Validate] has no topo instance but carries link attributes` |
| portNums条数超过endpoint数 / 非降序 / 含0或超限 | WARNING | `[PhysicalLevel][Validate] portNum` |
| protocols未去重升序 | WARNING | `[PhysicalLevel][Validate] protocols is not sorted and unique` |
| `ref.netLayer` 无效 | WARNING | `[PhysicalLevel][Validate] has no valid netLayer` |
| `partitionSizes` 含0项 / 未降序 / 与rankSize不符 | WARNING | `[PhysicalLevel][Validate] partitionSizes` |
| 范围链断裂（相邻两级不满足包含关系） | WARNING | `[PhysicalLevel][Validate] does not contain level` |
| 身份 `(netLayer, topoInstId)` 重复 | WARNING | `[PhysicalLevel][Validate]` + `share the same source` |
| 反序列化时Level数超限（>32） | WARNING | `[TopoInfo][DeSerialize] implausible physicalLevelNum` |
| `physicalLevels` 为空导致走旧Matcher（降级面统计用） | WARNING | `[TopoMatch]` + `fallback` |
| `MatchTopo` 返回不支持导致回退（降级面统计用） | WARNING | `[TopoMatch]` + `not support` |

**最后两条是灰度期的核心观测点**：它们直接给出"新路径实际命中率"。旧Matcher的删除动作以这两条日志
在现网无命中为前提，不能只凭功能用例通过就删。

## 2.8 资料描述

不涉及对外接口变更，无需修改用户文档。

内部设计文档：

- `topo-parse-refactor-design.md` / `topo-parse-impl-plan.md`
- `topo-match-refactor-design.md` / `topo-match-impl-plan.md`
- `link-binding-refactor-design.md`（下游建链）
- `docs/design/device_form_factor.md`（整机形态两种取法的对比与切换条件）

## 2.9 性能&&质量

| 项 | 说明 |
|----|------|
| 计算开销 | Parse新增步骤在 `CalcTopoShape` 内执行，**每个通信域（每个tag）一次**，不在算子下发路径上。新增查询为：每层1次 `GetRanksByLayer`、每个TopoInstance 3次、每个iface 1次 `GetEndpointInfo`，外加1次ACL形态查询；层数≤4、实例与iface数均个位数。**不调用 `GetLinks`**（避免O(N²)） |
| Match开销 | 纯函数、无系统调用、无RankGraph查询，复杂度为Level数与rank数的线性量级 |
| 序列化体积 | 增量以 `EndpointDesc`（160字节/条）为主；`protocols`/`portNums`/`deviceFormFactor` 均为个位数字节。**合一后Level数减少**（超节点32卡从6级降为3级），总量较合并前下降。Level数上限32 |
| 性能基线 | Squeeze与PCIe-Mix旧分层影响算法性能，`baseLevelIdx` 保证分组不变，但仍需建立性能基线确认无回退 |
| 质量目标 | 全rank一致性：同一主rankList内各rank看到的该层主rankList完全相同；输入顺序无关：打乱RankGraph返回顺序不改变输出；Mesh与NHR得到相同Hierarchy |

**链路属性与整机形态是局部量，不承诺跨rank一致。** `portNums` 尤其如此——同一台机器上不同die的rank
链路就可能不同（HCOMM `TopoGetClosPort` 里die0是4个口、die1是2个口）。消费侧（cost model）若要用它们
驱动算法选择，一致性需自行处理；本结构只提供 `partitionSizes` 一个跨rank锚点。

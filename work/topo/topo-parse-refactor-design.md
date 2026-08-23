# Topo信息解析与标准化重构需求设计

- 文档状态：草案
- 更新日期：2026-08-17（按已落地实现修订：NetInstance与TopoInstance合一、新增链路属性与整机形态）
- 适用仓库：`cann/hccl`、`cann/hcomm`
- 上游输入：HCOMM RankGraph
- 下游设计：[TopoMatch统一重构](./topo-match-refactor-design.md)、[物理建链统一重构](./link-binding-refactor-design.md)

---

## 1. 概要

本文档只定义Topo信息解析与标准化：保留现有`CalcTopoShape`和Topo数据结构，在其基础上生成统一的
`physicalLevels`视图，供TopoMatch和物理建链使用。

```text
HCOMM RankGraph                       ACL
    |                                  |
    +-> CalcTopoShape                  |
        +-> ExtractNetLayerDetails（现有）
        +-> ExtractTopoDetails（现有）
        +-> CalcDeviceFormFactor  <----+   新增，查本卡整机形态（POD/A_K/A_X/PCIE_CARD）
        +-> BuildPhysicalLevels           新增
            +-> BuildLayerCandidates      逐netLayer把ranktable层级与topo层级合一
            +-> NormalizePhysicalLevels   三键排序 + 链校验
            +-> ValidatePhysicalLevels    不变量自检
    -> TopoInfoWithNetLayerDetails.physicalLevels + deviceFormFactor
```

**"合一"是本设计的核心。** topo文件描述server内层级、ranktable描述server间层级，在HCCL看来都是
RankGraph的查询结果，但分属两组能力互补的接口：NetInstance只给得出全局分区，TopoInstance只给得出
互联形态、位置、协议与端口数。`physicalLevels`把二者按rank范围对齐后合成一条链，每个物理层级出现一次。

本需求不生成算法Hierarchy，不决定算法层数，也不查询任意rank对的全部Link。

标准化失败不是致命错误：`physicalLevels`为空时，全部现有字段和旧执行路径必须保持可用（见第9章）。

## 2. 目标与非目标

### 2.1 目标

1. 保留`CalcTopoShape`、`ExtractNetLayerDetails`和`ExtractTopoDetails`现有入口；
2. 保留`TopoInfo`、`NetLayerDetails`和`TopoInstDetails`名称与内容；
3. 仅在`TopoInfoWithNetLayerDetails`尾部追加`physicalLevels`；
4. 将NetInstance（ranktable层级）与TopoInstance（topo层级）**合一**成当前rank可见的范围链：
   rank集合相同的合并为一个Level，更细的TopoInstance单独成级，没有TopoInstance的层用标志位注明；
5. 每个Level携带该层的全局分区、互联形态、Device/Host位置、协议集合与物理端口数，并用`view`
   标明是否知道该层的全局划分；
6. PhysicalLevel按`(块大小, view, topoType)`三键排序，不引入固定P0/P1/P2/P3槽位；
7. HCOMM返回的裸指针数据立即复制到HCCL管理的容器；
8. 支持当前rank局部视角下规则嵌套的TopoInstance范围；
9. 沿用现有HCOMM接口调用方式：TopoInstance与Endpoint系列经`src/common/hcomm_dlsym/hccl_rank_graph_dl.h`
   弱符号调用，`GetLayers`、`GetInstSizeListByLayer`、`GetRanksByLayer`沿用现有直接调用，均不引入HCOMM私有头；
10. `physicalLevels`纳入现有`Serialize/DeSerialize`，且不改变既有字段的编码顺序；
11. 标准化失败时降级为空视图，不影响现有字段和旧执行路径；
12. 额外提取本卡整机形态`deviceFormFactor`（供性能建模判断是否POD机型，POD的交换机层为2:1收敛），
    取不到时停在`UNKNOWN`，同样不影响任何现有字段。

### 2.2 非目标

1. 计算Level1、Level2或Level3算法分组；
2. 选择具体Link、Channel、协议或HostDPU；
3. 修改HCOMM RankGraph内部语义；
4. 删除、重命名或调整现有Topo字段顺序；
5. 首期支持互相重叠但互不包含的范围（例如同一NetLayer上的2D Mesh x/y TopoInstance）；
6. 首期支持任意非连续、非对称且缺少稳定坐标的Instance布局；
7. 全量复制可能达到O(N²)的rank对Link；
8. 改造`InitRankInfo`中已有的`GetPairLinkCounter`等既有Link查询逻辑。

## 3. 输入与HCOMM接口

### 3.1 使用接口

| 接口 | 提取内容 | 返回顺序 | 调用方式 |
|------|----------|----------|----------|
| `HcclRankGraphGetLayers` | 实际NetLayer编号 | 升序（源自`std::set`） | 直接调用 |
| `HcclRankGraphGetInstSizeListByLayer` | 某NetLayer的全局NetInstance大小列表 | **无序**（见3.2） | 直接调用 |
| `HcclRankGraphGetRanksByLayer` | 当前rank所在NetInstance的完整rank列表 | 升序（源自`std::set`） | 直接调用 |
| `HcclRankGraphGetTopoInstsByLayer` | 当前rank所在NetInstance内的全部TopoInstance | **无序**（见3.2） | dlsym弱符号 |
| `HcclRankGraphGetTopoType` | TopoInstance类型 | — | dlsym弱符号 |
| `HcclRankGraphGetRanksByTopoInst` | 指定TopoInstance的rank列表 | 升序（源自`std::set`） | dlsym弱符号 |
| `HcclRankGraphGetEndpointNum` | 当前rank在该TopoInstance上的Endpoint数量 | — | dlsym弱符号 |
| `HcclRankGraphGetEndpointDesc` | 当前rank在该TopoInstance上的Endpoint协议、地址和位置 | — | dlsym弱符号 |
| `HcclRankGraphGetEndpointInfo` | Endpoint所属iface的物理端口数（`ENDPOINT_ATTR_BW_COEFF`） | — | dlsym弱符号 |

**返回内存的生命周期**：HCOMM为`GetLayers`、`GetRanksByLayer`、`GetInstSizeListByLayer`各持有**一个**成员
vector，每次调用先`clear()`再重填，然后返回其`data()`。因此：

- 同一接口的下一次调用会使上一次返回的指针失效（含扩容导致的重分配）；
- 不同接口之间互不影响。

提取函数必须在**同一接口的下一次调用之前**完成复制，不得将HCOMM裸指针保存在Topo结构中。逐层循环提取时，
必须每层调用后立即复制，不能先收集各层指针再统一复制。

### 3.2 接口能力限制

- `GetInstSizeListByLayer`提供全局Instance大小，但不提供每个Instance成员；
- `GetRanksByLayer`只提供当前rank所在Instance成员；
- **`GetTopoInstsByLayer`返回的是当前rank所在NetInstance内的全部TopoInstance，不是"当前rank所属的
  TopoInstance"**。返回列表中可能包含当前rank并不属于的兄弟实例，消费方必须按成员关系自行过滤；
- **两个接口的返回顺序无任何保证**：HCOMM内部`netInsts_`是
  `vector<unordered_map<string, NetInstance>>`、`topoInsts_`是`unordered_map<u32, TopoInstance>`，
  遍历顺序是哈希序，既不是rank序也不是ID序。因此`instSizeList`的下标不对应任何稳定的分区编号，
  `topoInsts`的下标也不对应任何稳定的实例编号；
- **`GetEndpointDesc(comm, layer, topoInstId, ...)`返回的是当前rank在该TopoInstance上的Endpoint，
  不能枚举对端rank的Endpoint集合**；
- `GetEndpointNum`返回的是"当前rank在该TopoInstance上的各连接接口所支持协议数之和"，它是
  `GetEndpointDesc`实际写入条数的**上界**而不一定相等（Endpoint按`(地址, 协议)`去重，`GetEndpointNum`
  求和时不去重）。缓冲不足时`GetEndpointDesc`返回`HCCL_E_PARA`，不做截断；
- Endpoint能力摘要不能证明任意rank对一定存在Link；
- Rank ID连续不等价于任意物理Instance成员连续。

- **`GetEndpointDesc`的输出顺序同样是哈希序**：其内部遍历的是
  `unordered_map<(CommAddr, CommProtocol), iface>`。因此Endpoint快照必须排序后保存（见4.2）；
- **一个iface有N种协议就产生N个EndpointDesc，且全部映射回同一个iface**（HCOMM
  `rank_graph_builder.cc`的`SetEndpointDesc`按`iface × protocol`双层循环注册，键是`(commAddr, protocol)`）。
  因此`GetEndpointInfo(ENDPOINT_ATTR_BW_COEFF)`对这N条会返回**同一个端口数**——它是iface的属性，不是
  endpoint的属性。逐endpoint累加会把同一条物理链路的端口数重复计入（`ub_ctp` + `ub_mem`的8口iface
  会被算成16口）。端口数必须按`commAddr`去重到iface粒度后统计（见4.2）;
- **`ENDPOINT_ATTR_BW_COEFF`名为"带宽系数"，实现返回的是`iface->GetPorts().size()`**（HCOMM
  `rank_graph.cc`的`GetEndpointInfo`），即portGroupSize。本仓`op_common.cc`建链时填
  `channel.portGroupSize`用的是同一个属性。

哈希序这几条直接决定了本设计的三个选择：`physicalLevels`中一切来自哈希序的数据都必须规范化后保存
（`partitionSizes`降序、Endpoint快照按稳定键排序，见4.2）、Level排序必须用与输入顺序无关的全序
（三键排序，见5.3），以及"RankGraph返回顺序变化不影响标准化结果"必须作为强制不变量（见第6章）
而不是可选目标。

因此，本需求输出的是当前rank的标准化局部范围链。全局一致性通过HCOMM构造契约和全rank测试验证。
对端Endpoint归属判断的边界由[物理建链统一重构](./link-binding-refactor-design.md)第7章约束。

## 4. 数据结构

### 4.1 兼容原则

| 数据结构 | 本次处理 |
|----------|----------|
| `TopoInfo` | 完全不变 |
| `NetLayerDetails` | 完全不变 |
| `TopoInstDetails` | 完全不变 |
| `TopoInfoWithNetLayerDetails` | 保留全部现有内容，仅在尾部追加`physicalLevels` |
| PhysicalLevel相关类型 | 全部采用新增独立结构 |

`PhysicalLevelInfo::localRanks`对TopoInstance来源确实与`TopoInstDetails::ranksInTopo`内容重复。这是有意保留的
冗余：`ranksInTopo`按`GetTopoInstsByLayer`的哈希返回序索引，按下标回查会随rehash静默错位（见3.2），
消费侧需要一份与Level绑定、下标语义稳定的成员列表。序列化体积的实际增量见8.4——主导项是`EndpointDesc`
而不是`localRanks`。

**`partitionSizes`与现有`instSizeListOfLayer[layer]`内容同源**，是本设计唯一一处有意的持久化重复。
理由：`instSizeListOfLayer`按哈希序保存、按NetLayer组织，而`partitionSizes`降序规范化、按Level组织，
且是跨rank一致性校验的锚点（见6.1）。若改为"只存netLayer号、消费侧回查`instSizeListOfLayer`"，
消费侧就得自己再做一次降序规范化，等于把这条容易漏掉的约束扩散到每个消费者。

`partitionSizes`的派生量（GCD、是否全等等）**不进入本结构**。topo-parse只承载提取到的拓扑事实，
分层维度与非对称判断属于TopoMatch，由它按需自行计算。

### 4.2 新增辅助结构

```cpp
// 该Level是否知道"整个通信域在这个粒度上的完整划分"。
// 合一之后它不再是"来源"的区分，而是"知不知道全局分区"的区分，与partitionSizes是否为空等价
enum class PhysicalLevelView : uint32_t {
    LOCAL  = 0, // 只知道当前rank所在的那一块；partitionSizes恒为空
    GLOBAL = 1, // 知道该netLayer的完整分区；partitionSizes非空
};

// 该Level在RankGraph中的原始身份，用于回查。
// netLayer恒有效；topoInstId仅在该Level有TopoInstance支撑时有效
struct PhysicalSourceRef {
    uint32_t netLayer   = INVALID_UINT;
    uint32_t topoInstId = INVALID_UINT;
};

struct PhysicalLevelInfo {
    std::vector<uint32_t> localRanks;      // 升序去重，必含当前rank
    PhysicalLevelView     view;
    std::vector<uint32_t> partitionSizes;  // GLOBAL：该层全部Instance大小，降序；LOCAL：空
                                           // 不存GCD/是否全等等派生量，见下
    PhysicalSourceRef     ref;

    // ---- 链路属性：由该Level的TopoInstance提供，全部随hasTopoInst一起生效 ----
    bool                       hasTopoInst = false;                        // 见下
    CommTopo                   topoType    = COMM_TOPO_RESERVED;           // 排序第三键
    EndpointLocType            locType     = ENDPOINT_LOC_TYPE_RESERVED;   // Device / Host
    std::vector<CommProtocol>  protocols;                                  // 去重升序
    std::vector<uint32_t>      portNums;                                   // 按iface去重，降序
    std::vector<EndpointDesc>  endpoints;                                  // 按稳定键排序
};
```

**没有`PhysicalSourceType`，也没有独立的`PhysicalSourceInfo`。** 合一之后一个Level可能同时对应一个
NetInstance和一个TopoInstance，"来源类别"这个判别式不再成立；`ref`同时持有两者的身份，链路属性直接平铺
在Level上。

**`view`不是标注习惯，而是RankGraph两组接口的能力差异，无法互相推导**：

- `GetInstSizeListByLayer`返回该netLayer上**全部**NetInstance的大小 → 该层可以`GLOBAL`；
- `GetTopoInstsByLayer`内部是`GetNetInstanceByRankId(netLayer, myRank_)`（`rank_graph.cc`），
  只看得到当前rank所属NetInstance内的TopoInstance，兄弟NetInstance的拓扑结构从这个接口拿不到。
  因此**比NetInstance更细**的TopoInstance级永远补不出全局分区 → 只能`LOCAL`。

合一只发生在rank集合相同的一对上：此时该级既拿到分区（来自NetInstance）又拿到链路属性
（来自TopoInstance），`view = GLOBAL`。更细的TopoInstance级`view = LOCAL`、`partitionSizes`为空。

**`hasTopoInst`是显式标志位，不由其他字段推断。** 某netLayer上`GetTopoInstsByLayer`返回0个实例时，
该层只有ranktable给出的分区，没有任何链路属性可言，此时`hasTopoInst = false`、下面五个字段全部保持
无效值。做成独立标志位而不是让消费侧去猜"`topoType == RESERVED`是不是就代表没有"，因为那两件事不等价：
有TopoInstance但endpoint查询失败时，`topoType`仍然有效而`endpoints`为空。

**`locType`与`protocols`从`endpoints`提炼而来，是SE模型里每层必备的两项。**

- `locType`：该级的链路落在Device还是Host。消费侧据此判断**"是否需要使用host网卡"——看最高一级的
  `locType`是否为`HOST`**。各endpoint位置不一致时保持`RESERVED`并告警：一个Level对应一种网络平面，
  位置本应唯一，不一致说明混进了不同平面的接口，此时给出任何一个都会误导该判断；
- `protocols`：是**集合**而不是单值。同一个iface可以同时跑多种协议（如`ub_ctp`与`ub_mem`），
  HCOMM会为每种协议各生成一个EndpointDesc但它们指向同一个iface。去重升序保存。

**`portNums`按iface去重，一条物理链路一项，降序。** 该层只有一条8口链路记作`{8}`，有两条则记作`{6,2}`，
求和为本卡在该级的总物理端口数。**必须按`commAddr`去重到iface粒度**，理由见3.2：逐endpoint统计会把
多协议iface的端口数重复计入。取不到时"全有或全无"——只要有一条查询失败就整个清空，残缺数组会让消费侧
算出一个"看着合理但偏小"的总端口数，这种错误比空数组难查得多。

`endpoints`保留完整快照供建链侧回查，`locType`/`protocols`/`portNums`是它的提炼结果而不是替代品。

**`endpoints`必须按稳定键排序后保存**：`GetEndpointDesc`内部遍历的是`unordered_map<(addr, protocol), iface>`，输出顺序是哈希序，既不稳定也不跨进程一致。若原样保存，第6章不变量11（"RankGraph返回顺序变化不影响标准化结果"）在`endpoints`上不成立。构建阶段必须按`(protocol, loc.locType, commAddr字节)`字典序排序，排序后再写入。同理`protocols`去重升序、`portNums`降序，都是同一条不变量的要求。

**不存`partitionSizes`的任何派生量。** 早期版本曾保存`partitionGcd`（分层维度依据）与`partitionUniform`
（Instance大小是否全等，供Level3拒绝非对称拓扑）两个派生缓存。二者已移除，理由有两条：

- **职责**：非对称的判断与处理归TopoMatch，topo-parse只做拓扑信息提取，不在提取侧固化决策口径；
- **一致性**：存储的派生值可能与来源不一致，而这种不一致本地无法察觉。`partitionSizes`始终保留，
  消费侧一行即可重算——`uniform`等价于全部元素相等，`gcd`是一次欧几里得归约。

同理**不存"是否POD"这类由`deviceFormFactor`派生的布尔**，也不存"是否需要host网卡"——后者由消费侧
读最高一级的`locType`推导。本结构只承载提取到的事实。

需要注意`uniform`并非独立信息：设分区数为`n`、`gcd`为`g`、总数为`N`，则`uniform ⟺ g × n == N`
（各块全等于`s`时`g = s`、`s × n = N`；反之若`g × n = N`而各块都是`g`的倍数且不小于`g`，
则每块只能恰好是`g`）。因此保留`partitionSizes`一项即已完备。

**`partitionSizes`保留完整列表，但必须降序规范化后保存。** `GetInstSizeListByLayer`的返回顺序是哈希序（见3.2，HCOMM侧是`Level2Id2NetInst = vector<unordered_map<string, shared_ptr<NetInstance>>>`的遍历），列表下标不对应任何稳定的分区编号。**原样保存会导致同一拓扑在不同进程得到不同字节流，跨rank一致性锚点直接失效**；降序排序后按multiset语义使用，则既稳定又无损。

只存派生量不够：`[16,4]`与`[12,8]`的GCD都是4、是否全等都是false、rank数都是20，仅凭派生字段无法区分这两种完全不同的分区。而`partitionSizes`是本结构唯一的**全局量**——同一netLayer上跨rank逐字节相同——因此也是反序列化后做跨rank一致性校验的唯一可用锚点。这既是保留完整列表的理由，也是不必再存派生量的理由。

构建期临时输入直接使用候选数组，不额外定义包装结构：

```cpp
// 仅在CalcTopoShape期间存在，不序列化
using PhysicalLevelCandidates = std::vector<PhysicalLevelInfo>;
```

### 4.2.1 整机形态

`deviceFormFactor`是与`physicalLevels`独立的一个标量，挂在`TopoInfoWithNetLayerDetails`上：

```cpp
// 取值与acl_rt.h的ACL_DEVICE_FORM_FACTOR_*逐一对应
enum class DeviceFormFactor : uint32_t {
    POD = 0, A_K = 1, A_X = 2, PCIE_CARD = 3,
    UNKNOWN = INVALID_UINT,   // 本仓的降级值，不是ACL取值
};
```

三点设计约束：

1. **`UNKNOWN`不能用0代替。**`ACL_DEVICE_FORM_FACTOR_POD`就是0，与`uint32_t`默认值、`memset`结果、
   反序列化失败后的残留值全部撞车。任何"没取到就保持默认"的路径都会被读成"这是POD机型"，而POD恰好
   是消费侧要走特殊分支（2:1收敛）的那个值，**误判方向正好是最坏的**；
2. **存原始形态，不存`bool isPod`。** 与`partitionSizes`同理：本结构只承载事实，"POD ⇒ 交换机层2:1收敛"
   属于消费侧的知识；且存bool会在那一刻丢掉A_K/A_X的区分。判POD用`TopoInfoWithNetLayerDetails::IsPodForm()`；
3. **枚举值是ACL宏的一份复制**（`alg_param.h`同时参与device/AICPU侧编译，引不进host侧的`acl_rt.h`），
   一致性由`topo_host.cc`里的`static_assert`守护——那个编译单元同时看得到两套定义，ACL改值会直接编译失败；
4. **取值时的设备号必须是`logicDevId`**。ACL/驱动有三套设备号：`phyDevId`（物理编号）、
   `userDevId`（用户可见，`aclrtSetDevice`/`aclrtGetDevice`这一套，受`ASCEND_RT_VISIBLE_DEVICES`影响）、
   `logicDevId`（驱动内部）。`aclrtGetDeviceInfo`与底层`halGetDeviceInfo`要的是**logicDevId**
   （`ascend_hal_base.h`：除`INFO_TYPE_MASTERID`外一律使用logical device ID；HCOMM的
   `CcuGetMainboardId`形参名即`deviceLogicId`）。
   而`aclrtGetDevice`返回的是**userDevId**，必须经`aclrtGetLogicDevIdByUserDevId`转换。
   **少这一步在默认部署下测不出来**——两者恰好相等；只有配了`ASCEND_RT_VISIBLE_DEVICES`或容器只挂载
   部分设备时才会去读另一张卡的形态，且读到的是合法值，不报错、只静默拿到错的机型。

### 4.3 现有结构增量扩展

`NetLayerDetails`和`TopoInstDetails`定义保持不变。

`TopoInfoWithNetLayerDetails`仅追加一个尾部成员：

```cpp
struct TopoInfoWithNetLayerDetails : public TopoInfo {
    // 全部现有成员、名称和顺序保持不变

    DeviceFormFactor deviceFormFactor = DeviceFormFactor::UNKNOWN; // 新增标量
    uint32_t physicalLevelNum = 0;                                 // 与physicalLevels.size()同步
    std::vector<PhysicalLevelInfo> physicalLevels;                 // 新增尾部成员

    bool IsPodForm() const { return deviceFormFactor == DeviceFormFactor::POD; }
};
```

`deviceFormFactor`声明在标量区（与其它整机属性放在一起可读性更好），但**序列化追加在
`physicalLevels`之后**。这个不对称是有意的：守的是"新字段只追加在字节流尾部"的约定。
代价是`physicalLevelNum`超限触发的早返回会连带丢掉它（停在`UNKNOWN`）——流是纯位置流，跳过变长的
`physicalLevels`段就无法定位其后的字段。两者的降级态都是"该字段不可用"，消费侧本就必须处理。

字段关系：

- 现有字段继续服务未迁移的Selector和Executor；
- `physicalLevels`服务新TopoMatch和LinkBindingResolver；
- `physicalLevels`由Normalizer单向生成，完成后只读；
- 不从`physicalLevels`反向修改现有字段；
- 全覆盖通过`localRanks.size() == userRankSize`计算，不新增派生布尔字段；
- Level下标只在当前Topo对象内有效，原始身份由`PhysicalSourceRef`保存。

### 4.4 消费侧读取契约

| 消费者 | 允许读取 |
|--------|----------|
| TopoMatch | `localRanks`、`view`、`partitionSizes`（分层维度与非对称性由它自行推导） |
| LinkBindingResolver | `localRanks`、`ref`、`topoType`、`locType`、`protocols`、`endpoints`、Level下标 |
| Selector | `endpoints`（判断Device/Host能力，替代重复调用RankGraph） |
| 性能建模（cost model） | `topoType`、`locType`、`protocols`、`portNums`，以及外层的`deviceFormFactor`；<br>"是否需要使用host网卡"由它读**最高一级**的`locType`自行推导 |

TopoMatch只读`localRanks`/`view`/`partitionSizes`，因此协议、Endpoint、TopoType、端口数和整机形态在
原理上都不可能影响算法分组——这是可以静态检查的验收项。

**链路属性与`deviceFormFactor`都是局部量，跨rank不保证相同。** `portNums`尤其如此：同一台机器上不同die
的rank链路就可能不同（HCOMM `TopoGetClosPort`里die0是4个口、die1是2个口）。消费侧若要用它们做全域
一致的决策（如cost model驱动算法选择），必须自行处理一致性，本结构不提供该保证。非对称性判断由TopoMatch从`partitionSizes`自行推导，只用于**准入判定**（是否返回不支持），不参与任何维度或成员计算，因此不破坏"Hierarchy = F(physicalLevels, algTopoRequest, myRank)"这一不变量的分组部分。

## 5. 解析与标准化流程

### 5.1 CalcTopoShape改造

现有实现的实际顺序为：

```text
ExtractNetLayerDetails
CalcLevel1Nhr
ExtractTopoDetails
CalcLevel0TopoShape
Is2DieFullMesh
IsLevel0PcieMix
CalcLevel0MeshType
CalcLevel2Uboe
CalcLevel2Ubg
```

保留上述全部步骤及其顺序，在末尾追加：

```text
...（现有9个步骤保持不变）
CalcDeviceFormFactor           # 新增，查ACL整机形态，与comm无关
BuildPhysicalLevels            # 新增
  ├─ BuildLayerCandidates      #   逐netLayer合一ranktable层级与topo层级
  ├─ NormalizePhysicalLevels   #   三键排序 + 链校验
  └─ ValidatePhysicalLevels    #   不变量自检
```

新增步骤放在最后，保证现有字段的提取和派生逻辑完全不受影响，也保证新步骤可以复用已提取的
`netLayerDetails`和`topoInstDetailsOfLayer`。

`HcclRankGraphGetRanksByLayer`原本只使用返回数量。新逻辑在不改变
`localNetInsSizeOfLayer[layer] = rankNum`的同时，将完整rank列表复制到临时候选数组。这是`CalcTopoShape`中
唯一新增的RankGraph查询；新增逻辑不调用`GetLinks`。

`CalcDeviceFormFactor`不依赖`HcclComm`，只查本设备，恒返回`HCCL_SUCCESS`：取不到时停在`UNKNOWN`，
现有字段与旧执行路径完全不受影响。这与同文件里其它`Calc*`的失败即返回是有意的区别——那些是算法分层的
输入，这个不是。

### 5.2 逐层合一

**每个netLayer产出一到多个Level，规则如下**（`BuildLayerCandidates`）：

| 情形 | 产出 | `view` | `partitionSizes` | `hasTopoInst` |
|------|------|--------|------------------|---------------|
| TopoInstance的rank集合 **== 本层NetInstance** | 合并成一个Level | `GLOBAL` | 该层完整分区 | `true` |
| TopoInstance的rank集合 **比NetInstance更细** | 单独成级 | `LOCAL` | 空 | `true` |
| 该层**没有**同范围TopoInstance | NetInstance单独成级 | `GLOBAL` | 该层完整分区 | `false` |

**除netLayer 0外每层只有一个TopoInstance，因此合并是确定的。** netLayer 0在特殊机型上可能挂多个
（典型是同范围的一个Mesh一个CLOS），此时按"有多少写多少"各出一个Level，由`topoType`区分。
同范围的多个TopoInstance**各自都持有该层的分区**——分区是该层的全局事实，不专属于其中某一种互联形态。

各字段的取数：

- NetInstance侧的`localRanks`来自`GetRanksByLayer`；
- NetInstance侧的`partitionSizes`来自`instSizeListOfLayer[layer]`的**局部拷贝**，降序排序后保存。
  必须先拷贝再排：该字段会被序列化下发，就地重排会导致现有Topo字段Golden不一致，且只在存在非对称
  拓扑时才显现；
- TopoInstance侧的`localRanks`来自`GetRanksByTopoInst`，`topoType`来自`GetTopoType`，
  **取不到必须整体降级**（理由见5.3：它是排序第3键）；
- `endpoints`按4.2约定排序后保存；`locType`与`protocols`从中提炼；`portNums`按`commAddr`去重到iface
  粒度后逐条查`GetEndpointInfo(ENDPOINT_ATTR_BW_COEFF)`。

**`GetTopoInstsByLayer`返回失败一律整体降级；"该层没有TopoInstance"的形态是成功且`topoInstNum == 0`。**

早期版本写作"其余netType上HCOMM返回`HCCL_E_PARA`，按per-layer局部降级处理"，该结论只适用于
`CommunicatorImpl::GetTopoInstsByLayer`的netType闸门，而那是**非950路径**：

- `shouldGoOutPlace`（`hccl_common.h`）把本流程限定在`DEV_TYPE_950`/`DEV_TYPE_960`，其余设备在算子
  入口就走了老流程，`CalcTopoShape`不执行；
- 这两者的SoC名都命中`hrtGetHcclV2Support`的strstr判定（`adapter_rts.cc`），因此HCOMM侧必定走
  `IRankGraph::GetTopoInstsByLayer`（`rank_graph_interface.cc`）——它**不判netType**，只在
  `netLayer`不属于`GetLevels(myRank)`时返回`HCCL_E_PARA`；
- 而本流程遍历的`netLayers`正是`HcclRankGraphGetLayers`的结果，即同一个`GetLevels(myRank)`，
  因此该分支不可达。

真正的"该层没有TopoInstance"来自`NetInstance::GetTopoInstsByLayer`（`net_instance.cc`）——它只是遍历
`topoInsts_`这个map，空map返回0且不报错。此时该层仍然出一个Level（带分区），只是`hasTopoInst = false`、
没有任何链路属性可供建链，消费侧须容忍。

因此实现上分三条互斥路径：返回非成功→整体降级；`topoInstNum == 0`→该层出一个`hasTopoInst = false`的级；
`topoInstNum != 0`但指针为空→整体降级。局部跳过一个真实错误会让该rank比其余rank少若干Level，
而少掉之后这条链单看仍然合法，本地校验一个都拦不住。
现有`ExtractTopoDetails`对同一调用**根本不检查返回值**（该层拿到0个实例就过去了），若新步骤在此整体降级，
会出现"旧路径正常、新路径全空"的不对称失效。

**必须按成员关系过滤**：`GetTopoInstsByLayer`返回的是当前rank所在NetInstance内的全部TopoInstance
（见3.2），其中不包含当前rank的兄弟实例必须丢弃。保留兄弟实例会在范围链中引入与当前rank无关、
且互不包含的范围，导致本可支持的拓扑被误判为不支持。

HCOMM对两类TopoInstance的注册范围并不相同，过滤在二者上的性质也不同：

| 来源 | HCOMM注册范围 | 过滤的性质 |
|------|---------------|------------|
| layer0 peer2peer（直连Mesh） | `UpdateTopoInstForMyRankOnly`只注册myRank所属的那一个 | **防御性**：正常路径上返回的本就只有myRank的Mesh |
| peer2net Fabric（CLOS、PCIe-SW等） | 为NetInstance内**全部rank**注册，rank按所连Fabric分别落到不同topoInstId | **必需的正确性步骤** |

第二行有真实场景：一个16卡NetInstance内挂两个PCIe switch，rank 0～7连switch A、rank 8～15连switch B，
则该层有两个Fabric TopoInstance，`GetTopoInstsByLayer`对rank 0会同时返回A和B，而rank 0不属于B。

**过滤逻辑不得依赖上表第一行**：它是HCOMM当前的构造实现细节，且与其公开头注释本就不一致（头文件写的是
"myRank所在的topoInstance集合"，实现返回的是整个NetInstance的）。按成员关系过滤在两种语义下都正确，
这正是选择它而不是选择"信任接口语义"的原因。

### 5.3 三键排序

对候选的rank集合排序去重、`partitionSizes`与`portNums`降序、`protocols`去重升序之后，按以下三个键
依次比较排序：

| 键 | 内容 | 取值来源 |
|---|---|---|
| 1 | 当前rank在该级的**块大小**升序 | `localRanks.size()` |
| 2 | `view`：`LOCAL`(0)在前，`GLOBAL`(1)在后 | 只知道本块的排在知道全局分区的之前 |
| 3 | `topoType`：按互联紧密度递减，`1DMESH`(0) < `CLOS`(1) < 无TopoInstance(2) | 仅在前两键全部打平时生效 |

**三键缺一不可。** 任一键缺失时排序会退化到RankGraph的哈希返回序，各rank排出的下标语义就不一致了——而分叉之后每一级单看都合法（rank集合真实、嵌套成立、含当前rank），没有任何本地校验拦得住。

**第3键对`GLOBAL`级同样生效。** 这是合一带来的变化：合并前"GLOBAL级永远不会与LOCAL级在第二键上打平，
同一netLayer也只出一个GLOBAL级"，第3键只需服务LOCAL级；合并之后，netLayer 0上同范围的Mesh与CLOS两级
**都带全局分区**（`view`同为`GLOBAL`），前两键全部打平，定序完全依赖`topoType`。
无TopoInstance的级没有形态可言，取一个与`1DMESH`/`CLOS`不重叠的序值（2）排在末尾——它携带的信息最少。

**第3键不得用枚举值代替**：`COMM_TOPO_CLOS = 0 < COMM_TOPO_1DMESH = 1`（`hccl_rank_graph.h`），枚举序与"直连在内层"正好相反。必须用显式优先级函数，且该函数对预期外的类型返回失败而非静默排到末尾——排序键取不到值时，各rank对未知类型的相对顺序就没有共同依据。走到本模块的必然是910_95，只可能出现`1DMESH`与`CLOS`，因此该分支是哨兵。

由此推出一条实现约束：**有TopoInstance的级其`GetTopoType`失败必须整体降级，不能保留`RESERVED`继续**。
在早期"topoType只是建链元数据"的设计下容忍是对的，但它升格为排序键之后，一个停留在`RESERVED`的级会让
第3键失去取值依据。910_95上该调用不会失败，代价为零。注意该约束的作用范围随合一从"LOCAL级"扩大到了
"所有`hasTopoInst == true`的级"。

排序用`sort`而非`stable_sort`：上述键构成全序，结果不依赖输入顺序——这正是目的，输入顺序来自RankGraph的哈希遍历，本就不可依赖。

兜底键在三键之后依次为`localRanks`字典序、`ref.netLayer`、`ref.topoInstId`。正常输入上永不决定顺序，
只为让比较器在不同Level上构成全序——`std::sort`对等价元素的相对顺序未指定，各rank排出的下标语义会分叉。

### 5.4 合一的边界

**合一只发生在rank集合完全相同的NetInstance与TopoInstance之间。** 二者承载的信息互补而非重叠：
NetInstance给全局分区，TopoInstance给互联形态、位置、协议与端口数，合并后这一级同时具备两者。

不合并的两种情形：

- **更细的TopoInstance**（如8卡NetInstance内的4卡Mesh）：它看不到兄弟NetInstance，没有全局分区可言，
  单独成`LOCAL`级；
- **同范围的多个TopoInstance**（netLayer 0的Mesh与CLOS）：rank集合虽然相同，但互联形态不同、
  端口数与协议也不同，合成一级会丢掉其中一份。各出一级，由第3键定序。

代价与收益：

- 链上相邻两级的rank集合**允许相等**（不是严格递增）。netLayer 0上同范围的Mesh与CLOS两级、以及两个
  netLayer的本地NetInstance恰好同范围，都是合法的相等相邻对；
- 相比合并前"每个netLayer必出一个GLOBAL级 + 每个TopoInstance必出一个LOCAL级"的1:1模型，级数明显变少：
  常规2机×8卡从4级降到2级，超节点32卡从6级降到3级。消费侧不再需要按`view`筛掉重复范围。

同一rank集合的多个netLayer给出不同全局分区统计是合法拓扑（例如某netLayer把通信域切成`[8,8,8,8]`，
另一netLayer切成`[8,24]`，而当前rank的本地成员恰好相同）。这两份统计各自留在自己的Level上，
`ref.netLayer`区分它们，`[16,4]`与`[12,8]`这类此前被GCD抹平的差异也一并保住。

### 5.5 局部TopoInstance视角

仅含TopoInstance的PhysicalLevel即使没有全局分区（`partitionSizes`为空），也保留其完整`localRanks`，
不得因此降级为只能用于Link的能力来源。

规则嵌套场景中，当前rank只需要看到自身的范围链：

```text
S0 = 当前rank所在最小TopoInstance范围
S1 = 严格包含S0的下一范围
...
Sn = 当前rank可见的全通信域范围
```

范围链要求相邻元素相等或严格包含。同一NetLayer上互相重叠但互不包含的TopoInstance（典型为2D Mesh的x/y环）
不属于范围链，首期不纳入支持范围，按第9章降级处理，对应算法继续使用旧Matcher。

任意非连续、非对称且无法由NetInstance全局统计证明一致的TopoInstance布局同样不纳入首期支持范围。

## 6. 不变量

标准化成功（`physicalLevels`非空）必须满足：

1. `userRankSize > 0`且`userRank < userRankSize`；
2. 每个候选和Level的rank ID无重复、无越界、升序并包含当前rank；
3. **`view`与payload自洽**：`LOCAL` ⟺ `partitionSizes`为空；`GLOBAL` ⟺ `partitionSizes`非空。
   合一之后`view`的含义是"知不知道全局分区"，与来源类别脱钩，因此这一条从三方自洽收缩为两方等价；
   - **3a.** `view`必须是`LOCAL`或`GLOBAL`之一。`PhysicalLevelView`底层类型固定为`uint32_t`，任何`uint32_t`
     值都是合法表示，不显式白名单则非法值会静默落进`GLOBAL`分支被当成全局分区处理；
   - **3b.** `ref.netLayer`恒有效（`!= INVALID_UINT`）。每个Level必然归属某一层，且取值来自`GetLayers`
     的实际结果；无效值说明构建侧漏填，消费侧回查原始对象时会拿到错误的层；
   - **3c.** `hasTopoInst`与链路属性自洽。`true`时`ref.topoInstId != INVALID_UINT`，且
     `portNums.size() <= endpoints.size()`（按iface去重后条数不会超过endpoint数）、每个`portNum`非0且
     不超`PORT_NUM_SANITY_LIMIT`、`portNums`降序、`protocols`去重升序；
     `false`时`ref.topoInstId`、`topoType`、`locType`、`protocols`、`portNums`、`endpoints`**必须全部保持
     无效值**。留一个"半有"的状态最危险——消费侧按`hasTopoInst`判定为不可用，却又能从字段里读出看似合理的值；
4. `GLOBAL`级的`partitionSizes`之和等于`userRankSize`；
   - **4b.** `GLOBAL`级的`partitionSizes`不含0项。0能穿过不变量4（不改变求和）与不变量5（降序时排在末尾），
     于是幽灵空分区会被消费侧当成一个真实Instance计入；
5. `GLOBAL`级的`partitionSizes`降序排列；
6. `GLOBAL`级的`localRanks.size()`必须是`partitionSizes`中的一项。`localRanks`是本地NetInstance的
   rank集合，其大小本就是该层某个分区，因此这是等式而非整除关系；
7. Level按大小**非递减**排列（相等相邻对合法，见5.4）；
8. 每个`hasTopoInst == true`的级其`topoType`可由5.3第3键的优先级函数定序。
   **注意作用范围随合一扩大**：合并前只需校验`LOCAL`级，现在同范围的Mesh与CLOS两级都是`GLOBAL`，
   仍然要靠第3键定序；
9. Level的`ref.netLayer`来自`GetLayers`实际结果；
10. 相邻Level满足包含关系`levels[i] ⊇ levels[i-1]`（允许相等）；
11. RankGraph返回顺序变化不影响标准化结果。这是强制项而非期望项：`GetInstSizeListByLayer`和
    `GetTopoInstsByLayer`的返回顺序本就是哈希序（见3.2），`GetEndpointDesc`的输出顺序同样是哈希序，
    任何依赖返回下标或返回顺序的实现都会产生不稳定结果。`endpoints`必须按4.2的稳定键排序、
    `partitionSizes`与`portNums`必须降序、`protocols`必须去重升序，才能满足本条；
12. 现有Topo字段与重构前相同输入下的Golden一致（无论标准化成功还是降级）；
13. 身份`(netLayer, topoInstId)`全域唯一。每个Level要么对应一个TopoInstance，要么是某层唯一的
    无TopoInstance级，本就不该出现两个同身份的Level；更要紧的是5.3兜底键正是靠这两项才构成全序——
    若它们能重复，两个Level在比较器下等价，而`std::sort`对等价元素的相对顺序未指定，各rank排出的
    下标语义就会分叉。本条把该前提落成受检不变量。
    > 合并前本条是`(view, netLayer, topoInstId)`三元组：那时同一netLayer会同时出一个NetInstance级
    > （`topoInstId == INVALID_UINT`）和若干TopoInstance级，需要`view`才能区分。合一之后
    > 无TopoInstance的级每层至多一个，`(netLayer, topoInstId)`已经唯一。

> **构建阶段另有两条检查，性质不同，不要混为一谈：**
>
> - **NetInstance的rank列表大小等于`localNetInsSizeOfLayer[layer]`——真实的跨调用一致性校验。**
>   `localNetInsSizeOfLayer`由`ExtractNetLayerDetails`中的一次`GetRanksByLayer`写入，标准化阶段是
>   **另一次**独立调用，两者不同源。不一致说明RankGraph在两次调用之间发生了变化，是活跃分支；
> - **`instSizeList`之和等于`userRankSize`——哨兵，正常路径永不触发。**`ExtractNetLayerDetails`已用
>   同一等式做过硬`CHK_RET`（不满足时返回`HCCL_E_PARA`，通信域直接起不来），且它先于标准化阶段执行。
>   保留它只为在HCOMM改变分层语义时第一时间暴露——`instSizeList`是`partitionSizes`的唯一输入，静默取到
>   错误统计的代价远高于这几行。**不要把它计入"降级路径已覆盖"的证据**，也不要为它设计触发用例。
>   代码注释中必须写明它不是活跃分支，否则后来人会误以为这里可能失败。
>
> 不变量8同样是哨兵（910_95上`topoType`只可能是`1DMESH`或`CLOS`），但它与上面那条哨兵不同：
> 它守护的是排序键的可取值性，一旦失效后果是跨rank下标分叉而非本地错误，因此仍必须显式检查并降级。

### 6.1 本地校验覆盖不到的部分

上述各条（1~13，含3a与4b）全部是**单rank本地**校验。有一类失效它们结构性地拦不住：

**同一netLayer上TopoInstance的种类结构跨rank不一致。** 例如某rank的layer0返回了MESH+CLOS两个实例、
另一rank只返回了MESH一个（下层接口异常、或通信域跨了机型）。此时两个rank的`physicalLevels`级数不同、
同一下标的语义不同，但**各自单看每一级都完全合法**——rank集合真实、嵌套成立、含当前rank、payload自洽。

各rank级数一致靠的是"同一netLayer上TopoInstance的种类结构由机型和配置保证一致"这条**外部契约**，
而不是链的形状。契约不成立时需要跨rank校验兜住：

- **锚点**：`partitionSizes`（同一netLayer上跨rank逐字节相同）与各netLayer上`LOCAL`级`topoType`的多重集；
- **时机**：反序列化后；
- **处置**：不一致即降级并打错误日志，按"下层接口或配置问题"上报，不继续往下走。

本期先落地上述本地各条与锚点字段，跨rank校验的具体落点随TopoMatch一并确定。

## 7. 示例

本章只给出标准化规则的最小示例。**三个真实机型的完整`TopoInfoWithNetLayerDetails`内容**（含现有字段、
`physicalLevels`逐字段取值、TopoMatch产出与序列化增量）见
[实现方案附录A](./topo-parse-impl-plan.md)，实施与验收请以附录A为准。

记号：每级写作 `localRanks | view | partitionSizes | topoType, locType, protocols, portNums`。

### 7.1 常规机型：2台Server × 8卡（rank 0）

```text
原始：netLayer0 = [8,8]  本地NetInstance {0..7}   TopoInstance 1DMESH {0..7}
      netLayer1 = [16]   本地NetInstance {0..15}  TopoInstance CLOS   {0..15}

标准化（每层的TopoInstance与NetInstance同范围，各自合一）：
  [0] {0..7}  GLOBAL {8,8}  1DMESH, device, [hccs], portNums={1}   L0/inst
  [1] {0..15} GLOBAL {16}   CLOS,   device, [roce], portNums={1}   L1/inst
```

合并前这里是4级（LOCAL/GLOBAL交替），合一之后每个物理层级只出现一次。

rank 8的`[0]`的`localRanks`变为`{8..15}`，`partitionSizes`一字不改——这正是它作为跨rank一致性锚点的依据。

### 7.2 超节点：4台Server每2台组成一个超节点 = 32卡（rank 0）

```text
原始：netLayer0 = [8,8,8,8]  TopoInstance 1DMESH {0..7}
      netLayer1 = [16,16]    TopoInstance CLOS   {0..15}
      netLayer2 = [32]       TopoInstance CLOS   {0..31}

标准化：
  [0] {0..7}  GLOBAL {8,8,8,8}  1DMESH, device, [hccs],  portNums={1}
  [1] {0..15} GLOBAL {16,16}    CLOS,   device, [ub_ctp],portNums={8}
  [2] {0..31} GLOBAL {32}       CLOS,   host,   [roce],  portNums={1}
```

**最高一级的`locType`是`HOST`，消费侧据此判定"需要使用host网卡"。** 本结构不存该结论。

### 7.3 SE设计中的完整形态：ranktable层级与topo层级合一

SE给出的目标形态是两组查询结果合成一条链。以4机2超节点、server内8卡再分2个4卡Mesh为例（rank 0）：

```text
ranktable侧（NetInstance，给分区）：
  netLayer0 = [8,8,8,8]   netLayer1 = [16,16]   netLayer2 = [32]

topo侧（TopoInstance，给链路属性）：
  L0: Mesh{0..3} ub_ctp portNum=1、CLOS{0..7} pcie portNum=4
  L1: CLOS{0..15} ub_ctp portNum=8
  L2: CLOS{0..31} roce  portNum=1（host）

合一后：
  [0] {0..3}  LOCAL  {}          Mesh, device, [ub_ctp],          portNums={1}   L0/mesh
  [1] {0..7}  GLOBAL {8,8,8,8}   CLOS, device, [pcie],            portNums={4}   L0/clos
  [2] {0..15} GLOBAL {16,16}     CLOS, device, [ub_ctp, ub_mem],  portNums={8}   L1/clos
  [3] {0..31} GLOBAL {32}        CLOS, host,   [roce],            portNums={1}   L2/clos
```

三点对照SE模型：

- `[0]`比本层NetInstance（8卡）更细，拿不到全局分区，因此是`LOCAL`——对应SE的`topoInfo.local.level0`；
- `[1]`与netLayer 0的NetInstance同范围，合一后同时持有`{8,8,8,8}`与pcie链路属性。
  **SE原稿把它写作`local.level1:[8]`是笔误**，按"知道全局分区即为global"的规则它是`GLOBAL`；
- `[2]`的`protocols`是**集合**`[ub_ctp, ub_mem]`而`portNums`只有一项`{8}`：这是同一个iface跑两种协议，
  端口数按iface计一次。这正是`portNums`必须按`commAddr`去重的场景。

### 7.4 netLayer 0上Mesh与CLOS同范围（UBX/PC16）

单Server 16卡，NetInstance = {0..15}，其内同时挂Mesh{0..15}与CLOS{0..15}（rank集合完全相同）：

```text
  [0] {0..15} GLOBAL {16}  1DMESH, device, [hccs|ub_mem], portNums={1}  L0/mesh
  [1] {0..15} GLOBAL {16}  CLOS,   device, [ubc_tp],      portNums={8}  L0/clos
```

**两级都带全局分区**——分区是netLayer 0的全局事实，不专属于其中某一种互联形态。前两键（块大小、view）
全部打平，定序完全依赖第3键`topoType`。**这是合一之后第3键必须对`GLOBAL`级生效的直接依据**，也是
"有多少写多少"这条规则的来源：两种形态的端口数与协议都不同，合成一级会丢信息。

### 7.5 非对称：netLayer0 = [16,4]

20卡。16卡侧NetInstance内有Mesh{0..3}与CLOS{0..15}，4卡侧有Mesh{16..19}与CLOS{16..19}：

```text
rank 0：
  [0] {0..3}   LOCAL  {}       1DMESH   L0/mesh
  [1] {0..15}  GLOBAL {16,4}   CLOS     L0/clos
  [2] {0..19}  GLOBAL {20}     CLOS     L1/clos
rank 16：
  [0] {16..19} GLOBAL {16,4}   1DMESH   L0/mesh   <- 与[1]的rank集合完全相同，靠第3键定序
  [1] {16..19} GLOBAL {16,4}   CLOS     L0/clos
  [2] {0..19}  GLOBAL {20}     CLOS     L1/clos
```

三点值得注意：

- **两个rank的级数相同、下标语义逐位对齐**。非对称只体现在块大小（16 vs 4），不体现在级数上。
  这依赖6.1所述的外部契约；
- **`partitionSizes = {16,4}`在两个rank上完全相同**，而`localRanks`处处不同；
- rank 16上Mesh与NetInstance恰好同范围（都是4卡），因此`[0]`也升为`GLOBAL`并带上分区——
  合一规则只看rank集合是否相同，不看它来自哪一类。

### 7.6 某netLayer没有TopoInstance

```text
  [k] {0..31} GLOBAL {32}  hasTopoInst=false, topoType=RESERVED, locType=RESERVED,
                           protocols={}, portNums={}, endpoints={}
```

该层只有ranktable给出的分区，没有任何链路属性。**这是正常形态而非异常**，用`hasTopoInst`显式注明，
不整体降级。代价是该层没有Endpoint与`topoType`可供建链、cost model也拿不到该层的端口数，消费侧须容忍。

### 7.7 同层2D Mesh（首期降级）

当前rank为0时，`GetTopoInstsByLayer(0)`可能返回该NetInstance内的全部实例：

```text
TopoInstance A: {0,1,2,3}      // x环，含rank0
TopoInstance B: {0,4,8,12}     // y环，含rank0
TopoInstance C: {4,5,6,7}      // 兄弟x环，不含rank0 -> 按5.2过滤丢弃
TopoInstance D: {1,5,9,13}     // 兄弟y环，不含rank0 -> 按5.2过滤丢弃
```

过滤后剩下A与B，二者只在当前rank处相交、互不包含，不构成范围链。`CalcTopoShape`返回成功，
`physicalLevels`为空，现有字段与旧Matcher行为完全不变。

注意对比：若某层是4个互不相交的Mesh8（典型的服务器内分组），过滤后只剩当前rank所属的那一个，
不会触发降级。只有当前rank**同时**属于多个互不包含的实例时才会。

## 8. 序列化

### 8.1 现有序列化路径

`TopoInfoWithNetLayerDetails::Serialize/DeSerialize`当前只有两个使用场景，均在同一版本二进制内部：

1. **Host侧缓存往返**：`HcclCalcTopoInfo`首次调用`InitRankInfo`（内含`CalcTopoShape`）后把序列化结果写入
   EngineCtx；此后每个算子都从EngineCtx反序列化重建topoInfo；
2. **Host到Device下发**：`AlgResourceCtxSerializable::Serialize`把topoInfo字节追加在尾部，并用
   `topoInfoSeqSize`定位。

不存在跨rank的topo字节交换，也不存在与不同版本HCCL对端的协商需求。

### 8.2 处理方式

`physicalLevels`直接加入`Serialize/DeSerialize`尾部：

```text
现有全部字段（顺序、编码方式完全不变）
physicalLevelNum
physicalLevels[]   逐级：localRanks / view / partitionSizes / ref.netLayer / ref.topoInstId /
                          hasTopoInst / topoType / locType / protocols / portNums / endpoints
deviceFormFactor   （声明在标量区，但编码在最尾部，见4.3）
```

规则：

1. 既有字段的编码顺序和方式一律不变，新字段只追加在最后；
2. `physicalLevels`必须与其他字段一起序列化。若只保留在首次构建的对象里，第二个算子从EngineCtx反序列化
   得到的将是空视图，TopoMatch会稳定失败；
3. `PhysicalLevelInfo`按**字段**编码，不使用结构体裸拷贝跨Host/Device传输——它含多个`std::vector`，
   走`BinaryStream`的泛型重载会把堆指针写进字节流。`EndpointDesc`本身是POD，可整体编码，由
   `static_assert(std::is_trivially_copyable<EndpointDesc>::value)`守护；
   每个Level的编码顺序见上方框图，共11项；
4. `AlgResourceCtxSerializable`侧无需改动：`topoInfoSeqSize`随之变化，尾部定位逻辑仍然成立；
5. 不引入magic、版本号或能力协商——没有跨版本对端，这些机制在本场景内不产生收益。

### 8.3 已知问题：截断流的反序列化行为（本期暂不处理）

`BinaryStream::operator>>(std::vector<T>&)`的实现是：

```cpp
size_t size;        // 未初始化
*this >> size;      // 流已耗尽或failbit置位时，stream.read不写入size
vec.resize(size);   // 用栈上的垃圾值resize
```

因此**任何导致topo字节流被截断的情况**（Device侧ctx缓冲短于Host实际写入、memcpy截断、`topoInfoSeqSize`
与尾部实际长度不一致）都会让`resize`拿到未初始化值，抛出`length_error`/`bad_alloc`且无人捕获。

这不是本次改动引入的问题：现有的`netLayers`、`instSizeListOfLayer`、`ranksInTopo`走的是同一条路径，
`physicalLevels`只是增加了触发站点的数量。**本期不修**，记录在此以免后续排查时误判为新特性引入。

若后续处理，可选方案（按侵入性排序）：读完计数后判一次`stream`状态并终止；或把新增计数字段读成显式
初始化为0的`u32`；或在`BinaryStream`层面把`size`初始化为0并在`resize`前校验剩余字节数。最后一个方案
能一次性覆盖全部既有字段，但会改变现有字段的行为，需要单独评估。

### 8.4 序列化体积

新字段的增量以`EndpointDesc`为主而不是`localRanks`：

| 组成 | 单位大小 | 说明 |
|------|----------|------|
| `EndpointDesc` | **160字节** | `protocol`4 + `CommAddr`40 + `EndpointLoc`64 + `raws`52（`hcomm_res_defs.h`） |
| `localRanks` | 4字节/rank | 每个Level一份 |
| `partitionSizes` | 4字节/Instance | 仅`GLOBAL`级，长度等于该层Instance数，通常个位数 |
| `protocols` / `portNums` | 4字节/项 | 均为个位数长度，可忽略 |
| `deviceFormFactor` | 4字节 | 整个结构一份 |

**合一之后Level数明显变少**（7.2的超节点场景从6级降到3级），但**Endpoint总量不变**——它按TopoInstance计，
合并只改变它挂在哪一级。按3个Level（各4个Endpoint）估算：Endpoint部分约1.9KB，`localRanks`部分在
32 rank下约230字节，其余字段合计不足100字节；`ENDPOINT_NUM_SANITY_LIMIT`取64时单级最坏10KB。
**Endpoint仍是主导项**，容量评估以它为准。

两点可优化（本期不做，记录备选）：

- 全覆盖Level的`localRanks`恒等于`{0, 1, ..., userRankSize-1}`，是O(N)的纯冗余，可用一个标志位替代；
- `EndpointDesc`尾部的`raws[52]`在当前实现中未被赋值，序列化的是未初始化字节。它不影响往返一致性，
  但会让"相同拓扑的字节流是否相同"这类断言不成立，写测试时需要按字段而不是按字节比较。

作为参照：`topoInstDetailsOfLayer.ranksInTopo`今天已经在下发`O(N × 实例数)`的数据，基线本就不低，
新字段不改变量级。实施时在Device下发用例中实测并记录基线（见11.2第6条）。

## 9. 失败语义

**新增标准化步骤的任何失败一律降级，不改变`CalcTopoShape`的返回值。**

| 失败场景 | 处理 |
|----------|------|
| `GetRanksByLayer`、`GetRanksByTopoInst`查询失败 | 整体降级 |
| 两次查询结果不一致（如本地NetInstance大小对不上） | 整体降级 |
| `GetTopoType`失败或返回预期外类型 | 整体降级（排序第3键失去依据，见5.3） |
| rank集合重叠但不形成范围链 | 整体降级 |
| 非连续或无法证明一致的Instance布局 | 整体降级 |
| Source或rank ID非法（数据自相矛盾） | 整体降级 |
| `GetTopoInstsByLayer`返回**失败** | 整体降级（见5.2：910_95上该分支不可达） |
| `GetTopoInstsByLayer`成功但`topoInstNum == 0`（该层无TopoInstance） | **不是失败**：该层出一个`hasTopoInst = false`的级 |
| Endpoint接口返回失败或数量不合理 | 局部降级：该级`endpoints`/`locType`/`protocols`/`portNums`为空 |
| 端口数查询失败或取到0/超限 | 局部降级：该级`portNums`**整体**清空（全有或全无，不留残缺数组） |
| `aclrtGetDevice` / `aclrtGetLogicDevIdByUserDevId` / `aclrtGetDeviceInfo` 任一步失败 | 局部降级：`deviceFormFactor`停在`UNKNOWN`，`physicalLevels`不受影响 |

- **整体降级**：记录WARNING日志，`physicalLevels`保持为空，`CalcTopoShape`返回`HCCL_SUCCESS`。全部现有
  字段照常输出，未迁移的Selector、Matcher和Executor完全不受影响。
- **局部降级**：Level照常生成，仅链路属性为空。建链阶段按
  [物理建链设计](./link-binding-refactor-design.md)7.2处理，Topo解析不因此清空整个视图。
  `portNums`的局部降级必须是**该级整体**清空而不是跳过失败项：残缺数组会让消费侧算出一个偏小的
  总端口数，这种错误比空数组难查得多——空数组至少能让消费侧明确识别为不可用并走保守分支。

新增步骤不返回`HCCL_E_PARA`或`HCCL_E_INTERNAL`：`userRankSize`、`myRank`等参数合法性在此之前的现有步骤
中已经保证，标准化阶段再次校验只用于判定是否降级。

之所以不把RankGraph查询失败提升为致命：现有`ExtractTopoDetails`对`HcclRankGraphGetTopoInstsByLayer`
的返回值本来就未做检查，把新步骤的查询失败变成致命错误会让原本可用的通信域起不来，与"绝不影响旧路径"
的目标冲突。

`HCCL_E_NOT_SUPPORT`不在本阶段返回。是否支持由消费侧判断：TopoMatch和LinkBindingResolver在
`physicalLevels`为空或不足以构造目标形态时返回`HCCL_E_NOT_SUPPORT`。

`physicalLevels`在临时对象中构建，全部校验通过后再赋值；降级或失败时保持为空。

## 10. 实施步骤

1. 新增PhysicalLevel相关辅助结构与`DeviceFormFactor`（不含任何派生量、不含`PhysicalSourceType`）；
2. 在`CalcTopoShape`末尾追加`CalcDeviceFormFactor`与`BuildPhysicalLevels`；
3. 逐netLayer合一NetInstance与TopoInstance：同范围合并、更细单独成级、无TopoInstance置标志位；
4. Endpoint快照按4.2稳定键排序，`locType`/`protocols`从中提炼，`portNums`按`commAddr`去重到iface粒度；
5. 实现`NormalizePhysicalLevels`与`ValidatePhysicalLevels`，含三键排序、降级路径与第6章不变量；
6. 在`TopoInfoWithNetLayerDetails`尾部追加`physicalLevels`与`deviceFormFactor`并接入`Serialize/DeSerialize`；
7. 补充EngineCtx缓存往返与Host/Device下发的序列化用例；
8. 为TopoMatch、LinkBindingResolver与性能建模提供只读输入；
9. 保留全部原有字段及其提取逻辑。

## 11. 测试与验收

### 11.1 标准化测试

| 输入 | 预期 |
|------|------|
| 7.1的2台Server × 8卡 | **2个Level**，均为`GLOBAL`，`partitionSizes`为`{8,8}`/`{16}` |
| 7.2的超节点32卡 | **3个Level**，`partitionSizes`为`{8,8,8,8}`/`{16,16}`/`{32}`，最高级`locType == HOST` |
| 7.3的SE完整形态 | 4个Level，`[0]`为`LOCAL`、其余为`GLOBAL`；`[2].protocols == {ub_ctp, ub_mem}`且`[2].portNums == {8}` |
| TopoInstance16 CLOS、NetInstance16（同rank集合） | **1个Level**（合一），`GLOBAL` + 带`topoType`/`endpoints` |
| 同范围的Mesh与CLOS两个TopoInstance，输入顺序颠倒 | 输出恒为Mesh在前（第3键；守护"不得用枚举值"），且**两级都带`partitionSizes`** |
| 有TopoInstance的级其`topoType`为`RESERVED`或其它预期外类型 | 返回`HCCL_E_NOT_SUPPORT`，整体降级 |
| 某netLayer的`topoInstNum == 0` | 该层出一个`hasTopoInst == false`的级，其余链路属性全为无效值，**不**整体降级 |
| 一个iface跑两种协议 | `protocols`两项、`portNums`**一项**（守护按iface去重；逐endpoint统计会得到两项） |
| 某条endpoint的端口数查询失败 | 该级`portNums`**整体为空**，而非只少一项 |
| 端口数取到0或超`PORT_NUM_SANITY_LIMIT` | 该级`portNums`整体为空并告警 |
| 同一级各endpoint的`locType`不一致 | `locType == RESERVED`并告警，其余字段照常 |
| size相同、rank集合不同且互不包含 | 返回成功，`physicalLevels`为空 |
| RankGraph返回顺序变化 | 输出不变 |
| `GetEndpointDesc`返回顺序变化 | `endpoints`/`protocols`/`portNums`排序后输出不变 |
| `instSizeList`按`[4,16]`顺序返回 | `partitionSizes == {16,4}` |
| 全等统计`[8,8,8,8]`与非全等统计`[8,24]` | 二者`partitionSizes`不同且原样保留（GCD都是8，若只存派生量则不可区分） |
| `[16,4]`与`[12,8]`两组输入 | `partitionSizes`不同（二者gcd与是否全等均相同，仅凭派生量无法区分） |
| `partitionSizes`含0项 | 返回`HCCL_E_NOT_SUPPORT`，整体降级（不变量4b） |
| `localRanks.size()`不在`partitionSizes`中 | 返回`HCCL_E_NOT_SUPPORT`，整体降级（不变量6） |
| `view`为`LOCAL`/`GLOBAL`之外的值 | 返回`HCCL_E_NOT_SUPPORT`，整体降级（不变量3a） |
| `ref.netLayer == INVALID_UINT` | 返回`HCCL_E_NOT_SUPPORT`，整体降级（不变量3b） |
| `hasTopoInst == false`但带了任一链路属性 | 返回`HCCL_E_NOT_SUPPORT`，整体降级（不变量3c） |
| `portNums.size() > endpoints.size()` | 返回`HCCL_E_NOT_SUPPORT`，整体降级（不变量3c） |
| 两个Level的`(netLayer, topoInstId)`相同 | 返回`HCCL_E_NOT_SUPPORT`，整体降级（不变量13） |
| `LOCAL`级 | `partitionSizes`为空 |
| 同rank集合的两级相邻 | 链校验通过（守护"允许相等"） |
| 非连续NetLayer编号 | `ref.netLayer`保留实际编号 |
| 同层2D Mesh（x/y互不包含） | 返回成功，`physicalLevels`为空 |
| 构建`partitionSizes`后`instSizeListOfLayer`未被重排 | 该字段与调用前逐元素相等 |

整机形态：

| 输入 | 预期 |
|------|------|
| ACL返回0/1/2/3 | `deviceFormFactor`为`POD`/`A_K`/`A_X`/`PCIE_CARD`；`IsPodForm()`仅在0时为true |
| ACL返回4~7（RSV/装备/EVB） | `deviceFormFactor == UNKNOWN`并告警，**不得**落到`POD` |
| `aclrtGetDevice` / `aclrtGetLogicDevIdByUserDevId` / `aclrtGetDeviceInfo` 任一失败 | `deviceFormFactor == UNKNOWN`，`CalcTopoShape`仍返回`HCCL_SUCCESS` |
| `userDevId != logicDevId`（模拟`ASCEND_RT_VISIBLE_DEVICES`） | 取到的是**logicDevId**对应的形态；默认部署下两者相等，这是唯一能测出设备号用错的用例 |
| 反序列化一个`physicalLevelNum`超限的字节流 | 整段丢弃且`deviceFormFactor`停在`UNKNOWN`（尾部追加的必然代价，见4.3） |

### 11.2 兼容测试

1. 重构前后所有现有Topo字段Golden完全一致，降级场景同样一致；
2. `TopoInfo`、`NetLayerDetails`和`TopoInstDetails`定义不变；
3. `TopoInfoWithNetLayerDetails`原有成员声明顺序不变，新字段在尾部；
4. 序列化字节中既有字段部分保持不变，新字段只出现在尾部；
5. **EngineCtx缓存往返用例**：第一个算子构建、第二个算子从缓存重建后，`physicalLevels`内容一致；
6. `AlgResourceCtxSerializable`往返后Device侧`physicalLevels`与Host侧一致；
7. 降级场景下缓存往返仍然成功，`physicalLevels`保持为空。

### 11.3 验收标准

1. `CalcTopoShape`继续输出全部现有Topo字段；
2. `physicalLevels`只作为新增尾部成员存在，并随现有序列化一同传递；
3. 同范围的NetInstance与TopoInstance合一为一个Level，该Level同时具备全局分区与链路属性；
   同范围但形态不同的多个TopoInstance各自成级，不互相覆盖；
4. TopoInstance和NetInstance来源及Endpoint能力不丢失，且Endpoint快照顺序稳定；
4c. 每个有TopoInstance的级都能给出`topoType`、`locType`、`protocols`与`portNums`；
    `portNums`按iface去重，多协议iface不重复计数；
4d. `deviceFormFactor`取到时如实反映本卡形态，取不到时为`UNKNOWN`且绝不落到`POD`；
4b. `partitionSizes`原样保留完整列表，使消费侧能区分GCD相同的不同分区（`[8,8,8,8]` vs `[8,24]`）；
5. 新增标准化逻辑不调用`GetLinks`，不缓存rank对Link（`InitRankInfo`中已有的
   `GetPairLinkCounter`等既有逻辑不在本次改造范围内）；
6. 标准化失败时降级为空视图，现有序列化行为和旧执行路径保持可用；
7. HCCL未引入HCOMM私有头或编译期依赖；
8. 相关UT和Host/Device序列化测试通过。

## 12. 风险与开放问题

### 12.1 风险

- 现有字段与`physicalLevels`不一致：只允许Normalizer单向生成并增加一致性校验；
- Endpoint实验接口不可用：按第9章降级，精确TopoInstance场景由建链阶段返回不支持；
- 当前rank局部TopoInstance视图不一致：增加全rank范围链测试。注意没有全局分区（`partitionSizes`为空）
  的Level其维度只能来自纯本地量（`localRanks.size()`），TopoMatch是纯函数、运行期无法发现各rank取值
  不一致，症状是**挂死而不是报错**。缓解手段见TopoMatch设计§5.2的交叉校验规则；
- 序列化体积增加：见8.4，主导项是`EndpointDesc`（160字节/条）而非`localRanks`，需在Device下发用例中
  确认实际ctx大小；
- 构建`partitionSizes`时就地排序污染现有字段：必须先拷贝`instSizeListOfLayer[layer]`再排序，见5.2，实现方案的Review清单单列一条；
- **`portNums`按endpoint而不是按iface统计**：多协议iface会被重复计入，总端口数翻倍。这是个静默错误——
  数值看着合理、校验全过、只有和实际带宽对不上时才会暴露。守护手段是"一个iface两种协议"的专项用例；
- **`deviceFormFactor`把`UNKNOWN`写成0**：0就是`POD`，所有取不到的场景都会被读成POD机型，
  而POD是消费侧要走2:1收敛特殊分支的那个值。守护手段是初值/降级值一律`INVALID_UINT` + `switch`白名单
  + `IsPodForm()`统一判定；
- **把`userDevId`当`logicDevId`传给`aclrtGetDeviceInfo`**：默认部署下两者相等，功能用例全绿；
  只在配了`ASCEND_RT_VISIBLE_DEVICES`或容器挂载部分设备的环境上才会读到另一张卡的形态，
  且返回的是合法值不报错。守护手段是必经`aclrtGetLogicDevIdByUserDevId`、变量名如实区分、
  INFO日志把两个id都打出来；
- **第3键漏给`GLOBAL`级用**：合并前它只服务`LOCAL`级，合一之后同范围的Mesh与CLOS都是`GLOBAL`，
  漏掉会让这两级定序退回兜底键，跨rank下标语义分叉。守护手段是7.4的专项用例。

### 12.2 开放问题

1. 哪些Endpoint扩展属性需要进入稳定契约（当前保存`EndpointDesc`本身，并提炼出`locType`/`protocols`/`portNums`）；
1b. `hasTopoInst == false`的层是否需要用`GetLinks`兜底补出`locType`/`protocols`——SE当前的结论是
    "设一个标志位注明即可"。若后续发现带host/roce属性的ranktable层恰好落在这类层上，
    "是否需要使用host网卡"就判不出来，届时需要重新评估；
1c. `ACL_DEV_ATTR_DEVICE_FORM_FACTOR`（409）在HCOMM全仓无使用者，其形态判断一律走
    `ACL_DEV_ATTR_MAINBOARD_ID`（407）再取`bit[7:5]`。二者读的是同一段硬件位域、取值逐一相等，
    但409在目标CANN/驱动版本上是否真的打通需要上板确认，详见
    [整机形态获取方案对比](./docs/design/device_form_factor.md)；
2. 任意非连续Instance布局是否需要HCOMM提供稳定坐标或全局成员查询；
3. 同层互不包含范围（2D Mesh）后续是否需要独立的"并行范围"表达，还是保持由旧Matcher承载。

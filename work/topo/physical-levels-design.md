# PhysicalLevel（topo-parse）设计与实现说明

对应提交：`topo parse`（`e5f707ad`）
涉及文件：

| 文件 | 作用 |
| --- | --- |
| `src/ops/op_common/inc/alg_param.h` | `PhysicalLevelInfo` 等数据结构、序列化 |
| `src/ops/op_common/topo/physical_level.h` | 对外声明与两个合理性阈值 |
| `src/ops/op_common/topo/physical_level_build.cc` | 依赖 `HcclComm`/RankGraph 的采集与合一 |
| `src/ops/op_common/topo/physical_level_normalize.cc` | 纯函数：归一、排序、校验（可离线 UT） |
| `src/ops/op_common/topo/topo_host.cc` | `CalcDeviceFormFactor` 与 `CalcTopoShape` 挂载点 |
| `src/ops/op_common/executor/executor_v2_base.*` | 消费侧两个访问器 |
| `src/common/adapter_acl.*` | `hcalrtGetDeviceInfo` 新增 `quiet` 参数 |

本文档承载源码中被压缩掉的详细论证。代码里只留结论，"为什么是这样"看这里。

---

## 1. 解决什么问题

RankGraph 给出的是两套彼此独立的原始视图：

- **NetInstance**：来自 ranktable，按 `netLayer` 分组的可连通范围，描述 server 间层级；
- **TopoInstance**：来自 topo 文件，某种链路形态下的实际互联范围，描述 server 内层级。

二者可能同层、可能同范围、也可能一大一小嵌套。改动前每处消费者都自己去调 RankGraph 重新解释一遍，同一个物理事实被翻译成了多种口径：有的按固定 `netLayer` 编号取，有的默认取 `links[0]`，有的把 rank 范围当成链路存在性。

`physicalLevels` 把这些原始视图整理成一条 **"当前 rank 可见的、从小到大的范围链"**，让下游（TopoMatch 算算法分层、LinkBindingResolver 选物理链路、cost model 建模）只读这一份标准化结果，不再各自去碰 RankGraph。

---

## 2. 数据结构

```cpp
enum class PhysicalLevelView : u32 { LOCAL = 0, GLOBAL = 1 };

struct PhysicalSourceRef {
    u32 netLayer    = INVALID_UINT;
    u32 topoInstId  = INVALID_UINT;
};

struct PhysicalLevelInfo {
    std::vector<u32>          localRanks;           // 当前 rank 在该范围内可见的全部 rank
    PhysicalLevelView         view;
    std::vector<u32>          instSizeListByLayer;  // 该 netLayer 的完整分区布局（仅 GLOBAL）
    PhysicalSourceRef         ref;
    // ---- 链路属性，随 hasTopoInst 一起生效 ----
    bool                      hasTopoInst = false;
    CommTopo                  topoType;
    EndpointLocType           locType;
    std::vector<CommProtocol> protocols;
    std::vector<u32>          portNums;
    std::vector<EndpointDesc> endpoints;
};
```

### 2.1 `view` 为什么不是"来源"标注

`view` 表达的是 **"这一级知不知道整个通信域在这个粒度上的完整划分"**。它不是标注习惯，而是 RankGraph 两组接口的**能力差异**，无法互相推导：

| 接口 | 看得到什么 | 可达的 view |
| --- | --- | --- |
| `HcclRankGraphGetInstSizeListByLayer` | 该 `netLayer` 上**全部** NetInstance 的大小 | GLOBAL |
| `HcclRankGraphGetTopoInstsByLayer` | 内部是 `GetNetInstanceByRankId(netLayer, myRank_)`，只看得到**当前 rank 所属** NetInstance 内的 TopoInstance | 只能 LOCAL |

兄弟 NetInstance 的拓扑结构从第二个接口根本拿不到。合一之后，`view` 与 `instSizeListByLayer` 是否为空严格等价（不变量 3d）。

### 2.2 `instSizeListByLayer` 是布局，不是 multiset

该字段**原样透传** `HcclRankGraphGetInstSizeListByLayer` 的返回序，不做任何重排。

返回序的语义是"按最小 rankId 升序的分区布局"。例如 5+3 拓扑中若 rank 0–4 在 5 卡 server，则为 `{5,3}`；若 rank 0–2 在 3 卡 server，则为 `{3,5}`。

**不重排的理由**：布局语义是 HCCL 主干既有的契约，不是本次新引入的假设——

- `topo_host.cc` 的 `CalcGroupIdx` 已经在用前缀和从这个列表定位 `serverIdx`；
- `GetCurrentServerStartRank` / `GetCurrentServerEndRank` 在其上求本 server 的 rank 区间。

重排（例如按大小降序归一）会把布局降级成一个只剩 multiset 语义的量，反而丢掉消费侧要用的信息，且与主干其余代码口径分叉。

**已知的接口约定缺口**：该顺序由 HCOMM 保证，但**未在 `hccl_rank_graph.h` 的接口注释中写明**，属于待补齐的接口约定。为此构建侧与校验侧**各有一次前缀和自检**（见不变量 6），不成立时整体降级为空。

### 2.3 不放派生量

结构里不放 `instSizeListByLayer` 的派生量（GCD、是否全等、分层维度、非对称判断等）。本结构只承载**提取到的拓扑事实**；分层维度与非对称判断属于 TopoMatch，由它按需自行计算。派生值存在这里既会与来源产生不一致的可能，也会把决策口径固化在提取侧。

同理，"是否需要使用 host 网卡"这个结论也不存——由消费侧看最高一级的 `locType` 是否为 HOST 自行推导。

### 2.4 `portNums` 的口径

- **取值来源**：`ENDPOINT_ATTR_BW_COEFF`。该属性名为"带宽系数"，但 HCOMM 侧实现就是 `iface->GetPorts().size()`，即 `portGroupSize`（见 HCOMM `rank_graph.cc` 的 `GetEndpointInfo`）。本仓 `op_common.cc` 建链时填 `channel.portGroupSize` 用的是同一个属性。
- **按 iface 去重，一条链路一项**：该层只有一条 8 口链路记作 `{8}`，两条链路记作 `{6,2}`，求和为本卡在该级的总物理端口数。
- **必须按 iface 去重而不是按 endpoint 计数**：一个 iface 有 N 种协议就有 N 个 `EndpointDesc`，且全部映射回同一个 iface（见 HCOMM `rank_graph_builder.cc` 的 `SetEndpointDesc`）。逐 endpoint 查会把同一条链路的端口数重复计入——一个跑 ub_ctp + ub_mem 的 8 口 iface 会被算成 16 口。
- **iface 身份用 `commAddr` 判定**：HCOMM 的 `endpointToIfaceMap` 正是以 `(commAddr, protocol)` 为键，同 addr 不同 protocol 必然指向同一个 iface。
- **不与 `endpoints` 保持下标对应**：`endpoints` 按 `EndpointDescLess` 定序，`portNums` 按端口数降序，两者是同一批 iface 的两种独立排列。
- **数值不去重**：两条都是 8 口的链路就是 `{8,8}`。去重的是 iface，不是端口数值。
- **局部量**：跨 rank 不保证相同——同一台机器上不同 die 的 rank 链路就可能不同（HCOMM 的 `TopoGetClosPort` 里 die0 是 4 个口、die1 是 2 个口）。
- **空表示未取到**，不是"0 个端口"。0 在采集侧就被判为不可信丢弃了。

---

## 3. 构建流程

挂载点在 `CalcTopoShape` 末尾：

```
ExtractNetLayerDetails
  → ExtractTopoDetails
  → ... 既有的各 Calc*
  → CalcDeviceFormFactor      (新增，只查本卡)
  → BuildPhysicalLevels       (新增，复用已提取的 netLayerDetails)
```

放最后的理由：现有字段的提取与派生逻辑完全不受影响，且可复用已提取的 `netLayerDetails`。

`BuildPhysicalLevels` 的内部顺序：

```
BuildPhysicalLevelCandidates   逐 netLayer 采集候选
  → NormalizePhysicalLevels    归一 → 三键排序 → 链校验
  → ValidatePhysicalLevels     不变量校验
  → 赋值给 topoInfo->physicalLevels
```

全程在临时对象中构建，全部校验通过后再赋值；降级或失败时 `physicalLevels` 保持为空。

只遍历 `netLayerDetails.netLayers` 里的 layer：一方面规避 HCOMM 对非法 layer 的抛异常分支，另一方面构造性地保证每个 Level 的 `ref.netLayer` 都来自 `GetLayers` 的实际结果。逐层收集的先后无所谓——`LevelLess` 是全序，不依赖输入顺序（输入顺序来自 RankGraph 的哈希遍历，本就不可依赖）。

---

## 4. 合一规则

对每个 `netLayer L`：

1. 取该层本地 NetInstance 的 rank 集合与全层分区；
2. 取该层含当前 rank 的每个 TopoInstance；
3. 按 rank 集合关系分派：

| TopoInstance 的 rank 集合 | 处理 | 结果 |
| --- | --- | --- |
| 与 NetInstance **相同** | 与之合并成一个 Level | 既有分区又有链路属性，`view = GLOBAL` |
| 比 NetInstance **小** | 单独成 Level | 无全局分区可言，`view = LOCAL` |
| 该层**没有任何** TopoInstance | NetInstance 单独成 Level | `hasTopoInst = false`，链路属性全部无效 |

除 `netLayer 0` 外每层只有一个 TopoInstance，因此合并是确定的。`netLayer 0` 在特殊机型上可能挂多个（典型是同范围的一个 Mesh 一个 CLOS），此时按"有多少写多少"各出一个 Level，由 `topoType` 区分。

同层多个同范围的 TopoInstance **各自都持有**这份分区——分区是该层的全局事实，不专属于其中某一个形态。

### 4.1 一个完整例子

4 台 Server，每 2 台组成一个超节点 = 32 卡，server 内 8 卡再分 2 个 4 卡 Mesh，rank 0 视角：

原始：

```
NetInstance   netLayer0 = [8,8,8,8]   netLayer1 = [16,16]   netLayer2 = [32]
TopoInstance  L0 = {0..3} Mesh 与 {0..7} CLOS
              L1 = {0..15} CLOS
              L2 = {0..31} CLOS
```

标准化后：

| idx | localRanks | view | instSizeListByLayer | ref | locType | protocol | portNums |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | {0..3} | LOCAL | {} | L0 / Mesh | device | ub_ctp | {1} |
| 1 | {0..7} | GLOBAL | {8,8,8,8} | L0 / CLOS | device | pcie | {4} |
| 2 | {0..15} | GLOBAL | {16,16} | L1 / CLOS | device | ub_ctp | {8} |
| 3 | {0..31} | GLOBAL | {32} | L2 / CLOS | host | roce | {1} |

**Level 下标与 `netLayer` 编号没有固定对应关系**：一个 `netLayer` 可能贡献一级（只有同范围 TopoInstance）、两级（还有更细的 TopoInstance），或在没有 TopoInstance 时只贡献一个无链路属性的级。要拿 `netLayer` 必须经 `ref.netLayer` 回查，不能拿下标当层号用——这正是 `GetPhysicalLevelNetLayer` 存在的理由。

---

## 5. 排序：三键 + 两个兜底键

`LevelLess`（`physical_level_normalize.cc`）：

| 序号 | 键 | 说明 |
| --- | --- | --- |
| 1 | `localRanks.size()` 升序 | 当前 rank 在该级的块大小。GLOBAL 级的 `localRanks` 就是本地 NetInstance 的 rank 集合，其 size 等于 `localNetInsSizeOfLayer[layer]`，与 LOCAL 级同量纲，可直接比较 |
| 2 | `view`：LOCAL(0) 在前，GLOBAL(1) 在后 | |
| 3 | `topoType` 定序 | 仅在前两键全部打平时生效（`netLayer 0` 同范围的 Mesh 与 CLOS）。顺序按互联紧密度递减：1DMESH 在 CLOS 之前 |
| 兜 1 | `localRanks` 字典序 | 正常输入上永不决定顺序 |
| 兜 2 | `(ref.netLayer, ref.topoInstId)` | 使比较器成为全序 |

### 5.1 `topoType` 的定序不能用枚举值

`hccl_rank_graph.h` 里 `COMM_TOPO_CLOS = 0 < COMM_TOPO_1DMESH = 1`，与所需顺序**正好相反**。因此单独写 `TopoTypeOrder`：

```
COMM_TOPO_1DMESH → 0
COMM_TOPO_CLOS   → 1
无 TopoInstance  → 2   (TOPO_TYPE_ORDER_NO_TOPO_INST)
其它             → 返回 false，整体降级
```

`TOPO_TYPE_ORDER_NO_TOPO_INST` 的取值必须与 `TopoTypeOrder` 的输出空间不重叠，否则两类 Level 会在第三键上打平，定序退回兜底键。

合一之后，第三键对 GLOBAL 级同样必须参与——`netLayer 0` 上"同一 NetInstance 同时挂 MESH 和 CLOS 且两者 rank 集合相同"时，这两级都带全局分区（`view` 同为 GLOBAL），前两键全部打平，定序**完全依赖**这一键。不能像合一前那样只给 LOCAL 级用。

### 5.1.1 `topoType` 必须走按 topoInst 的接口

RankGraph 有两个查 `topoType` 的接口，**返回值口径不同**，取错会让整个 physicalLevels 静默降级为空：

| 接口 | 查的是 | A5 上 Mesh 层的返回值 |
| --- | --- | --- |
| `HcclRankGraphGetTopoTypeByLayer(comm, netLayer, &t)` | `NetType`（ranktable 给出的层类型） | `COMM_TOPO_CUSTOM` |
| `HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &t)` | `Hccl::TopoType`（topo 文件给出的实例形态） | `COMM_TOPO_1DMESH` |

两者背后是**两个不同的枚举**：

```
NetType  : CLOS, MESH_1D, MESH_2D, A3_SERVER, A2_AX_SERVER, TOPO_FILE_DESC
TopoType : CLOS, MESH_1D, MESH_2D, A3_SERVER, A2_AX_SERVER, TOPO_TYPE_RESERVED
```

A5 的 Mesh 层由 topo 文件描述，`NetType` 就是 `TOPO_FILE_DESC`，按层查映射到 `COMM_TOPO_CUSTOM`；而按实例查的 `TopoType` 里根本没有 `TOPO_FILE_DESC` 这一项，落到 `MESH_1D → COMM_TOPO_1DMESH`。CLOS 层两条路径都返回 `COMM_TOPO_CLOS`，所以只有 Mesh 侧会露出差异。

`FetchTopoInstances` 用的是**按 topoInst** 的那个，与 `topo_match_ubx.cc`、`topo_match_multilevel.cc`、`topo_host.cc` 的 host-DPU 判定完全一致——`topo_match_ubx.cc` 的 MESH1DCLOS 分支正是靠它区分 `netLayer 0` 上同范围的 Mesh 与 CLOS 两个实例，也就是本文第三键要处理的那个场景。反过来，`topo_match_1d.cc` 与 `topo_match_concurrent.cc` 用的是按层的接口，它们判的条件写的就是 `COMM_TOPO_CUSTOM || COMM_TOPO_CLOS`。

若将来有人把这里改成按层查，`TopoTypeOrder` 会对 `COMM_TOPO_CUSTOM` 返回 `false`，`NormalizePhysicalLevels` 直接整体降级，`physicalLevels` 恒为空。失败是静默的（只有一条 WARNING），因此调用点写了注释钉住这一点。

### 5.1.2 A5 上 `topoType` 的实际取值

取值链：topo 文件的 `topo_type` 字符串 → `Hccl::TopoType` → `IRankGraph::GetTopoType` 的映射表 → `CommTopo`。

A5 上 `PhysicalLevelInfo::topoType` 只会是这三个值（日志 `DescribeLevel` 直接打数值，现场按数值认）：

| 值 | 含义 | 来源 |
| --- | --- | --- |
| `1` `COMM_TOPO_1DMESH` | Mesh 层的 TopoInstance | topo 文件 `topo_type: "1DMESH"`；另外单卡通信域被 HCOMM 强制置为 `MESH_1D` |
| `0` `COMM_TOPO_CLOS` | CLOS 层的 TopoInstance | topo 文件 `topo_type: "CLOS"`，**或该字段缺失时的默认值** |
| `-1` `COMM_TOPO_RESERVED` | 该 Level 没有 TopoInstance | `hasTopoInst == false`，由不变量 3c 强制保持此值 |

`COMM_TOPO_CUSTOM` **不会**出现在这个字段里——那是 5.1.1 里按层接口的 `NetType` 口径。

其余 `topo_type` 取值都会导致整体降级，两条不同的路径：

| topo 文件写法 | 失败点 | 结果 |
| --- | --- | --- |
| `"2DMESH"` | `IRankGraph::GetTopoType` 的映射表里没有 `MESH_2D`，返回 `HCCL_E_PARA` | `FetchTopoInstances` 返回 `HCCL_E_INTERNAL`，整体降级 |
| `"A3_SERVER"` / `"A2_AX_SERVER"` | 能取到 `COMM_TOPO_910_93` / `COMM_TOPO_A2AXSERVER`，但 `TopoTypeOrder` 不认 | `NormalizePhysicalLevels` 返回 `HCCL_E_NOT_SUPPORT`，整体降级 |
| 表外的任意字符串 | `EdgeInfo::GetTopoType` 直接 `THROW` | RankGraph 建不出来，轮不到本模块 |

注意第二行：这两个值本模块**取得到但排不了序**。A5 范围内碰不到，但这套东西若要上 A3，第三键的定序表必须先扩。

还要注意 `topo_type` 字段缺失时 HCOMM 静默按 `CLOS` 处理（只有一条 WARNING）。此时 Mesh 层会以 `COMM_TOPO_CLOS` 的身份进入本模块：排序不受影响（该层只有一个实例，没有同范围竞争），但消费侧读 `topoType` 建模会拿到错的形态。

### 5.2 为什么排序键缺一不可

任一键缺失时排序会退化到 RankGraph 的哈希返回序，**各 rank 排出的下标语义就不一致了**。而分叉之后每一级单看都合法（rank 集合真实、嵌套成立、含当前 rank），**没有任何本地校验拦得住**，最终表现为静默 hang。

这也是"有 TopoInstance 却取不到 `topoType` 必须整体降级、不能保留 RESERVED 继续"的原因。

### 5.3 为什么用 `sort` 而不是 `stable_sort`

`LevelLess` 在不同 `(netLayer, topoInstId)` 上是**全序**，等价元素只可能是同一个 Level，因此结果与输入顺序无关——这正是目的。输入顺序来自 RankGraph 的哈希遍历，本就不可依赖。

兜底键 2 的必要性：前面所有键都相同的两级并非同一个 Level。`(netLayer, topoInstId)` 才唯一确定一个 Level，此处仍可能是两个不同的 Level：

- 同一 rank 集合上挂两张同 `topoType` 的并行 Fabric；
- 两个 `netLayer` 的本地 NetInstance 恰好同范围（`instSizeListByLayer` 不同）。

少了这一键时二者在比较器下等价，而 `std::sort` 对等价元素的相对顺序是**未指定**的（且随输入顺序变化），各 rank 排出的下标语义会分叉。`netLayer` 与 `topoInstId` 都是跨 rank 一致的量，补上后比较器成为全序。不变量 8 把这条隐含前提落成受检不变量。

---

## 6. 不变量清单

`ValidatePhysicalLevels` 逐条校验。任一条不成立 → 返回 `HCCL_E_NOT_SUPPORT` → 调用方整体降级为空。

| # | 内容 | 不加会怎样 |
| --- | --- | --- |
| 1 | `userRankSize != 0` 且 `userRank < userRankSize`；`levels` 非空 | — |
| 2 | `localRanks` 非空、严格升序（等价于无重复）、无越界、含当前 rank | — |
| 3a | `view` 必须是有效枚举值 | 底层类型是 u32，任何 u32 值都是合法表示；不白名单则非法值会静默落进 else 分支被当成 GLOBAL |
| 3b | `ref.netLayer != INVALID_UINT` | 构建侧漏填时，消费侧回查原始对象会拿到错误的层 |
| 3c | `hasTopoInst` 与链路属性自洽 | 错开之后消费侧会把一堆无效值（`topoType=RESERVED`、`locType=RESERVED`、空协议集）当成真实链路事实建模 |
| 3d | `view == LOCAL` ⟺ `instSizeListByLayer` 为空 | 消费侧会把只知道本块的级当成全局分区来切算法 |
| 4 | GLOBAL 级 `sum(instSizeListByLayer) == userRankSize` | 每个 `netLayer` 都是对整个通信域的一次完整划分 |
| 4b | `instSizeListByLayer` 中无 0 项 | 0 能完整穿过其余检查——不改变求和，也不影响前缀和定位到的块——于是幽灵空分区会被当成一个真实 Instance 计入 |
| 5 | `localRanks.size()` ∈ `instSizeListByLayer` | 见 6.1 |
| 6 | 前缀和布局自检 | 见 6.1 |
| 7 | 相邻级大小非递减 + 包含链（允许相等） | 见 6.2 |
| 8 | `(ref.netLayer, ref.topoInstId)` 全域唯一 | 见 5.3 |

`hasTopoInst == true` 时另有三条子检查：

- `portNums.size() <= endpoints.size()`——`portNums` 按 iface 去重，条数不会超过 endpoint 数；超过说明去重逻辑坏了，而多出来的项会让消费侧把总端口数算大；
- 每个 `portNum` 非 0 且 `<= PORT_NUM_SANITY_LIMIT`——0 能完整穿过降序检查（排在末尾），于是一条不存在的链路会被当成真实出口计入；
- `portNums` 降序、`protocols` 去重升序——采集顺序来自 endpoints 的哈希序，不规范化则同一拓扑在不同进程下得到不同字节流。

`hasTopoInst == false` 时要求全部链路属性保持无效值。留一个"半有"的状态最危险——消费侧按 `hasTopoInst` 判定为不可用，却又能从字段里读出看似合理的值。

### 6.1 不变量 5 与 6：为什么是两条

不变量 6 是**布局自检**：`instSizeListByLayer` 是按最小 rankId 升序的分区布局，因此用 `userRank` 做前缀和必然落进某一块，且那一块的大小必须等于 `localRanks.size()`。

两个量来源**相互独立**——前者来自该 `netLayer` 的**全部** NetInstance，后者来自本 rank **所在的**那一个 NetInstance——对得上才说明"按最小 rankId 升序"这个布局假设在本层成立。这是 `instSizeListByLayer` 唯一一处能在本地验证的跨 rank 性质。

不变量 5（成员检查）逻辑上被不变量 6 蕴含，**单列是为了把两类失败分开**：

- 5 不成立 → "这个大小根本不存在"，数据本身对不上；
- 5 过了、6 不过 → "大小存在，但不在 rank 序定位到的那个位置"，即布局假设出了问题。

两条的修法完全不同，合成一条会让现场从日志里分不出是哪一种。

同样的前缀和自检在构建侧 `FetchNetInstance` 里也做了一次——那里对的是 `GetRanksByLayer` 刚返回的 `ranks.size()`，属于更早的拦截点。

### 6.2 不变量 7：为什么允许相等

合法的相等相邻对有两类：

- 同范围的 LOCAL 级与 GLOBAL 级；
- `netLayer 0` 上 rank 集合相同的 MESH 与 CLOS 两级（UBX 场景）；
- 两个 `netLayer` 的本地 NetInstance 恰好同范围。

互相重叠但互不包含的范围（典型为 2D Mesh 的 x/y 环）在此被拒绝——这类拓扑不构成范围链，本结构表达不了。

---

## 7. 降级策略：两级，不能混

这是本次设计最关键的一条纪律。

### 7.1 整体降级（返回 `HCCL_E_INTERNAL` / `HCCL_E_NOT_SUPPORT`）

适用于**参与结构与排序**的量：

| 量 | 失败时 |
| --- | --- |
| `localRanks`（`GetRanksByLayer` / `GetRanksByTopoInst`） | 整体降级 |
| `instSizeListByLayer`（含各项自检） | 整体降级 |
| `topoType`（`GetTopoType`） | 整体降级 |
| `GetTopoInstsByLayer` 本身失败 | 整体降级 |
| 任一不变量不成立 | 整体降级 |

理由：这些量决定**级数与级的相对顺序**。它们一旦不对，各 rank 的 Level 下标语义会分叉；而分叉后每一级单看都合法，本地校验拦不住，最终是静默 hang。宁可整条链降级为空、让消费侧走旧路径。

### 7.2 局部降级（该字段留空，链照常生成）

适用于 **payload 叶子**：

| 量 | 失败时 |
| --- | --- |
| `endpoints`（`FetchEndpoints`，返回 `void`） | 该级 endpoints 留空 |
| `locType` / `protocols`（`FetchLocAndProtocols`） | 保持 RESERVED / 空 |
| `portNums`（`FetchPortNums`） | 该级 portNums 整体清空 |

理由：TopoMatch 只读 `localRanks` / `view` / `instSizeListByLayer`，这三样不受影响。链路属性缺失不会让各 rank 排出的 Level 下标语义分叉，因此不成比例地整体降级。

`portNums` 内部是**全有或全无**：任一条取不到就整个清空。残缺数组会让消费侧算出一个"看着合理但偏小"的总端口数，这种错误比空数组难查得多——空数组至少能让消费侧明确识别为不可用并走保守分支。

### 7.3 最外层：永不改变 `CalcTopoShape` 的返回值

`BuildPhysicalLevels` **恒返回 `HCCL_SUCCESS`**。任何内部失败一律降级为空 vector，不是错误。

`physicalLevels` 为空时全部现有字段和旧执行路径保持可用，消费侧回退到旧 Matcher。因此**新增字段不会让原本能起来的通信域起不来**。

### 7.4 消费侧因此要判几次空

`hasTopoInst == true` 只保证 `topoType` 有效。链路属性各自可能为空，消费侧要独立判：

1. `endpoints` 是否为空；
2. `portNums` 是否为空；
3. `locType` 是否为 `ENDPOINT_LOC_TYPE_RESERVED`；
4. `protocols` 是否为空。

任何一处漏判都会把降级值当成真实事实去建模。这四者是**独立**降级的——例如 `endpoints` 取到了但 `portNums` 因某条链路查询失败而整体清空，是完全可能的组合。

---

## 8. 跨 rank 一致性

| 量 | 性质 |
| --- | --- |
| `instSizeListByLayer` | **全局量**，同一 `netLayer` 上跨 rank 逐字节相同。本结构唯一可用的一致性锚点 |
| `localRanks` | **局部量**，跨 rank 必然不同（rank 0 看到 `{0..7}`，rank 9 看到 `{8..15}`）；但同一集合内的各 rank 看到的内容完全一致 |
| `topoType` / `view` / `ref` | 跨 rank 一致（由机型与配置决定） |
| `endpoints` / `portNums` / `locType` / `protocols` | **局部量**，跨 rank 不保证相同。消费侧若要做全域一致的决策需自行处理 |
| `isPod` | **局部量**，由本 rank 查本卡得到，异构组网下各 rank 可能不同 |

**级数的跨 rank 一致性靠外部契约，不靠链的形状。** 非对称场景（如 `netLayer0 = [16,4]`）下各 rank 的级数仍然相同，靠的是"同一 `netLayer` 上 TopoInstance 的种类结构由机型和配置保证一致"这条外部契约。契约不成立时（下层接口少返回一个实例、或通信域跨了机型）各 rank 的下标语义会分叉，须由跨 rank 校验兜住——**本地校验做不到**。

这也是 7.1 那条纪律的根据：凡是能影响级数的量，都不允许局部降级。

---

## 9. 序列化

### 9.1 尾部追加约定

`physicalLevels` 与 `physicalLevelNum` 必须放在结构体尾部，序列化也只能追加在尾部，否则会改变既有字段的字节偏移。

`isPod` 是个例外：**声明在标量区**（与其它整机属性放在一起可读性更好），但**序列化追加在 `physicalLevels` 之后**。这个不对称是有意的，守的是"只能追加在尾部"的约定。`DeSerialize` 侧必须保持同样的顺序。

### 9.2 `physicalLevelNum` 为什么独立成字段

对齐 `topoInstDetailsOfLayerSize` 的既有约定：反序列化侧先拿到一个可校验的上界，再据此决定是否 `resize`。

超过 `PHYSICAL_LEVEL_NUM_LIMIT`（= 10）时丢掉整个 physical level 段并 `return`。这会**连带丢掉后面的 `isPod`**，使其停在 `false`——这是尾部追加的必然代价：流是纯位置流，跳过了变长的 `physicalLevels` 段就无法定位其后的字段。可以接受，因为两者的降级态都是"该字段不可用"，消费侧本就必须处理。

`PHYSICAL_LEVEL_NUM_LIMIT` 的定位是"超过即判定字节流不可信"的**合理性阈值**，不是业务上限。

### 9.3 `static_assert(std::is_trivially_copyable<EndpointDesc>::value)`

`endpoints` 的序列化走的是 `BinaryStream` 的 vector 重载 → 对每个元素调用泛型 `operator<<`，而泛型重载的实现是 `stream.write(reinterpret_cast<const char*>(&t), sizeof(T))`——**整块裸拷贝**。

这对当前的 `EndpointDesc` 是正确的：它是纯 POD（`protocol` + `CommAddr` + `EndpointLoc` + `raws`，共 160 字节），拷贝这 160 字节就是拷贝它的全部内容。

会出事的例子——假设某天 HCOMM 把地址字段从定长数组改成变长容器：

```cpp
typedef struct {
    CommProtocol protocol;
    std::string  commAddrStr;   // 原来是 CommAddr commAddr;
    EndpointLoc  loc;
} EndpointDesc;
```

此时 `sizeof(EndpointDesc)` 仍是个固定值，裸拷贝仍然"能编过、能跑"，但写进字节流的是 `std::string` 内部那根**指向堆的指针**，而不是字符串内容。后果：

- Host 侧 EngineCtx 缓存往返：反序列化出来的 string 指向一块可能已被释放的堆内存；
- Host → Device 下发：那根指针是 Host 虚拟地址，在 Device 侧根本无效。

两种情况都不会在序列化时报错，而是在很久之后的某次访问上随机崩溃，现场离根因隔了两层。

加上这条 `static_assert` 后，上面那个改动会在编译 `alg_param.h` 时直接失败并给出明确提示，迫使改动者去把 `endpoints` 改成手写字段级编码，而不是等到运行期踩雷。

### 9.4 `EndpointDescLess` / `CommAddrEqual` 为什么按字段比较

两者都**不用 `memcmp` 整个结构体**：尾部 `raws` 在 HCOMM 侧从未赋值，比较未初始化字节会得到不稳定结果。

`GetEndpointDesc` 内部遍历的是 `unordered_map`，输出顺序是哈希序，既不稳定也不跨进程一致，必须归一化后再保存。

---

## 10. `isPod` 与 `CalcDeviceFormFactor`

### 10.1 为什么是 `bool` 而不是枚举

只判是不是 POD，不保留 A_K / A_X / PCIE_CARD 的区分：消费侧（cost model）当前只用"POD ⇒ 交换机层 2:1 收敛"这一条，其余形态对建模没有差异。将来若要按形态细分，再把原始取值一并存下来。

判定必须是**严格相等的正向判断**：

```cpp
topoInfo->isPod = (val == ACL_DEVICE_FORM_FACTOR_POD);
```

不要写成"非其它形态即 POD"之类的反向判断——ACL 将来新增形态时，未识别的取值必须落到 `false` 一侧。

### 10.2 恒返回 `HCCL_SUCCESS`

该字段是纯附加信息，取不到时停在 `false`，现有字段与旧执行路径完全不受影响，不应该因为它让通信域起不来。这与同文件里其它 `Calc*` 的失败即返回是**有意的区别**——那些是算法分层的输入，这个不是。

"取不到"与"取到了但不是 POD"**都落 `false`**，该字段不区分这两种情形。现场要分辨走的是哪一种，看 `CalcDeviceFormFactor` 的 INFO 日志（它把原始 `formFactor` 值一并打出）。

会取不到的现实场景：老驱动不支持该 infoType、容器内无权限、虚拟化设备。这些通常是整集群统一的，所以降级基本是全域同步发生的；但这是个**假设**，混合部署下各 rank 可能拿到不同结果，消费侧需自行处理。

### 10.3 三套设备号，不能混用

| 名称 | 含义 |
| --- | --- |
| `phyDevId` | 物理设备号，板上实际编号 |
| `userDevId` | 用户可见设备号，即 `aclrtSetDevice` / `aclrtGetDevice` 这一套，受 `ASCEND_RT_VISIBLE_DEVICES` 影响 |
| `logicDevId` | 逻辑设备号，驱动内部使用 |

`aclrtGetDeviceInfo`（以及底层的 `halGetDeviceInfo`）要的是 **`logicDevId`**：

- `ascend_hal_base.h` 对 `devId` 的说明是"除 `INFO_TYPE_MASTERID` 外，一律使用 logical device ID"；
- HCOMM 侧的 `CcuGetMainboardId` / `HrtGetMainboardId` 形参名就是 `deviceLogicId`，`hal_get_mainboard_id` 也是先把 phyId 转成 logicId 再调。

而 `aclrtGetDevice` 返回的是 **`userDevId`**，必须经 `aclrtGetLogicDevIdByUserDevId` 转换后才能传下去（HCOMM `communicator_impl.cc` 里同样是这么转的）。

**默认部署下两者恰好相等**，因此少这一步在大多数环境上"看起来是对的"；一旦配了 `ASCEND_RT_VISIBLE_DEVICES` 或容器只挂载部分设备，就会去读**另一张卡**的形态，而且读到的是一个**合法值**——不会报错，只会静默拿到错的机型。

日志里 `userDevId` 与 `logicDevId` 都打出来：二者不等时正是配了 VISIBLE_DEVICES 之类的场景，现场需要看得到。

### 10.4 编译期特性开关 `HCCL_SUPPORT_DEV_FORM_FACTOR`

`ACL_DEV_ATTR_DEVICE_FORM_FACTOR` 是新版 `acl_rt.h` 才有的 `aclrtDevAttr` 枚举值（本地 CANN 9.2.0 的 `acl_rt.h:688` 定义为 `409U`）。**CANN 9.0.0 的头文件里没有它**，直接引用会编译失败：

```
error: 'ACL_DEV_ATTR_DEVICE_FORM_FACTOR' was not declared in this scope
```

这正是 Jenkins `Compile_Ascend_ARM_monitor` 节点（装的是 CANN 9.0.0）上的构建失败原因——本地 9.2.0 能编过，线上编不过。

#### 为什么不用 `CANN_VERSION_NUM` 判版本

仓里已有 `CANN_VERSION(M, m, p)` 这套版本宏（`src/common/hcomm_dlsym/dlsym_common.h`），用它也能挡住。但要写 `>= CANN_VERSION(9, 2, 0)` 就得**先猜准该枚举是哪个版本引入的**——已知 9.0.0 没有、9.2.0 有，9.1.0 未知。猜保守了会在本来支持的版本上白白降级，而且这个判断和头文件的实际内容之间没有强制关联：`CANN_VERSION_NUM` 来自 CANN 包的 version 头，真正决定能不能编过的是**当前 include 路径上那份 `acl_rt.h`**。

#### 探测方式

枚举值对预处理器不可见，无法直接 `#ifdef`。但 `acl_rt.h` 在该枚举下方紧接着定义了一组**宏**，且这组宏只为它服务：

```c
    ACL_DEV_ATTR_DEVICE_FORM_FACTOR = 409U,   // device form factor (pod/server/pcie card)
    ...
} aclrtDevAttr;

// Device form factor returned by aclrtGetDeviceInfo with ACL_DEV_ATTR_DEVICE_FORM_FACTOR.
#define ACL_DEVICE_FORM_FACTOR_POD 0
#define ACL_DEVICE_FORM_FACTOR_A_K 1
#define ACL_DEVICE_FORM_FACTOR_A_X 2
#define ACL_DEVICE_FORM_FACTOR_PCIE_CARD 3
```

因此在 `adapter_acl.h`（`acl_rt.h` 之后）探测这个宏：

```cpp
#ifndef HCCL_SUPPORT_DEV_FORM_FACTOR
#ifdef ACL_DEVICE_FORM_FACTOR_POD
#define HCCL_SUPPORT_DEV_FORM_FACTOR 1
#else
#define HCCL_SUPPORT_DEV_FORM_FACTOR 0
#endif
#endif
```

这直接反映当前 include 路径上头文件的实际能力，不需要维护版本对照表。两者同批引入、且宏的注释明确写着它是该枚举的返回值，co-occurrence 是有依据的；万一将来不成立，失败方式是**编译报错**（响亮），不是静默取错值。

外层 `#ifndef` 留了**逃生口**：可以从命令行 `-DHCCL_SUPPORT_DEV_FORM_FACTOR=0` 强制关闭。这既是探测失效时的兜底，也让"老 CANN 分支"在新 CANN 机器上可被真实编译验证——不必只靠眼看。

#### 两个受保护的引用点

| 位置 | 保护方式 |
| --- | --- |
| `adapter_acl.cc` 的 `supportType` 白名单 | `#if HCCL_SUPPORT_DEV_FORM_FACTOR` 包住那一项 |
| `topo_host.cc` 的 `CalcDeviceFormFactor` 函数体 | `#if !HCCL_SUPPORT_DEV_FORM_FACTOR` 分支只打一条 INFO，`#else` 才是真正的查询 |

关闭时的行为：`isPod` 停在 `false`，即**一律按非 POD 建模**——与"查询失败"完全同一条降级路径，消费侧无需区分。

DFX 上两条分支**同为 INFO、同前缀**，现场 grep 一次 `[Topo][CalcDeviceFormFactor]` 就能分清走的是哪一条：

```
# 新 CANN，查询成功
[Topo][CalcDeviceFormFactor] userDevId[0] logicDevId[0] formFactor[0] isPod[1]
# 老 CANN，编译期就关掉了
[Topo][CalcDeviceFormFactor] acl has no device form factor attr, isPod stays false
```

### 10.5 `hcalrtGetDeviceInfo` 新增 `quiet` 参数

`quiet` 用于"取不到就降级"的可选属性：老驱动不支持某个 infoType 时会**稳定失败**，按 ERROR 打会在完全正常的老环境上持续刷错误日志，把真实故障淹掉。传 `quiet=true` 时改按 WARNING 打。

默认 `false`，保持所有既有调用方的行为不变。

**传 `quiet` 的调用方必须自己处理返回值并给出降级值**，否则失败会变得完全无声。

同时把 `ACL_DEV_ATTR_DEVICE_FORM_FACTOR` 加进了 `supportType` 白名单。

---

## 11. 消费侧访问器

放在 `InsCollAlgBase`（`executor_v2_base.h`）的 `protected` 区。选这个基类而不是 `ExecutorBase`：只有它拿得到 `TopoInfoWithNetLayerDetails`，`ExecutorBase` 只持有 `TopoInfo*`。

```cpp
u32 GetPhysicalLevelNetLayer(const TopoInfoWithNetLayerDetails* topoInfo, u32 levelIdx) const;
std::vector<u32> GetPhysicalLevelPortNums(const TopoInfoWithNetLayerDetails* topoInfo, u32 levelIdx) const;
```

`GetPhysicalLevelNetLayer` 存在的理由见 4.1：Level 下标不是层号。

**实现里刻意没用 `CHK_PTR_NULL`**：那个宏返回的是 `HcclResult`，在这个返回 `u32` 的函数里会被隐式转成一个**看起来像层号的小整数**（`HCCL_E_PTR` 是 3），静默当成 `netLayer[3]` 用。改成显式判空返回 `INVALID_UINT`。

`GetPhysicalLevelPortNums` 返回空有三种来源，从返回值上区分不了：下标越界、该级没有 TopoInstance 支撑、采集时局部降级。三者的共同含义是"端口数不可用"，**不表示该级有 0 个端口**——0 在采集侧就被判为不可信丢弃了。

---

## 12. DFX：关键日志

| 观测点 | 级别 | 搜索内容 |
| --- | --- | --- |
| 解析成功 | RUN_INFO | `[PhysicalLevel][Build]` + `physical levels` |
| 解析整体降级为空 | WARNING | `[PhysicalLevel][Build] normalize degraded` |
| 最终产物逐级明细 | RUN_INFO + INFO | `[PhysicalLevel][Build]` + `level[` |
| 整机形态取值 | INFO | `[Topo][CalcDeviceFormFactor]`（成功打 `userDevId`，老 CANN 打 `no device form factor attr`）|

**逐级明细打的是最终产物**——标准化、排序、校验全部走完之后真正落进 `topoInfo` 的内容。与 `BuildLayerCandidates` 里那条 DEBUG 不同，那条打的是**候选**，含后续会被剔除的级，故意留在 DEBUG。

RUN_INFO 与 INFO **各打一遍**：前者在默认日志级别下就可见，功能用例与上板验收直接搜得到；后者让这几行与同一批 INFO（如 `CalcDeviceFormFactor`）落在同一条时间线上，便于对照。

一行的格式：

```
[PhysicalLevel][Build] rank[0] level[1/3] rankNum[16] ranks[0..15] view[0]
  instSizeListByLayer[] ref[layer 0 inst 1] hasTopoInst[1] topoType[0]
  locType[0] protocolNum[2] portNums[8]
```

两个取舍：

- `localRanks` 只打首尾与个数，不整条展开。它的长度就是该级的 rank 数，顶层那一级等于整个通信域，万卡场景下整条打出来没人看得完。首尾足够辨认范围，升序与含 `myRank` 由校验侧保证。
- vector 展开上限 `LOG_VEC_MAX_ITEM = 16` 项，超出补 `,...`。大规模集群上 `instSizeListByLayer` 可能有上百项。

---

## 13. 两个合理性阈值

两者都是**异常告警阈值，不是防御性截断**。

### `ENDPOINT_NUM_SANITY_LIMIT = 64`

`GetEndpointNum` = (本 rank 在该 topoInst 上的接口数) × (每接口协议数)，是**纯本地量，不随 rankSize 增长**。协议数上界为 `CommProtocol` 的有效值个数（10 个），接口数经 `AddConnInterface` 去重后是个位数。

超过该量级只可能是 HCOMM 侧异常。此时**截断反而更糟**：`GetEndpointDesc` 会因缓冲不足返回 `HCCL_E_PARA`（它不截断而是报错），结果还是拿不到 endpoints，却把异常掩盖成"正常但数据少"。因此直接局部降级并告警。

### `PORT_NUM_SANITY_LIMIT = 64`

HCOMM 侧的 `MAX_PORT_NUM` 是 32（`topo_addr_info/src/rank_info_types.h`），驱动侧 UB 口上限是 36（`ascend_hal_base.h` 的 `HAL_UB_PORT_NUM`），一个 iface 实际持有的端口数远小于此。超过该量级说明 `ENDPOINT_ATTR_BW_COEFF` 返回的不是端口数，此时整个 Level 的 `portNums` 不可信。

`portNum == 0` 也判为不可信，口径对齐 `op_common.cc` 的 `BuildChannelInfo`——那里把 `portSize` 为 0 直接当作 `HCCL_E_INTERNAL`：一个真实存在的 iface 不可能持有 0 个端口。

---

## 14. 采集侧的几个易错点

### 14.1 HCOMM 返回的指针必须立即复制

`HcclRankGraphGetRanksByLayer`、`HcclRankGraphGetTopoInstsByLayer`、`HcclRankGraphGetRanksByTopoInst` 都只持有**一个成员 vector**，下一次调用会 `clear()` 并重填它。取到裸指针后必须立即 `assign` / 构造出自己的副本。

### 14.2 `GetEndpointDesc` 的 `descNum` 是回写量

`num`（`GetEndpointNum` 的结果）是实际写入条数的**上界**：Endpoint 按 `(addr, protocol)` 去重，而 `GetEndpointNum` 求和时不去重。按 `num` 开缓冲永远够，但必须以回写的 `descNum` 为准 `resize`。同时校验 `actualNum <= num`，超过则整体丢弃该级 endpoints。

### 14.3 `GetEndpointNum` 失败与 `num == 0` 要分开判

`num == 0` 是**合法结果**（当前 rank 在该 topoInst 上没有接口/协议），打 DEBUG。

而 `GetEndpointNum` 返回失败在 RankGraph 侧只有 `GetPeer(myRank_)` 为空这一条失败路径（`rank_graph.cc`），属于真异常，静默返回会让现场什么都不留，打 WARNING。

### 14.4 `GetTopoInstsByLayer` 返回 0 不是错误

`NetInstance::GetTopoInstsByLayer` 只是遍历 `topoInsts_` 这个 map，空 map 即返回 0 且不报错。意味着该层没有 endpoints 与 `topoType` 可供建链，调用方据此把 `hasTopoInst` 置 `false`。

### 14.5 不需要额外的能力探测

`FetchPortNums` 里 `endpoints` 为空有两种来源：该 Level 上本就没有接口，或 `FetchEndpoints` 已经降级。后者覆盖了"HCOMM 低版本没有这族符号"的情况——弱符号未命中时 `GetEndpointNum` 就已经失败了。所以不需要再引入 `HcommIsSupport*` 做一次能力探测（那会让 ST 多一处必须手写的桩）。

### 14.6 `FetchNetInstance` 里的跨调用一致性校验

`ranks.size() != details.localNetInsSizeOfLayer[layer]` 是**真实的**跨调用校验，不是自比。`localNetInsSizeOfLayer` 由 `ExtractNetLayerDetails` 中的**另一次** `GetRanksByLayer` 写入，此处是新发起的调用，两者不同源。不一致说明 RankGraph 在两次调用之间发生了变化，属于必须降级的场景。

### 14.7 `sum(instSizeListByLayer) == userRankSize` 是哨兵

正常路径永不触发：`ExtractNetLayerDetails` 已用同一等式做过 `CHK_RET` 并在不成立时返回 `HCCL_E_PARA`，而它在 `CalcTopoShape` 中先于本函数执行——等式不成立时通信域早已起不来，走不到这里。

保留它只为在 HCOMM 改变分层语义时第一时间暴露：`instSizeListByLayer` 是消费侧推导分层维度与非对称性的唯一输入，静默取到错误统计的代价远高于这几行。

注意这是"**全部** Instance"的和；当前 rank 所在的那一个由 `localNetInsSizeOfLayer` 校验，它才是 `<= userRankSize` 的那个量。

---

## 15. 构建配置

`physical_level_build.cc` 与 `physical_level_normalize.cc` 放在 `src/ops/op_common/topo/CMakeLists.txt` 的**无条件块**（不在 `if(NOT HCCL_CANN_COMPAT_850)` 里）：能力探测通过 support flag 与弱符号完成，不依赖版本宏。

`ACL_DEV_ATTR_DEVICE_FORM_FACTOR` 的可用性通过头文件探测而非版本宏判定，见 10.4。因此 `topo_host.cc` 与 `adapter_acl.cc` 都不需要新增 CMake 条件。

`physical_level_normalize.cc` 里的三个函数（`EndpointDescLess` / `CommAddrEqual` / `NormalizePhysicalLevels` / `ValidatePhysicalLevels`）**不依赖 `HcclComm` 与 RankGraph**，可离线 UT。归一步骤在构建侧已经做过一遍、正常路径上幂等，保留它就是为了让本函数不依赖调用方。

---

## 16. 已知开放问题

UBX 场景下（一个 Server 16 张卡、4 张卡组成一个 mesh），修改 ranktable 可以构造出 **NetInstance 比 TopoInstance 小**的情形：`netLayer 0` 的 NetInstance 是 4 卡，但同层能取到 4 卡和 16 卡两个 TopoInstance；而 `netLayer 1` 的 NetInstance 是 16 卡，其 TopoInstance 与 `netLayer 0` 那个 16 卡的内容相同。

当前代码在该场景下**校验全过、跨 rank 一致**——管线里没有任何地方假设 `TopoInstance ⊆ NetInstance`。真正的影响是 `serverNum` 会变成 4，以及那个 16 卡的级被拆成两个条目导致 `levels.back().locType` 可能是 RESERVED。

详细分析见 `docs/design/ubx_net_smaller_than_topo.html`。处理方式（是否按 `(localRanks, topoType)` 合并同范围级）尚未定稿。

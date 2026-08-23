# Topo信息解析与标准化 — 编码实现方案

- 文档状态：草案
- 更新日期：2026-08-17（按已落地实现修订：NetInstance与TopoInstance合一、新增链路属性与整机形态）
- 对应设计：[Topo信息解析与标准化](./topo-parse-refactor-design.md)
- 适用仓库：`cann/hccl`

---

## 1. 范围

只实现设计文档第4/5/6/8/9章：`physicalLevels`的结构定义、构建（含ranktable层级与topo层级合一）、
标准化、校验、序列化与降级，以及`deviceFormFactor`的提取。
不涉及TopoMatch、LinkBindingResolver与性能建模的消费逻辑。

完成标准：`physicalLevels`可被正确构建、正确往返序列化，且在任何输入下都不改变现有字段与旧执行路径的行为。

## 2. 实施前已确认的代码事实

这些事实决定了下面的方案，实施时若发现与实际不符需要先回到本节复核。

> **行号说明**：本表的行号锚定于`2ef7d69a [build] format all C/C++ files`之前的代码。该提交对全仓做了
> 格式化，多数引用会向后偏移20-50行（例如F5实际在`topo_host.cc:866-935`，F6实际在`:853-856`）。
> `op_common.cc:3502`（F12）是准的。实施前请按符号名重新定位，不要直接按行号跳转。

| # | 事实 | 位置 | 影响 |
|---|------|------|------|
| F1 | `BinaryStream`的泛型`operator<<`是`stream.write(&t, sizeof(T))`裸拷贝 | `src/common/binary_stream.h:35` | 含`std::vector`的新结构**绝不能**走泛型重载 |
| F2 | `vector<T>`重载逐元素递归；`map`重载已支持 | `binary_stream.h:43,94` | `vector<u32>`、`vector<EndpointDesc>`可直接用 |
| F3 | `DeSerialize`返回`void`，无法报错，现有代码用clamp防御 | `alg_param.h:271,310` | 新字段也只能clamp+清空+日志 |
| F4 | topo序列化有两条路径：EngineCtx缓存往返、`AlgResourceCtxSerializable`下发Device | `op_common.cc:1097`、`alg_param.h:506` | 新字段必须进`Serialize`，否则第二个算子拿到空视图 |
| F5 | `ExtractTopoDetails`未保存`topoInstId`，只保存了按`topoInstIdx`排列的三个数组 | `topo_host.cc:890-940` | 构建`PhysicalSourceRef::topoInstId`必须重新调`GetTopoInstsByLayer`，且**不能按下标与已有数组对齐**（见F14） |
| F6 | `ExtractNetLayerDetails`调用了`GetRanksByLayer`但只取`rankNum`，丢弃了rank列表 | `topo_host.cc:826-829` | 需要重新调用一次拿完整列表（见4.2） |
| F7 | 实验接口是弱符号，未加载时**返回`(HcclResult)(-1)`并打错误日志，不崩溃** | `dlsym_common.h:151-160`、`hccl_rank_graph_dl.cc` | **不做能力探测**：目标环境必然提供这些符号，返回值检查已经覆盖"未加载"这一路径（弱符号桩返回非SUCCESS，走既有的降级分支）。不引入`HcommIsSupportXxx()`调用，也不给`hccl_rank_graph_dl.h`补`DECL_SUPPORT_FLAG` |
| F8 | `GetEndpointNum`返回`Σ(本rank在该topoInst上的iface的协议数)`，是`GetEndpointDesc`实际写入条数的**上界**（Endpoint按`(addr, protocol)`去重、求和不去重）；缓冲不足时`GetEndpointDesc`返回`HCCL_E_PARA`不截断 | hcomm `rank_graph.cc:343-359`、`:398-446` | 按`num`开缓冲永远够；必须以回写的`descNum`为准resize |
| F9 | `CalGCD(u32,u32)`与`CalGCD(std::vector<u32>&)`已存在，声明在`topo.h` | `topo.h:30-31`、`topo.cc:44-51` | **本方案不使用**。派生量（GCD、是否全等等）已移出topo-parse，由TopoMatch自行计算；`physical_level_build.cc`因此也不再include`topo.h` |
| F10 | `alg_param.h`已`#include "hccl_rank_graph_dl.h"`，且被`-DAICPU_COMPILE`目标编译 | `alg_param.h:29`、`test/st/algorithm/utils/src/aicpu/CMakeLists.txt:303` | `EndpointDesc`/`CommTopo`在两侧都可见；结构体定义必须在AICPU下可编译 |
| F11 | ST的AICPU目标编译`topo.cc`与`topo_match_1d/_base/_multilevel/_ubx/_pcie_mix.cc`（**不含**`_concurrent`、`_ubx_1d`、`_3_level`、`_squeeze_2d`），且**不编译**`topo_host.cc` | `.../aicpu/CMakeLists.txt:189-194` | 新增的`.cc`是Host-only，不需进AICPU源表。注意AICPU源表是白名单而非通配，新增文件若需要在AICPU下编译必须逐个显式追加 |
| F12 | 现有`CheckHostDPUOnly`用VLA`EndpointDesc endPointDescs[endPointNums]` | `op_common.cc:3502` | 新代码用`std::vector`，不复制这个写法 |
| F13 | HCOMM为`GetLayers`/`GetRanksByLayer`/`GetInstSizeListByLayer`各持有**一个**成员vector，每次调用`clear()`后重填再返回`data()` | hcomm `rank_graph_interface.cc:78-174` | 同一接口的下次调用即失效，不同接口互不影响；逐层循环必须每层调用后立即复制 |
| F14 | `GetTopoInstsByLayer`返回的是**当前rank所在NetInstance内的全部TopoInstance**，且顺序是`unordered_map`哈希序 | hcomm `rank_graph.cc:292-296`、`net_instance.cc:147-154`、`net_instance.h:229` | 必须按成员关系过滤；**不能按下标与`ranksInTopo`对齐**，只能按`topoInstId`重新查询 |
| F14b | HCOMM公开头`hccl_rank_graph.h`对该接口的注释写的是"myRank**所在**的topoInstance集合"，与F14的实现行为相反 | hcomm `include/hccl/hccl_rank_graph.h` | 以实现为准（已核对`net_instance.cc:147-154`遍历整个`topoInsts_`，无过滤）。该背离需同步给HCOMM：若将来按注释"修正"了实现，本方案的过滤退化为无害的no-op，但依赖该行为的其他代码（如`ExtractTopoDetails`填充的`ranksInTopo`含兄弟实例）语义会变 |
| F14c | `NetInstance::GetTopoInstsByLayer`只`push_back`不`clear`，靠C封装层的`topoInstsVec_.clear()`兜住 | hcomm `net_instance.cc:147-154`、`rank_graph_interface.cc:380` | 当前经`HcclRankGraphGetTopoInstsByLayer`调用是安全的；不得绕过C接口直接调内部方法 |
| F14d | **layer0的两类TopoInstance注册范围不同**：peer2peer（直连Mesh）由`UpdateTopoInstForMyRankOnly`注册，内部`if (srcRankId != myRank_ && dstRankId != myRank_) continue;`，**只注册myRank所属的那一个**；peer2net（Fabric，如CLOS/PCIe-SW）由`AddTopoDescFabricInfo`为netInstance内**全部rank**注册，同一Fabric的所有rank共用一个topoInstId | hcomm `rank_graph_builder.cc:497-540`（调用点`:666`）、`:216-280` | 决定了F14的过滤在两类来源上性质不同：对layer0的Mesh是**防御性**的（HCOMM已只给myRank的），对Fabric是**必需的正确性步骤**（一个netInstance内挂两个PCIe switch时，rank0会看到自己不属于的那个Fabric实例）。**过滤逻辑不得依赖F14d**——它是HCOMM当前的构造实现，且与公开头注释（F14b）本就不一致 |
| F18 | `GetEndpointDesc`遍历的`endpointToIfaceMap_`是`unordered_map<pair<CommAddr,CommProtocol>, iface>`，**输出顺序是哈希序** | hcomm `net_instance.h:98`、`rank_graph.cc:415-441` | `endpoints`必须排序后保存，否则违反设计文档§6不变量11（见4.2） |
| F15 | `GetInstSizeListByLayer`的顺序是`vector<unordered_map<string, NetInstance>>`的哈希序 | hcomm `rank_gph.h:28`、`rank_graph.cc:277-290` | 只能用GCD等与顺序无关的聚合，不能按下标解释 |
| F16 | `GetLayers`、`GetRanksByLayer`、`GetRanksByTopoInst`的返回均源自`std::set`，天然升序 | hcomm `rank_graph.cc:242-255`、`net_instance.cc:169-180` | `SortUnique`在正常路径上是幂等的，保留它只为纯函数可独立测试与异常兜底 |
| F17 | `GetLocalInstRanks`在找不到netInstance时**抛异常**而非返回错误码；但C接口层先校验了`levels.find(netLayer)`并返回`HCCL_E_PARA` | hcomm `rank_graph.cc:242-248`、`rank_graph_interface.cc:139-142` | 只对`netLayerDetails.netLayers`中的layer调用即可避免走到抛异常分支 |
| **F19** | **一个iface有N种协议就生成N个`EndpointDesc`，键为`(commAddr, protocol)`，全部映射回同一个iface**；`GetEndpointInfo(ENDPOINT_ATTR_BW_COEFF)`返回的是`iface->GetPorts().size()` | hcomm `rank_graph_builder.cc`的`SetEndpointDesc`（`iface × protocol`双层循环）、`rank_graph.cc`的`GetEndpointInfo` | **端口数是iface的属性，不是endpoint的属性**。必须按`commAddr`去重后统计，逐endpoint累加会把多协议iface重复计入（`ub_ctp`+`ub_mem`的8口iface算成16口）。这是`portNums`实现的核心约束 |
| **F20** | 除netLayer 0外每层只有一个TopoInstance；netLayer 0在特殊机型上可能挂多个，且同范围时必然是"一个Mesh一个CLOS" | SE确认 | 合一规则因此是确定的；同范围的多个TopoInstance按"有多少写多少"各出一级，靠`topoType`定序 |
| **F21** | `ACL_DEV_ATTR_DEVICE_FORM_FACTOR = 409`与四个`ACL_DEVICE_FORM_FACTOR_*`宏存在于CANN 9.2.0；`ACL_DEVICE_FORM_FACTOR_POD == 0` | `acl_rt.h:686-699` | **`POD`就是0**，与`u32`默认值撞车，降级值必须用`INVALID_UINT`。9.2.0上的存在性已由`topo_host.cc`的`static_assert`编译通过证明；**8.5.0未验证** |
| **F22** | `hcalrtGetDeviceInfo`有一张`supportType`白名单，不在表内的属性直接返回`HCCL_E_NOT_SUPPORT`；该函数原本全仓无调用者 | `src/common/adapter_acl.cc` | 用新属性前必须先加白名单；本次同时给它加了`quiet`参数（默认false，不改现有行为）供降级路径降噪 |
| **F23** | ST只把`hcomm_dlsym`当**include目录**，不编译其中的`.cc`；每个`HcommIsSupportXxx`都要在`hccl_stub.cc`里手写桩 | `test/st/algorithm/utils/src/CMakeLists.txt`、`hccl_proxy/hccl_stub.cc` | 印证F7的结论：引入能力探测会打断ST构建，且桩返回false会让新特性在ST里全程不生效 |
| **F24** | **三套设备号互不相同**：`aclrtGetDevice`返回`userDevId`；`aclrtGetDeviceInfo`/`halGetDeviceInfo`要的是`logicDevId`（`ascend_hal_base.h`：除`INFO_TYPE_MASTERID`外一律用logical device ID）；转换用`aclrtGetLogicDevIdByUserDevId`。另：`aclrtGetLogicDevIdByPhyDevId`**名不副实，实际返回UserDevId** | `acl_rt.h:1614/5086/5106`、`ascend_hal_base.h:1310-1311`、hcomm `hal.c`的`hal_get_logicid_from_phyid`与`load_dcmi`注释、`communicator_impl.cc:3601` | 直接把`aclrtGetDevice`的结果传给`aclrtGetDeviceInfo`在默认部署下恰好正确（两者相等），**配了`ASCEND_RT_VISIBLE_DEVICES`或容器只挂部分设备时会静默读到另一张卡**。本仓`adapter_acl.cc`的`haclrtGetDeviceIndexByPhyId`用错误接口填了名为`deviceLogicId`的出参，目前无调用者，属潜在隐患 |

## 3. 交付物清单

### 3.1 新增文件

| 文件 | 内容 | 是否依赖comm |
|------|------|--------------|
| `src/ops/op_common/topo/physical_level.h` | 常量、函数声明 | — |
| `src/ops/op_common/topo/physical_level_normalize.cc` | 排序、链校验、Validate（**纯函数**） | 否 |
| `src/ops/op_common/topo/physical_level_build.cc` | RankGraph提取、编排、降级 | 是 |
| `test/ut/topo_physical_level/CMakeLists.txt` | UT构建 | — |
| `test/ut/topo_physical_level/physical_level_test.cc` | 纯函数UT | — |
| `test/ut/topo_physical_level/physical_level_serialize_test.cc` | 序列化往返UT | — |

拆成`normalize`与`build`两个`.cc`是为了让UT只编译前者，不需要链接RankGraph与`hccl_comm`。

### 3.2 修改文件

| 文件 | 修改 |
|------|------|
| `src/ops/op_common/inc/alg_param.h` | 新增结构体、常量、`DeviceFormFactor`与`IsPodForm()`；`TopoInfoWithNetLayerDetails`追加成员并扩展`Serialize/DeSerialize` |
| `src/ops/op_common/topo/topo_host.cc` | `CalcTopoShape`末尾追加`CalcDeviceFormFactor`与`BuildPhysicalLevels`两次调用；新增`CalcDeviceFormFactor`实现与守护ACL取值的`static_assert` |
| `src/ops/op_common/topo/topo_host.h` | 声明`CalcDeviceFormFactor` |
| `src/common/adapter_acl.cc/.h` | `supportType`白名单加`ACL_DEV_ATTR_DEVICE_FORM_FACTOR`；`hcalrtGetDeviceInfo`加`quiet`参数（默认false） |
| `src/ops/op_common/topo/CMakeLists.txt` | 追加两个新`.cc`到**无条件**`src_list` |
| `test/ut/CMakeLists.txt` | `add_subdirectory(topo_physical_level)` |

不修改`src/common/hcomm_dlsym/hccl_rank_graph_dl.h`：不做能力探测（见F7、F23）。

不修改：`ExtractNetLayerDetails`、`ExtractTopoDetails`、`NetLayerDetails`、`TopoInstDetails`、`TopoInfo`，
以及`AlgResourceCtxSerializable`。

## 4. 详细设计

### 4.1 数据结构（`alg_param.h`）

插入位置：`TopoInstDetails`定义之后、`TopoInfo`定义之前。

```cpp
// 该Level是否知道该netLayer的完整划分；合一后与partitionSizes是否为空等价
enum class PhysicalLevelView : u32 { LOCAL = 0, GLOBAL = 1 };

// netLayer恒有效；topoInstId仅在hasTopoInst时有效
struct PhysicalSourceRef {
    u32 netLayer   = INVALID_UINT;
    u32 topoInstId = INVALID_UINT;
};

struct PhysicalLevelInfo {
    std::vector<u32>  localRanks;                      // 升序去重
    PhysicalLevelView view = PhysicalLevelView::LOCAL;
    std::vector<u32>  partitionSizes;                  // GLOBAL：该层全部Instance大小，降序；LOCAL：空
    PhysicalSourceRef ref;

    // ---- 链路属性：随hasTopoInst一起生效 ----
    bool                      hasTopoInst = false;
    CommTopo                  topoType    = CommTopo::COMM_TOPO_RESERVED;
    EndpointLocType           locType     = EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
    std::vector<CommProtocol> protocols;                // 去重升序
    std::vector<u32>          portNums;                 // 按iface去重，降序
    std::vector<EndpointDesc> endpoints;                // 按EndpointDescLess排序
};

// 取值与acl_rt.h的ACL_DEVICE_FORM_FACTOR_*逐一对应，一致性由topo_host.cc的static_assert守护
enum class DeviceFormFactor : u32 {
    POD = 0, A_K = 1, A_X = 2, PCIE_CARD = 3,
    UNKNOWN = INVALID_UINT,
};
```

**已删除`PhysicalSourceType`与`PhysicalSourceInfo`。** 合一之后一个Level可能同时对应一个NetInstance和
一个TopoInstance，"来源类别"这个判别式不再成立；链路属性直接平铺在Level上，`ref`同时持有两者的身份。
早期版本的`source.xxx`访问路径全部变成`level.xxx`。

约束：

- `EndpointDesc`按POD整体编码，`alg_param.h`保留
  `static_assert(std::is_trivially_copyable<EndpointDesc>::value, ...)`；
- 设计文档"按字段编码，不使用结构体裸拷贝"的要求落在`PhysicalLevelInfo`上——它含5个`std::vector`，
  必须手写字段级编码（见4.5）；
- **`DeviceFormFactor::UNKNOWN`必须是`INVALID_UINT`而不是0**：`ACL_DEVICE_FORM_FACTOR_POD`就是0（F21），
  用0当"未知"会让每一次取值失败都被读成"这是POD机型"。判POD一律走`IsPodForm()`，
  不要写`!deviceFormFactor`或`static_cast<u32>(x) == 0`；
- `alg_param.h`同时参与device/AICPU侧编译，**不能include`acl_rt.h`**，因此`DeviceFormFactor`的取值是
  ACL宏的一份复制，由`topo_host.cc`（host-only，两套定义都可见）的`static_assert`钉死。

**序列化上限**：`PHYSICAL_LEVEL_NUM_LIMIT = 32`，`DeSerialize`读到超限值时丢弃整个physicalLevels段并
提前返回。定位是"超过即判定字节流不可信"，不是业务上限——每个netLayer最多贡献若干级，而netLayer数本身
受`HCCL_LOGIC_TOPO_LEVEL_NUM`约束，现网形态远小于32。

> **已知问题（本期不处理）**：`BinaryStream::operator>>(std::vector<T>&)`读的是未初始化的`size_t size`，
> 流被截断时`vec.resize(size)`会拿到栈上垃圾值并抛`length_error`/`bad_alloc`。这不是本次引入的——
> `netLayers`、`instSizeListOfLayer`、`ranksInTopo`走的是同一条路径，新字段只增加了触发站点数量。
> 详见设计文档§8.3（含后续可选修法）。

### 4.2 逐层合一（`physical_level_build.cc`）

```cpp
HcclResult BuildLayerCandidates(HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo,
                                u32 layer, std::vector<PhysicalLevelInfo>& candidates);
```

`BuildPhysicalLevelCandidates`退化为对`netLayerDetails.netLayers`的一层循环，逐层调用上面这个函数。
只遍历`netLayers`里的layer：规避F17的抛异常分支，同时构造性地保证每个Level的`ref.netLayer`都来自
`GetLayers`的实际结果（不变量9）。

**第一步：取本层NetInstance**（`FetchNetInstance`）

```text
(raw, num) = GetRanksByLayer(comm, layer)
netRanks   = copy(raw, num)                        # F13：必须本次调用后立即复制
校验 netRanks.size() == localNetInsSizeOfLayer[layer]   # 活跃分支，见4.4
校验 myRank ∈ netRanks
partitionSizes = instSizeListOfLayer[layer]        # 必须是局部拷贝，下面要就地排序
校验 sum(partitionSizes) == userRankSize           # 哨兵，见4.4
sort(partitionSizes, greater<u32>())               # 显式降序规范化
```

**`partitionSizes`必须是局部拷贝**。`instSizeListOfLayer[layer]`是**会被序列化下发的现有字段**，
就地重排会导致Golden不一致，且只在存在非对称拓扑（元素本就不同序）时才显现。

**`partitionSizes`必须显式降序排序。** `GetInstSizeListByLayer`的返回顺序是哈希序（F15），
不规范化则同一拓扑在不同进程得到不同字节流，跨rank锚点失效而无人察觉——序列化仍然成功，各rank的字节流
不同，问题要到TopoMatch层才显形。

**不计算也不保存任何派生量**（GCD、是否全等等）。非对称的判断与处理归TopoMatch，
`physical_level_build.cc`因此不include`topo.h`（F9）。

关于F6：设计文档描述的是"在`ExtractNetLayerDetails`中顺手把rank列表存进临时结构"。实现上改为
**重新调用一次**`HcclRankGraphGetRanksByLayer`——代价是每层一次额外查询（层数≤4，可忽略），
收益是`ExtractNetLayerDetails`保持零改动。

**第二步：取本层全部含当前rank的TopoInstance**（`FetchTopoInstances`）

```text
(rawIds, num) = GetTopoInstsByLayer(comm, layer)   # F5：必须重新查询才能拿到topoInstId
若失败            -> 整体降级                       # 见下方第3点
若 num == 0       -> 返回空列表（不是错误）          # 调用方据此置 hasTopoInst = false
若 rawIds == null -> 整体降级
for instId in copy(rawIds, num):
    ranks = GetRanksByTopoInst(comm, layer, instId)   # 按instId查询，与返回顺序无关
    若 myRank ∉ ranks -> 跳过                          # F14：必须过滤，见下方第2点
    topoType = GetTopoType(comm, layer, instId)
    若失败 -> 整体降级                                  # 排序第3键，见下方第4点
    level.localRanks     = ranks
    level.ref            = {layer, instId}
    level.hasTopoInst    = true
    level.topoType       = topoType
    FetchEndpoints(comm, layer, instId, level.endpoints)          # 局部降级
    FetchLocAndProtocols(level.endpoints, level.locType, level.protocols)
    FetchPortNums(comm, myRank, level.endpoints, level.portNums)  # 局部降级
```

**第三步：合一**（`BuildLayerCandidates`本体）

```text
merged = false
for level in topoLevels:
    if sorted(level.localRanks) == sorted(netRanks):
        level.view           = GLOBAL          # 同范围：把该层的全局分区并进来
        level.partitionSizes = partitionSizes
        merged = true
    else:
        level.view = LOCAL                     # 更细：看不到兄弟NetInstance，无全局分区可言
        level.partitionSizes.clear()
    candidates.push_back(level)

if not merged:                                  # 没有同范围TopoInstance
    level = {netRanks, GLOBAL, partitionSizes, {layer, INVALID_UINT}, hasTopoInst=false}
    candidates.push_back(level)
```

四点必须说明：

1. **同范围的多个TopoInstance各自都持有该层的分区**（F20）。netLayer 0上的Mesh与CLOS描述的是同一批rank
   的不同互联形态，分区是该层的全局事实、不专属于其中某一个形态。不要试图"只给第一个"——那会让哪一级
   拿到分区取决于哈希返回序；
2. **必须按成员关系过滤**。`GetTopoInstsByLayer`返回的是当前rank所在NetInstance内的**全部**
   TopoInstance，不是当前rank所属的那些（F14）。不过滤会把兄弟实例当成候选，它们与当前rank的范围
   互不包含，会直接触发"不构成范围链"的整体降级，把本可支持的拓扑误判为不支持；
3. **`GetTopoInstsByLayer`返回失败是整体降级；`topoInstNum == 0`不是失败**。设计文档5.2已论证910_95上
   netType闸门分支不可达，真正的"该层没有TopoInstance"是`NetInstance::GetTopoInstsByLayer`遍历空map的
   结果——成功且返回0。此时该层照常出一个Level（带分区），只是`hasTopoInst = false`。
   > 早期版本把这里写成"per-layer局部降级：`continue`，该层只出GLOBAL级"。合一之后表现相同
   > （该层仍然只有一个带分区、无链路属性的级），但语义更准确：它不是降级，是正常形态，
   > 且现在有显式标志位而不是靠"没有LOCAL级"来暗示；
4. **`GetTopoType`失败必须整体降级，不能保留`RESERVED`继续**。它是排序第3键。
   **注意作用范围随合一扩大**：合并前只有`LOCAL`级需要，现在netLayer 0上同范围的Mesh与CLOS两级
   都是`GLOBAL`，前两键全部打平，定序仍然完全依赖它。取不到就没有跨rank共同的定序依据，
   各rank排出的下标语义会分叉——而分叉后每一级单看都合法，本地校验一个都拦不住。

**Endpoint提取**（不变）：

```cpp
constexpr u32 ENDPOINT_NUM_SANITY_LIMIT = 64;   // 合理性阈值，不是安全边界

void FetchEndpoints(HcclComm comm, u32 layer, u32 instId, std::vector<EndpointDesc>& out)
{
    // GetEndpointNum失败 / num为0 / num超限 -> 局部降级，out留空
    std::vector<EndpointDesc> buf(num);   // F8：num是实际写入条数的上界
    u32 actual = num;
    // GetEndpointDesc失败或actual > num -> 局部降级
    buf.resize(actual);
    std::sort(buf.begin(), buf.end(), EndpointDescLess);   // F18：输出是哈希序，必须归一化
    out = std::move(buf);
}
```

**位置与协议提炼**：

```cpp
void FetchLocAndProtocols(const std::vector<EndpointDesc>& endpoints,
                          EndpointLocType& locType, std::vector<CommProtocol>& protocols);
```

- `locType`取首个endpoint的`loc.locType`，逐条比对；**不一致时置`RESERVED`并告警**——一个Level对应
  一种网络平面，位置本应唯一，给出任何一个都会误导"是否需要host网卡"的判断；
- `protocols`收集全部`desc.protocol`后**去重升序**。同一协议可能出现在多个iface上，因此
  `endpoints`已按protocol为首键排过也不能省掉这一步。

**端口数提取（本方案最容易写错的一处）**：

```cpp
constexpr u32 PORT_NUM_SANITY_LIMIT = 64;   // 参考：HCOMM MAX_PORT_NUM=32、驱动 HAL_UB_PORT_NUM=36

void FetchPortNums(HcclComm comm, u32 myRank, const std::vector<EndpointDesc>& endpoints,
                   std::vector<u32>& out)
{
    out.clear();
    if (endpoints.empty()) return;      // 已覆盖"HCOMM低版本没有这族符号"（GetEndpointNum先失败）

    std::vector<CommAddr> seenAddrs;    // 已计入的iface；规模个位数，线性查找即可
    std::vector<u32> portNums;
    for (const auto& desc : endpoints) {
        if (任一 seenAddrs 与 desc.commAddr 相等) continue;   // 同一iface的另一种协议，已计过
        EndpointAttrBwCoeff portNum{};
        if (GetEndpointInfo(comm, myRank, &desc, ENDPOINT_ATTR_BW_COEFF, ..., &portNum) != SUCCESS)
            return;                     // 全有或全无：整个清空
        if (portNum == 0 || portNum > PORT_NUM_SANITY_LIMIT) return;
        seenAddrs.push_back(desc.commAddr);
        portNums.push_back(portNum);
    }
    std::sort(portNums.begin(), portNums.end(), std::greater<u32>());
    out = std::move(portNums);
}
```

三条硬性要求：

1. **必须按`commAddr`去重到iface粒度**（F19）。一个iface有N种协议就有N个`EndpointDesc`，
   `GetEndpointInfo(BW_COEFF)`对这N条返回同一个端口数——它是`iface->GetPorts().size()`，是iface的属性。
   逐endpoint累加会把同一条物理链路重复计入：`ub_ctp` + `ub_mem`的8口iface会被算成`{8,8}`、求和16。
   **这是个静默错误**，数值看着合理、校验全过，只有和实际带宽对不上时才会暴露；
   iface身份用`commAddr`判定的依据是HCOMM的`endpointToIfaceMap`正是以`(commAddr, protocol)`为键；
2. **全有或全无**。任一条取不到就`return`（`out`保持为空），不要跳过失败项继续。残缺数组会让消费侧
   算出一个"看着合理但偏小"的总端口数，比空数组难查得多——空数组至少能让消费侧明确识别为不可用；
3. **不做能力探测**。`endpoints`为空已经覆盖了"HCOMM低版本没有这族符号"的情况（弱符号未命中时
   `GetEndpointNum`就已经失败），再加`HcommIsSupportXxx()`不但重复，还会打断ST构建（F23）。

`portNums`语义：该层只有一条8口链路记作`{8}`，两条链路记作`{6,2}`，求和为本卡在该级的总物理端口数。

**排序键**（F18）：

```cpp
bool EndpointDescLess(const EndpointDesc& a, const EndpointDesc& b)
{
    if (a.protocol != b.protocol) return a.protocol < b.protocol;
    if (a.loc.locType != b.loc.locType) return a.loc.locType < b.loc.locType;
    if (a.commAddr.type != b.commAddr.type) return a.commAddr.type < b.commAddr.type;
    return memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws)) < 0;
}

// iface身份判定，与上面同源：按字段而非memcmp整个EndpointDesc
bool CommAddrEqual(const CommAddr& a, const CommAddr& b)
{
    return a.type == b.type && memcmp(a.raws, b.raws, sizeof(a.raws)) == 0;
}
```

不要用`memcmp`比较整个`EndpointDesc`：尾部`raws[52]`在HCOMM侧从未赋值，比较未初始化字节会得到
不稳定结果——这恰好是要消除的问题。同理，序列化往返用例必须按字段比较而不是按字节比较。

`ENDPOINT_NUM_SANITY_LIMIT = 64`的来源：`num = (本rank在该topoInst上的接口数) × (每接口协议数)`，
是纯本地量。协议数上界是`CommProtocol`的有效值个数10个，接口数经`AddConnInterface`去重后是个位数，
10 × 6 ≈ 60，取64。**定位是异常告警阈值不是防御性截断**：超过只可能是HCOMM侧异常，
此时截断会让`GetEndpointDesc`因缓冲不足返回`HCCL_E_PARA`（F8：它不截断而是报错），结果还是拿不到
endpoints，却把异常掩盖成"正常但数据少"。

### 4.2.1 整机形态（`topo_host.cc`）

```cpp
HcclResult CalcDeviceFormFactor(TopoInfoWithNetLayerDetails* topoInfo)
{
    topoInfo->deviceFormFactor = DeviceFormFactor::UNKNOWN;

    s32 userDevId = 0;                              // aclrtGetDevice返回的是userDevId
    if (aclrtGetDevice(&userDevId) != ACL_SUCCESS) { WARNING; return HCCL_SUCCESS; }

    s32 logicDevId = 0;                             // aclrtGetDeviceInfo要的是logicDevId
    if (aclrtGetLogicDevIdByUserDevId(userDevId, &logicDevId) != ACL_SUCCESS) {
        WARNING; return HCCL_SUCCESS;
    }

    s64 val = 0;
    // quiet=true：老驱动不支持该infoType时会稳定失败，按ERROR打会在正常的老环境上持续刷错误日志
    if (hcalrtGetDeviceInfo(logicDevId, ACL_DEV_ATTR_DEVICE_FORM_FACTOR, val, true) != HCCL_SUCCESS) {
        WARNING; return HCCL_SUCCESS;
    }
    switch (val) {                                  // 白名单已知取值
        case ACL_DEVICE_FORM_FACTOR_POD: case ACL_DEVICE_FORM_FACTOR_A_K:
        case ACL_DEVICE_FORM_FACTOR_A_X: case ACL_DEVICE_FORM_FACTOR_PCIE_CARD:
            topoInfo->deviceFormFactor = static_cast<DeviceFormFactor>(val); break;
        default: WARNING; return HCCL_SUCCESS;      // 保持 UNKNOWN
    }
    return HCCL_SUCCESS;
}
```

四点：

1. **恒返回`HCCL_SUCCESS`**。该字段是纯附加信息，取不到时停在`UNKNOWN`，现有字段与旧执行路径完全不受
   影响。这与同文件里其它`Calc*`失败即返回是有意的区别——那些是算法分层的输入，这个不是；
2. **`switch`必须白名单已知取值**。`bit[7:5]`的取值空间有8个（还有RSV/装备/EVB），ACL只定义了4个宏；
   未识别的值落`default` → `UNKNOWN`。**绝不能让`default`落到`POD`**（F21：POD就是0）；
3. **设备号必须转换：`aclrtGetDevice`给的是userDevId，`aclrtGetDeviceInfo`要的是logicDevId**（F24）。
   少这一步在默认部署下"看起来是对的"（两者恰好相等），一旦配了`ASCEND_RT_VISIBLE_DEVICES`或容器只挂载
   部分设备就会去读**另一张卡**的形态——而且读到的是一个合法值，不报错、只静默拿到错的机型；
4. `hcalrtGetDeviceInfo`的白名单要先加`ACL_DEV_ATTR_DEVICE_FORM_FACTOR`（F22），否则直接返回
   `HCCL_E_NOT_SUPPORT`。

> **三套设备号不要混用**（F24）：
>
> | 设备号 | 含义 | 获取 |
> |--------|------|------|
> | `phyDevId` | 物理设备号，板上实际编号 | `aclrtGetPhyDevIdByUserDevId` / `...ByLogicDevId` |
> | `userDevId` | 用户可见设备号，`aclrtSetDevice`/`aclrtGetDevice`这一套，受`ASCEND_RT_VISIBLE_DEVICES`影响 | `aclrtGetDevice` |
> | `logicDevId` | 逻辑设备号，驱动内部使用 | `aclrtGetLogicDevIdByUserDevId` |
>
> **`aclrtGetDeviceInfo` / `halGetDeviceInfo` 要的是 `logicDevId`**：`ascend_hal_base.h`对`devId`的说明是
> "除`INFO_TYPE_MASTERID`外一律使用logical device ID"；HCOMM的`CcuGetMainboardId`/`HrtGetMainboardId`
> 形参名就是`deviceLogicId`，`hal_get_mainboard_id`也是先`phyId → userDevId → logicDevId`再调
> （`hal.c`的`hal_get_logicid_from_phyid`）。
>
> 另注意 **`aclrtGetLogicDevIdByPhyDevId`名不副实**：HCOMM `hal.c`的注释写明"该接口语义错误，实际返回的
> 是UserDevId"，并优先改用`aclrtGetUserDevIdByPhyDevId`。本仓`adapter_acl.cc`的
> `haclrtGetDeviceIndexByPhyId`正是用它填一个叫`deviceLogicId`的出参——**该函数目前无调用者，属于潜在
> 隐患**，将来若要用它拿logicDevId须先补一次`aclrtGetLogicDevIdByUserDevId`。

守护ACL取值一致性的`static_assert`放在`topo_host.cc`（host-only，同时看得到`DeviceFormFactor`与
`ACL_DEVICE_FORM_FACTOR_*`两套定义）：

```cpp
static_assert(static_cast<u32>(DeviceFormFactor::POD) == ACL_DEVICE_FORM_FACTOR_POD && ...,
              "DeviceFormFactor must mirror ACL_DEVICE_FORM_FACTOR_* in acl_rt.h");
```

> **实现选型备注**：HCOMM全仓没有一处使用`ACL_DEV_ATTR_DEVICE_FORM_FACTOR`（409），其形态判断一律走
> `ACL_DEV_ATTR_MAINBOARD_ID`（407）再取`bit[7:5]`（`CcuGetMainboardId` / `HrtGetMainboardId`）。
> 二者读的是同一段硬件位域、取值逐一相等，409只是驱动替调用方做了移位。选409的理由与切回407的触发条件
> 见[整机形态获取方案对比](./docs/design/device_form_factor.md)。

### 4.3 标准化（`physical_level_normalize.cc`，纯函数）

```cpp
HcclResult NormalizePhysicalLevels(
    std::vector<PhysicalLevelInfo>& candidates, u32 userRank, u32 userRankSize,
    std::vector<PhysicalLevelInfo>& levels);
```

步骤：

1. **归一**：每个候选`localRanks`排序去重、`partitionSizes`与`portNums`降序、`protocols`去重升序。
   构建侧已做过同样的规范化，这一步在正常路径上是幂等的；保留它是为了让纯函数不依赖调用方，可离线UT。
   同时剔除不含`userRank`的候选，并确认每个`hasTopoInst == true`候选的`topoType`可由`TopoTypeOrder`定序
   ——这是第2步第3键的前提，不满足直接返回`HCCL_E_NOT_SUPPORT`；

   > **`portNums`不去重**：该层若有两条链路都是8口就是`{8,8}`，求和才是总端口数。
   > 去重的对象是**iface**（在构建侧按`commAddr`做），不是端口数值。

2. **三键排序**：`std::sort`，比较函数`LevelLess`依次比较

   | 键 | 内容 |
   |---|---|
   | 1 | `localRanks.size()`升序 |
   | 2 | `view`：`LOCAL`(0) < `GLOBAL`(1) |
   | 3 | `LevelTopoOrder`：`1DMESH`=0 < `CLOS`=1 < 无TopoInstance=2，仅在前两键打平时生效 |
   | 兜底1 | `localRanks`字典序 |
   | 兜底2 | `ref.netLayer`，再`ref.topoInstId` |

   **第3键对`GLOBAL`级同样生效**——这是合一带来的改动。合并前的实现是
   `if (level.view == LOCAL) TopoTypeOrder(...)`，GLOBAL级统一取固定值；合一之后netLayer 0上同范围的
   Mesh与CLOS两级都是`GLOBAL`，前两键全部打平，仍然要靠`topoType`定序。现在的判据是`level.hasTopoInst`。

   无TopoInstance的级取`TOPO_TYPE_ORDER_NO_TOPO_INST = 2`，**必须与`TopoTypeOrder`的输出空间不重叠**，
   否则两类Level会在第3键上打平、定序退回兜底键。

   用`sort`而非`stable_sort`：上述键构成全序，结果不依赖输入顺序——这正是目的。第3键**不得用枚举值
   代替**：`COMM_TOPO_CLOS = 0 < COMM_TOPO_1DMESH = 1`，与"直连在内层"正好相反；

3. **合一已在构建侧完成**，标准化阶段不再做任何合并或拆分，只排序与校验；
4. **链校验**：`for i in [1, n)`，用`std::includes(levels[i].localRanks, levels[i-1].localRanks)`
   （两侧已排序）校验包含关系，**允许相等**——netLayer 0上同范围的Mesh与CLOS两级、以及两个netLayer的
   本地NetInstance恰好同范围，都是合法的相等相邻对。任一对不满足返回`HCCL_E_NOT_SUPPORT`。

复杂度：候选数K = Σ(每层的TopoInstance数或1)，实际≤10；总体`O(K log K + K·N)`。

### 4.4 校验（`physical_level_normalize.cc`，纯函数）

```cpp
HcclResult ValidatePhysicalLevels(
    const std::vector<PhysicalLevelInfo>& levels, u32 userRank, u32 userRankSize);
```

逐条对应设计文档第6章：

| 不变量 | 实现 |
|--------|------|
| 1 | `userRankSize > 0 && userRank < userRankSize` |
| 2 | 每个Level：升序严格递增（等价于无重复）、`back() < userRankSize`、`binary_search(userRank)` |
| 3 | `view`与payload等价：`LOCAL` ⟺ `partitionSizes.empty()` |
| 3a | `view`必须是`LOCAL`或`GLOBAL`之一（底层类型固定为u32，非法值否则会静默落进`GLOBAL`分支） |
| 3b | `ref.netLayer != INVALID_UINT` |
| 3c | `hasTopoInst == true`：`ref.topoInstId != INVALID_UINT`、`portNums.size() <= endpoints.size()`、每个`portNum`非0且`<= PORT_NUM_SANITY_LIMIT`、`portNums`降序、`protocols`已排序且无相邻重复（`is_sorted` + `adjacent_find`）<br>`hasTopoInst == false`：`ref.topoInstId`、`topoType`、`locType`、`protocols`、`portNums`、`endpoints`**六项全部为无效值** |
| 4 | `GLOBAL`级：`accumulate(partitionSizes) == userRankSize` |
| 4b | `GLOBAL`级：`partitionSizes`不含0项（0能穿过求和与降序两条检查） |
| 5 | `GLOBAL`级：`std::is_sorted(partitionSizes, greater<u32>())` |
| 6 | `GLOBAL`级：`localRanks.size()`是`partitionSizes`中的一项（等式而非整除关系） |
| 7 | 大小**非递减**：`levels[i].size() >= levels[i-1].size()` |
| 8 | 每个`hasTopoInst`级：`TopoTypeOrder(topoType)`返回true（哨兵，910_95上恒成立） |
| 9 | 每个`ref.netLayer`出现在`netLayerDetails.netLayers`中（构造性保证，见下） |
| 10 | `std::includes(levels[i], levels[i-1])`，**允许相等** |
| 11 | 由三键排序 + `endpoints`/`protocols`/`portNums`各自的规范化共同保证 |
| 12 | 由"不写现有字段"保证，ST Golden验证 |
| 13 | `(netLayer, topoInstId)`两两不重复。**从合并前的三元组`(view, netLayer, topoInstId)`收缩而来**：那时同一netLayer会同时出一个NetInstance级和若干TopoInstance级，需要`view`才能区分；合一之后无TopoInstance的级每层至多一个 |

不变量9需要`netLayers`，为保持`ValidatePhysicalLevels`是纯函数，把它放在构建阶段保证（构建时只遍历
`netLayers`里的layer，属于构造性保证）。

**构建阶段另有两条检查，性质不同**，实施时不要一起当成"反正不会触发"处理：

- **本地NetInstance大小校验是活跃分支**。`localNetInsSizeOfLayer`由`ExtractNetLayerDetails`的一次
  `GetRanksByLayer`写入，构建阶段发起的是**另一次**独立调用，两者不同源。不一致说明RankGraph在两次
  调用之间发生了变化，是必须降级的真实场景；
- **`instSizeList`求和校验是哨兵，正常路径永不触发**。`ExtractNetLayerDetails`已用同一等式做过硬
  `CHK_RET`（不满足即返回`HCCL_E_PARA`，通信域起不来），且先于本阶段执行。
  **不要**把它计入"降级路径已覆盖"的证据，也不要为它设计触发用例；代码注释中必须写明它不是活跃分支。

不变量8同样是哨兵，但**处理方式不同**：它守护的是排序键的可取值性。一旦失效，后果是各rank排出的下标
语义分叉，而分叉后每一级单看都合法，本地校验一个都拦不住。因此必须显式检查并降级，不能省略。

### 4.4.1 本地校验覆盖不到的部分

上表全部是**单rank本地**校验，有一类失效它们结构性地拦不住：同一netLayer上TopoInstance的种类
结构跨rank不一致（下层接口少返回一个实例、或通信域跨了机型）。此时各rank的级数与下标语义不同，
但各自单看每一级都完全合法。

各rank级数一致靠的是"同一netLayer上TopoInstance的种类结构由机型和配置保证一致"这条**外部契约**。
契约不成立时需跨rank校验兜住：锚点为`partitionSizes`（跨rank逐字节相同）与各netLayer上`topoType`的
多重集；时机在反序列化后；不一致即降级并按"下层接口或配置问题"打错误日志。
本期先落地本地各条与锚点字段，跨rank校验的落点随TopoMatch一并确定。

**链路属性与`deviceFormFactor`同样是局部量**，本地校验只保证自洽、不保证跨rank一致。
`portNums`尤其如此：同一台机器上不同die的rank链路就可能不同（HCOMM `TopoGetClosPort`里die0是4个口、
die1是2个口）。消费侧（cost model）若要用它们驱动算法选择，一致性需自行处理。

### 4.5 序列化（`alg_param.h`）

在`Serialize()`现有最后一个循环之后追加，`DeSerialize()`对称位置追加：

```cpp
// Serialize
physicalLevelNum = static_cast<u32>(physicalLevels.size());
binaryStream << physicalLevelNum;
for (const auto& level : physicalLevels) {
    binaryStream << level.localRanks;          // vector<u32>，走F2重载
    binaryStream << level.view;
    binaryStream << level.partitionSizes;      // vector<u32>
    binaryStream << level.ref.netLayer;
    binaryStream << level.ref.topoInstId;
    binaryStream << level.hasTopoInst;
    binaryStream << level.topoType;
    binaryStream << level.locType;
    binaryStream << level.protocols;           // vector<CommProtocol>
    binaryStream << level.portNums;            // vector<u32>
    binaryStream << level.endpoints;           // vector<EndpointDesc>，POD
}
binaryStream << deviceFormFactor;              // 声明在标量区，编码在最尾部
```

`DeSerialize`对称，并在开头把`physicalLevelNum`/`physicalLevels`/`deviceFormFactor`三者复位；
读到`physicalLevelNum > PHYSICAL_LEVEL_NUM_LIMIT`时丢弃整段并提前`return`。

> **`deviceFormFactor`放在尾部的代价**：上面那个提前`return`会连带跳过它，使其停在`UNKNOWN`。
> 这是纯位置流的必然结果——跳过了变长的`physicalLevels`段就无法定位其后的字段。可以接受：两者的
> 降级态都是"该字段不可用"，消费侧本就必须处理。代码里必须写明这一点，否则后来人会当成bug去"修"。

三条硬性要求：

1. **手写字段级循环，不写`binaryStream << physicalLevels;`**。后者会命中`vector<T>`重载并对
   `PhysicalLevelInfo`调用泛型裸拷贝重载（F1），把5个`std::vector`的堆指针写进字节流，反序列化后是
   野指针——而且不会有编译错误，只会在运行期随机崩溃。这是本次改动最容易踩的坑；
2. 新字段严格追加在最后，既有字段的编码顺序一律不动；
3. `vector<u32>`、`vector<CommProtocol>`、`vector<EndpointDesc>`都可以直接用现成重载，只需保证读写对称。

`AlgResourceCtxSerializable`无需改动：`topoInfoSeqSize`随之变化，尾部定位逻辑仍成立（F4）。

### 4.6 编排与降级（`physical_level_build.cc`）

```cpp
HcclResult BuildPhysicalLevels(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo)
{
    topoInfo->physicalLevels.clear();

    std::vector<PhysicalLevelInfo> candidates;
    std::vector<PhysicalLevelInfo> levels;

    HcclResult ret = BuildPhysicalLevelCandidates(comm, topoInfo, candidates);
    if (ret == HCCL_SUCCESS) {
        ret = NormalizePhysicalLevels(candidates, topoInfo->userRank, topoInfo->userRankSize, levels);
    }
    if (ret == HCCL_SUCCESS) {
        ret = ValidatePhysicalLevels(levels, topoInfo->userRank, topoInfo->userRankSize);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[PhysicalLevel] normalize degraded, ret[%d], rank[%u]. "
                     "physicalLevels stays empty, legacy path unaffected.", ret, topoInfo->userRank);
        return HCCL_SUCCESS;   // 降级，不改变CalcTopoShape的返回值
    }
    topoInfo->physicalLevels = std::move(levels);
    return HCCL_SUCCESS;
}
```

接入点（`topo_host.cc:771`）：

```cpp
HcclResult CalcTopoShape(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo)
{
    ... 现有9个步骤完全不动 ...
    CHK_RET(CalcLevel2Ubg(comm, topoInfo));
    CHK_RET(CalcDeviceFormFactor(topoInfo));        // 新增，与comm无关，恒返回SUCCESS
    CHK_RET(BuildPhysicalLevels(comm, topoInfo));   // 新增，恒返回SUCCESS
    return HCCL_SUCCESS;
}
```

> **与设计文档一致**：设计文档§9已明确"新增标准化步骤的任何失败一律降级，不改变`CalcTopoShape`的
> 返回值"，本节实现与之对齐，无偏差。理由：现有`ExtractTopoDetails`对
> `HcclRankGraphGetTopoInstsByLayer`的返回值本来就未做`CHK_RET`，把新步骤的查询失败提升为致命会让
> 原本能跑的通信域起不来，与"绝不影响旧路径"的目标冲突。

### 4.7 不做能力探测

RankGraph的TopoInstance与Endpoint系列接口在目标环境中必然提供，**不引入`HcommIsSupportXxx()`判断**，
也不给`hccl_rank_graph_dl.h`补`DECL_SUPPORT_FLAG`。

理由有两条：

1. 这些接口是弱符号，未加载时桩函数打错误日志并返回`(HcclResult)(-1)`（F7），并不崩溃。因此
   "符号未加载"这一路径**已经被既有的返回值检查覆盖**——`!= HCCL_SUCCESS`分支对"接口不存在"和"接口
   调用失败"的处理本就相同（整体降级或局部降级）。再加一层能力探测只是把同一个分支判断两遍；
2. **能力探测会打断ST构建**（F23）。ST只把`hcomm_dlsym`当include目录、不编译其中的`.cc`，每个
   `HcommIsSupportXxx`都必须在`hccl_stub.cc`里手写桩，漏一个就是链接失败；即便补了桩，桩通常返回
   `false`，新特性在ST里会全程不生效——用例照常通过，但测的是降级路径。

> 实施过程中曾一度为`FetchPortNums`加过`HcommIsSupportHcclRankGraphGetEndpointInfo()`门控，
> 因上述第2条回退。`FetchPortNums`改为以`endpoints.empty()`作为前置条件——弱符号未命中时
> `GetEndpointNum`早已失败、`endpoints`为空，该情况天然被覆盖。

对实现的要求相应变成：**每一个RankGraph调用都必须检查返回值**，不得出现忽略返回值的调用
（现有`ExtractTopoDetails`对`HcclRankGraphGetTopoInstsByLayer`就没检查，新代码不复制这个写法）。

### 4.8 构建改动

`src/ops/op_common/topo/CMakeLists.txt`：加入**无条件**`src_list`（第10-16行的块），不放进
`NOT HCCL_CANN_COMPAT_850`分支——新代码只经弱符号调用RankGraph，符号未加载时桩函数返回失败并走降级
（见4.7），不依赖版本宏。

```cmake
set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/topo.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_level_build.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_level_normalize.cc
    ...
)
```

AICPU侧不需要改动（F11）：ST的AICPU目标不编译`topo_host.cc`，新增的两个`.cc`同属Host-only。
但结构体在`alg_param.h`中，会被AICPU编译，因此结构体定义里不得出现Host-only的头依赖。

## 5. 测试方案

### 5.1 UT（`test/ut/topo_physical_level/`）

复刻`test/ut/reduce_scatter_birs/CMakeLists.txt`的写法：独立gtest可执行、直接编译被测`.cc`、
探测`CANN_VERSION_NUM`、`run_llt_test`。只编译`physical_level_normalize.cc`，不链接RankGraph。

`physical_level_test.cc`覆盖设计文档§11.1全表：

| 用例 | 输入 | 断言 |
|------|------|------|
| `NormalServerTwoLevels` | 附录A.1的候选（2×8卡） | **2个Level**，均`GLOBAL`，`partitionSizes`为`{8,8}`/`{16}` |
| `SuperPodThreeLevels` | 附录A.2的候选（4机2超节点32卡） | **3个Level**，逐级`localRanks`/`partitionSizes`/`locType`比对 |
| `SameRangeMerged` | TopoInst{0..15}/CLOS + NetInst{0..15} | **1个Level**（合一），`view==GLOBAL`且同时有`partitionSizes`与`topoType`/`endpoints` |
| `FinerTopoInstStaysLocal` | NetInst{0..7} + TopoInst Mesh{0..3} | 2个Level：`{0..3}`为`LOCAL`且`partitionSizes`为空，`{0..7}`为`GLOBAL` |
| `NoTopoInstLayerFlagged` | 某layer的`topoInstNum == 0` | 该层出1级，`hasTopoInst==false`且六项链路属性全为无效值，**不**整体降级 |
| `MeshBeforeClosOnFullTie` | 同范围的Mesh与CLOS，输入顺序颠倒 | 恒为Mesh在前（第3键），且**两级都带`partitionSizes`**（守护"第3键对GLOBAL级同样生效"） |
| `UnorderableTopoTypeDegrades` | `hasTopoInst`级的`topoType`为`RESERVED`或`COMM_TOPO_310P` | 返回`HCCL_E_NOT_SUPPORT` |
| `PortNumsDedupByIface` | 一个iface跑`ub_ctp`+`ub_mem`两种协议、8口 | `protocols`两项、`portNums == {8}`**一项**（逐endpoint统计会得到`{8,8}`） |
| `PortNumsTwoLinks` | 两个iface分别6口、2口 | `portNums == {6,2}`（降序） |
| `PortNumsAllOrNothing` | 三条endpoint中一条查询失败 | `portNums`**整体为空**，而非剩两项 |
| `PortNumsRejectZeroAndOverLimit` | 端口数返回0 / 返回65 | `portNums`整体为空并告警 |
| `MixedLocTypeMarkedUnknown` | 同级endpoint一个DEVICE一个HOST | `locType == ENDPOINT_LOC_TYPE_RESERVED`，其余字段照常 |
| `ProtocolsSortedUnique` | 同一协议出现在多个iface上 | `protocols`去重升序 |
| `PartitionSizesDistinguishAsymmetry` | `[16,4]`与`[12,8]`两组输入 | `partitionSizes`不同（二者gcd都是4、uniform都是false） |
| `PartitionSizesSortedDescending` | `instSizeList`按`[4,16]`顺序输入 | `partitionSizes == {16,4}` |
| `PartitionSizesKeptVerbatim` | `instSizeList`分别为`[8,8,8,8]`与`[8,24]` | 两者`partitionSizes`不同且原样保留 |
| `EqualAdjacentRangesAccepted` | 同范围的Mesh与CLOS相邻 | 链校验通过（守护"允许相等"） |
| `EndpointsSortedStably` | 同一组endpoint按不同顺序输入 | `endpoints`逐字段相等 |
| `BuildDoesNotMutateInstSizeList` | 调用构建路径后检查`instSizeListOfLayer[layer]` | 与调用前逐元素相等（守护就地排序陷阱） |
| `OrderIndependent` | 同一组候选打乱输入顺序 | 输出逐字段相等（守护三键全序） |
| `NonContiguousNetLayerId` | layer编号{0,3} | `ref.netLayer`保留3 |
| `OverlapNotNestedReturnsNotSupport` | {0,1,2,3}与{0,4,8,12} | 返回`HCCL_E_NOT_SUPPORT` |
| `ValidateRejectsDuplicateRank` | localRanks含重复 | 失败（不变量2） |
| `ValidateRejectsOutOfRange` | rank ≥ userRankSize | 失败（不变量2） |
| `ValidateRejectsMissingMyRank` | 不含myRank | 失败（不变量2） |
| `ValidateRejectsViewPayloadMismatch` | `view==LOCAL`但`partitionSizes`非空 | 失败（不变量3） |
| `ValidateRejectsInvalidView` | `view`为`LOCAL`/`GLOBAL`之外的值 | 失败（不变量3a） |
| `ValidateRejectsInvalidNetLayer` | `ref.netLayer == INVALID_UINT` | 失败（不变量3b） |
| `ValidateRejectsNoTopoInstWithLinkAttr` | `hasTopoInst==false`但`topoType`非RESERVED | 失败（不变量3c） |
| `ValidateRejectsPortNumsExceedEndpoints` | `portNums`比`endpoints`多一项 | 失败（不变量3c） |
| `ValidateRejectsUnsortedProtocols` | `protocols`未排序或有重复 | 失败（不变量3c） |
| `ValidateRejectsZeroPartition` | `partitionSizes`含0项 | 失败（不变量4b） |
| `ValidateRejectsDuplicateSource` | 两个Level的`(netLayer, topoInstId)`相同 | 失败（不变量13） |
| `ValidateRejectsPartitionSizesSumMismatch` | `sum(partitionSizes) != userRankSize` | 失败（不变量4） |
| `ValidateRejectsUnsortedPartitionSizes` | `partitionSizes == {4,16}` | 失败（不变量5） |
| `ValidateRejectsLocalRankNumNotAPartition` | `localRanks.size()`不在`partitionSizes`中 | 失败（不变量6） |
| `ValidateRejectsDecreasingSize` | 大小递减的相邻两级 | 失败（不变量7） |

整机形态（可与上面同一个可执行，不依赖RankGraph）：

| 用例 | 输入 | 断言 |
|------|------|------|
| `FormFactorKnownValues` | ACL返回0/1/2/3 | 依次为`POD`/`A_K`/`A_X`/`PCIE_CARD`；`IsPodForm()`仅0时true |
| `FormFactorUnknownValueNotPod` | ACL返回6（装备）或7（EVB） | `UNKNOWN`，**且`IsPodForm()`为false**（守护"POD==0"陷阱） |
| `FormFactorQueryFailStaysUnknown` | `aclrtGetDeviceInfo`失败 | `UNKNOWN`，`CalcDeviceFormFactor`仍返回`HCCL_SUCCESS` |
| `FormFactorLogicIdConversionFail` | `aclrtGetLogicDevIdByUserDevId`失败 | `UNKNOWN`，仍返回`HCCL_SUCCESS` |
| `FormFactorUsesLogicIdNotUserId` | 桩令`userDevId=1`、`logicDevId=3`，并让`aclrtGetDeviceInfo`按id返回不同形态 | 取到的是**logicDevId=3**对应的形态（守护F24；默认部署下两者相等，这条用例是唯一能测出该缺陷的手段） |
| `FormFactorDefaultIsUnknown` | 默认构造的`TopoInfoWithNetLayerDetails` | `deviceFormFactor == UNKNOWN`且`IsPodForm()`为false |

`physical_level_serialize_test.cc`（需编译`alg_param.h`，无需被测`.cc`）：

| 用例 | 断言 |
|------|------|
| `RoundTripEmpty` | `physicalLevels`为空时往返一致，且不影响其他字段 |
| `RoundTripFull` | 3个Level（附录A.2形态）、11个字段全覆盖、多Endpoint往返后逐字段相等 |
| `RoundTripPreservesAllLevelFields` | 逐项断言`view`/`partitionSizes`/`hasTopoInst`/`topoType`/`locType`/`protocols`/`portNums`往返一致——手写编码最容易漏掉其中一行 |
| `RoundTripPreservesDeviceFormFactor` | `deviceFormFactor`往返一致；它编码在最尾部，最容易被漏 |
| `OverLimitDropsLevelsAndFormFactor` | 构造`physicalLevelNum > PHYSICAL_LEVEL_NUM_LIMIT`的流 | `physicalLevels`为空且`deviceFormFactor == UNKNOWN`（守护尾部追加的已知代价，避免被当bug"修"） |
| `LegacyPrefixUnchanged` | 构造两个对象，一个`physicalLevels`为空、一个非空，比较字节流前缀完全相同 |
| `EndpointDescIsTriviallyCopyable` | 编译期`static_assert`（放在头文件即可，此处再加一条运行时占位断言便于定位） |

### 5.2 ST（`test/st/algorithm/`）

利用现有`topo_model`模拟拓扑，覆盖设计文档§11.2：

1. **EngineCtx缓存往返**（F4，最高优先级）：同一通信域连续执行两个算子，断言第二个算子拿到的
   `physicalLevels`与第一个一致且非空。这条用例直接守住本次改动的头号风险；
2. **Golden不变**：对现有全部ST拓扑，逐个断言改动前后`netLayerDetails`、`topoInstDetailsOfLayer`、
   `topoLevelNums`、`level0Topo`等字段完全一致；
3. **降级路径**：在`topo_model`中构造当前rank同时属于的两个互不包含TopoInstance（2D mesh的x/y环），
   断言`CalcTopoShape`返回`HCCL_SUCCESS`、`physicalLevels`为空、现有Matcher行为不变；
3b. **兄弟实例过滤**（对应F14）：构造同一NetInstance内多个互不相交的TopoInstance（如4个Mesh8），
   断言只有含当前rank的那个进入`physicalLevels`，且**不**触发降级。这条与用例3是一对，缺了它就分不清
   "正确过滤"和"碰巧没触发"；
4. **Endpoint能力缺失**：令`GetEndpointDesc`桩返回失败，断言`physicalLevels`仍然生成，
   该级`endpoints`/`locType`/`protocols`/`portNums`全部为空、`hasTopoInst`仍为`true`
   （守护"有TopoInstance但endpoint查不到"与"没有TopoInstance"是两回事）；
5. **Device下发往返**：AICPU路径下断言Device侧反序列化得到的`physicalLevels`与`deviceFormFactor`与Host一致；
6. **合一形态**：对每个ST拓扑断言级数等于"Σ每层的含myRank TopoInstance数（该层无实例时记1）"，
   且不存在"同范围但一个有分区一个没有"的相邻对——这是合一是否落实的最直接判据。
   注意本条会让**全部现有ST的Level数期望值发生变化**（常规2×8从4级变2级、超节点32卡从6级变3级），
   迁移时需要一并更新断言；

ST桩需要新增的能力：`HcclRankGraphGetEndpointInfo`桩当前对`ENDPOINT_ATTR_BW_COEFF`固定返回1
（`hccl_stub.cc`），要覆盖`PortNumsDedupByIface`与`PortNumsTwoLinks`需让它按endpoint地址返回不同值。

`topo_model`需要新增的能力：构造"同一layer多个互不包含TopoInstance"的场景（现有`is2D`标志可能已覆盖，
实施时先确认；不足则扩展）。

### 5.3 验收对照

设计文档§11.3的验收标准与用例的映射：

| 验收项 | 对应用例 |
|--------|----------|
| 1 现有字段照常输出 | ST-2 |
| 2 新字段只在尾部且随序列化传递 | UT`LegacyPrefixUnchanged` + ST-1 + ST-5 |
| 3 同范围合一、同范围不同形态各自成级 | UT`SameRangeMerged` + `FinerTopoInstStaysLocal` + `MeshBeforeClosOnFullTie` + ST-6 |
| 4 来源与Endpoint不丢失 | UT`RoundTripFull` + ST-4 |
| 4c 链路属性齐备、`portNums`按iface去重 | UT`PortNumsDedupByIface` + `PortNumsTwoLinks` + `ProtocolsSortedUnique` |
| 4d `deviceFormFactor`如实/降级不落POD | UT`FormFactorKnownValues` + `FormFactorUnknownValueNotPod` |
| 5 不调用GetLinks | 代码走查 + `physical_level_*.cc`中grep`GetLinks`为空 |
| 6 失败降级 | ST-3 |
| 7 无HCOMM私有依赖 | 编译期（新文件只include已有公开头） |
| 8 UT与序列化测试通过 | 全部 |

## 6. 提交拆分

每个PR独立可合入、独立安全：

| PR | 内容 | 风险 | 验证 |
|----|------|------|------|
| PR1 | 结构体 + 常量 + `static_assert` + 序列化 + 上限防御 | 零（`physicalLevels`恒为空，字节流前缀不变） | 序列化UT + 全量ST回归 |
| PR2 | `physical_level_normalize.cc` + 纯函数UT | 零（未被调用） | UT |
| PR3 | `physical_level_build.cc`（含合一） + 接入`CalcTopoShape` + 降级 | 低（只新增一次调用，恒SUCCESS） | ST-1/2/3/4/5/6 |
| PR4 | `DeviceFormFactor` + `CalcDeviceFormFactor` + `adapter_acl`白名单与`quiet`参数 | 低（新字段独立，取不到即`UNKNOWN`） | UT形态白名单 + 上板确认INFO日志 |

PR1先行的好处：序列化格式一旦确定并合入，后续PR不会再引起字节流变化，Device侧兼容性只需验证一次。
PR4与PR1-3无数据依赖，可并行。

## 7. 风险与回滚

| 风险 | 触发条件 | 缓解 | 回滚 |
|------|----------|------|------|
| `BinaryStream`裸拷贝导致野指针 | 误写`stream << physicalLevels` | 4.5的手写循环 + Review检查项 + `RoundTripFull`UT | — |
| 第二个算子拿到空视图 | 忘记进`Serialize` | ST-1专项用例 | — |
| **`portNums`按endpoint而非按iface统计** | 漏掉`commAddr`去重 | 多协议iface的端口数翻倍。**静默错误**：数值看着合理、校验全过，只有和实际带宽对不上才暴露。守护：`PortNumsDedupByIface`用例 + Review清单单列一条 | — |
| **`deviceFormFactor`把`UNKNOWN`写成0** | 用0当降级值、或`switch`的`default`落到POD | 所有取不到的场景被读成POD，而POD是消费侧要走2:1收敛的那个值。守护：初值/降级值一律`INVALID_UINT` + 白名单`switch` + `IsPodForm()`统一判定 + `FormFactorUnknownValueNotPod`用例 | — |
| **第3键漏给`GLOBAL`级用** | 沿用合并前的`if (view == LOCAL)`判据 | netLayer 0上同范围的Mesh与CLOS定序退回兜底键，跨rank下标语义分叉。守护：判据改为`hasTopoInst` + `MeshBeforeClosOnFullTie`用例断言两级都带分区 | — |
| **`hasTopoInst==false`却带了链路属性** | 构建侧漏清、或复用了带属性的对象 | 消费侧按标志位判定不可用、却又能读出看似合理的值。守护：不变量3c六项全查 | — |
| Device侧ctx增大触发资源上限 | 主导项是`EndpointDesc`（**160字节/条**）；`SANITY_LIMIT=64`时单级最坏10KB | 合一后Level数减少，总量较合并前下降；PR1合入后测量实际ctx大小并记录基线 | 结构体加`#if`开关关闭新字段 |
| 就地重排`instSizeListOfLayer` | 直接对该字段排序而非其局部拷贝 | 先拷贝再排；`BuildDoesNotMutateInstSizeList`用例；Review清单单列一条 | — |
| 各rank排出的Level下标语义分叉 | 排序键不全、或第3键用了枚举值 | 4.3三键表；`MeshBeforeClosOnFullTie`/`OrderIndependent`用例 | — |
| `partitionSizes`退回哈希序 | 漏掉显式`sort` | 4.2硬性要求 + `PartitionSizesSortedDescending`用例 | — |
| `endpoints`哈希序导致跨rank/跨进程不一致 | 未按稳定键排序 | 4.2的`EndpointDescLess`；`EndpointsSortedStably`用例 | — |
| 引入能力探测打断ST构建 | 加了`HcommIsSupportXxx()`调用 | 4.7；ST桩是白名单式手写，漏一个即链接失败 | — |
| 新增RankGraph查询影响初始化耗时 | 每层1次`GetRanksByLayer` + 每TopoInstance 3次 + 每iface 1次`GetEndpointInfo` | 层数≤4、实例与iface数均个位数；PR3后测一次初始化耗时 | — |
| HCOMM返回顺序变化导致结果不稳定 | `unordered_map`哈希序（F14/F15/F18） | 全部按`instId`查询、输出全部规范化；`OrderIndependent`用例 | — |
| HCOMM裸指针失效 | 逐层循环时前一层指针被下一次调用覆盖（F13） | 每次调用后立即复制；Review清单单列一条 | — |
| 409形态属性在目标版本未打通 | HCOMM全仓无使用者（见4.2.1备注） | 上板确认`[Topo][CalcDeviceFormFactor]`的INFO日志；不通则切`ACL_DEV_ATTR_MAINBOARD_ID` + `>>5 & 0x7` | — |
| **把`userDevId`当`logicDevId`传给`aclrtGetDeviceInfo`** | 直接用`aclrtGetDevice`的返回值（F24） | **默认部署下测不出来**——两者恰好相等；只在配了`ASCEND_RT_VISIBLE_DEVICES`或容器挂载部分设备时才会读到另一张卡，且返回的是合法值不报错。守护：必经`aclrtGetLogicDevIdByUserDevId`；变量名如实叫`userDevId`/`logicDevId`；INFO日志把两者都打出来，不等时现场可见 | — |
| 降级面过大导致新特性形同虚设 | 现网拓扑大量命中降级 | PR3合入后统计降级日志占比 | — |

回滚手段：PR3单独revert即可让`physicalLevels`恒为空，PR4单独revert即可让`deviceFormFactor`恒为`UNKNOWN`，
PR1-2保留也不产生任何行为差异。

## 8. Code Review检查清单

序列化：

- [ ] `alg_param.h`中没有出现`binaryStream << physicalLevels`或`>> physicalLevels`整体读写
- [ ] `Serialize`中新字段全部位于既有字段之后，既有字段顺序一行未改
- [ ] 每个Level的11项编码与`DeSerialize`逐项对称；`deviceFormFactor`在最尾部且两侧对称
- [ ] `DeSerialize`开头复位了`physicalLevelNum`/`physicalLevels`/`deviceFormFactor`
- [ ] `EndpointDesc`的`static_assert`存在

合一与构建：

- [ ] 同范围的TopoInstance与NetInstance合并成**一个**Level，且该Level同时有`partitionSizes`与链路属性
- [ ] 同范围的**多个**TopoInstance各自都拿到了`partitionSizes`（不是只给第一个）
- [ ] 比NetInstance更细的TopoInstance其`view == LOCAL`且`partitionSizes`为空
- [ ] 该层无TopoInstance时仍产出一个Level，`hasTopoInst == false`且六项链路属性全为无效值
- [ ] TopoInstance候选按`topoInstId`查询，**没有**按下标与`ranksInTopo`对齐
- [ ] TopoInstance候选丢弃了不含`myRank`的兄弟实例
- [ ] 逐层提取时每次`GetRanksByLayer`后立即复制
- [ ] **排序对象是`instSizeListOfLayer`的局部拷贝**，该字段在构建前后逐元素相等
- [ ] `partitionSizes`有独立的`std::sort(..., greater)`
- [ ] 构建侧不计算任何派生量，`physical_level_build.cc`不include`topo.h`

链路属性（本次新增，最易错）：

- [ ] **`portNums`按`commAddr`去重到iface粒度**，不是逐endpoint统计
- [ ] `portNums`失败时**整个清空**，没有跳过失败项继续
- [ ] `portNums`降序、非0、不超`PORT_NUM_SANITY_LIMIT`
- [ ] `protocols`去重升序
- [ ] `locType`各endpoint不一致时置`RESERVED`并告警，没有静默取第一个
- [ ] `hasTopoInst`是显式赋值，不由`topoType != RESERVED`之类推断

排序与校验：

- [ ] 第3键的判据是`hasTopoInst`而**不是**`view == LOCAL`（合一后GLOBAL级也要用）
- [ ] 无TopoInstance的序值与`TopoTypeOrder`输出空间不重叠
- [ ] 第3键用显式优先级函数而**不是**`CommTopo`枚举值（枚举序与直连优先相反）
- [ ] 排序用`sort`而非`stable_sort`
- [ ] 链校验用`std::includes`且**允许相邻相等**
- [ ] 不变量13比的是`(netLayer, topoInstId)`，**没有**残留`view`
- [ ] `GetTopoType`失败走整体降级，**没有**保留`COMM_TOPO_RESERVED`继续
- [ ] `GetTopoInstsByLayer`返回0不当作失败

整机形态：

- [ ] `DeviceFormFactor::UNKNOWN == INVALID_UINT`，**不是0**
- [ ] `switch`白名单四个已知取值，`default`保持`UNKNOWN`且不落`POD`
- [ ] 判POD一律走`IsPodForm()`，全仓没有`!deviceFormFactor`之类写法
- [ ] `topo_host.cc`有守护ACL取值的`static_assert`
- [ ] `hcalrtGetDeviceInfo`白名单已加`ACL_DEV_ATTR_DEVICE_FORM_FACTOR`
- [ ] `CalcDeviceFormFactor`全部返回路径都是`HCCL_SUCCESS`
- [ ] **`aclrtGetDevice`的返回值经`aclrtGetLogicDevIdByUserDevId`转换后才传给`hcalrtGetDeviceInfo`**（F24）
- [ ] 变量名如实区分`userDevId`与`logicDevId`，没有把`aclrtGetDevice`的出参直接命名为`deviceId`/`logicDevId`
- [ ] INFO日志同时打了`userDevId`与`logicDevId`（二者不等即VISIBLE_DEVICES场景，现场需可见）

其它：

- [ ] `physical_level_*.cc`中不出现`HcclRankGraphGetLinks`
- [ ] `BuildPhysicalLevels`的所有返回路径都是`HCCL_SUCCESS`
- [ ] `ExtractNetLayerDetails`、`ExtractTopoDetails`零改动
- [ ] **每个RankGraph调用都检查了返回值**；没有`HcommIsSupportXxx()`能力探测（见4.7）
- [ ] 新增`.cc`加入`CMakeLists.txt`的无条件`src_list`，未误加进AICPU源表
- [ ] 序列化往返用例按字段比较，没有按字节比较`EndpointDesc`（尾部`raws[52]`未初始化）

---

## 附录A：四个典型机型的完整实例

本附录给出实现完成后`TopoInfoWithNetLayerDetails`的实际内容，用于对照验收。除A.4另标注外都以
**rank 0**为视角，覆盖本方案最容易出错的四处：合一的判定、`portNums`的按iface去重、
三键排序（尤其第3键对`GLOBAL`级生效）、以及非对称场景下的跨rank下标对齐。

**建级规则回顾**：逐netLayer —— 同范围的TopoInstance与NetInstance合一（`GLOBAL`，既有分区又有链路属性）、
更细的TopoInstance单独成级（`LOCAL`）、无TopoInstance的层出一个`hasTopoInst == false`的级；
按`(块大小, view, topoType)`三键排序。

每级记作 `localRanks | view | partitionSizes | topoType, locType, protocols, portNums | ref`。

> **与合并前的差异**：早期方案是"每个netLayer出一个GLOBAL级 + 每个TopoInstance出一个LOCAL级、不合并"，
> A.1为4级、A.2为6级。合一之后同范围的两条并成一条，A.1降为2级、A.2降为3级。
> **迁移时全部现有ST/UT的级数期望值都要更新。**

### A.1 常规机型：2台Server × 8卡 = 16卡

物理拓扑：机内8卡HCCS fullmesh，机间RoCE。

```text
netLayer1: [16]      全域
netLayer0: [8, 8]    每台Server
```

现有字段（后三个数组按**layer编号**索引，与合并前完全一致，本次改动零影响）：

```cpp
userRankSize = 16
topoLevelNums = 2
level0Topo    = MESH_1D
level0Symmetric = true             // all_of([8,8])
level1Symmetric = true             // all_of([16])
topoInstDetailsOfLayerSize = 2

netLayerNum = 2
netLayers   = [0, 1]
netInstNumOfLayer      = [2,      1   ]
instSizeListOfLayer    = [[8, 8], [16]]
localNetInsSizeOfLayer = [8,      16  ]

topoInstDetailsOfLayer:
  [0]: topoInstNum=1, sizeOfTopo=[8],  typeOfTopo=[1DMESH], ranksInTopo=[{0..7}]
  [1]: topoInstNum=1, sizeOfTopo=[16], typeOfTopo=[CLOS],   ranksInTopo=[{0..15}]

deviceFormFactor = A_K            // 假定为A+K Server；IsPodForm() == false
```

每层的TopoInstance与NetInstance同范围，各自合一，得到**2个Level**：

```cpp
physicalLevels = [
  { localRanks={0..7},  view=GLOBAL, partitionSizes={8,8},
    ref={0, <mesh id>}, hasTopoInst=true,
    topoType=1DMESH, locType=DEVICE, protocols={HCCS}, portNums={1} },

  { localRanks={0..15}, view=GLOBAL, partitionSizes={16},
    ref={1, <clos id>}, hasTopoInst=true,
    topoType=CLOS,   locType=DEVICE, protocols={ROCE}, portNums={1} },
]
```

要点：

- 合并前这里是4级（`{0..7}`的LOCAL/GLOBAL对 + `{0..15}`的LOCAL/GLOBAL对）。合一之后每个物理层级
  只出现一次，消费侧不再需要按`view`筛掉重复范围；
- 换rank只改`localRanks`：rank 8看到`[0]`为`{8..15}`、`[1]`为`{0..15}`；
  **`partitionSizes`一字不改**（`{8,8}`与`{16}`），这正是它作为跨rank一致性锚点的依据；
- 最高一级的`locType == DEVICE`，因此**不需要使用host网卡**。

> 若netLayer1没有TopoInstance，`[1]`退化为`hasTopoInst=false`：仍然带`partitionSizes={16}`，
> 但`topoType`/`locType`/`protocols`/`portNums`/`endpoints`全为无效值。这是正常形态，不整体降级。

### A.2 超节点：4台Server，每2台组成一个超节点 = 32卡

```text
netLayer2: [32]            超节点间RoCE（host网卡）
netLayer1: [16, 16]        超节点内
netLayer0: [8, 8, 8, 8]    每台Server
```

```cpp
userRankSize = 32
topoLevelNums = 3
level0Symmetric = true             // all_of([8,8,8,8])
level1Symmetric = true             // all_of([16,16])

netLayers   = [0, 1, 2]
netInstNumOfLayer      = [4,          2,        1   ]
instSizeListOfLayer    = [[8,8,8,8],  [16,16],  [32]]
localNetInsSizeOfLayer = [8,          16,       32  ]
```

三层各自合一 → **3个Level**：

| idx | localRanks | view | partitionSizes | topoType | locType | protocols | portNums | ref |
|---|---|---|---|---|---|---|---|---|
| 0 | `{0..7}` | GLOBAL | `{8,8,8,8}` | 1DMESH | DEVICE | `{HCCS}` | `{1}` | L0/mesh |
| 1 | `{0..15}` | GLOBAL | `{16,16}` | CLOS | DEVICE | `{UBC_TP}` | `{8}` | L1/clos |
| 2 | `{0..31}` | GLOBAL | `{32}` | CLOS | **HOST** | `{ROCE}` | `{1}` | L2/clos |

rank 20看到`[0]`为`{16..23}`、`[1]`为`{16..31}`、`[2]`为`{0..31}`，三处`partitionSizes`不变。

**最高一级`locType == HOST`，消费侧据此判定"需要使用host网卡"。** 本结构不存该结论。

TopoMatch取算法维度时只读`localRanks`/`view`/`partitionSizes`；cost model读`topoType`/`locType`/
`protocols`/`portNums`与`deviceFormFactor`。

### A.3 UBX机型：单Server 16卡，Mesh与CLOS同在layer0

物理拓扑：0～7与8～15各自fullmesh直连，0～15经UB CLOS全连接，**两种链路都在netLayer0**。

由F14d，rank 0的`GetTopoInstsByLayer(0)`返回**2个**实例——Mesh是peer2peer、只注册myRank所属的那一个，
CLOS是Fabric、全部16个rank共用一个topoInstId。这正好命中`TOPO_INST_NUM_MESH_1D_CLOS = 2`，
`CalcLevel0TopoShape`才能识别出`MESH_1D_CLOS`。

```cpp
userRankSize = 16
topoLevelNums = 1                  // netInstNumOfLayer[0]==1，第一层就break
level0Topo    = MESH_1D_CLOS
level0BigClosRange = true          // CLOS范围16 > BIG_CLOS_RANGE(8)

netLayers   = [0]
netInstNumOfLayer      = [1]
instSizeListOfLayer    = [[16]]
localNetInsSizeOfLayer = [16]
```

Mesh（8卡）比NetInstance（16卡）更细 → `LOCAL`；CLOS（16卡）同范围 → 合一为`GLOBAL`。**2个Level**：

```cpp
physicalLevels = [
  { localRanks={0..7},  view=LOCAL,  partitionSizes={},
    ref={0, <mesh id>}, hasTopoInst=true,
    topoType=1DMESH, locType=DEVICE, protocols={HCCS, UB_MEM}, portNums={1} },

  { localRanks={0..15}, view=GLOBAL, partitionSizes={16},
    ref={0, <clos id>}, hasTopoInst=true,
    topoType=CLOS,   locType=DEVICE, protocols={UBC_TP},       portNums={8} },
]
```

两点：

- **同一个netLayer产生了2个Level**，因此Level下标与netLayer编号没有固定对应关系——这是TopoMatch设计
  §4.1坚持"`baseLevelIdx`精确不扫描"的直接依据；
- `[0]`的`protocols`有两项而`portNums`只有一项：Mesh的iface同时跑HCCS与UB_MEM，是同一条物理链路。
  **逐endpoint统计会得到`portNums={1,1}`，是错的**——这正是`PortNumsDedupByIface`用例要守的行为。

### A.4 非对称机型：netLayer0 = [16, 4]，共20卡

UBX类机型，两侧NetInstance内都同时挂Mesh和CLOS。**本节同时给出rank 0与rank 16两个视角。**

```cpp
userRankSize = 20
netLayers   = [0, 1]
netInstNumOfLayer      = [2,        1   ]
instSizeListOfLayer    = [[16, 4],  [20]]      // 哈希序，也可能是[4, 16]
localNetInsSizeOfLayer = rank 0: [16, 20]      // 局部量，跨rank不同
                         rank 16: [4, 20]
level0Symmetric = false            // all_of([16,4]) == false
```

**rank 0（16卡块）**：Mesh是4卡、比NetInstance（16卡）细 → `LOCAL`；CLOS是16卡、同范围 → `GLOBAL`。

| idx | localRanks | view | partitionSizes | topoType | ref |
|---|---|---|---|---|---|
| 0 | `{0..3}` | LOCAL | `{}` | 1DMESH | L0/mesh |
| 1 | `{0..15}` | GLOBAL | `{16,4}` | CLOS | L0/clos |
| 2 | `{0..19}` | GLOBAL | `{20}` | CLOS | L1/clos |

**rank 16（4卡块）**：Mesh与CLOS都是4卡、都与NetInstance同范围 → **两级都合一为`GLOBAL`**。

| idx | localRanks | view | partitionSizes | topoType | ref |
|---|---|---|---|---|---|
| 0 | `{16..19}` | GLOBAL | `{16,4}` | **1DMESH** | L0/mesh |
| 1 | `{16..19}` | GLOBAL | `{16,4}` | **CLOS** | L0/clos |
| 2 | `{0..19}` | GLOBAL | `{20}` | CLOS | L1/clos |

三点必须看清：

1. **两个rank的级数相同（都是3级）、下标语义逐位对齐**：`[0]`=机内直连、`[1]`=机内交换、`[2]`=层1。
   非对称只体现在块大小（16 vs 4）与`[0]`的`view`（LOCAL vs GLOBAL）上，不体现在级数上。
   `view`的差异是**合一规则的正确结果**：rank 16的Mesh恰好与NetInstance同范围，所以它确实知道全局分区；
2. **`partitionSizes = {16,4}`在两个rank上逐字节相同**，而`localRanks`处处不同。`{16,4}`与`{12,8}`的
   GCD都是4、是否全等都是false、rank数都是20，**只有保留完整列表才能区分**；
3. **rank 16的`[0]`与`[1]`是排序第3键唯一真实生效的场景**，且合一之后它们**都是`GLOBAL`**：
   块大小(4)与`view`(GLOBAL)全部打平，只有`topoType`能定序。
   **这正是"第3键的判据必须从`view == LOCAL`改成`hasTopoInst`"的直接依据**——沿用旧判据的话，
   这两级会双双取到"GLOBAL级不参与第3键"的固定值，定序退回兜底键，rank 16与rank 0的下标语义就分叉了。
   若第3键用了`CommTopo`枚举值（`CLOS=0 < 1DMESH=1`，与直连优先相反），结果同样是分叉——
   而分叉后每一级单看都合法，本地校验一个都拦不住。

### A.5 从这四个例子读出的结论

1. **`physicalLevels`的下标与netLayer编号没有固定对应关系。** A.3中2个Level全部来自`netLayer 0`。
   这是TopoMatch设计§4.1坚持"`baseLevelIdx`精确不扫描"的直接依据；
2. **相邻Level的`localRanks`相等仍可能出现**，虽然比合并前少得多（合并前每层必有一对）。
   A.4的rank 16有一对。链校验必须用`⊇`而不是`⊃`；
3. **`view == LOCAL`（无全局分区）在UBX上是常态**，交叉校验是主路径；
4. **跨rank一致性只有两个锚点**：`partitionSizes`与各netLayer上`topoType`的多重集。
   其余字段（`localRanks`、`localNetInsSizeOfLayer`、`portNums`、`locType`、`deviceFormFactor`）
   全是局部量，跨rank不保证相同，不能用来比对；
5. **`portNums`的项数是iface数，不是endpoint数、也不是协议数。** A.3的`[0]`最直观：2种协议、1条链路、
   `portNums`一项。

### A.6 对应的ST用例

在5.2的ST清单上追加：

| 用例 | 构造 | 断言 |
|------|------|------|
| `NormalTwoServer` | A.1的2×8拓扑 | **2级**，均`GLOBAL`；`[0].partitionSizes=={8,8}`、`[1].partitionSizes=={16}`；两级均`hasTopoInst` |
| `SuperPodFourServer` | A.2的4机2超节点 | **3级**；`partitionSizes`分别为`{8,8,8,8}`/`{16,16}`/`{32}`；`[2].locType==HOST` |
| `UbxSingleServer` | A.3的16卡Mesh+CLOS | **2级**；两级`ref.netLayer`都是0；`[0].view==LOCAL`、`[1].view==GLOBAL`且`partitionSizes=={16}` |
| `UbxMeshMultiProtocolOnePort` | A.3的`[0]`，Mesh iface跑HCCS+UB_MEM | `protocols`两项、`portNums`**一项**（守护按iface去重） |
| `UbxLocalDimCrossCheck` | A.3但令Mesh为6卡 | `CalcTopoShape`返回SUCCESS；TopoMatch的`{LEVEL_2}`返回`HCCL_E_NOT_SUPPORT` |
| `AsymmetricTwoViews` | A.4的20卡，同时跑rank 0与rank 16 | 两者级数均为**3**；`[2].partitionSizes`两者逐字节相等；rank 16的`[0].view==GLOBAL`而rank 0的`[0].view==LOCAL` |
| `AsymmetricMeshBeforeClosOnGlobal` | A.4的rank 16 | `[0].topoType==1DMESH`且`[1].topoType==CLOS`，**且两级`view`均为GLOBAL**（守护第3键对GLOBAL级生效） |
| `LayerWithoutTopoInstance` | 令某netLayer的`topoInstNum==0` | 该层出1级、`hasTopoInst==false`、六项链路属性全为无效值，其余层不受影响，**不**整体降级 |
| `FabricSiblingFiltered` | 一个16卡NetInstance内两个PCIe switch Fabric | rank 0的`physicalLevels`不含`{8..15}`那个Fabric，且**不**触发整体降级 |
| `HostNicDetectedFromTopLevel` | A.2拓扑 | 最高一级`locType==HOST`；A.1拓扑下则为`DEVICE`（守护"是否需要host网卡"的推导口径） |

`FabricSiblingFiltered`是F14d指出的真实过滤场景，与5.2的`3b`兄弟实例用例是一对：后者验证Mesh类
（HCOMM已过滤，走防御路径），前者验证Fabric类（HCOMM不过滤，必须由本方案过滤）。

`AsymmetricMeshBeforeClosOnGlobal`是本次合一改动**唯一会被静默破坏**的行为，务必保留。

# topo-parse 虚拟层级链设计（v2）

> 本文是 topo-parse 的**第二份**设计文档，覆盖"生成 TopoMatch 可直接使用的虚拟拓扑"这一新增诉求。
> 已有的 `topo-parse-refactor-design.md` 与 `topo-parse-impl-plan.md` 描述的是**物理**范围链
> （`physicalLevels`），本文不修改它们的任何结论——虚拟链是物理链的**纯派生产物**。
>
> 输入来源：`cross-pod-topology-analysis.md`（SE 的 VT 方案）。本文采纳其问题定义与拆分思想，
> 但重构了数据结构与构建规则，理由逐条列在第 9 章。

---

## 1. 背景与目标

### 1.1 SE 方案要解决的问题

HCCL 执行器用模运算推算各层 rank 下标：

```cpp
rankIdxLevel0_ = (myRank_ % intraSuperpodDeviceNum) % rankSizeLevel0_;
rankIdxLevel1_ = (myRank_ % intraSuperpodDeviceNum) / rankSizeLevel0_;
rankIdxLevel2_ = myRank_ / intraSuperpodDeviceNum;
```

这段算术成立的前提是：rank 空间可以按**统一基数**逐层切开，且每层的块在 rank 编号上**连续且对齐**。
server-major 编号下 pod 内 rank 非连续（pod0 = `{0,1,2,3, 8,9,10,11}`），前提被破坏，
`TopoForLayer2` 的 `rankId % (l0*l1)` 选错组，执行器的 `rankIdxLevel1_` 越界。

SE 的方案是**虚拟拓扑归一化**：把非连续层按连续性拆成虚拟 group，使模运算在虚拟 group 内自动正确，
执行器与模板零改动。本文认同这个方向。

### 1.2 本文强化的诉求

> 生成 TopoMatch 可直接使用的虚拟 topo，但 endpoint 等信息仍然保留物理信息。
> 比如在非对称的物理 topo 上直接生成对称的、TopoMatch 可用的虚拟 topo。

两点强化：

1. **不止"修正非连续"，而是"生成一个规整的虚拟拓扑"**。非对称物理拓扑（`netLayer0 = [16,4]`）
   同样应产出对称的虚拟拓扑，而不是像现在这样在 `TopoMatchMultilevel::MatchTopo` 里
   直接 `HCCL_E_NOT_SUPPORT` 拒掉（`topo_match_multilevel.cc:308-314`）。
2. **物理信息不丢**。虚拟层只承载"算法怎么分组"，Endpoint、topoType、netLayer、topoInstId
   这些建链要用的物理身份仍然来自 `physicalLevels`，由虚拟层反向引用。

### 1.3 与 physicalLevels 的分工

| | physicalLevels | virtualLevels |
|---|---|---|
| 是什么 | 当前 rank 可见的**物理范围链**，与 RankGraph 的 NetInstance/TopoInstance 一一对应 | 一条**对齐块大小链**，是 rank 空间的一次统一基数分解 |
| 形状 | 嵌套的 rank 集合，可能非连续、可能不等长 | 连续对齐块，长度整除递进 |
| 谁消费 | LinkBinding（读 `source`）、Selector（读 `source.endpoints`） | TopoMatch（算法分组）、Selector（选层数/算法） |
| 生成 | `BuildPhysicalLevels(comm, topoInfo)`，依赖 RankGraph | `BuildVirtualLevels(physicalLevels, ...)`，**纯函数，不碰 RankGraph** |
| 失败 | 降级为空，旧路径可用 | 降级为空，TopoMatch 回退旧 Matcher |

关键取舍：**虚拟链只从物理链派生，不新增任何 RankGraph 调用**。理由见 §9.4。

---

## 2. 核心结论：虚拟拓扑就是一条块大小链

### 2.1 执行器施加的约束

执行器读的是 `myRank_` 本身，不读我们产出的任何东西。因此无论虚拟拓扑怎么设计，
它必须让下面这组算术**恰好正确**：

```
rank = i₀ + B₀·i₁ + B₁·i₂ + …          （混合基数展开）
第 k 层的通信组 = 只有第 k 位不同的那些 rank
```

写成块大小的形式：设

```
B₀ | B₁ | … | Bₙ ,   Bₙ = userRankSize
```

则第 k 层（0 ≤ k ≤ n-1）myRank 的通信组是

```
G_k(myRank) = { (myRank mod B_k) + j·B_k + ⌊myRank / B_{k+1}⌋·B_{k+1}
                : j ∈ [0, B_{k+1} / B_k) }
```

组大小 `s_k = B_{k+1} / B_k`，且 `∏ s_k = userRankSize`。

**结论：一条满足整除递进的块大小链 `[B₀, B₁, …, Bₙ]` 完全决定了 TopoMatch 需要的全部分组。**
不需要存 rank 列表，不需要存拆分标记，不需要 `GetLinks` 做二次过滤。

### 2.2 三个例子的验算

**(a) SE 文档的 16P 交叉 Pod**（4 server × 4 卡，2 pod，server-major）

物理链（rank 0 视角）：`{0..3}` ⊂ `{0,1,2,3,8,9,10,11}` ⊂ `{0..15}`

虚拟链 `B = [4, 4, 16]`，基数 `(4, 1, 4)`：

| rank | `r mod 4` | `(r/4) mod 1` | `r / 4` | L0 组 | L1 组 | L2 组 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | `{0,1,2,3}` | `{0}` | `{0,4,8,12}` |
| 8 | 0 | 0 | 2 | `{8,9,10,11}` | `{8}` | `{0,4,8,12}` |

与 SE 文档 §2.1 的表一致：L1 trivial 化，跨 server 通信全部落到 L2。
`{0,8}` 同 pod 这一事实**不进入算法分组**，但通道创建从 L0 向上搜索时仍会命中 L1 链路
（SE 文档 §2.3），物理路径不劣化。

**(b) 非对称 `netLayer0 = [16, 4]`，20 卡**

`partitionGcd(netLayer0) = 4`，`netLayer1 = [20]`。虚拟链 `B = [4, 20]`，基数 `(4, 5)`。

| rank | L0 组 | L1 组 |
|---|---|---|
| 0 | `{0,1,2,3}` | `{0,4,8,12,16}` |
| 16 | `{16,17,18,19}` | `{0,4,8,12,16}` |

**物理非对称，虚拟对称。** 这正是 §1.2 的诉求。代价：L1 组里 `0↔4` 是服务器内链路、
`0↔16` 是服务器间链路，带宽不均，算法按最慢边估时。这是**成本问题不是正确性问题**，
留给 Selector 的 cost model（见 §12.3）。

**(c) 普通 32 卡三级**（`rankgraph_example.pptx` slide 1）

`[8,8,8,8] / [16,16] / [32]`，全连续。虚拟链 `B = [8, 16, 32]`，基数 `(8, 2, 2)`，
与今天的行为逐位相同 —— `isIdentity = true`，零影响。

### 2.3 为什么不存 rank 列表

SE 文档 §3.2 的 `virtualRanks[]` 存的是 myRank 所在 VT group 的完整 rank 列表。本文不存，三个理由：

1. **可由 `B` 推出**，存两份就有对不上的可能。
2. **rank 列表是局部量**，跨 rank 逐字节不同，不能当一致性锚点；而 `B` 是全局量，
   跨 rank 逐字节相同 —— 这是 `physicalLevels` 里 `partitionSizes` 扮演的同一个角色。
3. **序列化体积随 rankSize 线性增长**。`virtualLevels` 要跟着 `TopoInfoWithNetLayerDetails`
   过 EngineCtx 缓存并下发 Device，`B` 是 O(层数) 个 u32，rank 列表是 O(rankSize)。

---

## 3. 数据结构

```cpp
// 虚拟层级链上的一环。整条链按blockSize严格递增排列, 相邻两环满足整除关系
struct VirtualLevelInfo {
    // 该层的对齐块大小。myRank所在的块是 [ (myRank/blockSize)*blockSize, +blockSize )。
    // 全局量: 同一通信域内所有rank得到的blockSize序列逐字节相同, 是本结构唯一的跨rank一致性锚点
    u32 blockSize = 0;

    // 该虚拟层由physicalLevels中的哪一环背书, 是下标不是拷贝。
    // Endpoint/topoType/netLayer/topoInstId一律去physicalLevels[backingIdx].source取,
    // 虚拟层自身不复制任何物理信息 —— 这是"虚拟分组、物理建链"的落点。
    // 无物理层背书时(顶层补齐, 见5.4)为INVALID_UINT
    u32 backingIdx = INVALID_UINT;

    // blockSize是否被连续性钳位过(见5.2)。true表示该层的虚拟块小于背书物理层的实际范围,
    // 即"物理上能连通更多rank, 但算法只在更小的对齐块内分组"。
    // 仅用于日志与Selector的成本判断, 不参与任何成员计算
    bool clamped = false;
};
```

挂载点（`TopoInfoWithNetLayerDetails`）：

```cpp
    // 标准化后的物理范围链, 语义见PhysicalLevel块注释
    std::vector<PhysicalLevelInfo> physicalLevels;

    // 由physicalLevels纯派生的虚拟层级链, 语义见VirtualLevel块注释。
    // 只允许由BuildVirtualLevels单向生成; physicalLevels为空时它必然也为空
    std::vector<VirtualLevelInfo> virtualLevels;

    // 虚拟链是否等同于物理链的GLOBAL分区(即没有发生任何钳位)。
    // true时新旧TopoMatch的分组结果必然一致, 是回归验证的判据
    bool virtualIsIdentity = true;
```

**没有 `levelSplit[]`**：`clamped` 已经表达了同一件事，且挂在层上而不是挂在一个按
"L1=下标0"约定编址的平行数组上（SE 文档 §3.2 的 `levelSplit`/`virtualRanks` 就是这种平行数组，
它重新引入了我们在 physicalLevels 上刻意消除的"层下标 = netLayer 编号"假设）。

**派生量（不落结构体，消费侧现算）**：

```cpp
// 第k层的算法组大小
s_k = virtualLevels[k+1].blockSize / virtualLevels[k].blockSize;
// 算法层数
algLevelNum = virtualLevels.size() - 1;   // 顶层只作为边界, 不是一个算法层
```

---

## 4. 消费侧契约

### 4.1 TopoMatch

**只读 `virtualLevels[].blockSize` 与 `userRank` / `userRankSize`。**
不读 `backingIdx`、不读 `clamped`、不读 `physicalLevels`、不调用任何 RankGraph 接口。

第 k 个算法层的 rank 组：

```cpp
const u32 inner = virtualLevels[k].blockSize;
const u32 outer = virtualLevels[k + 1].blockSize;
const u32 base  = (myRank / outer) * outer + (myRank % inner);
std::vector<u32> group;
for (u32 r = base; r < base + outer; r += inner) {
    group.push_back(r);
}
```

十行取代 `topo_match_multilevel.cc` 里 `TopoForLayer0/1/2` 三个函数、六次 RankGraph 调用
和两处模运算过滤器。**`GetLinks` 的存在性过滤整个消失**——连通性已经在建虚拟链时校验过
（§5.3），不必在分组时逐 rank 重查。

### 4.2 LinkBinding

只读 `physicalLevels[].source`，与 v1 设计一致，**不受本文影响**。
需要"某个算法层用哪条物理链路"时，经 `virtualLevels[k].backingIdx` 跳到物理层再取 `source`。

### 4.3 Selector

可读 `virtualLevels[].blockSize`（判断有效层数、trivial 层）、`virtualIsIdentity`、
`clamped`（判断是否发生了带宽不均的钳位，用于代价估算）、
以及 `physicalLevels[].source.endpoints`（协议/位置）。

### 4.4 显式排除

虚拟链**不表达**下列信息，消费侧不得从中推断：

- rank 之间的物理距离或带宽（`blockSize` 只是分组粒度，不是拓扑距离）
- 某个虚拟层对应哪个 netLayer（只能经 `backingIdx` 查，且顶层补齐时没有）
- 非对称性（虚拟链恒对称；非对称事实只在 `physicalLevels[].partitionUniform` 里）

---

## 5. 构建规则

`BuildVirtualLevels(const std::vector<PhysicalLevelInfo>& phys, u32 userRank, u32 userRankSize,
std::vector<VirtualLevelInfo>& out, bool& isIdentity)` —— 纯函数，无 `HcclComm`，可离线 UT。

### 5.1 第一步：候选块大小（全局锚点优先）

自小到大遍历 `physicalLevels`，为每一环取一个候选块大小 `C_i`：

| 物理层 view | 候选 `C_i` | 是否全局量 |
|---|---|---|
| GLOBAL | `partitionGcd` | **是**，由 `partitionSizes` 派生，跨 rank 逐字节相同 |
| LOCAL | `localRanks.size()` | 否，靠"同一 netLayer 上 TopoInstance 的种类结构由机型和配置保证一致"这条外部契约 |

**必须优先用 GLOBAL 的 `partitionGcd`，不能用 `localRanks.size()` 贪心取最大。**
反例：非对称 `[16,4]`，rank 0 的物理 L0 是 `{0..15}`，贪心会取到 10（`10 | 20` 且 `{0..9} ⊆ {0..15}`），
而 rank 16 只能取 4，两边分叉且**每一侧单看都合法**，本地校验一个都拦不住。
用 `partitionGcd = 4` 则两边一致。

同一范围上若同时存在 LOCAL 与 GLOBAL 环（普通机型 layer0 的常态），**取 GLOBAL 的**，
LOCAL 环只贡献 `backingIdx`（它才带 endpoints）。

### 5.2 第二步：连续性钳位

`C_i` 只保证"大小对"，不保证"myRank 的那个块在 rank 编号上连续且对齐"。逐环检查：

```
aligned(C) = [ (myRank / C) * C , (myRank / C) * C + C )
若 aligned(C_i) ⊆ physicalLevels[i].localRanks  → 不钳位
否则                                            → C_i ← 满足包含关系的最大真因子, clamped = true
```

因子只在 `C_i` 的因子里降序试，且必须仍是**前一个已定块大小的倍数**（保证整除递进）。
试到底仍不成立 → 该环丢弃（不产生虚拟层）；若丢弃的是最内层则整体降级。

16P 例子：`netLayer1` 的 `partitionGcd = 8`，但 rank 0 的 `aligned(8) = {0..7}`，
而物理 L1 = `{0,1,2,3,8,9,10,11}`，`{4,5,6,7}` 不在里面 → 钳位到 4。rank 8 同理钳位到 4。
两边都得到 4，**因为拓扑对称，所有 pod 的非连续形态相同**（SE 文档 §4.3 的保证链）。

> ⚠️ 钳位量是**局部**判定的。它跨 rank 一致靠的是对称性契约，强度弱于 `partitionGcd`。
> 这是本设计最大的一致性风险，单列在 §7。

### 5.3 第三步：连通性校验

虚拟块必须真的连通，否则算法会在不存在的链路上建通信组。
判据：`aligned(C_i)` 必须整个落在**某个**物理层的 `localRanks` 里（就是 §5.2 的包含检查本身）。

这一条替代了 `TopoForLayer1/TopoForLayer2` 里的 `HcclRankGraphGetLinks(...) == 0 → continue`：
物理层的 `localRanks` 本来就是"该层可连通范围"，包含关系成立即连通性成立，
不必对每个候选 rank 单独查链路。

**唯一的语义差**：旧代码在 `netLayer2` 上查 `GetLinks`，同 pod 的 rank 因为**在 L2 上没有直接链路**
被排除；新规则不查。这正是 SE 文档 §3.4 主动要改掉的行为（改为循环所有下层查链路），
本文用"包含即连通 + 通道创建自下而上找最佳路径"达到同一效果，且不引入那个循环。

### 5.4 第四步：去重、补齐、收尾

1. **去重**：相邻环钳位后可能得到相同的 `C`，只保留一个（优先保留有 GLOBAL 背书的那个）。
2. **补齐顶层**：链末尾必须是 `userRankSize`。若最大的 `C < userRankSize`，追加一环
   `blockSize = userRankSize, backingIdx = INVALID_UINT`。正常路径上顶层 netLayer 覆盖全域，
   这一步是幂等的。
3. **不合并 trivial 层**：`s_k == 1` 的层保留在链里。执行器靠 `skipLevel1_` 与
   `rankSizeLevel1_ == 1` 跳过（SE 文档 §4.1 已验证安全），保留它才能让"第 k 层对应哪个
   netLayer"这一映射不因基数为 1 而错位。
4. `isIdentity` = 全程未发生钳位 **且** 未丢弃任何环。

---

## 6. 不变量

`ValidateVirtualLevels(virtualLevels, userRank, userRankSize)` 逐条检查：

| # | 不变量 | 失败后果 |
|---|---|---|
| V1 | `userRankSize > 0`，`userRank < userRankSize` | 一切无从谈起 |
| V2 | 链非空，`blockSize` 严格递增 | 分组重复或空 |
| V3 | `blockSize[k] | blockSize[k+1]`（整除递进） | 混合基数展开不成立，执行器下标错乱 |
| V4 | `blockSize.back() == userRankSize` | 顶层组不覆盖全域 |
| V5 | `blockSize[0] >= 1`，且 `blockSize[0] | userRankSize` | 同 V3 |
| V6 | `backingIdx == INVALID_UINT` 或 `< physicalLevels.size()` | 悬垂引用 |
| V7 | 对每个 `backingIdx != INVALID_UINT` 的环：`aligned(blockSize)` ⊆ `physicalLevels[backingIdx].localRanks` | 在不存在的链路上建组 |
| V8 | `clamped == true` ⟺ `blockSize <` 背书物理层的 `localRanks.size()` | 标志与事实不符，Selector 代价估错 |

V3 是**最关键**的一条：它是"执行器的模运算恰好正确"这件事的全部内容。
V7 是**唯一需要 `physicalLevels` 参与**的一条，因此校验函数要同时收两个数组。

哨兵与活跃分支的区分（沿用 v1 设计 §6.1 的口径）：

- **活跃**：V3、V4、V7、V8 —— 钳位与去重逻辑写错就会触发。
- **哨兵**：V1、V2、V5、V6 —— 构造性成立，保留只为在派生规则被改动时第一时间暴露。

---

## 7. 跨 rank 一致性：三级契约

虚拟链的每一位 `blockSize` 都必须跨 rank 逐字节相同，否则各 rank 分出的组互不匹配，
而**每个 rank 单看自己的链都完全合法**——这类分叉本地校验拦不住，只能靠契约。

三个来源的强度**不同**，必须分开记账：

| 强度 | 来源 | 依据 | 失效场景 |
|---|---|---|---|
| **强** | GLOBAL 环的 `partitionGcd` | 由 `partitionSizes` 派生，`GetInstSizeListByLayer` 返回的是全层划分，跨 rank 逐字节相同 | 只有 HCOMM 改变分层语义时才会破 |
| **中** | LOCAL 环的 `localRanks.size()` | "同一 netLayer 上 TopoInstance 的种类结构由机型和配置保证一致"（v1 设计已记账） | 通信域跨机型；下层接口少返回一个实例 |
| **弱** | §5.2 的**连续性钳位量** | 对称拓扑下所有 pod 的非连续形态相同 | **非对称 + 非连续同时出现** |

最后一行是本设计的主要风险。SE 文档 §4.3 给出的保证链是：

```
对称拓扑 → 所有 server 卡数相同 → 每个 VT = 1 个 L0 组 → layer1Size = 1 全局一致
```

它只覆盖**对称**拓扑。而 §1.2 的诉求恰恰是要支持非对称。两者叠加时（例如
`netLayer0 = [16,4]` 且 pod 内 rank 非连续），钳位量可能逐 rank 不同，保证链断裂。

**本设计的处置**：把"非对称 ∧ 发生钳位"列为**不支持**，整体降级为空虚拟链，
TopoMatch 回退旧 Matcher。判据是本地可查的：

```cpp
if (clampedAnywhere && !allGlobalLevelsUniform) → 降级
```

放开这条限制需要一次跨 rank 校验（把 `blockSize` 序列做一次 allgather 比对），
属于独立议题，见 §12.1。

---

## 8. 失败与降级

沿用 v1 的三级粒度，`BuildVirtualLevels` **恒返回 `HCCL_SUCCESS`**，不改变
`CalcTopoShape` 的返回值。

| 场景 | 粒度 | 日志级别 | 结果 |
|---|---|---|---|
| `physicalLevels` 为空 | 整体 | DEBUG | `virtualLevels` 空（物理链都没有，虚拟链无从派生，不是新问题） |
| 某环钳位后仍不满足包含关系，且不是最内层 | 局部 | DEBUG | 丢弃该环，链变短 |
| 最内层无法成链 | 整体 | WARNING | `virtualLevels` 空 |
| 非对称 ∧ 发生钳位（§7） | 整体 | WARNING | `virtualLevels` 空 |
| `ValidateVirtualLevels` 任一条不过 | 整体 | WARNING | `virtualLevels` 空 |
| 虚拟层数 > 执行器支持的层数 | 整体 | WARNING | `virtualLevels` 空（见 §12.2） |

`virtualLevels` 为空时 TopoMatch 走今天的 `TopoMatchMultilevel` 路径，行为逐位不变。
**新增字段不会让原本能起来的通信域起不来。**

---

## 9. 与 SE 方案（cross-pod-topology-analysis.md）的差异

采纳的部分：问题根因（§1）、按连续性拆分的核心思想（§2.1）、trivial 层安全性论证（§4.1）、
通道创建自下而上找物理路径（§2.3）、正常场景零影响的要求（§4.2）。

改掉的部分，逐条给理由：

### 9.1 `VirtualTopologyInfo` 独立结构 → 折叠进 topo-parse 产物

SE 方案新增 `virtual_topology.h/.cc` + `topoInfo.vtInfo`，与 `physicalLevels` 并列，
两者都在 `InitRankInfo()` 末尾生成、都从 RankGraph 取数、都被 TopoMatch 消费。
**这是第三条平行的拓扑通道**（现有字段、physicalLevels、vtInfo），三者对同一物理事实各自解释一遍，
正是 v1 设计要消除的问题。

折叠后：虚拟链从物理链派生，物理链是唯一的 RankGraph 出口。

### 9.2 `virtualRanks[]` rank 列表 → `blockSize` 单值

理由见 §2.3（可推导、非全局量、序列化体积）。

### 9.3 `levelSplit[]` 按 "L1 = 下标 0" 编址 → `clamped` 挂在层上

SE 方案 §3.2/§3.3 用 `idx = level - 1` 在平行数组里定位，前提是"算法层 k ↔ netLayer k"。
v1 设计的附录 A.3 已经证伪：UBX 机型单个 netLayer 会产生 3 个 Level，
层下标与 netLayer 编号**没有固定对应关系**。

### 9.4 `NormalizeTopology` 直接调 RankGraph → 纯函数派生

SE 方案 §3.3 的伪码第 1 步是：

```
GetRanksByTopoInst(netLayer, topoInsts[0]) → ranks[]
```

三个问题，两个是硬错误：

1. **`GetTopoInstsByLayer` 在非 `TOPO_FILE_DESC` 的 netLayer 上直接返回 `HCCL_E_PARA`**
   （已核实：`communicator_impl.cc:3947-3953` 硬卡 netType）。而 VT 要处理的
   `L1 ~ L[top-2]` 在实机上普遍是 CLOS（见 `rankgraph_example.pptx` 两页，
   netLayer1/2 分别是 CLOS 与 HOST 网卡）。**伪码在它要修的那类机型上第一步就拿不到数据。**
   正确的取数接口是 `GetRanksByLayer`（NetInstance 粒度）。
2. **`topoInsts[0]` 是哈希序下标**。UBX 机型 layer0 返回 2 个实例（MESH + CLOS），
   `[0]` 取到哪个不确定，跨进程不稳定。v1 设计明令禁止按下标取 TopoInstance。
3. 顺带：`topoInfo->netLayerList` 字段不存在，实际是 `topoInfo->netLayerDetails.netLayers`。

派生方案完全绕开这三点——`physicalLevels` 已经把 netType 差异、哈希序、兄弟实例过滤都处理完了。

### 9.5 `TopoForLayer2` 循环查 `GetLinks` → 包含即连通

SE 方案 §3.4 把顶层链路检查从"只查 L2"改成"循环查 L1..L[top]"。这是一处**与 VT 正交的行为变更**：
它让"仅在 L1 连通"的 rank 对也能进入 L2 组。方向正确，但用循环实现有两个代价：
`GetLinks` 是 `O(rankSize)` 次成对查询；且它把"连通性"与"分组"两件事继续耦合在分组循环里。

本文的做法是在**建链时**用包含关系一次性判定连通（§5.3），分组时不再查链路。

### 9.6 改动清单差异

| 文件 | SE 方案 | 本文 |
|---|---|---|
| `virtual_topology.h/.cc` | 新建，调 RankGraph | 改为 `virtual_level.h` / `virtual_level_build.cc`，纯函数 |
| `alg_param.h` | 加 `vtInfo` 字段 | 加 `VirtualLevelInfo` + `virtualLevels` + `virtualIsIdentity` + **序列化**（SE 清单漏了序列化） |
| `topo_host.cc` | 末尾调 `NormalizeTopology` | 末尾调 `BuildVirtualLevels`（在 `BuildPhysicalLevels` 之后） |
| `topo_match_multilevel.*` | 加 `vtInfo` 参数，改两处 | 三个 `TopoForLayerN` 整体替换为 §4.1 的十行 |
| `CMakeLists.txt` | 加 `virtual_topology.cc` | 加 `virtual_level_build.cc` |

---

## 10. 序列化

`virtualLevels` 随 `TopoInfoWithNetLayerDetails` 过 EngineCtx 缓存并下发 Device，必须序列化。

**必须手写字段级编码**，与 `physicalLevels` 同样的约束：

```cpp
// 禁止写成 binaryStream << virtualLevels
// 那会命中BinaryStream的泛型重载 stream.write(&t, sizeof(T)), 对含vector的结构体是裸拷贝堆指针,
// 且不会有任何编译错误。VirtualLevelInfo当前是纯POD, 但结构一旦扩展就会静默出事
u32 virtualLevelNum = static_cast<u32>(virtualLevels.size());
binaryStream << virtualLevelNum;
for (const auto& level : virtualLevels) {
    binaryStream << level.blockSize;
    binaryStream << level.backingIdx;
    binaryStream << level.clamped;
}
binaryStream << virtualIsIdentity;
```

追加在 `physicalLevels` 之后、字节流末尾，不改动任何既有字段的偏移。

体积：每层 4 + 4 + 1 = 9 字节，三级拓扑约 30 字节。相比 SE 方案的 `virtualRanks[]`
（16 卡约 64 字节，1024 卡约 4KB）是常数量级。

---

## 11. 落地拆分与测试

### 11.1 PR 拆分

| PR | 内容 | 依赖 |
|---|---|---|
| 1 | `VirtualLevelInfo` 结构 + 序列化 + `BuildVirtualLevels` 纯函数 + UT | topo-parse v1 已合入 |
| 2 | `topo_host.cc` 接入（生成但无人消费，`virtualIsIdentity` 打日志） | PR 1 |
| 3 | `TopoMatchMultilevel` 改为消费 `virtualLevels`，保留旧路径作为空链时的回退 | PR 2 + ST 改造 |
| 4 | 放开 `MatchTopo` 对非对称三级拓扑的硬拒（`topo_match_multilevel.cc:308-314`） | PR 3 |

PR 1、2 可以现在做，且**对现网零行为影响**（没有消费者）。PR 3 被 ST 阻塞（§11.3）。

### 11.2 UT（纯函数，无需 comm）

| 用例 | 输入物理链 | 期望虚拟链 | 验证点 |
|---|---|---|---|
| 普通三级对称 | `{0..7}/{0..15}/{0..31}` gcd 8/16/32 | `[8,16,32]`，identity | 零影响回归 |
| 16P 交叉 pod，rank 0 | `{0..3}/{0..3,8..11}/{0..15}` | `[4,4,16]`，clamped | 钳位生效 |
| 16P 交叉 pod，rank 8 | `{8..11}/{0..3,8..11}/{0..15}` | `[4,4,16]` | **与 rank 0 逐位相同** |
| 非对称 `[16,4]`，rank 0 | `{0..15}/{0..19}` gcd 4/20 | `[4,20]` | 用 `partitionGcd` 不用 `localRanks.size()` |
| 非对称 `[16,4]`，rank 16 | `{16..19}/{0..19}` gcd 4/20 | `[4,20]` | 跨 rank 一致 |
| 非对称 ∧ 钳位 | 构造 | **空链** | §7 的降级 |
| UBX 四级 | `{0..3}/{0..15}/{0..31}/{0..63}` | 见 §12.2 | 层数超限处置 |
| 整除链断裂 | 构造 `C = [3, 8]` | 空链 | V3 |
| `physicalLevels` 为空 | `{}` | 空链，DEBUG | 不是新问题 |

全部可离线跑，不依赖 HCOMM 打桩。

### 11.3 ST 阻塞

SE 文档 §5.1 已指出：ST 模拟器 `TopoModel` 按 pod-major 顺序连续分配 rankId，
**无法产生非连续 pod rank**，因而 PR 3 的核心路径没有 ST 覆盖。
需要先给 ST 基础设施加 `rankOverride` 参数，属于独立任务。

在此之前只能用 4-pod / 每 pod 1 server 的拓扑验证 L1 trivial 时的 `skipLevel1_` 路径
（SE 文档 §5.2），验不到钳位逻辑本身。

---

## 12. 风险与开放问题

### 12.1 钳位量的跨 rank 一致性没有硬保证（**最高**）

§7 已详述。当前处置是"非对称 ∧ 钳位 → 降级"，把风险挡在支持范围外。
彻底解法是一次跨 rank 的 `blockSize` 序列比对（allgather + memcmp），
代价是通信域建立时多一次同步。**需要 SE 定夺是否值得。**

### 12.2 虚拟层数可能超过执行器支持的层数

UBX 机型（`rankgraph_example.pptx` slide 2，64 卡）的物理链是
`{0..3} / {0..15} / {0..31} / {0..63}`，派生出的虚拟链是 `[4,16,32,64]` —— **四级**，
基数 `(4,4,2,2)`。而 `MatchTopo` 当前硬限 `topoLevelNums <= COMM_LAYER_SIZE_3`
（`topo_match_multilevel.cc:238`），执行器也只有二级/三级两种。

必须有一条**合并规则**把四级压到三级：合并相邻两轴等价于把两个 `blockSize` 中间那个删掉。
删哪一个是**代价决策**（合并 UB 4 卡与 CLOS 16 卡，还是合并上面两个交换层，
步数与带宽完全不同），不该在 topo-parse 里拍。

**倾向**：topo-parse 产出完整的四级链 + 一个"建议合并点"，由 Selector 落最终决定。
但这需要先和 topo-match 的 `baseLevelIdx` 编址方案对齐——那个决策**至今未定**
（见 topo-match 设计 §5.1 与 topo-parse v1 impl-plan 附录 A.2 的自相矛盾）。

### 12.3 非对称虚拟化后的带宽不均没有进入代价模型

§2.2(b) 的 `[16,4]` 例子里，虚拟 L1 组 `{0,4,8,12,16}` 混合了服务器内与服务器间链路。
算法按均匀带宽估时会高估性能。`clamped` 标志和 `physicalLevels[].partitionUniform`
都能让 Selector 看见这件事，但**没有任何代码在用**。属于 Selector↔TopoMatch 联动的既有缺口。

### 12.4 `asymmetric-topology-analysis.md` 不存在

SE 文档 §4.3 与文末两处引用了它，作为"VT 的 GCD 扩展"的规范来源。
仓库里没有这个文件。而 GCD 扩展**正是**本文 §1.2 的核心诉求。
本文 §5.1 的"GLOBAL 锚点优先"是我对该扩展的重建，**需要 SE 确认与原意一致**。

### 12.5 A3（910_93）不在覆盖范围

SE 文档已声明：A3 走 `CalcGroupIdx` → `CalcGeneralTopoInfoForA3`，不经 `TopoMatchMultilevel`，
同样不支持非连续拓扑，需独立修复。本文同样不涉及。

### 12.6 与 `TopoForLayer0` 现有 GCD 拆分的关系

`topo_match_multilevel.cc:37-61` 已经有一个 layer0 的 GCD 拆分（`gcdInstSize`），
只处理第 0 层、只在 `topoInstNum == 1`（Mesh1D）时启用。
它是本文 §5.1 规则在单层上的特例。PR 3 落地时应**整体删除**而不是并存，
否则同一件事有两套实现。

---

*相关文档：`topo-parse-refactor-design.md`（物理链，v1）、`topo-parse-impl-plan.md`（v1 编码方案）、
`topo-match-refactor-design.md`（消费侧）、`cross-pod-topology-analysis.md`（SE 的 VT 方案）。*

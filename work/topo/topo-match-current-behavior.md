# 现有 TopoMatch 在 topoLevelNums 从 0 到 3 变化时的输出梳理

- 文档状态：现状分析
- 创建日期：2026-08-07
- 关联设计：`topo-match-refactor-design.md`

---

## 1. topoLevelNums 的计算方式

`topoLevelNums` 在 `topo_host.cc:ExtractNetLayerDetails` 中计算：

```cpp
topoLevelNum = 0;
for (auto layerIdx : netLayers) {
    topoLevelNum++;
    if (netInstNumOfLayer[layerIdx] == 1) {
        break;  // 遇到第一个全覆盖层即停止
    }
}
```

含义：从最低物理层开始计数，直到遇到第一个只有一个 NetInstance（即覆盖全部 rank）的层为止。

| topoLevelNums | 典型物理拓扑（32 rank） | 说明 |
|---------------|------------------------|------|
| 0 | 非法 | 校验失败，所有 matcher 返回 `HCCL_E_INTERNAL` |
| 1 | `NetLayer0: [32]` | 最低层已全覆盖（单服务器） |
| 2 | `NetLayer0: [8,8,8,8]` → `NetLayer1: [32]` | 需两层才覆盖全部 rank |
| 3 | `NetLayer0: [8,8,8,8]` → `NetLayer1: [16,16]` → `NetLayer2: [32]` | 需三层才覆盖全部 rank |

> 注意：`topoLevelNums` 可能小于 `netLayerDetails.netLayerNum`。例如 `netLayers={0,1,2}` 但 layer0 已有 1 个 instance 时，`topoLevelNums=1` 而 `netLayerNum=3`。

---

## 2. 统一示例场景

本文以 32 rank 为基准，使用以下物理拓扑场景：

### 场景 A：topoLevelNums=1（单服务器）

```text
netLayers = {0}
NetLayer0: [32]              # 1 instance，全覆盖
TopoInst(0): Mesh1D, [0..31]
```

### 场景 B：topoLevelNums=2（多服务器，无超节点）

```text
netLayers = {0, 1}
NetLayer0: [8, 8, 8, 8]      # 4 个服务器
NetLayer1: [32]               # 全覆盖
TopoInst(0): Mesh1D, [0..7]  # rank0 所在服务器
```

### 场景 C：topoLevelNums=3（多服务器 + 超节点）

```text
netLayers = {0, 1, 2}
NetLayer0: [8, 8, 8, 8]      # 4 个服务器
NetLayer1: [16, 16]           # 2 个超节点
NetLayer2: [32]               # 全覆盖
TopoInst(0): Mesh1D, [0..7]  # rank0 的服务器
TopoInst(1): CLOS,   [0..15] # rank0 的超节点
TopoInst(2): CLOS,   [0..31] # 全通信域
```

### 场景 D：topoLevelNums=3 + HostDPUOnly

```text
netLayers = {0, 1, 2}
NetLayer0: [8, 8, 8, 8]      # Device 链路
NetLayer1: [16, 16]           # Device 链路
NetLayer2: [32]               # 只有 Host 链路（无 Device 链路）
```

`CheckHostDPUOnly` 判定：最高层（layer2）的 TopoInst 为 CLOS 且全覆盖，但 Endpoint 全部位于 Host → `hostDPUOnly=true`。

---

## 3. 各 TopoMatch 输出详表

以下以 **rank0** 为当前 rank，展示每个 matcher 在各 `topoLevelNums` 下的输出。

数据结构回顾：

```cpp
struct AlgHierarchyInfoForAllLevel {
    std::vector<std::vector<std::vector<u32>>> infos;
    // infos[algorithmLevel][orthogonalGroup][rankIndex]
};
```

### 3.1 TopoMatch1D

**校验**：`0 < topoLevelNums <= 3`

**逻辑**：直接返回全部 rank，完全忽略物理层数量。

| topoLevelNums | infos 内容 | infos.size() | infos[0].size() | 形状 |
|:---:|---|:---:|:---:|---|
| 1 | `[[[0,1,...,31]]]` | 1 | 1 | 全 rank 单层 |
| 2 | `[[[0,1,...,31]]]` | 1 | 1 | 全 rank 单层 |
| 3 | `[[[0,1,...,31]]]` | 1 | 1 | 全 rank 单层 |

> 附加校验：对每个 netLayer 调用 `HcclRankGraphGetTopoTypeByLayer`，要求 topoType 为 `COMM_TOPO_CUSTOM` 或 `COMM_TOPO_CLOS`。

---

### 3.2 TopoMatchConcurrent

**校验**：`0 < topoLevelNums <= 2`（**不支持 topoLevelNums=3**）

**逻辑**：返回全 rank，但在 `infos[0]` 中复制两份表达两个并发平面。

| topoLevelNums | infos 内容 | infos.size() | infos[0].size() | 形状 |
|:---:|---|:---:|:---:|---|
| 1 | `[[[0..31], [0..31]]]` | 1 | **2** | 全 rank × 2 副本 |
| 2 | `[[[0..31], [0..31]]]` | 1 | **2** | 全 rank × 2 副本 |
| 3 | **报错** `HCCL_E_INTERNAL` | — | — | 不支持 |

> `infos[0][0]` 和 `infos[0][1]` 内容完全相同，用第二维表达并发。

---

### 3.3 TopoMatchMultilevel

**校验**：`0 < topoLevelNums <= 3`

**核心逻辑**：

1. 获取 layer0 instSizeList，判断 `isSymmetric`。
2. 若 `topoLevelNums > 1`，获取 layer1 instSizeList，判断 `layer1Symmetric`。
3. 若 `topoLevelNums >= 3` 且不对称 → 返回 `HCCL_E_NOT_SUPPORT`。
4. 调用 `CheckHostDPUOnly`，若 `hostDPUOnly && topoLevelNums==3` → **降级为 2 级**。
5. `commLayerSize = (topoLevelNums==3 && !needDowngrade) ? 3 : 2`。
6. `infos.resize(commLayerSize)`。
7. TopoForLayer0：查 layer0 的 TopoInst，按 topoInstNum 分 Mesh1D / Mesh2D / 单卡。
8. TopoForLayer1：查 `algLayer1NetLayer`（正常=layer1，HostDPU=最高层），取 `rankId % layer0Size == myRank % layer0Size` 的 rank。
9. TopoForLayer2（仅 3 级且未降级）：查最高层，取 `rankId % (layer0Size*layer1Size) == myRank % (layer0Size*layer1Size)` 的 rank。

#### 场景 B（topoLevelNums=2，对称，8×4）

| 属性 | 值 |
|------|---|
| isSymmetric | true |
| hostDPUOnly | false |
| commLayerSize | 2 |
| layer0Size | 8 |

```text
infos[0] = [[0,1,2,3,4,5,6,7]]           # rank0 所在服务器
infos[1] = [[0,8,16,24]]                  # 同序号跨服务器 rank
```

| topoLevelNums | infos.size() | infos[0].size() | infos[1].size() | infos[2] | 形状 |
|:---:|:---:|:---:|:---:|:---:|---|
| 2 | 2 | 1 | 1 | — | 8 × 4 |

#### 场景 C（topoLevelNums=3，对称，8×2×2）

| 属性 | 值 |
|------|---|
| isSymmetric | true |
| layer1Symmetric | true |
| hostDPUOnly | false |
| commLayerSize | 3 |
| layer0Size | 8 |

```text
infos[0] = [[0,1,2,3,4,5,6,7]]           # 服务器内 8 rank
infos[1] = [[0,8]]                        # 超节点内同序号 2 rank
infos[2] = [[0,16]]                       # 跨超节点同序号 2 rank
```

| topoLevelNums | infos.size() | infos[0].size() | infos[1].size() | infos[2].size() | 形状 |
|:---:|:---:|:---:|:---:|:---:|---|
| 3 | 3 | 1 | 1 | 1 | 8 × 2 × 2 |

#### 场景 D（topoLevelNums=3，HostDPUOnly，降级为 2 级）

| 属性 | 值 |
|------|---|
| hostDPUOnly | true |
| needDowngrade | true |
| commLayerSize | **2**（从 3 降级） |
| algLayer1NetLayer | **netLayerList.back()**（使用最高层而非 layer1） |
| layer0Size | 8 |

```text
infos[0] = [[0,1,2,3,4,5,6,7]]           # 服务器内 8 rank
infos[1] = [[0,8,16,24]]                  # 跨超节点同序号（走 Host 链路）
infos[2] = (不存在，已降级)
```

| topoLevelNums | infos.size() | 形状 | 说明 |
|:---:|:---:|---|---|
| 3+HostDPU | **2** | 8 × 4 | 物理层 3 级但算法层降级为 2 级 |

#### topoLevelNums=1 的行为

TopoMatchMultilevel 在 `topoLevelNums=1` 时行为不稳定：

- `commLayerSize` 计算为 2（因为 `topoLevelNums != 3`）。
- `infos.resize(2)`。
- TopoForLayer0 正常返回当前层全部 rank。
- TopoForLayer1 使用 `algLayer1NetLayer=1`（默认值），若 netLayers 中不存在编号 1 的层，HCOMM 查询可能失败。
- 若存在编号 1 的层但该层与 layer0 覆盖相同 rank 范围，则 `rankId % layer0Size` 过滤后只剩 myRank 自身 → **退化维度**。

| topoLevelNums | infos.size() | 问题 |
|:---:|:---:|---|
| 1 | 2 | layer1 可能退化或查询失败 |

> 实际使用中 topoLevelNums=1 的算法通常选择 TopoMatch1D 而非 Multilevel。

#### 非对称场景（topoLevelNums=2，layer0=[8,8,8,4]）

```text
gcdInstSize = gcd(8,8,8,4) = 4
TopoForLayer0: 将 rank0 所在的 8-rank pod 按 GCD=4 拆分
  infos[0] = [[0,1,2,3]]    # rank0 的 GCD 子组
  layer0Size = 4
TopoForLayer1: rankId % 4 == 0 → [0,4,8,12,16,20,24]
  infos[1] = [[0,4,8,12,16,20,24]]
```

结果：`4 × 7`（28 rank）。

---

### 3.4 TopoMatch3Level

**校验**：`0 < topoLevelNums <= 3`

**核心逻辑**：

1. **始终** `infos.resize(3)`，无论 topoLevelNums 为多少。
2. 要求对称，非对称直接返回 `HCCL_E_NOT_SUPPORT`。
3. TopoForLayer0：查 layer0 的 TopoInst。
4. 若 `layerNum >= 2`：TopoForLayerGeneric(layer1, baseModSize=layer0Size, targetIdx=1)。
5. 若 `layerNum >= 3`：TopoForLayerGeneric(layer2, baseModSize=layer1Size, targetIdx=2)。

**关键问题**：不检查 `topoLevelNums`，只检查 `layerNum`（物理层数量）。当物理层不足时，对应的 `infos[i]` 保持为空。

| topoLevelNums | layerNum | infos[0] | infos[1] | infos[2] | 形状 | 问题 |
|:---:|:---:|---|---|---|---|---|
| 1 | 1 | `[[0..31]]` | **空** | **空** | 32 × ? × ? | 两个空层 |
| 2 | 2 | `[[0..7]]` | `[[0,8,16,24]]` | **空** | 8 × 4 × ? | 一个空层 |
| 3 | 3 | `[[0..7]]` | `[[0,8]]` | `[[0,16]]` | 8 × 2 × 2 | 正常 |

> 空层意味着 `infos[i].empty()`，消费者访问 `infos[2][0]` 会越界。

#### TopoMatch3Level 的 layer2 baseModSize 计算

```cpp
uint32_t layer1Size = topoInfo->netLayerDetails.localNetInsSizeOfLayer[1]; // 当前rank的layer1 instance大小
uint32_t layer2Size = topoInfo->netLayerDetails.localNetInsSizeOfLayer[2]; // 当前rank的layer2 instance大小
if (layer1Size == 0 || layer2Size == 0 || (layer2Size < layer1Size)) {
    return HCCL_E_NOT_SUPPORT;
}
uint32_t baseModSizeL2 = layer1Size;  // 用 layer1 的 instance 大小作为模基
```

这里直接用 `localNetInsSizeOfLayer[1]`（当前 rank 的 layer1 instance 大小）而非 `infos[1][0].size()`（算法层 1 的 rank 数），两者在非标准布局下可能不一致。

---

### 3.5 TopoMatchUBX

**校验**：`0 < topoLevelNums <= 2`（**不支持 topoLevelNums=3**）

**核心逻辑**：

1. 要求对称。
2. `infos.resize(2)`。
3. TopoForLayer0：查 layer0 的 TopoInst，按 topoInstNum 分 Mesh1D / Mesh1DClos。
4. 若 `layerNum >= 2`：TopoForLayer3 —— **硬编码 NetLayer=3**（`NATLAYER_ROCE=3`），取 `rankId % layer0Size == myRank % layer0Size`。

#### Mesh1D 场景（topoLevelNums=2，8×4）

```text
infos[0] = [[0,1,2,3,4,5,6,7]]           # 服务器内
infos[1] = [[0,8,16,24]]                  # 跨服务器同序号（NetLayer3）
```

| topoLevelNums | infos.size() | infos[0].size() | infos[1].size() | 形状 |
|:---:|:---:|:---:|:---:|---|
| 2 | 2 | 1 | 1 | 8 × 4 |

#### Mesh1DClos 场景（layer0 有 2 个 TopoInst）

```text
infos[0] = [[mesh1d ranks], [clos ranks]]  # 两个 group！
infos[1] = [[0,8,16,24]]
```

| topoLevelNums | infos.size() | infos[0].size() | 说明 |
|:---:|:---:|:---:|---|
| 2 | 2 | **2** | infos[0] 有两个 rankList（mesh + clos） |

#### topoLevelNums=1

| topoLevelNums | 问题 |
|:---:|---|
| 1 | 若 `layerNum >= 2`，TopoForLayer3 查询硬编码的 NetLayer=3，可能不存在 |

---

### 3.6 TopoMatchUBX1d（继承 TopoMatchUBX）

**校验**：`0 < topoLevelNums <= 2`

**与 TopoMatchUBX 的区别**：重写了 `TopoForLayer3`，**不做 `rankId % layer0Size` 过滤**，直接包含 layer3 中所有有链路的 rank。

| topoLevelNums | infos[0] | infos[1] | 形状 | 与 UBX 的区别 |
|:---:|---|---|---|---|
| 2 | `[[0..7]]` | `[[0,1,2,...,31]]`（全部有链路的 rank） | 8 × 32? | UBX1d 的 infos[1] 包含全部 rank，不按序号过滤 |

> UBX1d 的 `infos[1]` 实际包含 layer3 中所有与 myRank 有链路的 rank，通常为全部 rank。这使得算法层 1 的维度远大于 `rankSize / layer0Size`。

---

### 3.7 TopoMatchPcieMix

**校验**：`topoLevelNums != 0`（无上限校验）

**核心逻辑**：

1. 要求 layer0 instSizeList 对称。
2. `infos.resize(2)`。
3. TopoForLayer0：查 layer0 的 TopoInst，按 TopoType 分为 1DMESH ranks 和 CLOS ranks。
4. `infos[0].push_back(ranksInMeshTopo)` —— mesh ranks 放入算法层 0。
5. `infos[1].push_back(ranksInClosTopo)` —— clos ranks 放入算法层 1。
6. Deduplicate：从 clos ranks 中移除 `rankId % meshSize != myRank % meshSize` 的 rank。
7. 若 `layerNum >= 2`：打印 warning 并**忽略**。

#### 场景：单服务器 16 rank，layer0 有 Mesh1D(4 rank) + CLOS(16 rank)

```text
ranksInMeshTopo = [0,1,2,3]
ranksInClosTopo = [0,1,...,15] → 去重后 [0,4,8,12]

infos[0] = [[0,1,2,3]]       # mesh ranks → 算法层 0
infos[1] = [[0,4,8,12]]      # clos ranks (同序号) → 算法层 1
```

| topoLevelNums | infos.size() | infos[0].size() | infos[1].size() | 形状 | 问题 |
|:---:|:---:|:---:|:---:|---|---|
| 1 | 2 | 1 | 1 | 4 × 4 | layer1 被忽略 |
| 2 | 2 | 1 | 1 | 4 × 4 | layer1 被 warning 忽略 |
| 3 | 2 | 1 | 1 | 4 × 4 | layer1/2 被 warning 忽略 |

> PcieMix 将同一物理层（layer0）的两种 TopoType 拆分到两个算法层，而非使用不同物理层。这与设计文档的"物理绑定应在 LinkBindingResolver 处理"原则冲突。

---

### 3.8 TopoMatchSqueeze2D

**校验**：`topoLevelNums == 3`（**仅支持 3 级，其他报错**）

**核心逻辑**：

1. 要求对称，要求 `layerNum >= 3`。
2. `infos.resize(2)`（输出 2 级，不是 3 级）。
3. TopoForLayer0：查 **layer1**（不是 layer0！）的 TopoInst，取全部 rank → `infos[0]`。
4. TopoForLayer1：查 **layer2** 的 TopoInst，取 `rankId % combinedSize == myRank % combinedSize` 的 rank → `infos[1]`。

#### 场景 C（topoLevelNums=3，8×2×2）

```text
combinedSize = 16 (layer1 的 rank 数)
infos[0] = [[0,1,...,15]]     # layer1 全部 rank（合并了 layer0 和 layer1）
infos[1] = [[0,16]]           # layer2 同序号 rank
```

| topoLevelNums | infos.size() | infos[0].size() | infos[1].size() | 形状 | 说明 |
|:---:|:---:|:---:|:---:|---|---|
| 3 | **2** | 1 | 1 | 16 × 2 | layer0 被"挤压"进 layer1 |

#### topoLevelNums≠3

| topoLevelNums | 结果 |
|:---:|---|
| 1 | **报错** `HCCL_E_INTERNAL` |
| 2 | **报错** `HCCL_E_INTERNAL` |

> Squeeze2D 是唯一将 layer1 作为算法层 0 的 matcher，将物理 layer0+layer1 合并为一个算法维度。设计文档要求删除此特例。

---

## 4. 汇总对比表

### 4.1 按 topoLevelNums 横向对比

以 32 rank、rank0 为例，场景 B/C/D。

#### topoLevelNums=1（单服务器全覆盖）

| Matcher | infos.size() | infos[0] | infos[1] | infos[2] | 状态 |
|---------|:---:|---|---|---|---|
| TopoMatch1D | 1 | `[0..31]` | — | — | 正常 |
| TopoMatchConcurrent | 1 | `[0..31]`, `[0..31]` | — | — | 正常（2 副本） |
| TopoMatchMultilevel | 2 | `[0..31]` | 退化/报错 | — | 不稳定 |
| TopoMatch3Level | 3 | `[0..31]` | **空** | **空** | 空层问题 |
| TopoMatchUBX | 2 | `[0..31]` | 查 layer3 可能失败 | — | 硬编码问题 |
| TopoMatchUBX1d | 2 | `[0..31]` | 查 layer3 可能失败 | — | 硬编码问题 |
| TopoMatchPcieMix | 2 | mesh ranks | clos ranks | — | 正常（单层拆分） |
| TopoMatchSqueeze2D | — | — | — | — | **报错** |

#### topoLevelNums=2（多服务器，8×4）

| Matcher | infos.size() | infos[0] | infos[1] | infos[2] | 形状 | 状态 |
|---------|:---:|---|---|---|---|---|
| TopoMatch1D | 1 | `[0..31]` | — | — | 32 | 正常 |
| TopoMatchConcurrent | 1 | `[0..31]×2` | — | — | 32 × 2副本 | 正常 |
| TopoMatchMultilevel | 2 | `[0..7]` | `[0,8,16,24]` | — | 8 × 4 | 正常 |
| TopoMatch3Level | 3 | `[0..7]` | `[0,8,16,24]` | **空** | 8 × 4 × ? | 空层问题 |
| TopoMatchUBX | 2 | `[0..7]` | `[0,8,16,24]` | — | 8 × 4 | 正常 |
| TopoMatchUBX1d | 2 | `[0..7]` | `[0,1,...,31]` | — | 8 × 32? | 未按序号过滤 |
| TopoMatchPcieMix | 2 | mesh ranks | clos ranks | — | 4 × 4 | 忽略 layer1 |
| TopoMatchSqueeze2D | — | — | — | — | — | **报错** |

#### topoLevelNums=3（8×2×2）

| Matcher | infos.size() | infos[0] | infos[1] | infos[2] | 形状 | 状态 |
|---------|:---:|---|---|---|---|---|
| TopoMatch1D | 1 | `[0..31]` | — | — | 32 | 正常 |
| TopoMatchConcurrent | — | — | — | — | — | **报错** |
| TopoMatchMultilevel | 3 | `[0..7]` | `[0,8]` | `[0,16]` | 8 × 2 × 2 | 正常 |
| TopoMatch3Level | 3 | `[0..7]` | `[0,8]` | `[0,16]` | 8 × 2 × 2 | 正常 |
| TopoMatchUBX | — | — | — | — | — | **报错** |
| TopoMatchUBX1d | — | — | — | — | — | **报错** |
| TopoMatchPcieMix | 2 | mesh ranks | clos ranks | — | 4 × 4 | 忽略 layer1/2 |
| TopoMatchSqueeze2D | 2 | `[0..15]` | `[0,16]` | — | 16 × 2 | 挤压 layer0→1 |

#### topoLevelNums=3 + HostDPUOnly

| Matcher | infos.size() | infos[0] | infos[1] | infos[2] | 形状 | 状态 |
|---------|:---:|---|---|---|---|---|
| TopoMatchMultilevel | **2** | `[0..7]` | `[0,8,16,24]` | — | 8 × 4 | **降级**（HostDPU 改变层数） |
| TopoMatch3Level | 3 | `[0..7]` | `[0,8]` | `[0,16]`? | 8 × 2 × 2 | 不感知 HostDPU |

---

### 4.2 按 Matcher 纵向对比

| Matcher | 支持的 topoLevelNums | 输出 infos.size() | 是否检查对称 | HostDPU 影响 | 硬编码层号 | 特殊问题 |
|---------|:---:|:---:|:---:|:---:|:---:|---|
| TopoMatch1D | 1~3 | 固定 1 | 否 | 否 | 否 | 无 |
| TopoMatchConcurrent | 1~2 | 固定 1 | 否 | 否 | 否 | infos[0] 有 2 个副本 |
| TopoMatchMultilevel | 1~3 | 2 或 3（动态） | 是 | **是（降级）** | 否 | 物理层数控制算法层数 |
| TopoMatch3Level | 1~3 | 固定 3 | 是 | 否 | 否 | 空层问题 |
| TopoMatchUBX | 1~2 | 固定 2 | 是 | 否 | **是（NetLayer=3）** | 硬编码 ROCE 层 |
| TopoMatchUBX1d | 1~2 | 固定 2 | 是 | 否 | **是（NetLayer=3）** | 不按序号过滤 |
| TopoMatchPcieMix | 1~3 | 固定 2 | 是 | 否 | 否 | 同物理层拆两算法层 |
| TopoMatchSqueeze2D | 仅 3 | 固定 2 | 是 | 否 | 否 | layer0 用 layer1 数据 |

---

## 5. 现有问题清单

### 5.1 物理层数反向控制算法层数

- `TopoMatchMultilevel`：`topoLevelNums=2` → 输出 2 级，`topoLevelNums=3` → 输出 3 级。算法层数由物理层数决定。
- `TopoMatch3Level`：固定输出 3 级，`topoLevelNums<3` 时产生空层。
- `TopoMatchSqueeze2D`：要求 `topoLevelNums==3`，否则报错。

### 5.2 HostDPU 改变算法层数

- `TopoMatchMultilevel`：`hostDPUOnly=true` 时将 3 级降级为 2 级，`infos.size()` 从 3 变为 2。
- 其他 matcher 不感知 HostDPU。

### 5.3 硬编码物理层编号

- `TopoMatchUBX`：`NATLAYER_ROCE=3`，硬编码查询 NetLayer 3。
- `TopoMatchUBX1d`：`NATLAYER_THREE=3`，同上。
- 不能处理 `netLayers={0,3}` 等非连续编号场景。

### 5.4 空层问题

- `TopoMatch3Level`：`topoLevelNums < 3` 时 `infos[1]` 或 `infos[2]` 为空，消费者越界。

### 5.5 数据契约不统一

- `TopoMatchConcurrent`：`infos[0].size()==2`（并发副本），其他 matcher `infos[0].size()==1`。
- `TopoMatchUBX`（Mesh1DClos）：`infos[0].size()==2`（mesh + clos），其他场景 `infos[0].size()==1`。
- `TopoMatchPcieMix`：将同一物理层的两种 TopoType 拆到两个算法层。
- `TopoMatchSqueeze2D`：算法层 0 使用物理 layer1 数据。

### 5.6 不按序号过滤

- `TopoMatchUBX1d`：`TopoForLayer3` 不做 `rankId % layer0Size` 过滤，`infos[1]` 可能包含全部 rank，维度乘积不等于 rankSize。

### 5.7 对称性校验不一致

- `TopoMatchMultilevel`：支持非对称（GCD 拆分），但 3 级非对称报错。
- `TopoMatch3Level`：完全不支持非对称。
- `TopoMatchUBX`/`PcieMix`：要求对称，非对称直接报错。
- `TopoMatchSqueeze2D`：要求对称。

### 5.8 层选择不一致

| Matcher | 算法层0 数据来源 | 算法层1 数据来源 | 算法层2 数据来源 |
|---------|----------------|----------------|----------------|
| Multilevel | layer0 TopoInst | layer1 或最高层（HostDPU） | 最高层 |
| 3Level | layer0 TopoInst | layer1 TopoInst | layer2 TopoInst |
| UBX | layer0 TopoInst | **layer3**（硬编码） | — |
| UBX1d | layer0 TopoInst | **layer3**（硬编码） | — |
| PcieMix | layer0 的 1DMESH | layer0 的 CLOS | — |
| Squeeze2D | **layer1** TopoInst | **layer2** TopoInst | — |

---

## 6. 与设计文档目标行为的对比

| 场景 | 现有行为（最接近的 matcher） | 设计目标行为 | 差异 |
|------|---------------------------|-------------|------|
| topoLevelNums=1, algLevel=1 | TopoMatch1D: `[[[0..N-1]]]` | TopoMatchLevel1: `[[[0..N-1]]]` | 无差异 |
| topoLevelNums=2, algLevel=2 | Multilevel: `[[[0..7]],[[0,8,16,24]]]` | TopoMatchLevel2: `[[[0..7]],[[0,8,16,24]]]` | 无差异（对称场景） |
| topoLevelNums=3, algLevel=2 | Multilevel: `[[[0..7]],[[0,8]],[[0,16]]]` (3级!) | TopoMatchLevel2: `[[[0..7]],[[0,8,16,24]]]` (固定2级) | **现有输出3级，目标应输出2级** |
| topoLevelNums=3, algLevel=3 | Multilevel/3Level: `[[[0..7]],[[0,8]],[[0,16]]]` | TopoMatchLevel3: `[[[0..7]],[[0,8]],[[0,16]]]` | 无差异（正常3级） |
| topoLevelNums=3+HostDPU, algLevel=3 | Multilevel: 降级为2级 | TopoMatchLevel3: 仍为3级或返回不匹配 | **HostDPU 不应改变算法层数** |
| topoLevelNums=2, algLevel=3 | 3Level: `[[[0..7]],[[0,8,16,24]],[]]` (空层) | TopoMatchLevel3: 返回不匹配 | **不应返回空层** |
| Concurrent | `infos[0].size()==2` | `infos[0].size()==1` | **并发副本移入 Executor** |
| UBX | 硬编码 NetLayer=3 | UBX 移入物理绑定 | **不硬编码层号** |
| PcieMix | 同层拆两算法层 | 移入物理绑定 | **不拆分算法层** |
| Squeeze2D | layer0 用 layer1 数据 | 统一 GCD 规则 | **不挤压** |

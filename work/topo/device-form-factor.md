# 整机形态获取方案对比：`ACL_DEV_ATTR_DEVICE_FORM_FACTOR` vs `CcuGetMainboardId` 式位域解析

> 背景：SE 需求「判断是否 POD 机型」（POD 的交换机层是 2:1 收敛，cost model 需要据此修正性能模型）。
> 本文对比两种在 HCCL 仓落地的方式。当前工作区已按**方式 A** 实现。

---

## 0. 结论速览

| | 方式 A：`ACL_DEV_ATTR_DEVICE_FORM_FACTOR` | 方式 B：`ACL_DEV_ATTR_MAINBOARD_ID` + 位域解析 |
|---|---|---|
| 读取的硬件字段 | 同一个 | 同一个 |
| HCCL 侧代码量 | 少（无位运算、无映射表） | 多（位移 + 掩码 + 8 值映射表） |
| 能拿到的信息 | 仅整机形态 | 整机形态 + 形态细分 + 主从/池化 |
| 取值空间覆盖 | ACL 只定义 4 个宏（0–3） | 完整 8 个取值（0–7） |
| 生产验证 | **hcomm 全仓零使用者** | hcomm 两处在用，跑在现网 |
| 参考实现 | 无 | `CcuGetMainboardId` / `HrtGetMainboardId` |
| 维护耦合 | 无本地表 | 需复制一张 8 值映射表 |

**两者读的是同一个硬件寄存器字段，语义等价。** 差别只在「谁做位运算」和「顺带能不能拿到另外两段位」。

---

## 1. 共同的事实基础：MAINBOARD_ID 是一个位域

这是理解两种方式关系的关键。`ACL_DEV_ATTR_MAINBOARD_ID`（`acl_rt.h:686`，枚举值 407）返回的**不是一个机型编号，而是一个 16 bit 位域**。

布局见 hcomm `src/base_comm/resources/ccu/ccu_device/ccu_res_specs.cc:186-208` 的注释：

| 位段 | 含义 | 取值 |
|---|---|---|
| `bit[7:5]` | **整机形态** | `000` 天成 POD / `001` A+K Server / `010` A+X Server / `011` PCIE 标卡 / `100`–`101` RSV / `110` 装备 / `111` EVB |
| `bit[4:1]` | 形态细分 | `0000`–`1111` |
| `bit[0]` | 主从或池化 | `0` 主从（NPU 作为某 Host 的从设备）/ `1` 池化（NPU 作为资源池，其它 Host 对等访问） |

注释另标注：**当前 POD 和 EVB 不区分 A+X 或 A+K**。

### 用这个布局解码 hcomm 的机型常量表，完全自洽

`hcomm/src/legacy/ascend950/framework/topo/topo_addr_info/src/hal.h:72-83` 里那 11 个 `MAIN_BOARD_ID_*` 常量按上述位域拆开：

```
hal.h 常量                  值      bin        [7:5]形态       [4:1]细分   [0]
POD_2D                    0x03   00000011   0=POD           1        池化
POD                       0x07   00000111   0=POD           3        池化
SERVER_TYPE1              0x23   00100011   1=A_K Server    1        池化
SERVER_8PMESH             0x25   00100101   1=A_K Server    2        池化
SERVER_8PMESH_UBOE        0x27   00100111   1=A_K Server    3        池化
SERVER_8PMESH_NOSP        0x29   00101001   1=A_K Server    4        池化
SERVER_8PMESH_NOSP_UBOE   0x2B   00101011   1=A_K Server    5        池化
SERVER_16PMESH / UBX      0x44   01000100   2=A_X Server    2        主从
CARD_NOMESH               0x68   01101000   3=PCIE标卡       4        主从
CARD_2PMESH               0x6A   01101010   3=PCIE标卡       5        主从
CARD_4PMESH               0x6C   01101100   3=PCIE标卡       6        主从
```

11 个常量的命名（POD / SERVER / CARD）精确落在 `bit[7:5]` 上，同形态的不同型号只差 `bit[4:1]`。这不是巧合，是同一份编码规范在两个模块的两处使用。

### 与 ACL 形态宏的对应关系

`acl_rt.h:696-699`：

```c
#define ACL_DEVICE_FORM_FACTOR_POD       0   // == bit[7:5] 的 000
#define ACL_DEVICE_FORM_FACTOR_A_K       1   // == 001
#define ACL_DEVICE_FORM_FACTOR_A_X       2   // == 010
#define ACL_DEVICE_FORM_FACTOR_PCIE_CARD 3   // == 011
```

**四个值与 `bit[7:5]` 的前四个取值逐一相等。** 因此可以判定：`ACL_DEV_ATTR_DEVICE_FORM_FACTOR`（409）就是驱动侧替调用方做了 `(mainboardId >> 5) & 0x7` 的便捷接口。

---

## 2. 方式 B 的参考实现（hcomm 现状）

hcomm 里有**两份完全复制粘贴**的实现，位域注释、常量、映射表一模一样：

- `src/base_comm/resources/ccu/ccu_device/ccu_res_specs.cc:229` — `CcuGetMainboardId`
- `src/legacy/ascend950/unified_platform/external_system/orion_adapter_rts.cc:350` — `HrtGetMainboardId`

以 `CcuGetMainboardId` 为例：

```cpp
HcclResult CcuGetMainboardId(uint32_t deviceLogicId, Hccl::HcclMainboardId& hcclMainboardId)
{
    constexpr aclrtDevAttr devAttr = ACL_DEV_ATTR_MAINBOARD_ID;
    constexpr uint64_t BITS_5 = 5;
    constexpr uint64_t MASK_7 = 0x7;
    int64_t val = 0;
    auto ret = aclrtGetDeviceInfo(deviceLogicId, devAttr, &val);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(...);
        return HcclResult::HCCL_E_RUNTIME;      // 不降级，直接失败
    }
    uint64_t mainboardId = (static_cast<uint64_t>(val) >> BITS_5) & MASK_7;  // 提取 bit[7:5]
    hcclMainboardId = MAINBOARD_OTHERS;                                       // 先置未知
    auto it = rtMainboardIdToHcclMainboardId.find(mainboardId);
    if (it != rtMainboardIdToHcclMainboardId.end()) {
        hcclMainboardId = it->second;                                         // 命中才覆盖
    }
    return HcclResult::HCCL_SUCCESS;
}
```

配套的 8 值映射表（`ccu_res_specs.cc:211-227`）：

```cpp
POD_MAINBOARD          0x0 -> MAINBOARD_POD
A_K_SERVER_MAINBOARD   0x1 -> MAINBOARD_A_K_SERVER
A_X_SERVER_MAINBOARD   0x2 -> MAINBOARD_A_X_SERVER
PCIE_STD_MAINBOARD     0x3 -> MAINBOARD_PCIE_STD
RSV1_MAINBOARD         0x4 -> MAINBOARD_RSV
RSV2_MAINBOARD         0x5 -> MAINBOARD_RSV      // 两个 RSV 合并成一个枚举
EQUIP_MAINBOARD        0x6 -> MAINBOARD_EQUIPMENT
EVB_MAINBOARD          0x7 -> MAINBOARD_EVB
```

几个值得注意的实现细节：

- **函数名有误导性**：叫 `GetMainboardId`，实际返回的是从 mainboard id 里抠出来的「整机形态」，不是 mainboard id 本身。
- **白名单而非 default 落值**：先置 `MAINBOARD_OTHERS`，命中才覆盖。因为 `MAINBOARD_POD` 是枚举第一项、原始值是 0，任何「未识别按默认处理」的写法都会滑向 POD。
- **两份实现有细微差异**：`HrtGetMainboardId` 多了一条 `val < 0` 的校验（`orion_adapter_rts.cc:359`），`CcuGetMainboardId` 没有。属于复制粘贴后单边演进。
- **错误不降级**：取不到直接返回 `HCCL_E_RUNTIME`。因为它的消费者（CCU 资源规格初始化、建链模式判断）拿不到值就真的没法继续。

### hcomm 侧的消费者

都是**设备能力 / 资源规格**判断，不涉及算法选择：

| 位置 | 用途 |
|---|---|
| `ccu_res_specs.cc:259` `CheckServeMode` | `A_X_SERVER` 或 `PCIE_STD` → `ServeMode::ARMX86`，否则 `NORMAL` |
| `communicator_impl.cc:2862` | 标卡（`MAINBOARD_PCIE_STD`）下配置 CCU_MS 加速模式时拦截报错 |
| `tp_manager.cc:63` | 置 `isPcieStd` 标志 |
| `ccu_dev_mgr_imp.cc:188/288/395` | CCU 设备管理 |

---

## 3. 方式 A 在 HCCL 仓的实现（当前工作区）

### 取值

`src/ops/op_common/topo/topo_host.cc` 的 `CalcDeviceFormFactor()`：

```cpp
s32 userDevId = 0;
aclrtGetDevice(&userDevId);                              // 返回的是 userDevId
s32 logicDevId = 0;
aclrtGetLogicDevIdByUserDevId(userDevId, &logicDevId);   // aclrtGetDeviceInfo 要的是 logicDevId
s64 val = 0;
hcalrtGetDeviceInfo(logicDevId, ACL_DEV_ATTR_DEVICE_FORM_FACTOR, val, /*quiet=*/true);
topoInfo->isPod = (val == ACL_DEVICE_FORM_FACTOR_POD);   // 严格相等，未识别值落 false
```

**没有位运算、没有映射表**——驱动侧已经把 `bit[7:5]` 抠好了。

### 落点

- `src/ops/op_common/inc/alg_param.h` — `TopoInfoWithNetLayerDetails::isPod`（`bool`，默认 `false`）
- `src/common/adapter_acl.cc:108` — 白名单放行 `ACL_DEV_ATTR_DEVICE_FORM_FACTOR`
- `src/ops/op_common/topo/topo_host.cc` — `CalcDeviceFormFactor()` 实现与 `CalcTopoShape` 挂载点

**只落一个 `bool`，不在 `alg_param.h` 里复制 ACL 的形态枚举。** 消费侧当前只用「POD ⇒ 交换机层
2:1 收敛」这一条，`A_K`/`A_X`/`PCIE_CARD` 之间的区分对建模没有差异；`alg_param.h` 同时参与
device/AICPU 侧编译、引不进 `acl_rt.h`，落 bool 也就省掉了一份必须与 ACL 同步维护的复制
（连带去掉了原先守护这份复制的 `static_assert`）。

### 与 hcomm 做法一致的地方

- 白名单已知取值，未识别落 UNKNOWN，绝不 default 到第一项（`POD == 0` 的坑两边都要防）。
- **入参用 logicDevId**。这一点两种方式完全相同——407 和 409 走的是同一个 `aclrtGetDeviceInfo`，
  底层 `halGetDeviceInfo` 对 `devId` 的要求是"除 `INFO_TYPE_MASTERID` 外一律使用 logical device ID"
  （`ascend_hal_base.h:1310-1311`）。hcomm 的 `CcuGetMainboardId` 形参名就是 `deviceLogicId`，
  `hal_get_mainboard_id` 则是 `phyId → userDevId → logicDevId` 转两次再调
  （`hal.c` 的 `hal_get_logicid_from_phyid`）。

  **注意 `aclrtGetDevice` 返回的是 userDevId 而不是 logicDevId**，必须经
  `aclrtGetLogicDevIdByUserDevId` 转换（hcomm `communicator_impl.cc:3601` 同样这么转）。
  默认部署下两者相等，漏转在大多数环境上测不出来。

  另：`aclrtGetLogicDevIdByPhyDevId` **名不副实**，hcomm `hal.c` 注明"该接口语义错误，实际返回的是
  UserDevId"并改用 `aclrtGetUserDevIdByPhyDevId`。本仓 `adapter_acl.cc` 的
  `haclrtGetDeviceIndexByPhyId` 正是用它填一个叫 `deviceLogicId` 的出参——目前无调用者，属潜在隐患。

### 与 hcomm 做法有意不同的地方

- **恒返回 `HCCL_SUCCESS`，取不到降级为 `UNKNOWN`**。因为 cost model 的输入属于附加信息，不该让通信域起不来；而 hcomm 那两个函数的消费者是硬依赖。
- 枚举值在 `alg_param.h` 是 ACL 宏的一份复制（该头文件参与 device/AICPU 侧编译，引不进 host 侧的 `acl_rt.h`），靠 `topo_host.cc:781` 的 `static_assert` 把两套值钉死——ACL 改值会直接编译失败。

---

## 4. 逐项差异

### 4.1 代码量

| | 方式 A | 方式 B |
|---|---|---|
| 位移/掩码常量 | 无 | `BITS_5`、`MASK_7` |
| 原始值→枚举映射表 | 无 | 8 条 |
| 取值分支 | 4 case 白名单 | 8 条表 + 未命中兜底 |

方式 B 多出来的那张表是**从 hcomm 复制的**。hcomm 自己已经复制了两份，HCCL 再复制就是第三份，且跨仓——上游改动没有任何编译期或运行期提示。

方式 A 没有这个问题：形态语义由驱动维护，HCCL 只有 4 个宏值的 `static_assert` 需要守。

### 4.2 能拿到的信息

| 信息 | 方式 A | 方式 B |
|---|---|---|
| 整机形态 `bit[7:5]` | ✅ | ✅ |
| 形态细分 `bit[4:1]` | ❌ | ✅ |
| 主从/池化 `bit[0]` | ❌ | ✅ |

**这是方式 B 唯一实质性的功能优势。**

当前需求（判 POD、算 2:1 收敛）只要形态，两者等价。但如果 cost model 后续需要：

- 区分同为 A_K Server 的 `SERVER_8PMESH` 与 `SERVER_8PMESH_NOSP`（细分位不同，网络拓扑不同）；
- 或区分主从/池化部署（`bit[0]`，影响 Host-Device 数据通路）；

那 409 给不了，必须换成方式 B。**建议向 SE 确认 cost model 的建模粒度**——如果只到「POD / 非 POD」，方式 A 足够；如果要到具体型号，现在就该切。

### 4.3 取值空间覆盖

`bit[7:5]` 有 8 个取值，但 ACL 只定义了 4 个宏。`100`/`101`(RSV)、`110`(装备)、`111`(EVB) 没有对应宏。

- 方式 A：这些值落进 `default` → `UNKNOWN` + WARNING。**这不是理论防御——EVB 评估板真实存在**，说明该分支可达。
- 方式 B：映射表覆盖全部 8 值，能明确区分「这是 EVB」和「这是个我没见过的值」。

对当前需求（cost model 判 POD）两者结果一样：非 POD 都走保守默认。方式 B 的额外区分度目前用不上。

### 4.4 生产验证成熟度

**这是选方式 B 的最强论据。**

- 在 hcomm 全仓 grep `FORM_FACTOR`：**零命中**。所有形态判断都走 `ACL_DEV_ATTR_MAINBOARD_ID` + 手动移位。
- `ACL_DEV_ATTR_DEVICE_FORM_FACTOR` 在 CANN 9.2.0 的 `acl_rt.h` 里有定义，但在本地能看到的代码里没有任何使用者。

也就是说，方式 B 走的是**跑在现网上的路径**，方式 A 走的是**有定义但未见生产使用**的路径。如果 409 在目标 CANN 版本的驱动侧还没真正打通，方式 A 会稳定失败并降级到 `UNKNOWN`——功能上安全（不会误判成 POD），但 cost model 永远拿不到 POD 信息，等于需求没实现，且现象隐蔽（只有 WARNING 日志）。

**缓解办法**：上板后确认一次 `HCCL_INFO` 里的 `[Topo][CalcDeviceFormFactor] userDevId[x] logicDevId[y] formFactor[z] isPod[w]`。POD 环境应打出 `formFactor[0] isPod[1]`；若打出的是「get device form factor failed」的 WARNING，说明 409 不通，需切方式 B。日志里带 ACL 原始取值，正是为了在 `isPod` 为 `false` 时分辨「确实不是 POD」还是「取到了个没见过的形态」。

### 4.5 版本兼容

| | 方式 A | 方式 B |
|---|---|---|
| 依赖的 ACL 枚举 | `ACL_DEV_ATTR_DEVICE_FORM_FACTOR` (409) | `ACL_DEV_ATTR_MAINBOARD_ID` (407) |
| 依赖的宏 | `ACL_DEVICE_FORM_FACTOR_POD` | 无（自己定义位移常量） |
| CANN 9.2.0 | 已验证存在（编译通过） | 已验证存在 |
| CANN 8.5.0 | **未验证** | **未验证**，但 407 是更老的属性，存在概率更高 |

两者在 8.5.0 上都没实测。若缺失，都需要按 `hccl_rank_graph_dl.h:18-32` 的方式补桩。407 作为更基础的属性，在老版本上存在的可能性更大——这也偏向方式 B。

### 4.6 错误语义

两种方式在 HCCL 仓都应该**降级而非失败**（`isPod` 是 cost model 的附加输入，不是算法分层的必需项），这一点与 hcomm 的硬失败不同。所以错误处理不构成两者的差异。

---

## 5. 若切换到方式 B，改动范围

当前实现改动很局部，切换成本低：

1. `src/common/adapter_acl.cc:103` 白名单：`ACL_DEV_ATTR_DEVICE_FORM_FACTOR` → `ACL_DEV_ATTR_MAINBOARD_ID`（或两个都放行）。
2. `CalcDeviceFormFactor()`：改为读 407，把相等判断换成 `((val >> 5) & 0x7) == 0`（POD 在位域里同样是 0）。

**`alg_param.h` 的 `isPod` 字段、序列化、`portNums` 全部不受影响**，因为存的是归一化后的结论，不是原始 `mainboardId`。

预计改动量：约 5–10 行——落 bool 之后比原先存形态枚举的方案更小（不再需要扩 `switch`、不再需要维护 `static_assert`）。

---

## 6. 建议

**短期保留方式 A**，理由：

- 语义与 SE 的需求（通过 MAINBOARD_ID 判 POD）完全等价——读的是同一个硬件字段的同一段位；
- HCCL 侧不引入第三份跨仓复制的映射表；
- 代码已实现、已编译通过。

**但必须做一次上板验证**：确认 `ACL_DEV_ATTR_DEVICE_FORM_FACTOR` 在目标 CANN/驱动版本上真的返回值，而不是稳定失败。验证点见 4.4。

**以下任一条成立时切方式 B**：

1. 上板验证发现 409 不通；
2. cost model 需要形态细分（`bit[4:1]`）或主从/池化（`bit[0]`）；
3. 目标版本包含 CANN 8.5.0 且实测 409 在该版本缺失（407 更可能可用）。

---

## 附：关键代码位置索引

**HCCL 仓（本次改动）**

| 内容 | 位置 |
|---|---|
| `isPod` 字段 | `src/ops/op_common/inc/alg_param.h` |
| ACL 白名单 | `src/common/adapter_acl.cc:103-108` |
| `CalcDeviceFormFactor()` | `src/ops/op_common/topo/topo_host.cc` |
| 挂载点 | `src/ops/op_common/topo/topo_host.cc`（`CalcTopoShape` 末尾） |

**hcomm 仓（参考实现）**

| 内容 | 位置 |
|---|---|
| 位域布局注释 | `src/base_comm/resources/ccu/ccu_device/ccu_res_specs.cc:186-208` |
| `CcuGetMainboardId` | `src/base_comm/resources/ccu/ccu_device/ccu_res_specs.cc:229` |
| `HrtGetMainboardId`（第二份复制） | `src/legacy/ascend950/unified_platform/external_system/orion_adapter_rts.cc:350` |
| `HcclMainboardId` 枚举 | `src/legacy/ascend950/common/types/dev_type.h:21` |
| 机型常量表 `MAIN_BOARD_ID_*` | `src/legacy/ascend950/framework/topo/topo_addr_info/src/hal.h:72-83` |
| `hal_get_mainboard_id`（HAL 层等价物） | `src/legacy/ascend950/framework/topo/topo_addr_info/src/hal.c:143` |

**CANN 头文件**

| 内容 | 位置 |
|---|---|
| `ACL_DEV_ATTR_MAINBOARD_ID = 407` | `include/acl/acl_rt.h:686` |
| `ACL_DEV_ATTR_DEVICE_FORM_FACTOR = 409` | `include/acl/acl_rt.h:688` |
| `ACL_DEVICE_FORM_FACTOR_*` | `include/acl/acl_rt.h:696-699` |
| `aclrtGetDeviceInfo` | `include/acl/acl_rt.h:3948` |
| `INFO_TYPE_MAINBOARD_ID`（HAL 层，=39） | `pkg_inc/driver/ascend_hal_base.h:406` |

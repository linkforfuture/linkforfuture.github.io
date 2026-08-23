# ParallelExecutor数据切分实现审查报告

## 1. 审查结论

复审结论：在本轮约定范围内，未发现新的编译阻断或会导致相关算子无法跑通的逻辑错误。首次审查中的2.1、2.2和2.5已修正；2.4的当前实现也不再吞掉 `HCCL_E_PARA`。2.3和2.6按本轮要求暂缓处理，不作为当前算子跑通的阻断项。

本轮完成以下验证：

- 使用项目 `compile_commands.json` 的实际编译参数，对本次修改的7个源文件执行C++14语法编译，全部通过。
- 使用当前不包含新增切分比例枚举的Hcomm头文件编译 `op_common.cc`，SFINAE降级分支生效。
- 使用最小C++14示例验证枚举存在和不存在两种场景均能通过模板实例化。
- 核对五个Executor的公式类型、Rank数、Channel Map和比例方向，与设计文档一致。
- `git diff --check`通过。

## 2. 审查发现

### 2.1 [已修复] C++14下使用 `std::clamp`

**复审结果：** 已改为C++14支持的 `std::max`/`std::min`，并使用 `std::isfinite` 处理非有限回退值。相关源文件语法编译通过。

**位置：** `src/ops/op_common/template/template_utils.cc:183`

项目在根 `CMakeLists.txt:45`中明确使用C++14，但新实现使用了C++17才引入的：

```cpp
const double validFallback = std::clamp(fallbackRatio, 0.0, 1.0);
```

使用 `g++ -std=c++14` 验证时会报错：

```text
error: 'clamp' is not a member of 'std'
```

**影响：** Host和相关测试目标无法编译。

**建议：** 改为C++14可用的显式限制，并同时处理非有限值：

```cpp
const double validFallback = std::isfinite(fallbackRatio)
    ? std::max(0.0, std::min(fallbackRatio, 1.0))
    : 0.5;
```

### 2.2 [已修复] 通信域配置能力不能仅通过接口符号或CANN版本判断

**复审结果：** 已删除兼容桩中虚构的切分比例枚举，改用C++14 SFINAE检测当前Hcomm头文件是否包含新增枚举。枚举不存在时不实例化查询分支并降级到环境变量或内置公式；枚举存在时仍需运行时检查 `HcclConfigGetInfo`。当前SDK实编译通过。

**位置：**

- `src/common/hcomm_dlsym/hccl_host_comm_dl.h:18-34`
- `src/ops/op_common/op_common.cc:2224-2225`

当前实现通过 `CANN_VERSION_NUM` 决定是否使用本地 `HcclConfigType` 兼容桩，并在旧版本兼容桩中补充了
`HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO`。对9.1 beta1及以上版本，则直接使用SDK的
`hccl_comm.h`。当前本机9.1 beta1头文件仅定义：

```cpp
HCCL_CONFIG_TYPE_INVALID = -1,
HCCL_CONFIG_TYPE_OP_EXPANSION_MODE = 0
```

编译引用新枚举的最小示例会报错：

```text
error: 'HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO'
       is not a member of 'HcclConfigType'
```

更关键的问题是，Hcomm需要支持独立升级：某个旧版本Hcomm可能已经导出 `HcclConfigGetInfo`，但其配套头文件中的
`HcclConfigType` 不一定包含新增的切分比例配置项。因此，存在接口符号并不表示能够查询切分比例；固定的CANN版本门槛
也不能准确描述Hcomm头文件实际具备的能力。

**影响：**

- 当前配套SDK会因为直接引用不存在的枚举成员而编译失败。
- 在兼容桩中自行补充枚举项或硬编码枚举数值，会让代码假定旧Hcomm支持一个实际不存在的ABI能力，存在错误调用风险。
- 仅用 `dlsym` 判断 `HcclConfigGetInfo` 是否存在，无法区分接口只支持展开模式还是同时支持切分比例。

**修改方案：基于头文件实际能力进行编译期检测，运行时继续检测接口符号。**

1. 旧版本兼容桩只镜像该版本真实存在的类型和枚举项，删除本地补充的
   `HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO`，不猜测或硬编码其ABI数值。
2. 使用C++14可用的SFINAE检测当前编译所用 `HcclConfigType` 是否包含
   `HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO`。
3. 通过模板重载隔离新增枚举的引用。枚举不存在时，该引用所在的函数模板不会实例化，从而保证旧头文件可以正常编译。
4. 枚举不存在时，通信域配置能力视为不可用，`GetCommMultipleDimensionSplitRatio` 返回成功且
   `isConfigured = false`，继续按“环境变量配置 > 内置公式”降级。
5. 枚举存在时，再检查运行时Hcomm是否导出 `HcclConfigGetInfo`。只有编译期枚举和运行时接口同时存在，才发起通信域查询。

能力检测可按以下方式实现：

```cpp
template <typename...>
using VoidT = void;

template <typename T, typename = void>
struct HasSplitRatioConfigType : std::false_type {};

template <typename T>
struct HasSplitRatioConfigType<T,
    VoidT<decltype(T::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO)>>
    : std::true_type {};
```

查询逻辑通过模板重载分离：

```cpp
template <typename ConfigType>
HcclResult QueryCommSplitRatio(
    HcclComm comm, double &ratio, bool &isConfigured, std::false_type)
{
    HCCL_INFO("Current Hcomm headers do not support split ratio config, skip comm config.");
    return HCCL_SUCCESS;
}

template <typename ConfigType>
HcclResult QueryCommSplitRatio(
    HcclComm comm, double &ratio, bool &isConfigured, std::true_type)
{
    return QuerySplitRatioByConfigGetInfo(
        comm,
        ConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO,
        ratio,
        isConfigured);
}
```

入口调用：

```cpp
return QueryCommSplitRatio<HcclConfigType>(
    comm,
    ratio,
    isConfigured,
    HasSplitRatioConfigType<HcclConfigType>{});
```

最终兼容行为如下：

| 编译时Hcomm头文件 | 运行时 `HcclConfigGetInfo` | 处理方式 |
|---|---|---|
| 不包含新增枚举 | 不存在或存在 | 不查询通信域，降级到环境变量或内置公式 |
| 包含新增枚举 | 不存在 | 不查询通信域，降级到环境变量或内置公式 |
| 包含新增枚举 | 存在 | 查询通信域配置，未配置时继续降级 |

该方案以实际参与编译的Hcomm头文件作为配置类型能力来源，以运行时符号作为接口可调用能力来源，避免将CANN版本、
接口存在性或未经确认的枚举数值等同于切分比例配置能力。

#### 新头文件编译、旧Hcomm SO运行时的行为

还需要考虑编译环境的 `HcclConfigType` 已包含
`HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO`，但部署环境中的Hcomm SO尚未提供该配置能力的场景。

枚举定义只参与编译，不需要由运行时SO导出。枚举成员在编译后会固化为对应的整数立即数。例如新增配置项的枚举值为1时，
运行时调用等价于：

```cpp
dlHcclConfigGetInfo(
    comm,
    static_cast<HcclConfigType>(1),
    sizeof(double),
    &commRatio);
```

因此兼容行为取决于运行时SO是否存在接口符号，以及旧接口如何处理未知的 `cfgType`：

1. 运行时SO未导出 `HcclConfigGetInfo`：`dlsym`得到空函数指针，代码不会调用该接口，直接降级。
2. 运行时SO导出了接口但只支持展开模式：接口会收到编译时固化的新增枚举整数值。
3. 已确认旧Hcomm实现包含以下判断：

```cpp
if (cfgType != HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE) {
    HCCL_ERROR("[%s] cfgType[%d] is not supported yet.", __func__, cfgType);
    return HcclResult::HCCL_E_NOT_SUPPORT;
}
```

旧SO收到切分比例配置类型后会返回 `HCCL_E_NOT_SUPPORT`。当前
`QuerySplitRatioByConfigGetInfo` 将该返回值解释为运行时不支持通信域切分比例，保持
`isConfigured = false`并返回成功，随后继续按“环境变量配置 > 内置公式”降级，不会阻断算子执行。

据此，当前方案覆盖以下组合：

| 编译环境 | 运行时Hcomm SO | 实际行为 |
|---|---|---|
| 头文件无新增枚举 | 任意版本 | 编译期选择SFINAE降级分支，不调用通信域切分比例配置 |
| 头文件有新增枚举 | SO无 `HcclConfigGetInfo` | 运行时符号检测失败，降级 |
| 头文件有新增枚举 | SO有接口但无新增配置能力 | 接口返回 `HCCL_E_NOT_SUPPORT`，降级 |
| 头文件有新增枚举 | SO有接口且支持新增配置能力 | 正常读取通信域切分比例 |

该运行时兼容结论依赖Hcomm接口契约：未知或尚未支持的 `cfgType` 必须返回
`HCCL_E_NOT_SUPPORT`，不能将该整数解释为其他配置类型，也不能按不匹配的数据布局写入输出缓冲区。当前旧Hcomm实现满足这一条件。

#### 2.2.1 编译期检测：`HcclConfigType` 是否包含切分比例枚举

当前项目中 `HcclConfigType` 有两个来源（见 `hccl_host_comm_dl.h:17-37`）：

- **旧版本**（`CANN_VERSION_NUM < 9.1.0_beta.1`）：使用本地兼容桩定义，只有 `HCCL_CONFIG_TYPE_INVALID` 和 `HCCL_CONFIG_TYPE_OP_EXPANSION_MODE`。
- **9.1 beta1 及以上**：直接使用 SDK 的 `hccl_comm.h`（通过 `#include "hccl_comm.h"` 引入），而本机 9.1 beta1 头文件中 `HcclConfigType` 也只定义了这两项，**没有** `HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO`。

如果代码里直接写 `HcclConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO`，当编译所用的 `hccl_comm.h` 不包含这个枚举成员时，会直接编译失败。而且后续 SDK 升级可能会增加该枚举，所以不能硬编码数值也不能预先猜测。

代码使用 C++14 可用的 SFINAE（Substitution Failure Is Not An Error）来检测当前参与编译的 `HcclConfigType` 是否包含 `HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO`：

```cpp
template <typename...>
using VoidT = void;                        // 辅助别名，等价于 void

template <typename T, typename = void>
struct HasSplitRatioConfigType : std::false_type {};   // 主模板：默认继承 false_type

template <typename T>
struct HasSplitRatioConfigType<T,
    VoidT<decltype(T::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO)>>
    : std::true_type {};                               // 特化模板：如果表达式合法则继承 true_type
```

编译器遇到 `HasSplitRatioConfigType<HcclConfigType>` 时，会同时尝试匹配主模板和特化模板。特化模板的第二个模板参数是 `VoidT<decltype(T::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO)>`，这要求编译器能成功解析 `HcclConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO` 这个表达式。

- **如果 `HcclConfigType` 包含该枚举成员**（后续SDK升级后）：`decltype(...)` 成功解析，特化模板的第二个参数推导为 `void`，与主模板的默认参数 `void` 匹配。特化比主模板更特殊，编译器选择特化，结果为 `std::true_type`。
- **如果 `HcclConfigType` 不包含该枚举成员**（当前本机SDK）：`HcclConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO` 不存在，`decltype(...)` 替换失败。但 SFINAE 规则保证**替换失败不是错误**——编译器只是丢弃这个特化，回退到主模板，结果为 `std::false_type`。

关键在于：特化模板中引用 `HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO` 的代码只有在 SFINAE 检测成功时才会进入类型系统，不会产生编译错误。

即使 SFINAE 检测通过（`true_type`），仍然需要一个地方实际写出 `HcclConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO` 来调用接口。这里通过两个模板重载函数来隔离：

```cpp
// 枚举不存在时调用（false_type 版本）
template <typename ConfigType>
HcclResult QueryCommSplitRatio(
    HcclComm comm, double &ratio, bool &isConfigured, std::false_type)
{
    HCCL_INFO("Current Hcomm headers do not support split ratio config, skip comm config.");
    ratio = 0.0;
    isConfigured = false;
    return HCCL_SUCCESS;       // 直接降级，不查询通信域
}

// 枚举存在时调用（true_type 版本）——只有这个函数引用了枚举成员
template <typename ConfigType>
HcclResult QueryCommSplitRatio(
    HcclComm comm, double &ratio, bool &isConfigured, std::true_type)
{
    return QuerySplitRatioByConfigGetInfo(
        comm,
        ConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO,  // 唯一引用枚举成员的地方
        ratio,
        isConfigured);
}
```

`std::false_type` 和 `std::true_type` 是不同类型。调用者通过 `HasSplitRatioConfigType<HcclConfigType>{}` 传入的标签类型决定了编译器实例化哪个重载。当枚举不存在时：

1. SFINAE 检测结果是 `std::false_type`
2. 调用 `QueryCommSplitRatio<HcclConfigType>(..., std::false_type{})`
3. 编译器只实例化 `std::false_type` 重载
4. `std::true_type` 重载中的 `ConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO` **永远不会被编译器看到**，因为它不在被实例化的函数中

因此即使当前 SDK 的 `HcclConfigType` 没有这个枚举成员，代码也能正常编译。

入口函数将编译期检测结果转为运行时标签，驱动模板重载选择：

```cpp
HcclResult GetCommMultipleDimensionSplitRatio(HcclComm comm, double &ratio, bool &isConfigured)
{
    return QueryCommSplitRatio<HcclConfigType>(
        comm,
        ratio,
        isConfigured,
        HasSplitRatioConfigType<HcclConfigType>{});
}
```

#### 2.2.2 运行时检测：`HcclConfigGetInfo` 接口是否可调用

即使编译期确认 `HcclConfigType` 包含切分比例枚举，运行时的 Hcomm 动态库不一定导出了 `HcclConfigGetInfo` 符号。这项检测在 `QuerySplitRatioByConfigGetInfo` 中完成：

```cpp
auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
if (!hcommFunction.dlHcclConfigGetInfo) {    // ← 运行时检测点
    HCCL_INFO("HcclConfigGetInfo is not supported, skip comm config.");
    return HCCL_SUCCESS;
}
```

`dlHcclConfigGetInfo` 是一个 `std::function`，它的值来自 `dlhcomm_function.cc:39`：

```cpp
dlHcclConfigGetInfo = (HcclResult(*)(HcclComm, HcclConfigType, uint32_t, void*))
    dlsym(handle_, "HcclConfigGetInfo");
```

`dlsym` 在动态库 `libhcomm.so`（见 `dlhcomm_function.cc:51`）中查找 `"HcclConfigGetInfo"` 符号：

- **符号存在**：`dlsym` 返回非空指针，`dlHcclConfigGetInfo` 被赋值为有效 `std::function`，`!hcommFunction.dlHcclConfigGetInfo` 为 `false`，接口可调用。
- **符号不存在**：`dlsym` 返回 `nullptr`，由于 `std::function` 初始化用的是花括号 `{}` 默认构造（见 `dlhcomm_function.h:30`），加上 `dlsym` 返回 `nullptr` 时 `std::function` 从空指针构造会保持空状态，`!hcommFunction.dlHcclConfigGetInfo` 为 `true`，跳过通信域查询。

所以运行时检测逻辑是：**先检查 `dlHcclConfigGetInfo` 是否为空 `std::function`，空则说明 Hcomm 没有导出 `HcclConfigGetInfo` 接口，直接降级。**

#### 2.2.3 两层检测的组合效果

完整调用链是 `GetCommMultipleDimensionSplitRatio` → `QueryCommSplitRatio<HcclConfigType>` →（根据编译期检测结果分支）→ `QueryCommSplitRatio(..., true/false_type)` → `QuerySplitRatioByConfigGetInfo` →（运行时检测 `dlHcclConfigGetInfo`）。组合行为如下：

| 编译时头文件 | 运行时 `HcclConfigGetInfo` | 实际执行路径 | 结果 |
|---|---|---|---|
| `HcclConfigType` 不含切分比例枚举 | 任意 | `HasSplitRatioConfigType` = `false_type` → `QueryCommSplitRatio(..., std::false_type)` | 打日志、`isConfigured=false`，降级到环境变量 |
| 含切分比例枚举 | `dlHcclConfigGetInfo` 为空 | `true_type` → `QueryCommSplitRatio(..., std::true_type)` → `QuerySplitRatioByConfigGetInfo` → `!dlHcclConfigGetInfo` 为 true | 打日志、`isConfigured=false`，降级到环境变量 |
| 含切分比例枚举 | `dlHcclConfigGetInfo` 非空 | `true_type` → `QueryCommSplitRatio(..., std::true_type)` → `QuerySplitRatioByConfigGetInfo` → 调用接口 | 查询通信域配置，成功时 `isConfigured=true` |

两层检测各自负责一个维度：编译期 SFINAE 判断**头文件类型能力**，运行时 `std::function` 判断**动态库接口能力**，两者必须同时满足才发起查询。

### 2.3 [暂缓] CCU FastLaunch绕过三级配置解析

**复审处理：** 按本轮范围暂缓。该问题可能导致FastLaunch使用缓存比例而不是本次配置，但未发现其阻止算子按已有缓存上下文执行。

**位置：**

- `src/ops/op_common/op_common.cc:95-127`
- `src/ops/op_common/op_common.cc:268-289`
- `src/ops/op_common/op_common.cc:188-216`
- 各算子入口中 `ShouldGoCcuFastLaunch` 早于 `Selector`的调用

`SetMultipleDimensionSplitRatio(comm, param)` 放在 `Selector`末尾，但AllGather、AllReduce、ReduceScatter、Reduce和Broadcast的CCU入口都在调用 `Selector` 之前尝试FastLaunch。一旦命中已缓存的 `CcuFastLaunchCtx`：

- 本次调用不会查询通信域切分比例。
- 本次调用不会设置环境变量来源。
- FastLaunch缓存键只包含tag、数据类型、reduce op、count和root，没有比例值或来源。

因此通信域配置变化后，可能继续使用按旧比例生成的缓存kernel参数，与“通信域配置 > 环境变量 > 内置公式”的要求不一致。

Broadcast FastLaunch还会从默认初始化的Executor成员重新计算切分，但FastLaunch Executor没有恢复Rank和Channel Map，内置公式会回退到0.5。当前计算出的 `SliceCountPart0/1` 未被后续使用，说明FastLaunch比例的真实来源仍是首次展开时缓存的kernel参数。

**建议：**

1. 将Host侧比例优先级解析移到FastLaunch判断之前。
2. 将最终切分比例纳入FastLaunch缓存键，或者在比例与缓存上下文不一致时禁用该缓存并重新展开。
3. 删除Broadcast FastLaunch中未使用的 `SliceCountPart0/1`，或将缓存创建时的最终比例显式保存到FastLaunch Context并校验。

### 2.4 [已修复] 将所有 `HCCL_E_PARA` 视为“未配置”

**复审结果：** 当前仅将 `HCCL_E_NOT_SUPPORT` 作为可降级场景，`HCCL_E_PARA` 和其他异常错误会返回调用方，不再被当作未配置处理。

**位置：** `src/ops/op_common/op_common.cc:2224-2228`

当 `HcclConfigGetInfo` 返回 `HCCL_E_PARA` 时，实现直接将其视为“通信域未配置或不支持”并继续使用环境变量。但 `HCCL_E_PARA` 也可能表示：

- `comm` 无效。
- `infoLen` 与接口要求不匹配。
- 配置类型或输出指针参数错误。

这会吞掉真实的配置查询故障，违反设计中“只有不支持/未配置才降级”的约定。

**建议：** 使用后续接口明确约定的“未配置”返回码或配置有效标志。在没有正式语义前，不应普遍吞掉 `HCCL_E_PARA`。

### 2.5 [已修复] NaN比例可被当作有效配置传递

**复审结果：** 通信域比例、环境变量比例和公式回退值均增加了 `std::isfinite` 检查，NaN和无穷值不会进入数据切分计算。

**位置：**

- `src/ops/op_common/op_common.cc:2234-2239`
- `src/ops/op_common/template/template_utils.cc:183`

对通信域比例的校验只使用：

```cpp
commRatio < 0.0 || commRatio > 1.0
```

当 `commRatio` 为NaN时，两个比较都为false，该值会被标记为 `COMM_CONFIG`并直接传入Executor。同样，`std::clamp` 也不能将NaN转换为有效回退值。

**影响：** 后续切分计数包含NaN浮点运算并转换为整数，结果不可靠。

**建议：** 所有外部比例和公式回退值都先通过 `std::isfinite` 检查，再检查 $[0,1]$ 范围。

### 2.6 [暂缓] 缺少新增逻辑的自动化测试

**复审处理：** 按本轮范围暂缓。当前通过实编译和静态逻辑核对完成验证，但公式及配置优先级仍缺少自动化回归保护。

**位置：** `test/`

本次改动增加了331行左右的配置决策和数学计算逻辑，但没有新增或修改任何测试文件。当前缺少对以下关键行为的回归保护：

- 通信域 > 环境变量 > 公式的优先级。
- ReduceScatter类和AllGather两种公式。
- 多Channel端口数求和与空Channel回退。
- AllGather从原反向存储改为Mesh-first存储。
- FastLaunch缓存与比例配置的一致性。

**建议：** 按设计文档第12.12节补齐Host配置决策、公式函数和Executor比例方向测试，并增加FastLaunch命中场景。

## 3. 正确实现点

以下部分与设计一致：

- `DevAicpuOpConfig` 增加了比例来源，能区分显式0.5和内置公式。
- Host普通展开路径按通信域、环境变量、公式的顺序决策。
- `CalcParallelDataSplitRatio` 中ReduceScatter类和AllGather的时间系数与文档公式一致。
- 端口数取单个remoteRank的第一个非空Channel组并求和；机内端口和随后乘以
  `intraRankSize - 1`，得到公式使用的机内总端口规模。
- 五个Executor均已接入，Scatter未被修改。
- AllGather的分片顺序已统一为第一项Mesh-first、第二项Clos-first。
- Reduce使用 `parallelDataSplitRatio_` 贯穿Scratch计算和每轮数据切分，没有继续误用原始配置值。

## 4. 验证情况

已执行：

- `git diff --check`：通过。
- 基于 `compile_commands.json`，对 `op_common.cc`、`template_utils.cc`及五个Executor源文件执行 `-fsyntax-only`：全部通过。
- 当前Hcomm头文件不包含新增枚举时，`op_common.cc`编译通过并选择SFINAE降级分支。
- 最小C++14示例覆盖新增枚举存在和不存在两种类型：均编译通过。
- 静态核对ReduceScatter类和AllGather公式、机内端口数缩放、五个Executor参数映射及Mesh-first比例方向：与设计一致。

未执行完整工程构建和UT。当前改动未提供针对新逻辑的测试用例，因此仍保留集成环境和真实拓扑上的运行验证风险。

## 5. 后续事项

1. 2.3按当前安排暂缓；后续若要求FastLaunch动态响应通信域或环境变量配置，需要处理缓存键与比例一致性。
2. 2.6按当前安排暂缓；建议在后续质量补强阶段补齐配置优先级、公式和Executor比例方向测试。
3. 在目标集成环境执行完整构建，并至少覆盖一次五个Parallel Executor的真实拓扑运行验证。

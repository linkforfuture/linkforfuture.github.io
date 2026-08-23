# HCCL AICPU 执行超时（HCCL_EXEC_TIMEOUT）方案分析

> 分析对象：`HCCL_EXEC_TIMEOUT` 环境变量在 AICPU 模式下的解析、传递与使用全链路。
> 代码基线：`cann/hccl`（`src/ops/op_common/`、`src/common/`）。

## 1. 概述

`HCCL_EXEC_TIMEOUT` 用于控制分布式训练/推理中设备间执行通信同步的等待时间（单位：秒）。在 AICPU 模式下，该值并非单一地作用于某一处，而是被**派生为多个子超时**，分别作用于：

- AICPU Kernel 启动超时（`aclrtLaunchKernel` 的 timeout 属性）
- Host 等待 Device 通知超时（host notify wait）
- Device 侧资源申请/队列满超时（thread res acquire timeout）
- Device 侧 notify wait 默认超时（set notify wait timeout）
- Device 侧算法编排中所有 channel/thread notify 等待超时

默认值 `CUSTOM_TIMEOUT = 1836` 秒（`src/ops/op_common/inc/alg_param.h:61`）。

## 2. 整体数据流

```
┌───────────────────────────────────────────────────────────────────────┐
│  阶段一：进程初始化（一次）                                              │
│  InitEnvConfig() ──> ParseExecTimeout()                                │
│    读 HCCL_EXEC_TIMEOUT → 校验 → 存 g_algEnvConfig (double, thread_local)│
│    读取接口：GetExternalInputExecTimeout(double&)                       │
└──────────────────────────────┬────────────────────────────────────────┘
                               │ (每次算子调用)
┌──────────────────────────────▼────────────────────────────────────────┐
│  阶段二：算子下发 Selector() [Host]                                     │
│  SetExecTimeout(param) ──> param.opConfig.execTimeout (u32)            │
│    未设置/越界 → CUSTOM_TIMEOUT(1836)；0 → 永不超时                      │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
           ┌───────────────────┴────────────────────┐
           ▼                                        ▼
┌─────────────────────────┐         ┌──────────────────────────────────┐
│ 阶段三：资源申请 [Host]   │         │ 阶段四：Kernel 启动 [Host]          │
│ HcclAllocAlgResourceAICPU│         │ HcclAicpuKernelEntranceLaunch     │
│  UpdateAicpuTimeoutCtx   │         │  DeriveAicpuTimeout               │
│   DeriveAicpuTimeout     │         │   → kernelLaunchTimeout(u16)      │
│    → waitTimeout         │         │   → hostNotifyTimeout             │
│    → fullTimeout         │         │   aclrtLaunchKernel(attr.timeout) │
│   序列化 resCtx → Device │         │   HcclSetNotifyWaitTimeOut        │
└─────────────────────────┘         │   HcclThreadNotifyWaitOnThread    │
                                    └──────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────────┐
│  阶段五：Device 侧 AICPU Kernel 执行                                     │
│  kernel_launch.cc (3 处入口):                                          │
│   HcclThreadResAcquireTimeOut(fullTimeout)                             │
│   HcclSetNotifyWaitTimeOut(waitTimeout)                                │
│   ExecTimeoutManager::SetExecTimeout(param->opConfig.execTimeout)      │
│   executor->Orchestrate()                                              │
│     └─ wrapper/alg_template_base:                                      │
│        ExecTimeoutManager::GetExecTimeout() → 各 notify wait 超时       │
└───────────────────────────────────────────────────────────────────────┘
```

## 3. 阶段一：环境变量解析（进程初始化）

**入口**：`InitEnvConfig()`（`src/common/alg_env_config.cc:250`）在 HCCL 初始化时调用 `ParseExecTimeout()`。

**`ParseExecTimeout()`**（`src/common/alg_env_config.cc:69-106`）逻辑：

| 步骤 | 处理 |
|------|------|
| 读取 | `GetEnv("HCCL_EXEC_TIMEOUT")`（`alg_env_config.cc:71`） |
| 未设置（`"EmptyString"`） | `execTimeOutSet=false`，`execTimeout=0`，返回 SUCCESS |
| 格式校验 | `IsValidNumberFormat(env, maxDecimal=2)`：仅允许数字，最多 2 位小数（`alg_env_config.cc:79`）。非法 → 告警 + 默认值 |
| 数值转换 | `SalStrToDouble` 转 double（`alg_env_config.cc:88`）。失败 → 默认值 |
| 上限校验 | `> UINT32_MAX` 视为过大 → 默认值（`alg_env_config.cc:96`） |
| 合法 | `execTimeOutSet=true`，`execTimeout=execTimeOut`(double) |

存储：全局 `g_algEnvConfig`（`thread_local AlgEnvConfig`，`alg_env_config.cc:27`），由 `g_algEnvConfigMutex` 保护。

读取接口 `GetExternalInputExecTimeout(double&)`（`alg_env_config.cc:108-117`）：未设置返回 `false`，已设置返回 `true` 并回填 double 值。

> 注意：解析阶段保留 double（支持十毫秒级精度，如 `0.05`），到阶段二才截断为 u32。

## 4. 阶段二：算子下发设置 OpParam

**入口**：`Selector()`（`src/ops/op_common/op_common.cc:150`）调用 `SetExecTimeout(param)`。

**`SetExecTimeout()`**（`src/ops/op_common/op_common.cc:2624-2645`）逻辑：

| 条件 | `param.opConfig.execTimeout` 赋值 |
|------|----------------------------------|
| `GetExternalInputExecTimeout` 返回 false | `CUSTOM_TIMEOUT`（1836） |
| `execTimeoutValue < 0` 或 `> UINT32_MAX` | `CUSTOM_TIMEOUT`（告警） |
| 其它 | `static_cast<uint32_t>(execTimeoutValue)` |
| └─ 截断后 == 0 | 保留 0（语义：永不超时） |

`opConfig` 类型为 `DevAicpuOpConfig`（`src/ops/op_common/inc/alg_param.h:499-503`），是 `OpParam` 的成员（`alg_param.h:584`）。`OpParam` 会随 kernel 参数一起拷贝到 device 侧。

## 5. 阶段三：AICPU 资源派生与序列化

**入口**：`HcclAllocAlgResourceAICPU()`（`src/ops/op_common/op_common.cc:1354`）在首次算子资源申请时调用 `UpdateAicpuTimeoutCtx(param, *resCtxHost)`（`op_common.cc:1367`）。

**`UpdateAicpuTimeoutCtx()`**（`src/ops/op_common/op_common.cc:66-75`）：

```cpp
AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
resCtx.waitTimeout = timeout.waitTimeout;      // = execTimeout
resCtx.fullTimeout = timeout.fullTimeout;      // = execTimeout + 20
```

写入 `AlgResourceCtxSerializable`（`alg_param.h:407`）的 `waitTimeout` / `fullTimeout` 字段，随后该 ctx 被 `Serialize()` 并 `HcclEngineCtxCopy` 拷贝到 device 内存（`op_common.cc:1346`），供 device 侧 kernel 读取。

## 6. AicpuTimeout 派生规则（核心）

**`DeriveAicpuTimeout()`**（`src/ops/op_common/aicpu_timeout.h:49-58`）：

| 子超时 | 计算 | 偏移常量 | 用途 |
|--------|------|---------|------|
| `waitTimeout` | `execTimeout` | — | Device notify wait 默认值 |
| `fullTimeout` | `execTimeout + 20` | `AICPU_FULL_TIMEOUT_OFFSET=20` | 队列满/资源申请超时 |
| `hostNotifyTimeout` | `execTimeout + 50` | `AICPU_HOST_NOTIFY_TIMEOUT_OFFSET=50` | Host 等 Device 通知 |
| `kernelLaunchTimeout` | `uint16(execTimeout + 30)` | `AICPU_KERNEL_TIMEOUT_OFFSET=30` | Kernel 启动超时（u16） |

辅助函数（`aicpu_timeout.h:30-47`）：
- `AddAicpuTimeoutOffset(timeout, offset)`：`timeout==0` 返回 0（永不超时）；溢出返回 `UINT32_MAX`；否则 `timeout+offset`。
- `ToKernelLaunchTimeout(timeout)`：超 `UINT16_MAX` 截断为 `UINT16_MAX`，否则转 `uint16_t`。

## 7. 阶段四：AICPU Kernel 启动超时（Host 侧下发）

**入口**：`HcclAicpuKernelEntranceLaunch` / `AicpuKernelLaunch`（`src/ops/op_common/op_common.cc:822, 848, 881`）。

> 本节的 `IsHcommDefaultTimeoutSupported()` 分支即"两条线"在 Host 侧的体现，两线总览与汇总表见第 9 节。

### 7.1 Kernel 启动超时 `kernelLaunchTimeout`（u16，秒）

```cpp
AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
u16 kernelLaunchTimeout = IsHcommDefaultTimeoutSupported() ? timeout.kernelLaunchTimeout
    : ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
```

- **HCOMM 支持 DefaultTimeout**：用 `DeriveAicpuTimeout` 派生值（offset = 30，`aicpu_timeout.h`）。
- **不支持（fallback）**：用 `op_common.cc:63` 本地常量 `KERNEL_TIMEOUT_OFFSET = 25`。

设置到 `aclrtLaunchKernelAttr` 的 `ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT`（`op_common.cc:886-887`），随 `aclrtLaunchKernelWithConfig` 下发。

### 7.2 Host 等待 Device 通知超时 `hostNotifyWaitTime`（u32，秒）

```cpp
u32 hostNotifyWaitTime = IsHcommDefaultTimeoutSupported() ? timeout.hostNotifyTimeout
    : AddAicpuTimeoutOffset(param.opConfig.execTimeout, HOST_NOTIFY_TIMEOUT_OFFSET);
```

- **支持**：用派生值（offset = 50）。
- **fallback**：用 `op_common.cc:62` 本地常量 `HOST_NOTIFY_TIMEOUT_OFFSET = 27`。

通过 `HcclSetNotifyWaitTimeOut(hostNotifyWaitTime)` 设置全局默认，再用 `HcclThreadNotifyWaitOnThreadDefault(cpuTsThread, idx, hostNotifyWaitTime)` 等待（`op_common.cc:851-854`）。

> ⚠️ **不对称点**：两条路径使用的 offset 常量不同（`aicpu_timeout.h` 的 30/50 vs `op_common.cc` 的 25/27）。由 `IsHcommDefaultTimeoutSupported()` 决定走哪套。这是历史演进遗留，改造时需同步两处。

## 8. 阶段五：Device 侧 Kernel 执行

**入口**：`src/ops/op_common/template/aicpu/kernel_launch.cc`，共 3 处（集合通信 OpsV2、P2P Send/Recv、旧路径），分别在 `kernel_launch.cc:423`、`675`、`952` 调用 `ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout)`。

每处入口在 `Orchestrate` 前完成：

1. **资源申请超时**：`HcclThreadResAcquireTimeOut(resCtxPtr->fullTimeout)`（`kernel_launch.cc:407`）—— 用序列化过来的 `fullTimeout`（= execTimeout + 20）。
2. **Notify wait 默认超时**：`HcclSetNotifyWaitTimeOut(resCtxPtr->waitTimeout)`（`kernel_launch.cc:410`）—— 设为 `waitTimeout`（= execTimeout），作为后续 `WithDefaultTimeout` 系列接口的隐式默认。
3. **主 thread 等 host 通知**：`HcclThreadNotifyWaitOnThreadDefault(thread, maxNotifyNum, resCtxPtr->waitTimeout)`（`kernel_launch.cc:414`）。
4. **写入全局单例**：`ExecTimeoutManager::SetExecTimeout(param->opConfig.execTimeout)`（`kernel_launch.cc:423/675/952`）—— 供算法编排阶段读取。

### 8.1 ExecTimeoutManager 单例

`src/ops/op_common/exec_timeout_manager.h` / `.cc`：

- Meyers 单例（`exec_timeout_manager.cc:25-28`）。
- `execTimeout_`、`timeoutSet_` 均为 `std::atomic`（`exec_timeout_manager.h:32-33`）。
- 构造默认值 `CUSTOM_TIMEOUT`（1836），`timeoutSet_=false`（`exec_timeout_manager.cc:16-20`）。
- `SetExecTimeout(u32)`：原子写入并置 `timeoutSet_=true`（`exec_timeout_manager.cc:30-35`）。
- `GetExecTimeout()`：`timeoutSet_` 为真返回已设值，否则返回 `CUSTOM_TIMEOUT`（`exec_timeout_manager.cc:37-42`）。

> 因为是 device 侧 aicpu 进程内单例，`SetExecTimeout` 只需在 kernel 入口设一次，后续整张算法编排图都能 `GetExecTimeout` 共享。

### 8.2 算法编排中的使用

`executor->Orchestrate()` 触发的数据传输模板通过 `ExecTimeoutManager::Instance().GetExecTimeout()` 取值，但**调用接口分两类**（两线判定见第 9 节）：

- `src/ops/op_common/template/alg_template_base.cc:90, 107, 127`：`AlgTemplateBase::ExecuteBarrier` 各重载，调 `HcclChannelNotifyWaitOnThreadDefault`（封装版，**分两线**：线 A 走 `WithDefaultTimeout` 忽略实参，线 B 显式传 `execTimeout`）。
- `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc`：18 处（行 141/166/188/212/255/295/318/395/471/489/539/552/598/642/655/738/952/984），直接调 dlsym 的 `HcommChannelNotifyWaitOnThread` / `HcommThreadNotifyWaitOnThread`，**始终显式传 `execTimeout`，不分两线**。

> 这些是 device 侧 aicpu 真正可能因对端未到而阻塞的同步点，是 `HCCL_EXEC_TIMEOUT` 生效的主战场。两类接口在两条线下的具体生效值见第 9 节"调用点归属表"。

## 9. 超时设置的两条线（汇总）

AICPU 超时在 HCOMM 接口层存在**两条并行线路**，由 `IsHcommDefaultTimeoutSupported()`（`src/common/hcomm_dlsym/hcomm_primitives_dl.cc:125-129`）判定：

- 该函数 = `HcommIsSupportHcommSetNotifyWaitTimeOut()` && `HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout()`。

### 线 A：使用 Hcomm 设超时接口（DefaultTimeout 支持）

- **前提**：`IsHcommDefaultTimeoutSupported() == true`。
- **机制**：先调 `HcclSetNotifyWaitTimeOut(T)`（`hcomm_primitives_dl.cc:131-141`）设置一个 HCOMM 内部全局默认超时 `T`；后续所有 `Hccl*NotifyWaitOnThreadDefault` 封装走 `*WithDefaultTimeout` 变体，**不传显式 timeout**，由 HCOMM 内部套用先前设的 `T`。
- **Host 侧 launch**（第 7 节）：用 `DeriveAicpuTimeout` 派生值，offset 取 `aicpu_timeout.h` 的 30/50。
- **Device 侧 kernel 入口**（第 8 节）：`HcclSetNotifyWaitTimeOut(resCtxPtr->waitTimeout)` 生效（`kernel_launch.cc:409-410`），主 thread 等待走 `WithDefaultTimeout`，生效值 = `waitTimeout`（= execTimeout）。
- **Device 侧编排 `*Default` 调用点**：走 `WithDefaultTimeout`，生效值 = 上面设的 `waitTimeout`；此时传给 `*Default` 的 `fallbackTimeout` 参数**被忽略**。

### 线 B：不调用 Hcomm 设超时接口（fallback）

- **前提**：`IsHcommDefaultTimeoutSupported() == false`（HCOMM 版本较旧，缺失 `SetNotifyWaitTimeOut` 或 `*WithDefaultTimeout`）。
- **机制**：`HcclSetNotifyWaitTimeOut` 返回 `HCCL_E_NOT_SUPPORT`（`hcomm_primitives_dl.cc:133-135`），调用方用 `HcommIsSupportHcommSetNotifyWaitTimeOut()` 守卫跳过（`kernel_launch.cc:409`）；所有 notify 等待**显式传 timeout 参数**。
- **Host 侧 launch**（第 7 节）：用 `op_common.cc` 本地 offset 25/27 派生。
- **Device 侧 kernel 入口**：跳过 `SetNotifyWaitTimeOut`，`HcclThreadNotifyWaitOnThreadDefault` fallback 到 `HcommThreadNotifyWaitOnThread(thread, idx, fallbackTimeout)`（`hcomm_primitives_dl.cc:161`），显式传 `resCtxPtr->waitTimeout`。
- **Device 侧编排 `*Default` 调用点**：fallback 到带显式 `fallbackTimeout` 的旧接口，值来自 `ExecTimeoutManager::GetExecTimeout()`。

### 始终显式传、不分两线的调用点

以下调用点直接使用 dlsym 得到的 `Hcomm*` 接口（非 `Hccl*Default` 封装），**两条线下都显式传 timeout**，不受 `IsHcommDefaultTimeoutSupported()` 影响：

- `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc`：18 处 `HcommChannelNotifyWaitOnThread` / `HcommThreadNotifyWaitOnThread`，显式传 `ExecTimeoutManager::GetExecTimeout()`（行 141/166/188/212/255/295/318/395/471/489/539/552/598/642/655/738/952/984）。
- `HcclThreadResAcquireTimeOut(fullTimeout)`（`kernel_launch.cc:407`）：独立的资源申请超时接口，单独由 `HcommIsSupportHcommThreadResAcquireTimeOut()` 守卫，不经过 notify 默认值通道。

> 即：线 A 的"设一次默认"仅对 `Hccl*NotifyWaitOnThreadDefault` 调用点（kernel 入口主 thread 等待、`AlgTemplateBase::ExecuteBarrier`）生效；wrapper 的数据传输同步始终显式传 `execTimeout`。

### 调用点归属表

| 调用点 | 位置 | 接口 | 线 A 生效值 | 线 B 生效值 |
|--------|------|------|------------|------------|
| Kernel 启动 | `op_common.cc:886` | `aclrtLaunchKernelAttr.timeout` | uint16(exec+30) | uint16(exec+25) |
| Host 等 Device 通知 | `op_common.cc:852-854` | `SetNotifyWaitTimeOut`+`ThreadNotifyWaitOnThreadDefault` | exec+50（设默认+WithDefault） | exec+27（显式） |
| Device 资源申请 | `kernel_launch.cc:407` | `ThreadResAcquireTimeOut` | fullTimeout(exec+20) | 不支持则跳过 |
| Device 设默认 notify | `kernel_launch.cc:410` | `SetNotifyWaitTimeOut` | waitTimeout(exec) | 跳过（NOT_SUPPORT） |
| Device 主 thread 等 host | `kernel_launch.cc:414` | `ThreadNotifyWaitOnThreadDefault` | WithDefault(exec) | 显式 waitTimeout(exec) |
| `ExecuteBarrier` | `alg_template_base.cc:92/94/110/114/129` | `ChannelNotifyWaitOnThreadDefault` | WithDefault(exec) | 显式 exec |
| 数据传输 wrapper（18 处） | `alg_data_trans_wrapper.cc` | `Hcomm*NotifyWaitOnThread` | 显式 exec | 显式 exec |

> 说明：`exec` = `ExecTimeoutManager::GetExecTimeout()` 或 `param.opConfig.execTimeout`，未设置时为 `CUSTOM_TIMEOUT`(1836)。

### 两线差异要点

1. **是否调 `HcclSetNotifyWaitTimeOut`**：线 A 调（设 HCOMM 全局默认），线 B 不调。
2. **Host launch offset 不同**：线 A 用 `aicpu_timeout.h`（30/50），线 B 用 `op_common.cc`（25/27）——这是两套常量不对称的根源。
3. **生效超时来源**：线 A 的 `*Default` 调用依赖"先前设的全局默认"；线 B 的 `*Default` 调用依赖"当场传的 fallbackTimeout"。
4. **wrapper 不受影响**：数据传输同步始终显式传 `execTimeout`，两线下行为一致。

## 10. 0 值与边界语义

| 输入 | `opConfig.execTimeout` | `DeriveAicpuTimeout` 各子超时 | 语义 |
|------|------------------------|------------------------------|------|
| 未设置 | 1836 | 1836 / 1856 / 1886 / uint16(1866) | 默认 |
| `0` | 0 | 全部 0（`AddAicpuTimeoutOffset(0,·)=0`） | **永不超时** |
| `0.05` | 0（截断） | 0 | 注意：u32 截断后 50ms 变 0，等同永不超时 |
| `> UINT32_MAX` | 1836 | 默认 | 越界回退 |
| 正常值 N | N | N / N+20 / N+50 / uint16(N+30) | 正常 |

> ⚠️ 文档（`docs/zh/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md`）宣称 AICPU 模式"支持十毫秒级精度"，但 `SetExecTimeout` 中 `static_cast<uint32_t>` 会把 `<1` 的值截断为 0（永不超时）。十毫秒精度实际只在 AIV 模式（`hccl_aiv_utils.cc:360` 的 `GetAivTimeout`，保留 us 级计算）生效。AICPU 模式建议配置整数秒。

## 11. AIV 模式对比（参考）

`GetAivTimeout()`（`src/ops/op_common/template/aiv/hccl_aiv_utils.cc:360-390`）同样读 `GetExternalInputExecTimeout`，但走独立路径：

- 默认 `AIV_TIMEOUT_DEFAULT = 1091` 秒，转 us。
- 通过 `aclrtGetOpTimeOutInterval` 获取硬件最短间隔 `interval`，有效范围 `[interval, 254*interval]`。
- 配置值向上对齐到 `interval * N`（N∈[1,254]），超出范围钳位到边界。
- 0 或越界 → 1091s。

AICPU 模式不经过此对齐逻辑。

## 12. 关键常量汇总

| 常量 | 值 | 位置 | 作用 |
|------|----|------|------|
| `CUSTOM_TIMEOUT` | 1836 | `alg_param.h:61` | AICPU 默认超时（秒） |
| `AICPU_FULL_TIMEOUT_OFFSET` | 20 | `aicpu_timeout.h:19` | fullTimeout 偏移 |
| `AICPU_KERNEL_TIMEOUT_OFFSET` | 30 | `aicpu_timeout.h:20` | kernelLaunchTimeout 偏移（DefaultTimeout 路径） |
| `AICPU_HOST_NOTIFY_TIMEOUT_OFFSET` | 50 | `aicpu_timeout.h:21` | hostNotifyTimeout 偏移（DefaultTimeout 路径） |
| `KERNEL_TIMEOUT_OFFSET` | 25 | `op_common.cc:63` | fallback 路径 kernel 启动偏移 |
| `HOST_NOTIFY_TIMEOUT_OFFSET` | 27 | `op_common.cc:62` | fallback 路径 host notify 偏移 |
| `AIV_TIMEOUT_DEFAULT` | 1091 | `hccl_aiv_utils.cc:362` | AIV 默认超时（秒） |

## 13. 关键文件索引

| 文件 | 职责 |
|------|------|
| `src/common/alg_env_config.cc:69-117` | 环境变量解析、全局存储与读取接口 |
| `src/common/alg_env_config.cc:250-256` | 初始化时调用解析 |
| `src/ops/op_common/op_common.cc:2624-2645` | `SetExecTimeout`：写入 OpParam |
| `src/ops/op_common/op_common.cc:66-75` | `UpdateAicpuTimeoutCtx`：派生并写入 resCtx |
| `src/ops/op_common/op_common.cc:822,848,881` | Host 侧 kernel 启动/host notify 超时 |
| `src/ops/op_common/aicpu_timeout.h` | `DeriveAicpuTimeout` 派生规则与偏移常量 |
| `src/ops/op_common/exec_timeout_manager.h/.cc` | Device 侧单例，算法编排阶段统一取值 |
| `src/ops/op_common/template/aicpu/kernel_launch.cc:423,675,952` | Device 侧 SetExecTimeout + resCtx 超时应用 |
| `src/ops/op_common/template/alg_template_base.cc:90,107,127` | `ExecuteBarrier` 使用 GetExecTimeout |
| `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc` | 数据传输 wrapper 使用 GetExecTimeout（18 处） |
| `src/common/hcomm_dlsym/hcomm_primitives_dl.cc:125-183` | DefaultTimeout 支持判定与 fallback 封装 |
| `src/ops/op_common/template/aiv/hccl_aiv_utils.cc:360-390` | AIV 模式超时（对比路径） |
| `docs/zh/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md` | 用户文档 |

## 14. 小结

AICPU 超时方案采用**一处解析、两线派生、分层应用**的设计：

1. **解析层**（`alg_env_config`）：double 精度保留，进程级一次。
2. **派生层**：`DeriveAicpuTimeout` 把单一 `execTimeout` 拆为 4 个子超时，分别覆盖 kernel 启动、host 等待、device 资源申请、device notify 等待；为不同子环节预留不同裕量（20/30/50）。
3. **应用层（两条线）**：
   - **线 A**（`IsHcommDefaultTimeoutSupported()==true`）：调 `HcclSetNotifyWaitTimeOut` 设 HCOMM 全局默认，`Hccl*NotifyWaitOnThreadDefault` 走 `WithDefaultTimeout`；Host launch offset 用 30/50。
   - **线 B**（fallback）：不调设默认接口，所有 `*Default` 调用显式传 `fallbackTimeout`；Host launch offset 用 25/27。
   - **wrapper 数据传输**：两条线下都直接调 `Hcomm*` 显式传 `execTimeout`，不经过默认值通道。
   - Device 侧 resCtx 序列化值用于资源/默认 notify；`ExecTimeoutManager` 单例供算法编排的 notify wait 共享。

需要关注的改造风险点：**两套 offset 常量不对称**（`aicpu_timeout.h` 30/50 vs `op_common.cc` 25/27，由两线分支决定），以及 **AICPU 模式下小数秒被 u32 截断为 0（永不超时）** 的行为。

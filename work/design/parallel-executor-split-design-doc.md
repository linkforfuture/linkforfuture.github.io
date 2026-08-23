# ParallelExecutor数据切分设计文档

## 1. 背景与需求范围

Parallel Executor采用数据并行的方式调用Template执行算法：把一整块数据拆成两份，第一份先执行Server内算法，再执行Server间算法；第二份先执行Server间算法，再执行Server内算法。两份数据的同一阶段并行执行，从而同时利用Server内Mesh链路和Server间Clos链路的带宽。Server间通信通常在不同Server的同号卡之间执行。

本文主要关注以下Parallel算法：

- `InsReduceScatterParallelMesh1DNHR`
- `InsAllGatherParallelMesh1DNHR`
- `InsAllReduceParallelRSAG`
- `InsBroadcastParallelMesh1DNHR`
- `ReduceParallelMesh1DNHR`

Scatter与Broadcast的第一阶段通信模型相同，因此本文也给出Scatter的结果。

## 2. 符号与基本假设

设集群拓扑为 $M \times N$：

- $N$：Server数量。
- $M$：每个Server内的Rank数量。
- $D$：每个Rank参与当前算子的输入数据量。对Scatter和Broadcast，$D$表示Root上待分发的总数据量。
- $r$：先走Mesh、后走Clos的数据比例，数据量为 $D r$。
- $1-r$：先走Clos、后走Mesh的数据比例，数据量为 $D(1-r)$。
- $B$：Server内Mesh链路的有效总带宽。
- $C$：Server间Clos链路的有效总带宽。
- $B_{LR}$：Mesh侧执行LocalReduce时的有效带宽，本文取 $B_{LR}=20B$。

推导时作如下假设：

1. 忽略同步、任务下发和固定启动延迟。除Reduce类算子的Mesh侧LocalReduce外，忽略其他计算开销。
2. 按照当前Parallel Executor的同步方式，通过使第一阶段并行的Mesh任务和Clos任务耗时相等来计算切分比例。
3. $B$ 和 $C$ 应为可同时使用的有效总带宽，而不是单条链路的标称带宽。

## 3. 通用切分公式

设某算子在 $P$ 个Rank上的第一阶段通信量系数为 $k(P)$。则对数据量 $X$，其通信时间近似为：

$$
T(P, X, W)=k(P)\frac{X}{W}
$$

其中 $W$ 是该通信域的有效总带宽。因此，第一阶段中两个并行任务的耗时分别为：

$$
T_{Mesh}=k(M)\frac{D r}{B}
$$

$$
T_{Clos}=k(N)\frac{D(1-r)}{C}
$$

令两者相等：

$$
k(M)\frac{D r}{B}=k(N)\frac{D(1-r)}{C}
$$

约去 $D$ 并整理得：

$$
r\left(\frac{k(M)}{B}+\frac{k(N)}{C}\right)=\frac{k(N)}{C}
$$

所以通用切分公式为：

$$
\boxed{
r=\frac{\frac{k(N)}{C}}
        {\frac{k(N)}{C}+\frac{k(M)}{B}}
}
$$

该公式的含义是：某一通信域越慢，分配给该通信域作为第一阶段处理的数据就应越少。

上述公式是只有通信时间时的基础形式。若Mesh侧还包含通信量系数为 $k_{LR}(M)$ 的
LocalReduce，则应将Mesh时间系数扩展为：

$$
\frac{k(M)}{B}+\frac{k_{LR}(M)}{B_{LR}}
$$

对应的切分公式为：

$$
\boxed{
r=\frac{\frac{k(N)}{C}}
        {\frac{k(N)}{C}+\frac{k(M)}{B}+\frac{k_{LR}(M)}{B_{LR}}}
}
$$

第4.2节中ReduceScatter的 $k_{LR}(M)=k_{RS}(M)=(M-1)/M$。

## 4. ReduceScatter

### 4.1 通信量系数

ReduceScatter在 $P$ 个Rank上执行时，每个Rank最终保留输入数据的 $1/P$，需要通信的数据量为：

$$
X-\frac{X}{P}=\frac{P-1}{P}X
$$

因此ReduceScatter的通信量系数为：

$$
k_{RS}(P)=\frac{P-1}{P}
$$

### 4.2 切分比例推导

第一份数据先在 $M$ 个机内Rank上执行ReduceScatter。Mesh算法完成数据交换后，还需对接收数据执行LocalReduce。
按照补充条件，通信和LocalReduce处理的数据量系数均为 $(M-1)/M$，因此：

$$
T_{Mesh}
=\frac{M-1}{M}\frac{D r}{B}
+\frac{M-1}{M}\frac{D r}{B_{LR}}
=\frac{M-1}{M}D r\left(\frac{1}{B}+\frac{1}{B_{LR}}\right)
$$

第二份数据先在 $N$ 个Server的同号卡之间执行ReduceScatter。Clos链路在传输过程中同时完成Reduce，
不需要增加独立的LocalReduce时间：

$$
T_{Clos}=\frac{N-1}{N}\frac{D(1-r)}{C}
$$

令 $T_{Mesh}=T_{Clos}$：

$$
\frac{M-1}{M}\left(\frac{1}{B}+\frac{1}{B_{LR}}\right)r
=\frac{N-1}{NC}(1-r)
$$

展开并整理：

$$
r\left[
\frac{M-1}{M}\left(\frac{1}{B}+\frac{1}{B_{LR}}\right)
+\frac{N-1}{NC}
\right]
=\frac{N-1}{NC}
$$

得到：

$$
\boxed{
r_{ReduceScatter}=
\frac{\frac{N-1}{NC}}
     {\frac{N-1}{NC}+
      \frac{M-1}{M}\left(\frac{1}{B}+\frac{1}{B_{LR}}\right)}
}
$$

代入 $B_{LR}=20B$：

$$
\frac{1}{B}+\frac{1}{B_{LR}}
=\frac{1}{B}+\frac{1}{20B}
=\frac{21}{20B}
$$

最终得到：

$$
\boxed{
r_{ReduceScatter}=
\frac{\frac{N-1}{NC}}
     {\frac{N-1}{NC}+\frac{21(M-1)}{20MB}}
}
$$

## 5. AllGather

### 5.1 通信量系数

AllGather与ReduceScatter对 $D$ 的含义不同。本文中 $D$ 是每个Rank的输入数据量。在 $P$ 个Rank上执行AllGather后，每个Rank从其他 $P-1$ 个Rank获取数据，通信数据量为：

$$
(P-1)X
$$

因此AllGather的通信量系数为：

$$
k_{AG}(P)=P-1
$$

### 5.2 切分比例推导

AllGather第一阶段的耗时为：

$$
T_{Mesh}=(M-1)\frac{D r}{B}
$$

$$
T_{Clos}=(N-1)\frac{D(1-r)}{C}
$$

令两者相等：

$$
\frac{M-1}{B}r=\frac{N-1}{C}(1-r)
$$

得到：

$$
\boxed{
r_{AllGather}=
\frac{\frac{N-1}{C}}
     {\frac{N-1}{C}+\frac{M-1}{B}}
}
$$

### 5.3 配置比例的方向

编码修改前，`InsAllGatherParallelMesh1DNHR`中的实际存放顺序为：

```cpp
splitDataSize.push_back(1 - multipleDimensionSplitRatio_); // 先Mesh后Clos
splitDataSize.push_back(multipleDimensionSplitRatio_);     // 先Clos后Mesh
```

因此，在修改前若配置项 `multipleDimensionSplitRatio` 记为 $q$，则AllGather需要配置：

$$
q=1-r_{AllGather}
$$

即：

$$
\boxed{
q_{AllGather}=
\frac{\frac{M-1}{B}}
     {\frac{N-1}{C}+\frac{M-1}{B}}
}
$$

第12章的编码方案会统一所有Parallel Executor的比例方向，使公共函数和
`multipleDimensionSplitRatio` 均表示先Mesh后Clos的比例。修改后的AllGather将直接使用
$r_{AllGather}$，不再使用上述反向配置值。

## 6. AllReduce

`InsAllReduceParallelRSAG`由两级ReduceScatter和反向的两级AllGather组成。第一阶段并行执行的是Server内ReduceScatter和Server间ReduceScatter；Mesh侧同样需要独立执行LocalReduce，Clos侧Reduce与传输融合。因此使用第4.2节修正后的公式：

$$
\boxed{
r_{AllReduce}=
\frac{\frac{N-1}{NC}}
     {\frac{N-1}{NC}+\frac{21(M-1)}{20MB}}
}
$$

`InsAllReduceParallelExecutor::GetParallelDataSplit` 中第一项是先Mesh后Clos的比例，因此可直接将 $r_{AllReduce}$ 用于 `multipleDimensionSplitRatio`。

## 7. Broadcast

`InsBroadcastParallelMesh1DNHR`由两级Scatter和反向的两级AllGather组成。对于总数据量 $X$，Scatter在 $P$ 个Rank上需要从Root发送其中的 $P-1$ 份，通信量为：

$$
\frac{P-1}{P}X
$$

因此：

$$
k_{Scatter}(P)=\frac{P-1}{P}
$$

Broadcast第一阶段执行Scatter，其切分公式为：

$$
\boxed{
r_{Broadcast}=
\frac{\frac{N-1}{NC}}
     {\frac{N-1}{NC}+\frac{M-1}{MB}}
}
$$

Broadcast不执行Reduce操作，因此Mesh侧没有LocalReduce附加时间，公式不乘 $21/20$。

`InsBroadcastParallelExecutor::GetParallelDataSplit` 中第一项是先Mesh后Clos的比例，因此可直接将 $r_{Broadcast}$ 用于 `multipleDimensionSplitRatio`。

## 8. Reduce

`ReduceParallelMesh1DNHR`由两级ReduceScatter和两级AllGather组成，只有Root保留最终结果。其第一阶段与ReduceScatter相同，Mesh侧需要执行LocalReduce，因此：

$$
\boxed{
r_{Reduce}=
\frac{\frac{N-1}{NC}}
     {\frac{N-1}{NC}+\frac{21(M-1)}{20MB}}
}
$$

`ReduceParallelExecutor`中第一份数据是先Mesh后Clos，因此可直接将 $r_{Reduce}$ 用于 `multipleDimensionSplitRatio`。

## 9. Scatter

Scatter的第一阶段通信量系数与Broadcast中的Scatter阶段相同，因此：

$$
\boxed{
r_{Scatter}=
\frac{\frac{N-1}{NC}}
     {\frac{N-1}{NC}+\frac{M-1}{MB}}
}
$$

当Scatter Executor的分片顺序为“第一项先Mesh后Clos”时，可直接使用上式作为配置值。

## 10. 公式汇总

| 算子 | 第一阶段模型 | $k(P)$ | 先Mesh后Clos的比例 $r$ |
| --- | --- | --- | --- |
| ReduceScatter | ReduceScatter + Mesh LocalReduce | $\frac{P-1}{P}$ | $\frac{\frac{N-1}{NC}}{\frac{N-1}{NC}+\frac{21(M-1)}{20MB}}$ |
| AllGather | AllGather | $P-1$ | $\frac{\frac{N-1}{C}}{\frac{N-1}{C}+\frac{M-1}{B}}$ |
| AllReduce | ReduceScatter + Mesh LocalReduce | $\frac{P-1}{P}$ | $\frac{\frac{N-1}{NC}}{\frac{N-1}{NC}+\frac{21(M-1)}{20MB}}$ |
| Broadcast | Scatter | $\frac{P-1}{P}$ | $\frac{\frac{N-1}{NC}}{\frac{N-1}{NC}+\frac{M-1}{MB}}$ |
| Reduce | ReduceScatter + Mesh LocalReduce | $\frac{P-1}{P}$ | $\frac{\frac{N-1}{NC}}{\frac{N-1}{NC}+\frac{21(M-1)}{20MB}}$ |
| Scatter | Scatter | $\frac{P-1}{P}$ | $\frac{\frac{N-1}{NC}}{\frac{N-1}{NC}+\frac{M-1}{MB}}$ |

AllGather需要特别注意：修改前其Executor将 `multipleDimensionSplitRatio` 存放为先Clos后
Mesh的比例，因此配置值为 $1-r_{AllGather}$；完成第12章的编码修改后，配置语义统一为
先Mesh后Clos，配置值改为 $r_{AllGather}$。

## 11. 用端口数估算带宽

若Mesh和Clos的单端口有效带宽分别为 $b$ 和 $c$，单个机内remoteRank对应Channel组的端口和为
$q_M$，机内共有 $M-1$ 个remoteRank，则公式使用的机内有效端口规模为：

$$
p_M=(M-1)q_M
$$

Server间第一个非空Channel组的原始端口和记为 $p_{C,raw}$。普通机型直接使用该值；若该
Channel组恰好包含两条Channel，且两条Channel的有效 `dieId` 不同，则判定为POD机型。
POD机型的Server间链路存在2:1收敛，两个Die上的端口不能按无收敛带宽直接相加，因此
公式使用的Server间有效端口规模为：

$$
p_C=
\begin{cases}
\frac{p_{C,raw}}{2}, & \text{两条Channel且dieId不同} \\
p_{C,raw}, & \text{其他情况}
\end{cases}
$$

当 `dieId` 查询失败或为无效值时，不判定为POD，保持 $p_C=p_{C,raw}$ 并打印Warning。
因此：

$$
B=p_M b
$$

$$
C=p_C c
$$

只有当两类端口的单端口有效带宽相同，或者端口数已经按带宽完成归一化时，才能直接以端口数代替 $B$ 和 $C$。否则应使用“端口数 $\times$ 单端口有效带宽”计算总带宽。

例如，当 $b=c$时，带LocalReduce的Reduce类算子公式可简化为：

$$
\boxed{
r=
\frac{\frac{N-1}{N p_C}}
     {\frac{N-1}{N p_C}+\frac{21(M-1)}{20M p_M}}
}
$$

代入 $p_M=(M-1)q_M$ 后，机内时间系数可化简为：

$$
\frac{21(M-1)}{20M p_M}=\frac{21}{20M q_M}
$$

Broadcast和Scatter不包含LocalReduce，仍使用：

$$
\boxed{
r_{ScatterLike}=
\frac{\frac{N-1}{N p_C}}
     {\frac{N-1}{N p_C}+\frac{M-1}{M p_M}}
}
$$

AllGather公式可简化为：

$$
\boxed{
r_{AllGather}=
\frac{\frac{N-1}{p_C}}
     {\frac{N-1}{p_C}+\frac{M-1}{p_M}}
}
$$

代入 $p_M=(M-1)q_M$ 后，AllGather的机内时间系数可化简为：

$$
\frac{M-1}{p_M}=\frac{1}{q_M}
$$

## 12. 编码方案

### 12.1 实现目标

实际执行时无法直接获取Channel的有效带宽，因此使用Channel上的
`portGroupSize` 作为带宽系数。该处理与
`CalcDataSplitByPortGroupCommon` 的数据切分方式一致：认为端口数与可用带宽成正比，
按照端口数分配数据。对于POD机型，Server间原始端口和还需按2:1收敛关系除以2后再进入公式。

本次实现覆盖以下Parallel Executor：

- `InsReduceScatterParallelExecutor`
- `InsV2AllGatherParallelExecutor`
- `InsAllReduceParallelExecutor`
- `InsBroadcastParallelExecutor`
- `ReduceParallelExecutor`

Scatter不在本次修改范围内。

切分比例保留三种获取方式，优先级从高到低为：

1. 通信域配置。
2. 环境变量 `HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO`。
3. 内置公式计算。

Host侧负责查询通信域配置和环境变量，并将比例值及来源通过
`OpParam::opConfig` 传递到Device侧。Device侧若收到显式配置，直接使用该配置；
只有在Host侧确认通信域和环境变量都没有配置时，才使用内置公式。

### 12.2 优先级与Host/Device数据流

整体数据流如下：

```text
Host:
    通信域切分比例有效？
        是 -> ratio = 通信域配置，source = COMM_CONFIG
        否 -> 环境变量切分比例有效？
                  是 -> ratio = 环境变量配置，source = ENV_CONFIG
                  否 -> source = BUILTIN_FORMULA
    将 ratio 和 source 写入 OpParam::opConfig

Device:
    source == COMM_CONFIG 或 ENV_CONFIG？
        是 -> 直接使用 OpParam 中的 ratio
        否 -> 根据Rank数和Channel端口数计算 ratio
```

不允许Device侧重新读取环境变量或查询通信域配置，以保证Host展开和
Device展开使用相同的优先级和配置快照。

### 12.3 OpParam配置字段

当前 `DevAicpuOpConfig` 中只有比例值，且 `SetMultipleDimensionSplitRatio`
在环境变量未配置时会直接写入默认值0.5。这会导致Device侧无法区分“显式配置
0.5”和“没有配置”，因此必须增加配置来源字段。

建议定义：

```cpp
enum class MultipleDimensionSplitRatioSource : uint8_t {
    BUILTIN_FORMULA = 0,
    ENV_CONFIG,
    COMM_CONFIG
};

struct DevAicpuOpConfig {
    u32 execTimeout = 0;
    double multipleDimensionSplitRatio = 0.5;
    MultipleDimensionSplitRatioSource multipleDimensionSplitRatioSource =
        MultipleDimensionSplitRatioSource::BUILTIN_FORMULA;
};
```

`multipleDimensionSplitRatio` 在三种来源下都统一表示“先Mesh、后Clos”的数据比例。
当 `source == BUILTIN_FORMULA` 时，该字段中的0.5仅作为公式无法计算时的最终
应急回退值，不表示已存在配置。

### 12.4 Host侧配置解析

将现有 `SetMultipleDimensionSplitRatio(OpParam &param)` 扩展为可以访问通信域的形式：

```cpp
HcclResult SetMultipleDimensionSplitRatio(HcclComm comm, OpParam &param);
```

`HcclExecOp`中的调用同步调整为：

```cpp
CHK_RET(SetMultipleDimensionSplitRatio(comm, param));
```

该函数按以下顺序执行。

#### 12.4.1 通信域配置

参考 `DecideHcclOpExpansionMode` 通过 `DlHcommFunction::dlHcclConfigGetInfo`获取展开模式的
方式，使用后续由HCCL接口提供的切分比例配置类型。本文中以
`HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO` 作为预期名称，最终以实际公开的
`HcclConfigType` 枚举为准：

```cpp
double commRatio = 0.0;
const uint32_t infoLen = sizeof(commRatio);
HcclResult ret = hcommFunction.dlHcclConfigGetInfo(
    comm,
    HcclConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO,
    infoLen,
    &commRatio);
```

处理规则：

- 接口存在、通信域已配置且比例在 $(0,1]$ 内：写入该比例，来源设为
  `COMM_CONFIG`，不再读取环境变量。
- `HcclConfigGetInfo` 成功返回但比例为0.0：表示通信域未配置切分比例，保持
  `isConfigured=false`并继续读取环境变量。
- `HcclConfigGetInfo` 未导出、不支持该配置类型或通信域未设置比例：继续读取
  环境变量。
- 返回值不是有限值、小于0.0或大于1.0：打印Error并返回
  `HCCL_E_PARA`，不降级使用环境变量。
- 查询发生“不支持/未配置”以外的异常错误：返回该错误，避免隐藏通信域故障。

由于接口尚未提供切分比例配置类型，编码时应将通信域查询封装为独立的
小函数，便于后续接入真实枚举和“未配置”返回语义：

```cpp
HcclResult GetCommMultipleDimensionSplitRatio(
    HcclComm comm, double &ratio, bool &isConfigured);
```

#### 12.4.2 环境变量配置

当通信域没有有效配置时，调用现有
`GetExternalInputMultipleDimensionSplitRatio`：

```cpp
double envRatio = 0.0;
if (GetExternalInputMultipleDimensionSplitRatio(envRatio)) {
    constexpr double defaultRatio = 0.5;
    if (envRatio < 0.0 || envRatio > 1.0) {
        HCCL_WARNING("multiple dimension split ratio[%f] is out of range, "
                     "use default ratio[%f]", envRatio, defaultRatio);
        envRatio = defaultRatio;
    }
    param.opConfig.multipleDimensionSplitRatio = envRatio;
    param.opConfig.multipleDimensionSplitRatioSource =
        MultipleDimensionSplitRatioSource::ENV_CONFIG;
    return HCCL_SUCCESS;
}
```

环境变量的现有解析和校验行为保持不变：

- 非数字等无法解析的值仍由 `ParseMultipleDimensionSplitRatio`/`InitEnvConfig`
  返回参数错误。
- 能够解析但超出 $[0,1]$ 的值仍打印Warning并使用默认0.5。此时来源仍标记为
  `ENV_CONFIG`，表示环境变量路径已经给出最终决策，Device侧不再使用内置公式。

只有当环境变量未设置时，才不把0.5标记为配置值，而是设置：

```cpp
param.opConfig.multipleDimensionSplitRatio = 0.5;
param.opConfig.multipleDimensionSplitRatioSource =
    MultipleDimensionSplitRatioSource::BUILTIN_FORMULA;
```

这表示Device侧应尝试使用内置公式，0.5只是公式失败时的保底值。

在通信域切分比例接口尚未对外提供的过渡阶段，Host侧查询函数应返回
`isConfigured = false`，实际优先级为“环境变量 > 内置公式”。后续接口可用时只需实现
`GetCommMultipleDimensionSplitRatio`，其余Host/Device数据流不需改动。

### 12.5 OpParam传递与兼容性

`DevAicpuOpConfig` 是 `OpParam` 的一部分，新增的来源字段必须随现有 `OpParam` 一起从
Host侧传递到AICPU/Device侧。实现时需检查所有 `OpParam` 拷贝、序列化、快速下发和
缓存上下文，确保 `multipleDimensionSplitRatioSource` 不会在中间环节丢失。

Host侧与Device侧必须共享同一枚举定义和结构布局。如果 `OpParam` 存在跨版本二进制兼容
要求，则需在对应版本号或长度校验中同步体现该字段变更。

### 12.6 Device侧公共数据切分接口

在 `src/ops/op_common/template/template_utils.h` 和
`src/ops/op_common/template/template_utils.cc` 中增加公共计算能力，避免每个Executor重复实现
端口统计、公式计算和异常处理。

建议定义通信量模型：

```cpp
enum class ParallelDataSplitType {
    REDUCE_SCATTER_WITH_LOCAL_REDUCE = 0,
    ALL_GATHER = 1,
    SCATTER = 2
};
```

其中：

- `REDUCE_SCATTER_WITH_LOCAL_REDUCE` 使用 $k(P)=(P-1)/P$，并在Mesh侧增加
  LocalReduce时间，适用于ReduceScatter、AllReduce和Reduce。
- `SCATTER` 使用 $k(P)=(P-1)/P$，但不增加LocalReduce时间，适用于Broadcast。
- `ALL_GATHER` 使用 $k(P)=P-1$，适用于AllGather。

公共比例计算函数建议定义为：

```cpp
double CalcParallelDataSplitRatio(
    uint64_t intraRankSize,
    uint64_t interRankSize,
    const std::map<u32, std::vector<ChannelInfo>> &intraChannels,
    const std::map<u32, std::vector<ChannelInfo>> &interChannels,
    ParallelDataSplitType splitType,
    double fallbackRatio);
```

该函数只负责内置公式计算，返回先Mesh后Clos的数据比例 $r$，返回值范围为
$[0,1]$。`fallbackRatio` 仅用于公式因Channel或Rank信息不完整而无法计算的场景。

### 12.7 端口数提取

`Channel Map` 的结构为：

```cpp
std::map<u32, std::vector<ChannelInfo>>
```

其中：

- `map` 的Key是remoteRank。
- `vector<ChannelInfo>` 表示本Rank与同一个remoteRank之间可并行使用的Channel组。
- `ChannelInfo::portGroupSize` 表示对应Channel的端口带宽系数。
- `ChannelInfo::dieId` 表示该Channel本端Endpoint所属的Die，默认值为
  `INVALID_VALUE_RANKID`。

Host侧在 `HcclGetChannelImpl` 构造 `ChannelInfo` 时，使用Channel的 `localEndpoint` 调用
`HcclRankGraphGetEndpointInfo(..., ENDPOINT_ATTR_DIE_ID, ...)` 获取 `dieId`。该字段随
`AlgResourceCtxSerializable::channels` 一起序列化到Device侧，因此Device侧公式计算不需要
持有 `HcclComm`。若运行时接口不支持 `ENDPOINT_ATTR_DIE_ID` 或查询失败，则打印Warning并保留
无效值，后续按非POD机型处理，不能因为该优化信息缺失而阻断算子执行。

端口统计规则与 `CalcDataSplitByPortGroupCommon` 保持一致：

1. 在Channel Map中查找第一个非空的Channel组。
2. 对该Channel组中所有 `ChannelInfo::portGroupSize` 求和。
3. 将Server内Channel组的端口和记为 $q_M$，再计算
   $p_M=q_M(M-1)$，即 `intraPortGroupSize *= intraRankSize - 1`。
4. 将Server间Channel组的端口和记为 $p_{C,raw}$。
5. 若Server间首个非空Channel组恰好有两条Channel、两条Channel的 `dieId` 均有效且不同，
   判定为POD机型，计算 $p_C=p_{C,raw}/2$；其他情况使用 $p_C=p_{C,raw}$。

建议增加内部辅助函数：

```cpp
bool GetPortGroupSize(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    uint64_t &portGroupSize);
```

当前设计假设同一通信域内不同remoteRank的Channel端口组配置一致，因此只取第一个非空
Channel组。这样既与各Template调用 `CalcDataSplitByPortGroupCommon` 时传入单个remoteRank
的Channel组保持一致，又通过乘以机内remoteRank数量 $M-1$ 得到公式所需的机内总端口规模。
POD判断也基于同一个首个非空Server间Channel组，避免端口统计和机型判断使用不同样本。
乘法前必须检查 `uint64_t` 溢出；发生溢出时打印Warning并返回应急回退比例。

POD折算使用浮点除法：

```cpp
const bool isPod = interChannelGroup.size() == 2 &&
    interChannelGroup[0].dieId != INVALID_VALUE_RANKID &&
    interChannelGroup[1].dieId != INVALID_VALUE_RANKID &&
    interChannelGroup[0].dieId != interChannelGroup[1].dieId;
const double effectiveInterPortGroupSize =
    static_cast<double>(interPortGroupSize) / (isPod ? 2.0 : 1.0);
```

使用浮点除法可避免原始端口和为奇数时发生整数截断。后续三类算子的Clos时间系数均使用
`effectiveInterPortGroupSize`。

### 12.8 比例计算

#### 带LocalReduce的Reduce类

ReduceScatter、AllReduce和Reduce使用：

$$
r=
\frac{\frac{N-1}{N p_C}}
     {\frac{N-1}{N p_C}+\frac{21(M-1)}{20M p_M}}
$$

编码时应先计算两个时间系数：

```cpp
intraPortGroupSize *= intraRankSize - 1;
const double effectiveInterPortGroupSize =
    static_cast<double>(interPortGroupSize) / (isPod ? 2.0 : 1.0);
const double meshTimeCoeff =
    21.0 * static_cast<double>(intraRankSize - 1) /
    (20.0 * static_cast<double>(intraRankSize) * intraPortGroupSize);
const double closTimeCoeff =
    static_cast<double>(interRankSize - 1) /
    (static_cast<double>(interRankSize) * effectiveInterPortGroupSize);
const double ratio = closTimeCoeff / (closTimeCoeff + meshTimeCoeff);
```

#### Scatter类

Broadcast使用原Scatter公式，不包含LocalReduce：

$$
r_{ScatterLike}=
\frac{\frac{N-1}{N p_C}}
     {\frac{N-1}{N p_C}+\frac{M-1}{M p_M}}
$$

编码时的 `meshTimeCoeff` 不乘 $21/20$。

#### AllGather

AllGather使用：

$$
r_{AllGather}=
\frac{\frac{N-1}{p_C}}
     {\frac{N-1}{p_C}+\frac{M-1}{p_M}}
$$

编码方式为：

```cpp
intraPortGroupSize *= intraRankSize - 1;
const double meshTimeCoeff =
    static_cast<double>(intraRankSize - 1) / intraPortGroupSize;
const double closTimeCoeff =
    static_cast<double>(interRankSize - 1) / interPortGroupSize;
const double ratio = closTimeCoeff / (closTimeCoeff + meshTimeCoeff);
```

AllGather在Server数量增大时，原始公式会继续向1收敛。但128P实测表明，继续提高
先Mesh后Clos的数据比例并不优于0.5，因此AllGather在量化前增加上限：

```cpp
if (splitType == ParallelDataSplitType::ALL_GATHER) {
    ratio = std::min(ratio, 0.5);
}
```

该限制只作用于内置公式计算结果，不影响通信域配置和环境变量配置；当原始公式结果
小于或等于0.5时保持原值。

在非POD、Server内固定8卡、$q_M=1$、$p_M=7$、$p_C=8$ 的假设下，Server数量
$N$ 从2增加到16时，三类公式的量化前ratio变化如下。AllGather限制后曲线表示
`min(rawRatio, 0.5)`，实际返回前还会继续执行八分位量化。

图例：

| 颜色 | 曲线名称 | 含义 |
| --- | --- | --- |
| <span style="display:inline-block;width:36px;height:3px;background:#1f77b4;vertical-align:middle;"></span> 蓝色 `#1f77b4` | Reduce类(带LocalReduce) | ReduceScatter、AllReduce、Reduce使用的带LocalReduce模型 |
| <span style="display:inline-block;width:36px;height:3px;background:#ff7f0e;vertical-align:middle;"></span> 橙色 `#ff7f0e` | Scatter类 | Broadcast使用的Scatter模型 |
| <span style="display:inline-block;width:36px;height:3px;background:#2ca02c;vertical-align:middle;"></span> 绿色 `#2ca02c` | AllGather限制前 | AllGather原始公式结果 |
| <span style="display:inline-block;width:36px;height:3px;background:#d62728;vertical-align:middle;"></span> 红色 `#d62728` | AllGather限制后 | AllGather执行 `min(rawRatio, 0.5)` 后的结果 |

<svg width="760" height="420" viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ratio随Server数量变化曲线图">
  <rect x="0" y="0" width="760" height="420" fill="#ffffff"/>
  <text x="380" y="22" text-anchor="middle" font-size="16" font-family="Arial, sans-serif">ratio随Server数量变化</text>
  <line x1="60" y1="330" x2="700" y2="330" stroke="#333333" stroke-width="1.2"/>
  <line x1="60" y1="30" x2="60" y2="330" stroke="#333333" stroke-width="1.2"/>
  <g stroke="#dddddd" stroke-width="1">
    <line x1="60" y1="287.1" x2="700" y2="287.1"/>
    <line x1="60" y1="244.3" x2="700" y2="244.3"/>
    <line x1="60" y1="201.4" x2="700" y2="201.4"/>
    <line x1="60" y1="158.6" x2="700" y2="158.6"/>
    <line x1="60" y1="115.7" x2="700" y2="115.7"/>
    <line x1="60" y1="72.9" x2="700" y2="72.9"/>
    <line x1="60" y1="30" x2="700" y2="30"/>
  </g>
  <g font-size="12" font-family="Arial, sans-serif" fill="#333333">
    <text x="52" y="334" text-anchor="end">0.0</text>
    <text x="52" y="291.1" text-anchor="end">0.1</text>
    <text x="52" y="248.3" text-anchor="end">0.2</text>
    <text x="52" y="205.4" text-anchor="end">0.3</text>
    <text x="52" y="162.6" text-anchor="end">0.4</text>
    <text x="52" y="119.7" text-anchor="end">0.5</text>
    <text x="52" y="76.9" text-anchor="end">0.6</text>
    <text x="52" y="34" text-anchor="end">0.7</text>
    <text x="60" y="350" text-anchor="middle">2</text>
    <text x="151.4" y="350" text-anchor="middle">4</text>
    <text x="242.9" y="350" text-anchor="middle">6</text>
    <text x="334.3" y="350" text-anchor="middle">8</text>
    <text x="425.7" y="350" text-anchor="middle">10</text>
    <text x="517.1" y="350" text-anchor="middle">12</text>
    <text x="608.6" y="350" text-anchor="middle">14</text>
    <text x="700" y="350" text-anchor="middle">16</text>
    <text x="380" y="385" text-anchor="middle">Server数量N</text>
    <text x="18" y="180" text-anchor="middle" transform="rotate(-90 18 180)">ratio</text>
  </g>
  <polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="60.0,191.6 105.7,163.7 151.4,151.3 197.1,144.9 242.9,140.6 288.6,137.6 334.3,135.0 380.0,133.7 425.7,132.0 471.4,131.1 517.1,130.3 562.9,129.4 608.6,129.0 654.3,128.1 700.0,127.7"/>
  <polyline fill="none" stroke="#ff7f0e" stroke-width="2.5" points="60.0,187.3 105.7,158.6 151.4,146.1 197.1,139.7 242.9,135.0 288.6,132.0 334.3,129.9 380.0,128.1 425.7,126.9 471.4,126.0 517.1,125.1 562.9,124.3 608.6,123.9 654.3,123.0 700.0,122.6"/>
  <polyline fill="none" stroke="#2ca02c" stroke-width="2.5" points="60.0,282.4 105.7,244.3 151.4,213.0 197.1,187.3 242.9,165.0 288.6,146.1 334.3,129.9 380.0,115.7 425.7,103.3 471.4,91.7 517.1,81.9 562.9,72.9 608.6,64.7 654.3,57.4 700.0,50.6"/>
  <polyline fill="none" stroke="#d62728" stroke-width="2.5" points="60.0,282.4 105.7,244.3 151.4,213.0 197.1,187.3 242.9,165.0 288.6,146.1 334.3,129.9 380.0,115.7 425.7,115.7 471.4,115.7 517.1,115.7 562.9,115.7 608.6,115.7 654.3,115.7 700.0,115.7"/>
</svg>

公共函数最终应检查计算结果是否为有限值并位于 $[0,1]$ 内。校验通过后，不再采用“保留两位小数”的方式，
而是将内置公式结果量化到以下候选集合：

$$
\left\{\frac{1}{8},\frac{2}{8},\frac{3}{8},\frac{4}{8},
\frac{5}{8},\frac{6}{8},\frac{7}{8}\right\}
$$

量化遵循就近原则。实现时先计算最接近的八分位索引，再将索引限制到 $[1,7]$：

```cpp
constexpr double ratioStep = 1.0 / 8.0;
const double nearestRatioIndex = std::round(ratio / ratioStep);
const double clampedRatioIndex =
    std::max(1.0, std::min(nearestRatioIndex, 7.0));
const double quantizedRatio = clampedRatioIndex * ratioStep;
```

例如：$0.33$ 距离 $3/8=0.375$ 最近，最终返回 $3/8$；小于 $1/8$ 的结果返回 $1/8$，
大于 $7/8$ 的结果返回 $7/8$。当原始值恰好位于两个候选值中点时，`std::round` 选择较大的候选值。

量化直接作用于原始公式结果，不先保留两位小数，避免临界点因十进制舍入发生偏移。通信域配置和环境变量配置
属于用户显式配置，直接使用配置值，不执行八分位量化。

### 12.9 内置公式回退策略

只有 `source == BUILTIN_FORMULA` 时才会进入本节。出现以下任意情况时，打印Warning并
返回 `fallbackRatio`：

- Server内或Server间Channel Map为空。
- 找不到非空的Channel组。
- 任意Channel组的端口总数为0。
- `intraPortGroupSize * (intraRankSize - 1)` 发生整数溢出或结果为0。
- `intraRankSize` 或 `interRankSize` 为0。
- 公式分母为0。
- 计算结果不是有限值或不在 $[0,1]$ 内。

`dieId` 查询失败不属于公式整体失败。该场景在Host侧打印Warning，Device侧因 `dieId` 无效而
不执行POD折算，继续使用 $p_C=p_{C,raw}$ 计算比例。

Warning中应至少包含：

- Executor或公共函数名称。
- 回退原因。
- Server内和Server间Rank数。
- Server内和Server间端口数。
- 最终使用的应急回退比例。

回退值也应限制在 $[0,1]$ 内。若配置值本身非法，建议限制到最近的边界值：

```cpp
const double validFallback = std::clamp(fallbackRatio, 0.0, 1.0);
```

CCU、AIV或其他无法恢复Channel Map的执行路径，若Host侧已提供通信域或环境变量
配置，仍直接使用配置值；只有在无显式配置且需要公式计算时，才因缺少Channel
信息回退到0.5。

### 12.10 Executor接入

五个Executor的 `GetParallelDataSplit` 首先检查Host侧传入的来源：

```cpp
double ratio = multipleDimensionSplitRatio_;
if (multipleDimensionSplitRatioSource_ ==
    MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
    ratio = CalcParallelDataSplitRatio(
        intraRankSize,
        interRankSize,
        intraChannels,
        interChannels,
        splitType,
        multipleDimensionSplitRatio_);
}
splitDataSize.push_back(ratio);
splitDataSize.push_back(1.0 - ratio);
```

每个Executor需从 `param.opConfig` 同时保存：

```cpp
multipleDimensionSplitRatio_ = param.opConfig.multipleDimensionSplitRatio;
multipleDimensionSplitRatioSource_ =
    param.opConfig.multipleDimensionSplitRatioSource;
```

为避免五个Executor重复来源判断，也可以在公共层增加一个包装函数，但端口公式
函数本身仍应保持只负责计算，便于独立测试。

#### ReduceScatter

`InsReduceScatterParallelExecutor::GetParallelDataSplit` 先处理Host侧配置来源，只在需要
内置公式时调用公共函数：

```cpp
double ratio = multipleDimensionSplitRatio_;
if (multipleDimensionSplitRatioSource_ ==
    MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
    ratio = CalcParallelDataSplitRatio(
        rankSizeLevel0_,
        rankSizeLevel1_,
        intraChannelMap_,
        interChannelMap_,
        ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE,
        multipleDimensionSplitRatio_);
}
splitDataSize.push_back(ratio);
splitDataSize.push_back(1.0 - ratio);
```

#### AllGather

`InsV2AllGatherParallelExecutor::GetParallelDataSplit` 使用
`ParallelDataSplitType::ALL_GATHER`。

公共函数返回值始终表示先Mesh后Clos的比例，因此调整当前存储方向为：

```cpp
splitDataSize.push_back(ratio);
splitDataSize.push_back(1.0 - ratio);
```

修改后不再把 `multipleDimensionSplitRatio` 解释为先Clos后Mesh的比例。通信域配置、
环境变量配置、内置公式和应急回退值都统一表示先Mesh后Clos的比例。

#### AllReduce

`InsAllReduceParallelExecutor::GetParallelDataSplit` 使用
`ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE`，端口信息分别来自 `intraLinks_` 和
`interLinks_`，Rank数分别使用Server内和Server间通信域的Rank数。

#### Broadcast

`InsBroadcastParallelExecutor::GetParallelDataSplit` 使用
`ParallelDataSplitType::SCATTER`。其第一阶段通信量系数虽然与ReduceScatter相同，
但不执行Reduce操作，因此不能增加Mesh侧LocalReduce时间。

#### Reduce

当前 `ReduceParallelExecutor` 在 `OrchestrateImpl` 中直接构造：

```cpp
std::array<long double, dataSplitPart_> dataSplitSize{
    multipleDimensionSplitRatio_,
    1.0 - multipleDimensionSplitRatio_};
```

建议为其补充 `GetParallelDataSplit`，统一调用公共函数，并在 `OrchestrateImpl` 中使用计算结果：

```cpp
const double ratio = GetParallelDataSplit();
std::array<long double, dataSplitPart_> dataSplitSize{
    ratio,
    1.0 - ratio};
```

同时，后续每轮数据计数切分也必须使用最终选定的成员比例，不能继续使用未经来源判断的
原始字段。推荐新增 `parallelDataSplitRatio_` 保存最终比例，保留
`multipleDimensionSplitRatio_` 和 `multipleDimensionSplitRatioSource_` 用于记录Host侧传入的原始决策。
其内置公式类型使用 `ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE`。

### 12.11 日志

Host侧配置解析完成时打印Info日志：

```text
ratioSource[COMM_CONFIG|ENV_CONFIG|BUILTIN_FORMULA], configuredRatio[value]
```

当通信域配置不可用而继续检查环境变量时，使用Info日志。通信域配置越界时
打印Error并返回参数错误。环境变量保留现有日志语义：格式不可解析时报错，数值越界时
打印Warning并使用0.5。

内置公式计算成功时打印Info日志：

```text
intraRankSize[M], interRankSize[N], intraPortGroupSize[pM=(M-1)qM],
interPortGroupSize[pCraw], effectiveInterPortGroupSize[pC], isPod[0|1],
splitType[type], rawRatio[raw], limitedRatio[limited], quantizedRatio[r]
```

各Executor打印最终两份数据的比例：

```text
meshFirstRatio[r], closFirstRatio[1-r]
```

日志中的比例语义统一为：

- 第一项：先Mesh后Clos。
- 第二项：先Clos后Mesh。

### 12.12 测试方案

Host侧配置优先级测试至少覆盖：

| 通信域配置 | 环境变量 | 期望来源 | 期望结果 |
| --- | --- | --- | --- |
| 有效 | 有效 | `COMM_CONFIG` | 使用通信域比例 |
| 有效 | 未设置 | `COMM_CONFIG` | 使用通信域比例 |
| 返回0.0（未配置） | 有效 | `ENV_CONFIG` | 使用环境变量比例 |
| 返回0.0（未配置） | 未设置 | `BUILTIN_FORMULA` | Device侧使用公式 |
| 未配置/不支持 | 有效 | `ENV_CONFIG` | 使用环境变量比例 |
| 无效 | 任意 | - | 返回参数错误 |
| 未配置/不支持 | 未设置 | `BUILTIN_FORMULA` | Device侧使用公式 |
| 未配置/不支持 | 格式不可解析 | - | 保留现有行为，环境变量解析返回参数错误 |
| 未配置/不支持 | 可解析但越界 | `ENV_CONFIG` | Warning后使用0.5，不调用内置公式 |
| 查询发生异常错误 | 任意 | - | 返回错误 |

Device侧来源选择测试至少覆盖：

1. `COMM_CONFIG` 直接使用Host侧比例，不访问Channel公式。
2. `ENV_CONFIG` 直接使用Host侧比例，不访问Channel公式。
3. `BUILTIN_FORMULA` 在Channel信息有效时使用公式。
4. `BUILTIN_FORMULA` 在Channel信息无效时Warning并回退到0.5。

公共计算函数应增加单元测试，至少覆盖以下场景：

| 场景 | 输入 | 期望 |
| --- | --- | --- |
| Reduce类等Rank等原始端口 | $M=N=P,\ q_M=p_C$ | 原始值 $\frac{20(P-1)}{20(P-1)+21}$，最终量化到最近的八分位 |
| Reduce类不同Rank数 | $M\ne N,\ p_M=p_C$ | 使用包含 $21/20$ LocalReduce系数的公式 |
| Reduce类不同端口数 | $p_M\ne p_C$ | 端口多的一侧获得更多首阶段数据，同时计入LocalReduce开销 |
| Broadcast等Rank等原始端口 | $M=N=P,\ q_M=p_C$ | $r=(P-1)/P$，不包含LocalReduce系数 |
| AllGather等Rank等原始端口 | $M=N=P,\ q_M=p_C$ | $r=(P-1)/P$ |
| AllGather不同Rank数 | $M\ne N$ | 使用 $P-1$ 系数，与带LocalReduce的Reduce类结果区分 |
| AllGather超过上限 | 原始公式结果大于0.5 | 量化前限制到0.5，最终返回0.5 |
| 多Channel端口组 | 单个对端有多个Channel | 先求各 `portGroupSize` 之和 $q_M$，机内再乘 $M-1$ |
| POD双通道 | Server间恰好两条Channel，`dieId` 均有效且不同 | $p_C=p_{C,raw}/2$ 后进入公式 |
| 双通道同Die | Server间恰好两条Channel，`dieId` 相同 | 不折算，$p_C=p_{C,raw}$ |
| Server间单通道 | Server间只有一条Channel | 不折算，$p_C=p_{C,raw}$ |
| dieId查询失败 | 任一Channel的 `dieId` 无效 | Host打印Warning，不判定为POD，算子继续执行 |
| 第一个Map项为空 | 后续Map项有效 | 使用第一个非空Channel组 |
| Channel Map为空 | 无有效端口 | Warning并返回应急回退比例 |
| 端口总数为0 | `portGroupSize` 全为0 | Warning并返回应急回退比例 |
| Rank数为0 | 任一Rank数为0 | Warning并返回应急回退比例 |
| 非法应急回退值 | 小于0或大于1 | 限制到 $[0,1]$ |
| 内置公式多位小数 | 例如原始值 $0.740740\ldots$ | 返回 $6/8=0.75$ |
| 内置公式结果约为 $0.33$ | 距离 $3/8$ 最近 | 返回 $3/8=0.375$ |
| 内置公式结果接近0 | 小于 $1/8$ | 返回 $1/8=0.125$ |
| 内置公式结果接近1 | 大于 $7/8$ | 返回 $7/8=0.875$ |

Executor级测试重点验证：

1. ReduceScatter、AllReduce和Reduce选择
   `REDUCE_SCATTER_WITH_LOCAL_REDUCE` 模型。
2. Broadcast选择 `SCATTER` 模型，不增加LocalReduce时间。
3. AllGather选择 `ALL_GATHER` 模型。
4. 所有Executor输出数组第一项均为先Mesh后Clos比例。
5. 存在显式配置时不调用内置公式。
6. 无显式配置且无Channel信息时回退到0.5并输出Warning。

# 1. SRS：AllGather NHR末步直出与LocalCopy消减

> 对应提交：`3e2f7159`（allgather nhr decrease localcopy in nhr，MR !956）、`29908bce`（open allgather nhr template optimization，MR !2076）
>
> 需求范围：`src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.{h,cc}`中的AICPU NHR模板内部优化。

## 1.1 介绍

AllGather NHR模板执行时，首先将本rank输入数据搬入HCCL Scratch Buffer，随后按NHR步序在rank间交换数据。优化前，每一步收到的数据都先写入本地Scratch；全部NHR通信完成后，再通过`PostLocalCopy`将Scratch中的所有rank slice逐份复制到最终Output。

该流程在最后一步存在冗余：最后一步收到的数据已经不再参与后续NHR转发，仍然先落到Scratch再执行Scratch→Output的LocalCopy，会增加本地搬运量；同时，原`PostLocalCopy`位于所有通信步骤之后，与末步通信串行执行，增加尾部时延。

本需求的目标如下：

1. 在满足条件时，将NHR最后一步所需的远端slice从对端CCL Buffer直接Read到本端Output，避免“远端→Scratch→Output”的中间落盘和LocalCopy。
2. 将Scratch中已具备数据的`PostLocalCopy`与最后一步Read并行执行，隐藏部分本地搬运时延。
3. 原地AllGather场景下，如果本rank输入已位于最终输出位置，则跳过本rank slice的冗余LocalCopy。
4. 将末步直出能力从原地场景扩展到非原地场景；非原地场景保留本rank slice的必要复制，保证输出正确。
5. 对PCIe DMA Read、RemoteMemAccess、Output即Scratch及小rank等不满足条件的场景保持原流程，不改变AllGather结果语义。

## 1.2 输入

| 输入 | 说明 |
| ---- | ---- |
| `OpParam param` | AllGather算子参数；本需求使用`DataDes.dataType`确定数据类型和单元素字节数 |
| `TemplateDataParams tempAlgParams` | 模板数据参数，包含`sliceSize`、`tailSize`、`repeatNum`、slice/repeat stride及RemoteMemAccess标志 |
| `BuffInfo` | 输入、输出和HCCL Buffer的指针、BufferType、大小及base offset |
| `TemplateResource templateResource` | 模板执行资源，包括Thread列表和按远端rank组织的Channel列表 |
| `subCommRanks_`、`templateRankSize_` | NHR子通信域rank映射和参与NHR的rank数量 |
| `channelsPerRank_` | 每个远端rank使用的通道数，决定多通道数据切分和并行线程数量 |
| `dataSplit_`/`dataOffset_` | 每个通道负责的普通slice大小和偏移 |
| `dataSplitTail_`/`dataOffsetTail_` | 尾块模式下最后一个rank slice在每个通道上的大小和偏移 |

## 1.3 处理

1. 资源计算阶段按每通道两条Thread申请资源：一条执行NHR通信，一条执行末步并行`PostLocalCopy`；为Thread同步申请对应Notify。
2. Kernel启动后根据通信协议、RemoteMemAccess标志及Output Buffer属性判断是否允许最后一步直出。
3. 根据输入输出指针、BufferType、base offset和stride判断本rank输入是否已位于最终输出位置，决定能否跳过本rank slice复制。
4. NHR非最后一步继续执行Scratch↔Scratch通信，确保收到的数据能够参与后续步骤转发。
5. 满足末步优化条件时，从`fromRank`的远端CCL Buffer批量Read最后一步全部slice到本端Output；同时在独立Thread上复制本地Scratch中前序步骤已具备的slice。
6. `PostLocalCopy`跳过已由末步Read写入Output的slice；仅在安全的原地布局下额外跳过本rank slice。
7. 不满足优化条件时，按原流程执行所有NHR步骤，并在通信结束后串行完成`PostLocalCopy`。

## 1.4 输出

1. AllGather最终Output：每个参与rank的输入slice按算法rank顺序写入对应输出区间。
2. 优化路径下，Output由两条互不重叠的数据路径共同完成：
   - 末步缺失slice：远端CCL Buffer→本端Output；
   - 前序步骤已存在于本地Scratch的slice：本地Scratch→本端Output。
3. 原地布局满足跳过条件时，本rank slice保留在原有Output位置，不重复覆盖。
4. 接口返回`HCCL_SUCCESS`或线程、通道、同步、Read、Write、LocalCopy过程中产生的相应`HcclResult`错误码。

## 1.5 约束分析

| 支持的算子名称 | AllGather |
| -------------- | ---- |
| 支持的算法名称 | NHR；不改变算法选择器，仅优化已选中`InsTempAllGatherNHR`后的模板内部数据流 |
| 支持的芯片类型 | 继承AICPU NHR模板原有芯片支持范围，本需求未新增芯片枚举或芯片专用分支 |
| 支持的展开模式 | AICPU模板展开 |
| 支持的拓扑形态 | 继承NHR模板原有拓扑选择范围；末步直出仅在Channel集合不含PCIe协议时启用 |
| 支持的调用类型 | 继承AllGather NHR执行器原有单算子、图模式调用能力，不改变对外API |
| 支持的数据类型 | 继承AllGather和`DATATYPE_SIZE_TABLE`支持的数据类型，本需求不引入新的类型限制 |
| 支持的数据量 | 无新增数据量阈值；`sliceSize`和`tailSize`同时为0时直接成功返回 |
| 是否支持绕路 | 不改变原NHR通道和路由选择；末步仍使用`fromRank`/`toRank`对应Channel完成同步和Read |
| 是否支持确定性计算 | 支持；AllGather仅做数据搬运，本需求不改变slice编号和输出位置 |

末步直出启用条件：

| 条件 | 要求 | 不满足时的行为 |
| ---- | ---- | -------------- |
| 通信模式 | `isDmaRead_ == false`，即当前Channel集合不是PCIe DMA Read模式 | 回退普通NHR步骤和串行`PostLocalCopy` |
| 远端内存访问 | `enableRemoteMemAccess_ == false` | 回退原流程 |
| 输出Buffer类型 | `outBuffType == BufferType::OUTPUT` | 回退原流程 |
| Output与Scratch关系 | `outputPtr != hcclBuff.addr` | Output即Scratch时原本不需要`PostLocalCopy`，不启用本优化 |
| NHR步位置 | 当前为最后一步且`stepInfo.nSlices > 1` | 非末步或末步只有1个slice时执行普通路径 |

跳过本rank slice复制需要额外同时满足：

| 条件 | 要求 |
| ---- | ---- |
| Buffer类型 | `inBuffType == OUTPUT`且`outBuffType == OUTPUT` |
| 输入输出地址 | `inputPtr == outputPtr` |
| base offset | `inBuffBaseOff == outBuffBaseOff` |
| slice布局 | `inputSliceStride == outputSliceStride` |
| repeat布局 | `inputRepeatStride == outputRepeatStride` |

资源约束：假设每个远端rank有`C = channelsPerRank_`条通道，需求实现后的资源量为：

| 资源 | 优化前 | 优化后 |
| ---- | ------ | ------ |
| Thread总数 | `C` | `2C` |
| Slave Thread数 | `C - 1` | `2C - 1` |
| 每个Slave Thread的Notify数 | 1 | 2 |
| Main Thread Notify数 | `C - 1` | `2C - 1` |
| Scratch倍数 | `templateRankSize_` | 不变，仍为`templateRankSize_` |

# 2. SD：AllGather NHR末步直出与LocalCopy并行设计

## 2.1 功能描述

本设计在`InsTempAllGatherNHR`内部增加“普通NHR步”和“最后一步直出”两种slice构造及执行模式，并增加每通道一条专用PostCopy Thread。

两个提交的职责关系如下：

| 提交 | 主要设计内容 |
| ---- | ------------ |
| `3e2f7159` | 建立末步优化框架：线程数扩展为每通道两条、增加Thread Notify、抽取slice计算与step执行函数、识别严格原地场景、末步采用“首slice Write到Scratch，其余slice Read到Output”，并行执行剩余`PostLocalCopy` |
| `29908bce` | 将“末步能否直出”与“本rank能否跳过复制”解耦，使非原地Output也可进入优化；末步由“Write+Read”收敛为“所有末步slice均Read到Output”，删除首slice中间落Scratch及其LocalCopy |

最终设计包含以下功能：

1. `CanReadLastStepToOutput`判断当前Buffer和通信模式是否允许末步直出。
2. `CanSkipOwnSliceCopy`独立判断本rank输入是否已经处于最终Output位置。
3. `CalcSliceInfo`统一计算普通slice、尾块、repeat及多通道场景下的Scratch偏移和数据长度。
4. `BuildStepSlices`通过`StepBuildMode`构造普通步Scratch↔Scratch描述或末步远端Scratch→Output描述。
5. `RunStepNHR`按step选择普通通信或`RunLastStepReadToOutput`。
6. `RunLastStepReadToOutput`同步并启动PostCopy Thread，然后在原通信Thread上批量Read末步数据，实现通信与LocalCopy并行。
7. `PostLocalCopy`基于`lastStepReadSliceIdxs_`和`skipOwnSliceCopy_`过滤已经就位的数据，仅复制仍需从Scratch输出的slice。

本需求仅修改HCCL AllGather AICPU模板层，继续通过既有wrapper接口使用HCOMM Thread、Channel、Read、Write和Notify能力，不引入HCCL对HCOMM私有头文件或实现的编译期依赖。

## 2.2 流程描述

### 2.2.1 资源申请流程

```text
CalcRes
  ↓
申请NHR Channel并计算channelsPerRank_ = C
  ↓
GetThreadNum() = 2C
  ↓
slaveThreadNum = 2C - 1
notifyNumPerThread = [2, 2, ..., 2]
notifyNumOnMainThread = 2C - 1
  ↓
前C条Thread用于各通道NHR通信
后C条Thread用于各通道PostLocalCopy
```

Notify用途：

- 每条Slave Thread的Notify索引0用于Kernel开始阶段Main Thread→Slave Thread的前同步。
- PostCopy Thread的`POST_COPY_NOTIFY_IDX = 1`用于对应通信Thread在前序NHR步骤完成后唤醒PostCopy Thread。
- Main Thread上的`2C - 1`个Notify槽位用于Kernel结束阶段各Slave Thread→Main Thread的后同步。

### 2.2.2 Kernel总体流程

```text
KernelRun
  ├─ sliceSize和tailSize均为0 → 直接返回HCCL_SUCCESS
  ├─ Thread数量不足 → 返回HCCL_E_INTERNAL
  ↓
准备数据类型、多通道切分、协议模式和Buffer参数
  ↓
readLastStepToOutput_ = CanReadLastStepToOutput()
skipOwnSliceCopy_ = readLastStepToOutput_ && CanSkipOwnSliceCopy()
  ↓
主从Thread前同步
  ↓
逐channel执行：
  ├─ LocalDataCopy：本rank Input → 本地Scratch
  ├─ templateRankSize_ > 1时执行RunAllGatherNHR
  └─ 特殊末步未启动PostCopy Thread时，在通信Thread串行执行PostLocalCopy
  ↓
主从Thread后同步
  ↓
返回HCCL_SUCCESS
```

### 2.2.3 普通NHR步骤

`GetStepInfo`计算每一步的收发rank和slice列表：

```text
nSteps = ceil(log2(templateRankSize_))
deltaRank = 1 << (nSteps - 1 - step)
fromRank = (myAlgRank + rankSize - deltaRank) % rankSize
toRank   = (myAlgRank + deltaRank) % rankSize
nSlices  = (rankSize - 1 + 2^(nSteps - 1 - step)) / 2^(nSteps - step)
```

非最后一步必须保留Scratch↔Scratch数据流，因为本步收到的slice可能在后续步骤继续发送：

```text
本地Scratch[txIdx] --SendRecvBatchWrite--> toRank远端Scratch[txIdx]
fromRank远端Scratch[rxIdx] -------------> 本地Scratch[rxIdx]
```

PCIe DMA Read模式继续调用普通`SendRecvRead`路径，不进入末步直出优化。

### 2.2.4 最后一步直出与并行PostLocalCopy

进入最后一步前，NHR调度保证`fromRank`的CCL Buffer中已经具备本rank最后一步所需的全部`rxSlice`：第一个slice是`fromRank`自己的数据，其余slice在前序步骤中到达`fromRank`。最后一步之后不再有数据转发，因此这些slice不需要写入本地Scratch。

```text
通信Thread[channelIdx]
  │
  ├─ 前序NHR步骤全部完成
  ├─ PreSyncInterThreads(POST_COPY_NOTIFY_IDX)
  │       └──────────────→ PostCopy Thread[C + channelIdx]
  │                           └─ PostLocalCopy：本地Scratch已有slice → Output
  │
  └─ SendRecvBatchRead：fromRank远端Scratch中的末步slice → 本端Output

两条Thread并行执行，最终由Kernel末尾PostSyncInterThreads汇合。
```

`BuildStepSlices(LAST_STEP_READ_TO_OUTPUT)`为每个repeat、每个末步slice构造：

```text
src = channelRecv.remoteCclMem.addr + rxScratchOff
dst = outputPtr + outBuffBaseOff
      + rpt * outputRepeatStride
      + rxIdx * outputSliceStride
      + rxPartialOffset
size = rxSliceSize
```

末步每个`rxIdx`在第一个repeat时记录到`lastStepReadSliceIdxs_`。rank集合对所有repeat相同，因此`PostLocalCopy`可据此在全部repeat中跳过这些slice。

### 2.2.5 PostLocalCopy过滤规则

`PostLocalCopy`遍历全部repeat和子通信域rank，按以下顺序过滤：

1. Output与HCCL Scratch为同一地址时，整个PostLocalCopy直接跳过。
2. 末步直出已启用、原地布局满足且`algRank == myAlgRank`时，跳过本rank slice。
3. `algRank`存在于`lastStepReadSliceIdxs_`时，跳过由末步Read直接写入Output的slice。
4. 其余slice执行本地Scratch→Output的`LocalCopy`。

该规则保证PostCopy Thread与通信Thread写入的Output区间互不重叠。

### 2.2.6 8卡示例

以8卡NHR、rank 0、原地AllGather为例，rank 0在最后一步从rank 7获得`7、5、3、1`四个slice。

| 流程 | 末步数据路径 | PostLocalCopy的slice | LocalCopy次数 |
| ---- | ------------ | ------------------- | ------------- |
| 优化前 | `7、5、3、1`先写本地Scratch | `0～7`全部复制 | 8 |
| `3e2f7159`阶段 | `7`写Scratch；`5、3、1`直接Read到Output | `4、6、2、7`，本rank`0`跳过 | 4 |
| `29908bce`最终设计 | `7、5、3、1`全部直接Read到Output | `4、6、2`，本rank`0`跳过 | 3 |

非原地场景下本rank`0`不能跳过，因此最终设计由PostLocalCopy复制`0、4、6、2`四个slice，另外四个slice由末步Read直接写入Output。

## 2.3 数据描述

### 2.3.1 模板输入数据

| 数据结构/字段 | 说明 |
| ------------- | ---- |
| `TemplateDataParams::sliceSize` | 常规rank slice大小，单位为字节 |
| `TemplateDataParams::tailSize` | 最后一个rank的不等长尾块大小，0表示无尾块 |
| `repeatNum` | 重复数据块数量 |
| `inputSliceStride`/`outputSliceStride` | 相邻rank slice在Input/Output中的间隔 |
| `inputRepeatStride`/`outputRepeatStride` | 相邻repeat数据块的间隔 |
| `enableRemoteMemAccess` | 是否使用RemoteMemAccess模式；为true时禁用末步直出 |
| `BuffInfo` | `inputPtr`、`outputPtr`、`hcclBuff`、三类BufferType及base offset |

### 2.3.2 NHR步骤数据

| 字段 | 说明 |
| ---- | ---- |
| `AicpuNHRStepInfo::step` | 当前NHR步骤编号 |
| `fromRank`/`toRank` | 当前步骤接收来源和发送目标的算法rank |
| `nSlices` | 当前步骤收发的slice数量 |
| `txSliceIdxs` | 本rank需要向`toRank`提供的slice编号 |
| `rxSliceIdxs` | 本rank需要从`fromRank`获得的slice编号 |

### 2.3.3 Slice计算数据

`SliceCalcInfo`封装单个slice在某repeat、某channel下的计算结果：

| 字段 | 说明 |
| ---- | ---- |
| `txIdx`/`rxIdx` | 发送和接收slice编号 |
| `txPartialOffset`/`rxPartialOffset` | 多通道切分后的通道内偏移；尾块使用`dataOffsetTail_` |
| `scratchBase` | 当前repeat在HCCL Scratch中的起始偏移 |
| `txScratchOff`/`rxScratchOff` | 发送、接收slice在Scratch中的绝对偏移 |
| `txSliceSize`/`rxSliceSize` | 当前通道负责的字节数；尾块使用`dataSplitTail_` |

`DataSlice`最终以`addr_ + offset_`描述源/目的地址，以`size_`描述字节数，以`count_ = size_ / DATATYPE_SIZE_TABLE[dataType_]`描述元素个数。

### 2.3.4 模板状态数据

| 字段 | 生命周期与用途 |
| ---- | -------------- |
| `readLastStepToOutput_` | 每次`KernelRun`重新计算；标识是否允许末步Read直出 |
| `skipOwnSliceCopy_` | 每次`KernelRun`重新计算；标识是否可跳过本rank slice复制 |
| `lastStepReadSliceIdxs_` | 每个channel进入`RunAllGatherNHR`时清空，记录该channel末步直出的算法rank编号 |
| `postLocalCopyLaunched` | 每个channel初始化为false；特殊末步成功启动PostCopy Thread后置true，防止Kernel重复串行执行`PostLocalCopy` |

## 2.4 依赖性描述

| 依赖项 | 说明 |
| ------ | ---- |
| NHR调度 | 依赖`GetNHRStepNum`和`GetStepInfo`生成的NHR步骤；核心不变量是最后一步开始前，`fromRank`已拥有全部末步`rxSlice` |
| Channel和远端内存 | 依赖`ChannelInfo::remoteCclMem.addr`可按计算偏移读取，并依赖`channelSend`/`channelRecv`完成Read模式的双端Channel Notify同步 |
| 数据搬运wrapper | 使用`LocalCopy`、`SendRecvBatchWrite`、`SendRecvBatchRead`和PCIe回退用`SendRecvRead`，不新增wrapper函数签名 |
| Thread同步wrapper | 使用`PreSyncInterThreads`和`PostSyncInterThreads`协调通信Thread与PostCopy Thread |
| 多通道数据切分 | 使用`CalcDataSplitByPortGroup`生成`dataSplit_`和`dataOffset_`，普通块与尾块分别计算 |
| 资源管理 | `AlgResourceRequest`必须按`GetRes`结果提供足够的Thread和Notify；Thread不足时Kernel返回`HCCL_E_INTERNAL` |
| HCOMM能力 | 继续通过HCCL既有wrapper和dlsym层调用HCOMM Read/Write/Notify能力，不新增HCOMM符号及编译期硬依赖 |
| 上层执行器 | 沿用AllGather执行器对`InsTempAllGatherNHR`的选择和参数组装，本需求不修改selector、executor及`include/hccl.h` |

## 2.5 接口描述

| 函数原型 | `bool CanReadLastStepToOutput() const` |
| ---------- | ---- |
| 函数功能 | 判断当前通信模式和Output Buffer是否允许最后一步直接Read到Output |
| 输入说明 | 使用成员`isDmaRead_`、`enableRemoteMemAccess_`及`tempAlgParams_.buffInfo` |
| 输出说明 | 无 |
| 返回值说明 | `true`：允许继续判断末步特殊路径；`false`：全部step按普通路径执行 |

| 函数原型 | `bool CanSkipOwnSliceCopy() const` |
| ---------- | ---- |
| 函数功能 | 判断本rank输入数据是否已经处于最终Output布局，可否跳过本rank Scratch→Output复制 |
| 输入说明 | 使用Input/Output的BufferType、指针、base offset、slice stride和repeat stride |
| 输出说明 | 无 |
| 返回值说明 | `true`：本rank slice可跳过；`false`：仍需由`PostLocalCopy`输出 |

| 函数原型 | `HcclResult BuildStepSlices(..., StepBuildMode mode, ...)` |
| ---------- | ---- |
| 函数功能 | 按普通步骤或末步直出模式，为全部repeat和当前channel构造收发`DataSlice`列表 |
| 输入说明 | 收发Channel、NHR步骤信息、channel索引、构建模式及四个slice容器 |
| 输出说明 | 普通模式生成tx/rx Scratch描述；末步模式生成远端Scratch→Output的rx描述，并记录末步Read的rank编号 |
| 返回值说明 | 成功返回`HCCL_SUCCESS`；内部计算或校验失败返回相应错误码 |

| 函数原型 | `HcclResult RunStepNHR(..., u32 step, u32 nSteps, bool &postLocalCopyLaunched)` |
| ---------- | ---- |
| 函数功能 | 计算当前step收发关系、校验Channel并选择普通通信或末步直出路径 |
| 输入说明 | Thread、Channel映射、channel索引、当前step、总step数及PostCopy启动标志 |
| 输出说明 | 完成当前NHR step；特殊末步可能将`postLocalCopyLaunched`置为true |
| 返回值说明 | `HCCL_SUCCESS`、`HCCL_E_INTERNAL`或下层通信错误码 |

| 函数原型 | `HcclResult RunLastStepReadToOutput(..., bool &postLocalCopyLaunched)` |
| ---------- | ---- |
| 函数功能 | 启动当前channel的PostCopy Thread，并将最后一步全部slice从远端CCL Buffer批量Read到本端Output |
| 输入说明 | Thread列表、收发Channel、末步信息、channel索引、step和启动标志 |
| 输出说明 | 末步slice写入Output；PostCopy Thread开始复制本地Scratch中的其余slice |
| 返回值说明 | 成功返回`HCCL_SUCCESS`；Thread越界、同步或BatchRead失败返回对应错误码 |

| 函数原型 | `HcclResult PostLocalCopy(const ThreadHandle &thread, const u32 &channelIdx)` |
| ---------- | ---- |
| 函数功能 | 将当前channel负责的本地Scratch数据复制到Output，并过滤已直出或无需复制的slice |
| 输入说明 | 执行LocalCopy的Thread和channel索引 |
| 输出说明 | 当前channel、全部repeat中剩余slice写入Output |
| 返回值说明 | `HCCL_SUCCESS`或`GetAlgRank`、`LocalCopy`返回的错误码 |

对外`HcclAllGather`接口、AllGather executor接口及wrapper接口签名均不变。

## 2.6 约束分析

| 支持的算子名称 | AllGather |
| -------------- | ---- |
| 支持的算法名称 | AICPU NHR |
| 支持的芯片类型 | 继承`InsTempAllGatherNHR`原支持范围 |
| 支持的展开模式 | AICPU模板展开 |
| 支持的拓扑形态 | NHR模板已支持拓扑；末步特殊路径要求非PCIe Channel模式 |
| 支持的调用类型 | 不改变现有调用范围 |
| 支持的数据类型 | 不改变现有AllGather类型范围 |
| 支持的数据量 | 不改变现有数据量范围，支持repeat、tail和多通道切分 |
| 是否支持绕路 | 不改变 |
| 是否支持确定性计算 | 不改变 |

其他设计约束：

| 约束 | 说明 |
| ---- | ---- |
| 末步不变量 | 只有最后一步的数据不再参与转发，因此仅最后一步允许绕过本地Scratch |
| 小rank处理 | 特殊路径要求`nSlices > 1`；最后一步`nSlices = floor(rankSize / 2)`，因此rankSize为2或3时不启用，rankSize为1时不执行NHR通信 |
| 输出区间互斥 | `PostLocalCopy`必须跳过`lastStepReadSliceIdxs_`；只有`CanSkipOwnSliceCopy`为true时才可跳过本rank，防止非原地场景漏写 |
| Thread顺序 | PostCopy Thread必须等待对应通信Thread的前序NHR步骤完成后才能读取本地Scratch；Kernel结束前必须等待全部Thread完成 |
| Channel边界 | `fromRank`/`toRank`必须存在对应`channelIdx`，PostCopy Thread索引必须小于Thread数组长度，否则返回`HCCL_E_INTERNAL` |
| 尾块处理 | 当slice编号为`templateRankSize_ - 1`且`tailSize != 0`时，必须使用Tail切分大小和偏移 |
| 多通道处理 | 各channel仅操作`dataSplit_[channelIdx]`对应区间，多个channel合并后覆盖完整slice |
| 资源开销 | 每通道固定多申请一条Thread；即使运行时条件不满足末步优化，也会占用已申请的额外Thread/Notify资源 |
| Scratch占用 | 本需求不缩减Scratch倍数，仍保留完整rankSize倍Scratch，以支持前序NHR步骤和回退路径 |

## 2.7 DFX设计

| 校验内容 | 级别 | 搜索内容 |
| -------- | ---- | -------- |
| 优化条件及本rank复制策略 | DEBUG | `[InsTempAllGatherNHR] Read last step to output[%d], skip own slice copy[%d]` |
| 确认进入可优化的最后一步 | DEBUG | `[InsTempAllGatherNHR] rank[%u] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]`；匹配`step == nSteps - 1`且`nSlices > 1` |
| 最后一步Read执行失败 | ERROR | `[InsTempAllGatherNHR] last step read failed (step=%u)` |

测试用例可按第一条日志区分关键分支：原地优化场景匹配`Read last step to output[1], skip own slice copy[1]`；非原地优化场景匹配`Read last step to output[1], skip own slice copy[0]`；PCIe、RemoteMemAccess或Output即Scratch等回退场景匹配`Read last step to output[0], skip own slice copy[0]`。第一条日志表明运行条件允许优化，第二条日志进一步确认当前执行到满足`nSlices > 1`的最后一步。

## 2.8 资料描述

不涉及。该需求不修改对外API、环境变量、算法配置方式及用户调用方式，无需新增用户资料。

## 2.9 性能&&质量

### 性能收益

设子通信域rank数为`P`，最后一步slice数为`L = floor(P / 2)`。在Output不等于Scratch、每个repeat、每个channel的情况下：

| 场景 | 优化前PostLocalCopy次数 | 优化后PostLocalCopy次数 | 减少次数 |
| ---- | ---------------------- | ---------------------- | -------- |
| 非原地AllGather | `P` | `P - L = ceil(P / 2)` | `L` |
| 满足布局条件的原地AllGather | `P` | `P - L - 1` | `L + 1` |

最后一步的跨rank数据量不一定减少：原来通过Write传输的末步数据改为Read传输；性能收益主要来自末步数据不再落本地Scratch、减少Scratch→Output LocalCopy，以及剩余LocalCopy与末步Read并行。最终设计还将首slice从“Write到Scratch后LocalCopy”改为直接Read到Output，使末步特殊路径由两个通信阶段收敛为一个BatchRead阶段。

需要关注的资源代价是Thread数由`C`增加到`2C`，且每个Slave Thread申请两个Notify。性能验收应同时观察算子时延、本地搬运带宽、Thread/Notify资源占用和大并发通信域下的资源压力。

### 正确性保证

1. 普通步骤仍写入Scratch，不破坏NHR后续转发依赖。
2. 末步Read和PostLocalCopy操作不同的输出slice集合，避免并发写冲突。
3. 非原地场景不跳过本rank slice，防止输出缺失。
4. 不满足协议、Buffer或步数条件时回退原流程，保持功能兼容。
5. repeat、tail及多通道均使用统一的`CalcSliceInfo`计算，保证地址和长度一致。

### 测试设计

两个需求提交本身仅修改模板`.h/.cc`文件，未新增UT/ST。建议补充或执行以下验收：

| 测试类型 | 场景 | 预期结果 |
| -------- | ---- | -------- |
| 功能ST | 非PCIe、非RMA、原地AllGather，rankSize=4/8 | 开启末步直出并跳过本rank；结果与基准AllGather一致 |
| 功能ST | 非PCIe、非RMA、非原地AllGather，rankSize=4/8 | 开启末步直出但不跳过本rank；结果正确 |
| 回退ST | PCIe Channel | `readLastStepToOutput_ == false`，走普通DMA Read路径，结果正确 |
| 回退ST | `enableRemoteMemAccess=true` | 禁用末步直出，结果正确 |
| 回退ST | `outputPtr == hcclBuff.addr` | 不启用末步直出，`PostLocalCopy`跳过，结果正确 |
| 小rankST | rankSize=1/2/3 | 不进入`nSlices > 1`特殊路径，无越界、死锁或重复复制 |
| 泛化ST | 非2次幂rankSize=5/6/7 | NHR slice集合完整，末步Read与PostCopy集合无重叠、无遗漏 |
| 泛化ST | `repeatNum > 1` | 每个repeat输出正确，`lastStepReadSliceIdxs_`可复用于全部repeat |
| 泛化ST | `tailSize != 0` | 最后rank按Tail切分大小和偏移读写，无越界 |
| 泛化ST | 多通道及不均匀切分 | 各channel覆盖区间互斥且合并完整，输出正确 |
| 异常UT/ST | Thread数不足、Channel缺失、channelIdx越界 | 返回`HCCL_E_INTERNAL`并输出对应ERROR日志 |
| 性能测试 | 典型中大数据量，原地/非原地，4/8卡，多通道 | 相比优化前尾部LocalCopy耗时下降，总时延无劣化；记录额外Thread/Notify资源成本 |

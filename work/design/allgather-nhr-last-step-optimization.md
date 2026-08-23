# AG NHR末步全Read优化工作交接

> 更新时间：2026-07-28  
> 仓库：`/home/ytz/CANN/hccl`  
> 状态：该修改已经合入主仓；本文档不绑定具体开发分支  
> 历史提交参考：`a6d2be4c open allgather nhr template optimization`

## 1. 新会话首先执行

```bash
cd /home/ytz/CANN/hccl
git status --short --branch
git branch --show-current
rg -n "RunLastStepReadToOutput|LAST_STEP_READ_TO_OUTPUT" \
    src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.{cc,h}
```

本文档不要求当前工作分支具有固定名称，也不要求HEAD等于历史开发提交。新会话可能位于主分支、
维护分支或后续新建的功能分支，只需确认当前代码包含以下实现：

```text
StepBuildMode::LAST_STEP_READ_TO_OUTPUT
RunLastStepReadToOutput()
最后一步不再调用SendRecvBatchWrite()
最后一步全部rx slice从fromRank CCL Read到本端Output
```

如果当前分支没有这些实现，应先确认它是尚未同步主仓的旧分支。是否merge、rebase或cherry-pick应由
用户决定，不要在交接阶段擅自改变分支历史。

随后优先阅读：

1. 本文档。
2. `/home/ytz/CANN/ZZ/InsTempAllGatherNHR_last_step_optimization/`
   `InsTempAllGatherNHR_last_step_diagram_prompts.md`的第15节。
3. `/home/ytz/CANN/ZZ/InsTempAllGatherNHR_last_step_optimization/`
   `InsTempAllGatherNHR_regression_test_plan.md`。
4. 原始需求分析：
   `/home/ytz/CANN/ZZ/HCCL设计文档/InsTempAllGatherNHR_optimization_analysis.md`。

## 2. 需求和最终目标

修改对象是AICPU通用AllGather NHR模板：

- `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc`
- `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.h`

原优化路径在最后一个NHR step中：

```text
rx[0]:
fromRank Write到本端CCL
    → 本端PostLocalCopy到Output

rx[1..]:
本端从fromRank CCL Read
    → 直接写入Output
```

最终目标是把最后一步负责接收的全部切片统一改为：

```text
rx[0..nSlices-1]:
本端从fromRank CCL Read
    → 直接写入本端Output
```

这样可删除每个repeat中`rx[0]`对应的：

```text
1次Write + 1次PostLocalCopy
```

并替换为：

```text
1次Read
```

净减少一次数据搬运，同时删除原末步Write对应的一套ACK/DATA_SIGNAL同步。

## 3. 当前代码状态

### 3.1 相关提交

以下提交号用于追溯原开发过程，不是使用本文档的分支前提。主仓可能通过merge、cherry-pick或
squash形成不同提交号，应以当前源码中的实现为准。

```text
c7423d34 ag nhr 修复
a6d2be4c open allgather nhr template optimization
```

`c7423d34`完成了两个条件的拆分：

- `CanReadLastStepToOutput()`：判断末步是否可以直接Read到Output。
- `CanSkipOwnSliceCopy()`：判断本Rank自己的切片是否已经位于正确Output位置，从而可以跳过
  `PostLocalCopy`。

需要注意：`CanReadLastStepToOutput()`条件变宽发生在`c7423d34`，不是最终删除Write的
`a6d2be4c`引入的。

`a6d2be4c`完成了最终末步全Read：

1. `LAST_STEP_WRITE_THEN_READ`重命名为`LAST_STEP_READ_TO_OUTPUT`。
2. `BuildStepSlices()`不再对`i == 0`生成Write切片。
3. 最后一步对全部`rx[0..nSlices-1]`生成`remote CCL → local Output`的Read切片。
4. 全部末步`rxIdx`加入`lastStepReadSliceIdxs_`，使`PostLocalCopy()`跳过这些已直接Read到
   Output的切片。
5. `RunLastStepWriteThenRead()`重命名为`RunLastStepReadToOutput()`。
6. 删除其中的`SendRecvBatchWrite()`，保留`SendRecvBatchRead()`。
7. `PostLocalCopy()`仍通过专用本地线程与最后一步Read并行。

### 3.2 优化启用条件

当前必须同时满足：

```text
isDmaRead_ == false
enableRemoteMemAccess_ == false
outBuffType == BufferType::OUTPUT
outputPtr != hcclBuff.addr
step是最后一个NHR step
stepInfo.nSlices > 1
```

最后一步的`nSlices`约为`floor(templateRankSize / 2)`：

| NHR rankSize | 最后一步nSlices | 全Read分支 |
|---:|---:|---|
| 2 | 1 | 不命中 |
| 3 | 1 | 不命中 |
| 4 | 2 | 最小正向命中 |
| 5 | 2 | 命中 |
| 6 | 3 | 命中 |
| 7 | 3 | 命中 |
| 8 | 4 | 命中 |

### 3.3 `LocalDataCopy()`不能被忽略

`KernelRun()`在每个Channel运行NHR前都会调用`LocalDataCopy()`：

```cpp
CHK_RET(LocalDataCopy(templateResource.threads, channelIdx));
CHK_RET(RunAllGatherNHR(...));
```

它只在下面条件成立时跳过：

```cpp
inputPtr == hcclBuff.addr && inOff == scOff
```

该条件表示输入数据已经位于本地CCL的正确地址，不是“数据没有放入CCL”。因此无论实际执行Copy
还是命中`continue`，NHR开始前本Rank自己的切片都已经存在于本地CCL正确槽位。

## 4. 末步全Read的数据可达性

对算法Rank `r`，最后一步满足：

```text
deltaRank = 1
fromRank = (r - 1 + rankSize) % rankSize
rx[i] = (r - 1 - 2i + rankSize) % rankSize
```

令前驱Rank：

```text
p = fromRank = r - 1
```

则：

```text
rx[0] = p
```

因此：

- `rx[0]`是`fromRank`自己的数据，NHR开始时已由`fromRank`的`LocalDataCopy()`放入其本地CCL。
- `rx[1..]`已通过前面的普通NHR step汇聚到同一个`fromRank` CCL。
- 最后一步开始时，当前Rank需要的全部`rx`都可以从`fromRank.remoteCclMem`直接Read。

已按`GetStepInfo()`相同公式静态检查rankSize 4～256的所有Rank，没有发现末步`rx`不在
`fromRank` CCL中的情况。

### 4.1 4-Rank示例

初始LocalDataCopy后：

| CCL | 已有切片 |
|---|---|
| R0 | S0 |
| R1 | S1 |
| R2 | S2 |
| R3 | S3 |

Step 0，`deltaRank=2`：

```text
R0 --S0--> R2
R1 --S1--> R3
R2 --S2--> R0
R3 --S3--> R1
```

Step 0完成后：

| CCL | 已有切片 |
|---|---|
| R0 | S0、S2 |
| R1 | S1、S3 |
| R2 | S2、S0 |
| R3 | S3、S1 |

Rank 0最后一步：

```text
fromRank = R3
toRank = R1
rx = [S3, S1]
```

R3的CCL中已经有S3和S1，因此Rank 0可以直接读取二者。

## 5. 关于“Read到未初始化通信缓冲区”的检视意见

曾收到如下检视结论：

> 删除最后一步Write后，接收方会从此前未使用过的Rank间通信缓冲区读取未初始化或过期数据。

根据当前HCCL代码和CheckerL2内存模型，这个结论不成立，原因是它混淆了Channel和CCL内存的
所有权。

### 5.1 `remoteCclMem`不是Channel私有暂存区

`BuffInfo::hcclBuff`定义为跨Rank缓存Buffer。

Channel建立时，`ChannelInfo::remoteCclMem`通过：

```cpp
HcclChannelGetHcclBuffer(...)
```

获取对端导出的CCL buffer。

CheckerL2模型明确执行：

```cpp
locMem_ = npu.GetMemBlock(BufferType::CCL);
rmtMem_ = reverseChannel->GetLocMem();
```

所以：

```text
Rank 0的channelRecv.remoteCclMem
    = Rank 3本地hcclBuff的远端访问视图
```

某条Rank 0↔Rank 3 Channel此前没有传输过，不代表Rank 3的CCL没有被初始化。该CCL可以由：

- Rank 3自己的`LocalDataCopy()`；
- 其他Rank通过其他Channel的前序Write；

提前填充。Channel只是访问路径，不是数据存储所有者。

### 5.2 旧Write的数据方向也不支持该检视结论

旧版Rank 3最后一步的Write是：

```text
R3 local CCL[S3] → R0 remote CCL[S3]
```

Rank 0的Read源则是：

```text
R3 remote CCL
```

旧Write写入Rank 0的CCL，并没有填充Rank 3的CCL，因此它本来就不可能是Rank 0后续Read源的
初始化操作。

旧流程：

```text
R3 CCL[S3] --Write--> R0 CCL[S3] --PostLocalCopy--> R0 Output[S3]
R3 CCL[S1] ----------------Read---------------------> R0 Output[S1]
```

新流程：

```text
R3 CCL[S3] ----------------Read---------------------> R0 Output[S3]
R3 CCL[S1] ----------------Read---------------------> R0 Output[S1]
```

因此不建议因为这条检视意见恢复旧的严格`inputPtr == outputPtr`条件。更合适的处理是：

1. 在代码注释中明确`remoteCclMem`是`fromRank`本地CCL的远端视图。
2. 明确`rx[0]`来自`fromRank`的初始LocalDataCopy，`rx[1..]`来自前序NHR step。
3. 用真机正确性结果作为最终实证。

只有在HCOMM真机实现把`HcclChannelGetHcclBuffer()`解释为每条Channel独立、互不共享的暂存内存时，
上述检视意见才可能成立；但这会与当前HCCL接口用法和CheckerL2模型不一致。如真机出现数据错误，
优先检查远端内存注册映射、可见性和同步语义，而不是直接假定数据没有进入CCL。

## 6. 同步关系

普通步骤的`SendRecvBatchWrite()`包含：

```text
ACK Record/Wait
Batch Write
DATA_SIGNAL Record/Wait
```

因此前一个普通step返回时，其Write已经完成。

最后一步的`SendRecvBatchRead()`包含：

```text
ACK Record/Wait
Batch Read
DATA_SIGNAL Record/Wait
```

对端自己的LocalDataCopy和前序step均排在最后一步ACK之前。末步不再写本地CCL后：

- 通信线程从`fromRank` CCL Read到本端Output。
- PostLocalCopy线程读取本端前序step已经准备好的CCL切片。
- 两条路径均为读已有CCL数据，不会相互覆盖。
- 专用线程的`LocalPostTo/LocalWaitFrom`同步数量保持不变。

## 7. 模板影响范围

直接实例化`InsTempAllGatherNHR`的算子：

| 算子 | 可能命中末步全Read | 测试定位 |
|---|---|---|
| AllGather | 是 | P0核心验证 |
| AllReduce | Parallel最终AG阶段、OrderPreservedGroup可能命中 | P0核心验证 |
| Broadcast | 复用模板，但当前最终AG通常`outBuffType=INPUT` | P1保护性回归 |
| Reduce | 复用模板，但AG结果通常落CCL | P1保护性回归 |
| Scatter | 不直接实例化 | 非定向范围 |
| ReduceScatter | 不直接实例化 | 非定向范围 |

以下实现不直接执行本次修改的`RunLastStepReadToOutput()`：

- CCU、DPU各自的AllGather NHR模板。
- `InsTempAllGatherOmniPipeNHR`，虽然继承本模板，但重写了`KernelRun`和NHR step执行过程。

## 8. 已完成验证

### 8.1 构建

现有分析文档记录Host包已成功编译和打包：

```bash
bash build.sh --pkg -j8 -p /home/ytz/CANN/Ascend/cann-9.1.0
```

### 8.2 CheckerL2/ST日志

最终全Read代码对应：

```text
allgather_new_opt_4x4_32mb_int8.log
allgather_new_opt_4x5_32mb_int8.log
allgather_new_opt_4x6_32mb_int8.log
allgather_new_opt_4x7_32mb_int8.log
allgather_new_opt_4x8_32mb_int8.log
```

五个用例均为：

```text
32MiB
int8
InsAllGatherParallelMesh1DNHR
```

结果：

| Topo | 总Rank数 | 结果 | GTest时间 |
|---|---:|---|---:|
| 4×4 | 16 | PASS | 1039ms |
| 4×5 | 20 | PASS | 1649ms |
| 4×6 | 24 | PASS | 2028ms |
| 4×7 | 28 | PASS | 3321ms |
| 4×8 | 32 | PASS | 5045ms |

日志确认：

1. 前序普通step的Write数量和peer不变。
2. 末步发往`toRank`的Write消失。
3. 原末步全部`rx`改为从`fromRank` Read到`BufferType::OUTPUT`。
4. 原`rx[0]`对应的PostLocalCopy消失。
5. 五种Topo均无ERROR、FAILED和timeout。

### 8.3 任务数量变化

第一轮Write+Read优化与最终全Read对比：

| Topo | Write | Read | LocalCopy | 三类任务总计 |
|---|---:|---:|---:|---:|
| 4×4 | 480→240 | 960→1200 | 1056→816 | 2496→2256，-9.6% |
| 4×5 | 1200→800 | 1840→2240 | 2240→1840 | 5280→4880，-7.6% |
| 4×6 | 1440→960 | 2976→3456 | 2784→2304 | 7200→6720，-6.7% |
| 4×7 | 2800→2100 | 4760→5460 | 4900→4200 | 12460→11760，-5.6% |
| 4×8 | 3840→2880 | 8064→9024 | 6912→5952 | 18816→17856，-5.1% |

这些变化证明数据路径和任务结构已按预期重组，但CheckerL2的GTest时间不能证明真机性能收益。

## 9. 16P真机可验证的重点

用户当前有16P真机环境。优先覆盖：

| 优先级 | 算子/算法 | 建议Topo | 建议数据 | 预期 |
|---|---|---|---|---|
| P0 | AllGather `InsAllGatherParallelMesh1DNHR` | 4×4，16P | 32MiB int8 | 命中全Read |
| P0 | AllGather同算法保护边界 | 4×3，12P子通信域 | 32MiB int8 | `nSlices==1`，不命中 |
| P0 | AllReduce `InsAllReduceParallelRSAG` | 4×4，16P | 64MiB fp16/sum | 最终AG命中全Read |
| P1 | AllGather Sole `InsAllGatherNHR` | 每Server取1P，共4P | 101个fp16元素 | 最小Sole正向命中 |
| P1 | Broadcast `InsBroadcastParallelMesh1DNHR` | 4×4，16P | 32MiB int8，root 0/非0 | 不命中，保护回归 |
| P1 | Reduce `ReduceParallelMesh1DNHR` | 4×4，16P | 64MiB fp16/sum，root 0/非0 | 不命中，保护回归 |

建议至少额外覆盖：

- `32MiB+3B`之类非对齐尾片；
- int8、fp16、fp32；
- 单loop和多loop；
- out-of-place；
- 如果接口和算法支持，再覆盖in-place；
- OFFLOAD/RemoteMemAccess保护场景。

16P不能完整覆盖：

- 4×5～4×8真机Topo；
- `rankSize > 32`的AllReduce OrderPreservedGroup；
- 33/36 Rank的大通信域。

每个用例必须从日志确认实际`algName`，不能只根据Topo和数据量推断算法已经命中。

## 10. 完整回归测试设计

完整测试矩阵位于：

```text
/home/ytz/CANN/ZZ/InsTempAllGatherNHR_last_step_optimization/
    InsTempAllGatherNHR_regression_test_plan.md
```

其中包含：

- T1：两层`MESH_1D + NHR`，`Build4xNTopo(3～8)`。
- T2：每Server 1 Device的Pure NHR。
- T5：`2×4×4`三层普通Topo。
- T7：33/36 Rank逻辑一维通信域。
- AllGather、AllReduce P0矩阵。
- Broadcast、Reduce保护性回归。
- 数据量、dtype、Buffer和执行模式覆盖原则。
- 日志、任务数量和正确性验收标准。

本轮只完成了测试场景设计，没有把该矩阵固化为新的仓内自动化测试代码。

用户此前在“只分析影响范围/设计测试场景”的请求中明确要求不要使用
`hccl-auto-st-runner`。新会话不要自行执行测试；如用户后续明确要求运行，再按新的授权处理。

## 11. 辅助材料位置

所有本需求产生的辅助文件已从HCCL工作区迁移到：

```text
/home/ytz/CANN/ZZ/InsTempAllGatherNHR_last_step_optimization/
```

主要文件：

| 文件 | 用途 |
|---|---|
| `InsTempAllGatherNHR_last_step_diagram_prompts.md` | 完整方案演进、数据流、最终实现和日志分析；最终结论看第15节 |
| `InsTempAllGatherNHR_regression_test_plan.md` | 完整回归测试设计 |
| `InsTempAllGatherNHR_figures/01～12.svg` | 优化、Parallel全流程、4×4～4×8阶段2 NHR说明图 |
| `InsTempAllGatherNHR_figures/generate_phase2_nhr_rank0_flows.py` | 4×5～4×8 SVG生成脚本 |
| `allgather_opt_disabled_4x4_32mb_int8.log` | 原始未启用优化日志 |
| `allgather_opt_enabled_4x4～4x8_32mb_int8.log` | 第一轮末步Write+Read优化日志 |
| `allgather_new_opt_4x4～4x8_32mb_int8.log` | 最终末步全Read日志 |

注意：

- 说明文档第1～14节包含第一轮“末步Write+Read”方案及旧图，属于方案演进。
- 当前最终实现必须以第15节“末步全部Read”为准。
- 图8～图12主要描述第一轮方案，阅读时不要误认为仍代表当前最终代码。

## 12. 尚未完成和下一步建议

### 12.1 必做

1. 在16P真机执行AllGather 4×4主路径正确性验证。
2. 在16P真机执行AllReduce Parallel最终AG阶段验证。
3. 记录优化前后真实带宽、耗时和稳定性，采用多轮测试并去除首轮热身与日志开销。
4. 检查真机日志中的最终算法名、末步peer、Read目标Buffer和错误码。

### 12.2 建议

1. 根据检视反馈完善当前代码注释，明确`remoteCclMem`的内存归属和`rx[0]`来源。
2. 增加至少一个`inputPtr != outputPtr`、`nSlices > 1`且预填充CCL哨兵值的定向回归。
3. 补齐4×3不命中边界、非对齐tail、多loop、Broadcast和Reduce保护用例。
4. 单独核查AllReduce Parallel在OFFLOAD模式下是否正确设置`enableRemoteMemAccess`；现有参数构造中
   没有明显看到显式赋值，这是独立风险点。
5. 若检视仍坚持“Channel私有缓冲区”模型，应要求提供HCOMM接口契约或真机失败日志，不要仅根据
   Channel是否曾使用过来判断远端CCL是否初始化。

### 12.3 当前没有做的事情

- 没有修改HCOMM代码。
- 没有引入HCOMM私有头或编译期依赖。
- 没有提交新的UT/ST测试代码。
- 没有执行真机性能验证。
- 没有创建新的commit、PR或push操作。

## 13. 结论摘要

当前代码已经实现末步全Read，并在4×4～4×8 CheckerL2/ST中全部通过。静态数据可达性分析和
4-Rank实例均表明，`fromRank`在最后一步前已经拥有全部`rx`切片；旧末步Write写向当前Rank的
本地CCL，不是用于初始化Read源，因此“删除Write必然读取未初始化Channel缓冲区”的检视结论
不符合当前内存模型。

后续工作的核心不是继续修改算法路径，而是：

```text
16P真机正确性验证
    → 真机多轮性能对比
    → 补充关键回归
    → 用代码注释和测试证据关闭检视争议
```

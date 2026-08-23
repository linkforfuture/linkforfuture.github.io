# AllGather NHR rankSize=1 空 channels 崩溃分析报告

> 仓库：`cann/hccl_ratio`（分支 `ratio`，HEAD `8e6b8d11`）  
> 日期：2026-07-09  
> 涉及文件：`src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc`、`src/ops/op_common/inc/alg_param.h`  
> 涉及 commit：`8e6b8d11`（触发）、`85b5e483`（潜伏缺陷引入）

---

## 1. 问题概述

执行 `bash build.sh --st` 全量 ST 测试，20 个用例中 18 个通过、**2 个段错误失败**：

| # | 用例 | 结果 | 崩溃子用例 |
|---|------|------|-----------|
| 6 | `st_all_gather_aicpu_test` | **SEGFAULT**（19.1s） | `st_all_gather_a5_aicpu_meshnhr_2x1x8rank_int8_test` |
| 9 | `st_all_gather_3level_test` | **SEGFAULT**（26.0s） | `st_allgather_3level_2x1x4_int8_different_scale` |

两个失败用例的共同特征：**AllGather 算子 + int8 dtype + 拓扑含"1"维（2x1x8、2x1x4）**。

---

## 2. 崩溃定位

### 2.1 崩溃栈（gdb）

两个用例在同一指令地址崩溃：

```
Thread N received signal SIGSEGV
#0  ops_hccl::InsTempAllGatherNHR::LocalDataCopy(...)           ← 崩溃点
#1  ops_hccl::InsTempAllGatherNHR::KernelRun(...)
#2  InsV2AllGatherSequenceExecutor3Level<..., InsTempAllGatherNHR, InsTempAllGatherNHR>::OrchestrateLoop
#3  InsV2AllGatherSequenceExecutor3Level::Orchestrate
#4  HcclLaunchAicpuKernel (kernel_launch.cc:427)
...
#12 HcclAllGather
```

### 2.2 崩溃根因（运行时 gdb 证据）

崩溃指令：`mov (%rdx,%rax,8),%r13`，其中 `rdx=0x0`（NULL）。

`rdx` 来自 `mov 0x488(%rbx),%rdx`（NHR 对象偏移 0x488 = `dataSplit_` 向量的 data 指针）。

对应源码 `ins_temp_all_gather_nhr.cc:381`：

```cpp
u64 partialSliceSize = dataSplit_[channelIdx];   // dataSplit_ 为空 → NULL 解引用
```

运行时读取的关键成员值（`this = rbx`）：

| 成员 | 偏移 | 值 | 含义 |
|------|------|-----|------|
| `templateRankSize_` | 0x18 | **1** | rankSize=1 的 level |
| `dataSplit_` data ptr | 0x488 | **nil** | 空向量 |
| `dataSplit_` size | 0x490 | **0** | 0 个元素 |
| `dataOffset_` | 0x4a0 | nil | 空向量 |
| `dataSplitTail_` | 0x4b8 | nil | 空向量 |
| `tailSize` | 0x188 | 1 | 尾块模式开启 |
| `repeatNum` | 0x170 | 2 | Level1（repeatNum=levels_[2].rankSize） |

### 2.3 根因链

1. 拓扑 `2x1x8` → 3-level sequence 执行器 `levels_=[8,1,2]`，中间 **Level1 NHR 的 `templateRankSize_=1`**（"1"维）。
2. rankSize=1 → 该 level 子通信域只有 1 个 rank → 无通信对端 → **channels map 为空**。
3. 3-level 执行器对每个 level 设 `tailSize = sliceSize`（恒非 0），故 `KernelRun` 第 95 行早返回条件 `sliceSize==0 && tailSize==0` **永不触发**。
4. `PrepareDataSplitForMultiChannel`（第 82 行）用 `templateResource.channels.begin()->second` 调 `CalcDataSplitByPortGroup`——**对空 map 做 `begin()->second` 是未定义行为（UB）**。
5. 该 UB 在特定 `sizeof(ChannelInfo)` 下使 `dataSplit_` 为空 → `LocalDataCopy` 第 381 行 `dataSplit_[channelIdx]` 解引用空向量的 data 指针（nil）→ **SIGSEGV**。

### 2.4 对照：reduce_scatter NHR 同拓扑不崩

`st_reduce_scatter_a5_aicpu_meshnhr_2x1x8rank_int8_max_test`（同 2x1x8 int8）**通过**。原因：`ins_temp_reduce_scatter_nhr.cc:140` 有 `if (templateRankSize_ <= 1)` 早返回路径 + line 136-137 越界检查。**all_gather NHR 缺这两处保护**。

---

## 3. 触发点定位：8e6b8d11 的 dieId 字段

### 3.1 问题：崩溃仅在 8e6b8d11 之后出现

用户反馈：上一个 commit（`9c2aa604`）无此问题，带上 `8e6b8d11` 后才崩。需要确认是否 8e6b8d11 暴露了该缺陷。

### 3.2 静态分析

8e6b8d11（"parallel算法切分方案：comm配置+环境变量+公式计算"）改动文件：

| 文件 | 改动 | 是否触及崩溃路径 |
|------|------|-----------------|
| `ins_v2_all_gather_parallel_executor.cc/h` | parallel 执行器切分比例 0.8→0.5 + BUILTIN_FORMULA | 否（2x1x8 走 sequence-3level，非 parallel） |
| `template_utils.cc/h` | 纯新增 parallel 切分公式 | 否（仅 parallel 调用） |
| `op_common.cc` | `SetMultipleDimensionSplitRatio` + `HcclGetChannelImpl` 加 dieId 获取 | HcclGetChannelImpl 运行于所有执行器 |
| `alg_param.h` | ChannelInfo 加 `dieId` 字段 + ratioSource 枚举 + ratio 默认 0.8→0.5 | ChannelInfo 布局变化影响全局 |
| reduce/broadcast/reduce_scatter parallel 执行器 | parallel 切分 | 否 |

崩溃路径（sequence-3level + NHR 模板）**未被 8e6b8d11 直接修改**。NHR 模板和 3-level 执行器的最后修改是更早的 `85b5e483 hccl_aicpu_three_sequence`（8e6b8d11 的祖先）。

### 3.3 实证验证（4 次构建 + gdb）

| 变体 | dieId 字段 | dieId 获取 | ratio/公式 | 2x1x8 结果 |
|------|-----------|-----------|-----------|-----------|
| `9c2aa604`（父提交） | ✗ | ✗ | ✗ | **PASS** |
| `8e6b8d11`（全量） | ✓ | ✓ | ✓ | **SIGSEGV** |
| `8e6b8d11` − dieId 获取 | ✓ | ✗ | ✓ | **SIGSEGV** |
| `9c2aa604` + 仅 dieId 字段 | ✓ | ✗ | ✗ | **SIGSEGV** |

**结论：触发点是 `alg_param.h` 中给 `ChannelInfo` 新增 `u32 dieId` 字段这一项改动**，与 ratio/公式/dieId 获取逻辑无关。仅加一个 dieId 字段即可在 9c2aa604 上复现完全相同的崩溃（同一 LocalDataCopy、同一 dataSplit_ 空、templateRankSize_=1）。

### 3.4 dieId 字段的 sizeof 影响

`ChannelHandle = uint64_t`（8 字节对齐）。dieId(u32) 插在 `portGroupSize`(u32) 与 `handle`(uint64_t) 之间：

```
无 dieId: ... portGroupSize(off20,4B) | handle(off24,8B)          ← 24 已 8 对齐
有 dieId: ... portGroupSize(off20,4B) | dieId(off24,4B) | pad(28-31) | handle(off32,8B)
```

实测 sizeof（用 compile_commands.json 的编译参数独立编译探测）：

| 构建 | `sizeof(ChannelInfo)` | `handle` 偏移 |
|------|----------------------|--------------|
| 9c2aa604（无 dieId） | **152** | 24 |
| 8e6b8d11（有 dieId） | **160**（+8） | 32 |

dieId 使 `sizeof(ChannelInfo)` 增加 8 字节（4 字节字段 + 4 字节对齐填充）。

### 3.5 dieId 如何暴露 UB

**确证部分**：
- rankSize=1 level 的 channels map **本就为空**（0 对端 → 0 channel 请求 → 0 channel 创建）。9c2aa604 实测：map root=NULL, node_count=0；HcclGetChannelImpl level=1 channelRequest bytes=0。
- `PrepareDataSplitForMultiChannel` 第 82 行对空 map 做 `channels.begin()->second`（UB）。
- 9c2aa604（sizeof=152）：dataSplit_ n=1（UB 碰巧产生 1 个元素），测试通过。
- 8e6b8d11（sizeof=160）：dataSplit_ n=0（空），崩溃。
- `CalcDataSplitByPortGroupCommon` 对"最后一个 channel"分配全部数据：`dataSplit_[0] = totalDataCount × dataTypeSize = 1`（int8 sendCount=1，恰好正确），这是 9c2aa604 UB 碰巧良性的原因。

**推断部分（未完全验证）**：
- 推断模型：UB 的 `channels.size() = (end−data)/sizeof(ChannelInfo)`，dieId 的 sizeof 152→160 使该除法结果从 ≥1 翻为 0。
- 直接测量遇到困难：按 libstdc++ 布局推断 UB 向量在 `TemplateResource.threads` 位置（off48），但实测 threads 字节跨度仅 16（threads.count=2），16/152=0，与 n=1 矛盾；扫描 TemplateResource off0..408 也未找到字节差在 [150,1000] 的 (data,end) 对。
- 因此 `channels.size() = (end−data)/sizeof` 这一具体计算链**未被直接测量证实**，C∈[152,160) 的推断**不成立**。
- 精确定位 UB 的真实计算链需要 `-g` 调试构建单步 inspect `PrepareDataSplitForMultiChannel` 中 `channels`（UB 引用）的真实 data/end/size 值。

**核心结论（无论 UB 细节）**：
- dieId 字段**没有"把 channels 变空"**——rankSize=1 的 channels 本来就是空的。
- 真正的 bug 是 `ins_temp_all_gather_nhr.cc:82` 对空 map 做 `begin()->second`（UB），自 `85b5e483` 起潜伏。
- dieId 的 `sizeof+8`（152→160）**改变了 UB 的表现**：从"碰巧良性"（n=1，通过）变为"空 dataSplit_"（n=0，崩溃），因此表现为"带上 8e6b8d11 才崩"。

---

## 4. 修复方案

### 4.1 修复定位

根因在 NHR 模板（`ins_temp_all_gather_nhr.cc`），不在 dieId 字段（dieId 是合法改动，用于 POD 链路识别）。修复消除 UB 根因，dieId 字段无需回退。

### 4.2 修复内容（2 处改动）

**改动 1：`PrepareDataSplitForMultiChannel`（第 78 行）——处理空 channels**

```cpp
HcclResult InsTempAllGatherNHR::PrepareDataSplitForMultiChannel(const TemplateResource &templateResource) {
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalDataCount = tempAlgParams_.sliceSize / dataTypeSize;
    if (templateResource.channels.empty() || templateResource.channels.begin()->second.empty()) {
        dataSplit_.assign(1, tempAlgParams_.sliceSize);
        dataOffset_.assign(1, 0);
        if (tempAlgParams_.tailSize > 0) {
            dataSplitTail_.assign(1, tempAlgParams_.tailSize);
            dataOffsetTail_.assign(1, 0);
        }
        return HCCL_SUCCESS;
    }
    std::vector<u64> elemCountOut;
    CHK_RET(CalcDataSplitByPortGroup(...));  // 原逻辑
    ...
}
```

rankSize=1 / 空 channels 时，按单通道整块填充切分向量，使 `LocalDataCopy` 的 ccl→ccl 重排能正常执行（level1 需做数据重排供 level0 读取，不能整段跳过）。

**改动 2：`KernelRun`（第 135 行）——跳过 rankSize≤1 的远程交换**

```cpp
for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
    bool postLocalCopyLaunched = false;
    CHK_RET(LocalDataCopy(templateResource.threads, channelIdx));
    if (templateRankSize_ > 1) {                    // ← 新增保护
        CHK_RET(RunAllGatherNHR(templateResource.threads, templateResource.channels,
            channelIdx, postLocalCopyLaunched));
    }
    if (!postLocalCopyLaunched) {
        CHK_RET(PostLocalCopy(templateResource.threads[channelIdx], channelIdx));
    }
}
```

rankSize=1 时无远程对端，`RunAllGatherNHR` 无意义且 channels 空会再次崩溃，故跳过。遵循 reduce_scatter NHR 的 `if (templateRankSize_ <= 1)` 既有模式。

### 4.3 修复 patch

保存在 `/tmp/opencode/fix_all_gather_nhr_rankSize1.patch`（33 行）。

---

## 5. 验证

### 5.1 构建 + 安装

```
bash build.sh --pkg --full                              # 构建成功
./build_out/cann-hccl_9.1.0_linux-x86_64.run --full -q --pylocal \
    --install-path=/home/ytz/CANN/Ascend               # 安装成功
```

### 5.2 原失败用例验证

| 用例 | 修复前 | 修复后 |
|------|--------|--------|
| `st_all_gather_aicpu_test`（17 子用例） | 第 17 个 SEGFAULT | **17/17 PASSED** |
| `st_all_gather_3level_test`（23 子用例） | 第 4 个 SEGFAULT | **23/23 PASSED** |

### 5.3 全量 ST 回归

```
bash build.sh --st
→ 100% tests passed, 0 tests failed out of 20
→ Total Test time = 326.10 sec
```

**20/20 全部通过，无回归。**

---

## 6. 关键经验

1. **潜伏 UB 的暴露**：一个看似无关的结构体字段添加（dieId），通过改变 `sizeof` 改变了未定义行为的表现，把"碰巧能跑"变成崩溃。这类问题极难通过静态审查发现——UB 本身就是"不可预测"的。

2. **防御性编程的必要性**：`channels.begin()->second` 不检查 map 是否为空，是典型的 UB。reduce_scatter NHR 有 `templateRankSize_ <= 1` 保护而 all_gather NHR 没有，说明同族算法模板间缺乏一致性校验。

3. **实证优于推断**：本次分析中，静态推断"`channels.size()=(end-data)/sizeof` 翻转"看似合理，但直接测量（gdb 读内存）证伪了该模型。UB 的实际行为往往与推断不符，必须以运行时数据为准。

4. **修复应针对根因而非触发点**：dieId 字段是合法改动，不应回退。真正的 bug 是 NHR 模板对 rankSize=1 / 空 channels 的处理缺失，修复应在此处。

---

## 附录 A：涉及的关键文件与行号

| 文件 | 行 | 内容 |
|------|-----|------|
| `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc` | 82 | `channels.begin()->second`（UB 根因） |
| 同上 | 381 | `dataSplit_[channelIdx]`（崩溃点） |
| 同上 | 78 | `PrepareDataSplitForMultiChannel`（修复点 1） |
| 同上 | 135 | `KernelRun` 循环（修复点 2） |
| `src/ops/op_common/inc/alg_param.h` | 375 | `u32 dieId`（触发点，8e6b8d11 新增） |
| `src/ops/op_common/executor/executor_v2_base.cc` | 30 | `RestoreChannelMap`（channels 恢复） |
| `src/ops/op_common/op_common.cc` | 1617 | `HcclGetChannelImpl`（channel 创建） |
| `src/ops/op_common/template/template_utils.cc` | 36 | `CalcDataSplitByPortGroupCommon`（切分计算） |
| `src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_3level.cc` | 234/268 | `tailSize = sliceSize`（阻止早返回） |

## 附录 B：实证构建矩阵

| # | 基线 | dieId 字段 | dieId 获取 | ratio/公式 | 2x1x8 | 用途 |
|---|------|-----------|-----------|-----------|-------|------|
| 1 | 9c2aa604 | ✗ | ✗ | ✗ | PASS | 基准（父提交无问题） |
| 2 | 8e6b8d11 | ✓ | ✓ | ✓ | SIGSEGV | 全量（问题复现） |
| 3 | 8e6b8d11−dieId获取 | ✓ | ✗ | ✓ | SIGSEGV | 排除 dieId 获取逻辑 |
| 4 | 9c2aa604+dieId字段 | ✓ | ✗ | ✗ | SIGSEGV | **隔离确认 dieId 字段为触发点** |
| 5 | 8e6b8d11+修复 | ✓ | ✓ | ✓ | PASS | 修复验证 |
| 6 | 8e6b8d11+修复（全量ST） | ✓ | ✓ | ✓ | 20/20 PASS | 全量回归 |

## 附录 C：gdb 关键命令参考

```bash
# 环境准备
source ../../Ascend/cann/set_env.sh
BUILD_ST_DIR=/home/ytz/CANN/win/hccl_ratio/test/st/algorithm/build
export LD_LIBRARY_PATH="${BUILD_ST_DIR}/utils/src:${BUILD_ST_DIR}/utils/src/hccl_verifier:${BUILD_ST_DIR}/utils/src/hccl_depends_stub:${BUILD_ST_DIR}/utils/src/aicpu:${LD_LIBRARY_PATH}"

# 跑单个崩溃子用例
cd ${BUILD_ST_DIR}
./testcase/st_all_gather_aicpu_test \
    --gtest_filter=ST_ALL_GATHER_AICPU_TEST.st_all_gather_a5_aicpu_meshnhr_2x1x8rank_int8_test

# gdb 抓崩溃栈
gdb -batch -ex "set pagination off" -ex "run --gtest_filter=...2x1x8... > /dev/null 2>&1" \
    -ex "bt 20" --args ./testcase/st_all_gather_aicpu_test

# gdb 读 NHR 对象成员（偏移从反汇编获取）
gdb -batch ... -ex "printf \"tRS=%u dataSplit_ptr=%p size=%lu\n\", *(uint*)($rbx+0x18), *(void**)($rbx+0x488), *(ulong*)($rbx+0x490)" ...

# 探测 sizeof(ChannelInfo)
g++ -std=c++14 @compile_flags /tmp/sizeof_test.cc -o /tmp/sizeof_test
# sizeof_test.cc: #include "alg_param.h" ... printf("%zu", sizeof(ops_hccl::ChannelInfo));
```

# CheckerL2 失败用例分析报告

> 本文档供 CheckerL2 团队分析 Checker 失败原因使用。测试背景为验证 HCCL `hccl_ratio` 仓提交 `27eaf29a`（write-reduce-with-notify）在 AICPU 模式下的功能正确性。测试中发现 Checker 存在失败用例，经分析初步判断为 hccl-vm device 侧 SQE 解析不完整所致，非被测提交的问题。

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| HCCL 代码 | `/home/ytz/CANN/win/hccl_ratio`, commit `27eaf29a` |
| HCOMM 代码 | `/home/ytz/CANN/hcomm` (含 `HcommWriteReduceWithNotifyOnThread`) |
| hccl-vm | `/home/ytz/CANN/CheckerL2/hccl_vm_install` |
| 执行模式 | `HCCL_OP_EXPANSION_MODE=AI_CPU`（QEMU aarch64 仿真） |
| 跨机集群 | `ascend950_cluster_4_server_normal`, comm `122`（2 机 × 2 卡 = 4 ranks） |
| 单机集群 | `ascend950_cluster_1_server_hf`, comm `118`（1 机 × 8 卡 = 8 ranks） |

## 2. 测试中对 hccl-vm 代码的修改

测试过程中对 `src/device_arm/proxy/device_sqe_parse_stub.cc` 做了以下修改（否则 AICPU 跨机模式直接死锁）：

### 2.1 原始问题

`ParseDavidUDMASqe` 函数的 switch 仅处理 `UDMA_OPC_WRITE`(0x3)、`UDMA_OPC_WRITE_WITH_NOTIFY`(0x5)、`UDMA_OPC_READ`(0x6) 三种 opcode。遇到 `UDMA_OPC_SEND`(0x0) 等未覆盖 opcode 时，default 分支执行 `return`，**直接中断当前 jetty 队列中后续所有 WQE 的解析**，导致任务缺失和 Runner 死锁。

### 2.2 修改内容

```cpp
// device_sqe_parse_stub.h — 增加 isSend 参数
void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId,
                              bool isRead, bool isSend = false);

// device_sqe_parse_stub.cc — ParseDavidUDMASqe switch
case UDMA_OPC_SEND:              // 0x0  新增
case UDMA_OPC_SEND_WITH_IMM:     // 0x1  新增
case UDMA_OPC_SEND_WITH_INVALID: // 0x2  新增
    ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false, true);
    break;
case UDMA_OPC_WRITE:             // 0x3  原有
    ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false);
    break;
case UDMA_OPC_WRITE_WITH_IMM:    // 0x4  新增，归入 WITH_NOTIFY
case UDMA_OPC_WRITE_WITH_NOTIFY: // 0x5  原有
    ParseDavidUBWriteWithNotifySqe(wqeAddr, streamId, jettyId);
    index++;
    break;
case UDMA_OPC_READ:              // 0x6  原有
    ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, true);
    break;
default:
    HCCL_VM_WARN("skip unsupported opcode[{}], continue parsing.", ...);
    break;   // 原为 return，改为 break

// ParseDavidUBReadWriteSqe — SEND 的 rank 解析改用 EID 回退
uint32_t srcRankId = isSend ? curRankId : GetRankIdByDevAddr(srcOffset);
uint32_t dstRankId = isSend ? GetRmtRankIdByEid(ubWqe->comm.rmtEid[0])
                            : GetRankIdByDevAddr(dstOffset);
```

### 2.3 修改原因

- `UDMA_OPC_SEND` WQE 的 `rmtAddr` 字段不是设备内存地址，`GetRankIdByDevAddr` 无法解析，返回 0（误判为 rank 0）
- SEND 的本地地址（`locAddr`）也可能在 UB 传输缓冲区范围内，同样无法解析
- 因此对 SEND 使用 `curRankId`（发送方）和 `GetRmtRankIdByEid`（接收方）替代地址解析

### 2.4 遗留问题

此修改是**权宜之计**。SEND WQE 的数据传输部分可能未完整解析（`dstOffset` 仍使用 `rmtAddr`，该地址不在设备内存映射中），可能导致 Checker 的语义检查（如 buffer 范围追踪）不准确。完整的修复需要了解 UB 协议 SEND WQE 的确切格式。

## 3. 失败用例清单

### 3.1 总览

全部 10 个用例 mpirun 均无运行时错误（0 error），失败均来自 Checker。按 Checker 结果分两类：

| 类别 | 用例编号 | 数量 | 含义 |
|------|---------|------|------|
| **完全失败** | #01, #04, #05, #10 | 4 | syncIter=0 和 syncIter=1 均 Checker failed |
| **Partial Success** | #02, #03, #06, #07, #08, #09 | 6 | syncIter=0 失败，syncIter=1 Checker Success |

> 测试参数 `-n 1 -c 1` 产生 2 个 syncIter。syncIter=0 对应首轮（含初始化），syncIter=1 对应实际数据传输轮次。

### 3.2 完全失败用例（4 个）

#### Case #01 — 跨机 AllReduce 8M fp32 sum

| 参数 | 值 |
|------|-----|
| 算子 | `all_reduce_test` |
| 拓扑 | 跨机 2 机 × 2 卡 = 4 ranks (comm 122) |
| 数据量 | 8M |
| dtype | fp32 |
| reduce op | sum |
| mpirun errors | 0 |
| 内存冲突 | 0 |
| Checker 结果 | failed (两个 syncIter 均失败) |

**Checker ERROR：**

```
# syncIter=0 — GenGraph 阶段
[ErrorCode: 102] Local Record/Wait matching is stuck on this rank.
  Some Wait tasks are still blocked, but no new local Record task can unblock them,
  rankId=0, firstBlockedWaitNode=[TaskWaitAICPU] node=3, rank=0, stream=0, protocol=INVALID,
  notify={recordRank=0, waitRank=0, notifyId=9}, blockedWaitNodeCount=2

# syncIter=0 — SemanticCheck 阶段
[ErrorCode: 409] AllReduce result range [0x0,0x40000) for rank 0
  should combine inputs from 4 source ranks, but it actually combines 3.
```

**Device 侧任务统计（2 个 syncIter 合计）：**

| 任务类型 | 数量 |
|---------|------|
| NOTIFY_WAIT | 192 |
| NOTIFY_RECORD | 168 |
| MEM_CPY | 120 |
| taskType[9] (CCU/AIV_GRAPH) | 80 |
| REDUCE | 16 |

**NOTIFY_WAIT - NOTIFY_RECORD = 24**（缺少 24 个 NOTIFY_RECORD）

---

#### Case #04 — 跨机 ReduceScatter 8M fp32 sum

| 参数 | 值 |
|------|-----|
| 算子 | `reduce_scatter_test` |
| 拓扑 | 跨机 2 机 × 2 卡 = 4 ranks (comm 122) |
| 数据量 | 8M |
| dtype | fp32 |
| reduce op | sum |
| mpirun errors | 0 |
| 内存冲突 | 0 |
| Checker 结果 | failed (两个 syncIter 均失败) |

**Checker ERROR：** 同 Case #01 模式（ErrorCode 102 + 409）

**Device 侧任务统计：**

| 任务类型 | 数量 |
|---------|------|
| NOTIFY_WAIT | 48 |
| NOTIFY_RECORD | 40 |
| MEM_CPY | 32 |
| taskType[9] | 32 |
| REDUCE | 8 |

**NOTIFY_WAIT - NOTIFY_RECORD = 8**

---

#### Case #05 — 跨机 ReduceScatter 512M fp32 sum

| 参数 | 值 |
|------|-----|
| 算子 | `reduce_scatter_test` |
| 拓扑 | 跨机 2 机 × 2 卡 = 4 ranks (comm 122) |
| 数据量 | 512M |
| dtype | fp32 |
| reduce op | sum |
| mpirun errors | 0 |
| 内存冲突 | 0 |
| Checker 结果 | failed (两个 syncIter 均失败) |

**Checker ERROR：** 同 Case #01 模式（ErrorCode 102 + 409）

**Device 侧任务统计：**

| 任务类型 | 数量 |
|---------|------|
| NOTIFY_WAIT | 496 |
| NOTIFY_RECORD | 496 |
| MEM_CPY | 216 |
| REDUCE | 120 |
| taskType[9] | 36 |

**NOTIFY_WAIT - NOTIFY_RECORD = 0**（数量相等但仍匹配失败，说明是匹配关系错误而非数量不足）

---

#### Case #10 — 跨机 AllReduce 1M fp32 sum（边界用例）

| 参数 | 值 |
|------|-----|
| 算子 | `all_reduce_test` |
| 拓扑 | 跨机 2 机 × 2 卡 = 4 ranks (comm 122) |
| 数据量 | 1M |
| dtype | fp32 |
| reduce op | sum |
| mpirun errors | 0 |
| 内存冲突 | 0 |
| Checker 结果 | failed (两个 syncIter 均失败) |

**Checker ERROR：**

```
[ErrorCode: 102] Local Record/Wait matching is stuck on this rank.
  rankId=0, notify={recordRank=0, waitRank=0, notifyId=9}, blockedWaitNodeCount=2

[ErrorCode: 409] AllReduce result range [0x0,0x40000) for rank 0
  should combine inputs from 4 source ranks, but it actually combines 3.
```

**Device 侧任务统计：**

| 任务类型 | 数量 |
|---------|------|
| NOTIFY_WAIT | 80 |
| NOTIFY_RECORD | 72 |
| MEM_CPY | 48 |
| taskType[9] | 32 |
| REDUCE | 8 |

**NOTIFY_WAIT - NOTIFY_RECORD = 8**

---

### 3.3 Partial Success 用例（6 个，syncIter=1 通过）

以下用例的 syncIter=0 失败（错误与完全失败用例相同），但 syncIter=1 **Checker Success**。

| # | 用例 | 拓扑 | NOTIFY_WAIT | NOTIFY_RECORD | diff |
|---|------|------|:-----------:|:-------------:|:----:|
| 02 | cross-AllReduce-512M-fp32 | 跨机 2×2 | 984 | 980 | 4 |
| 03 | cross-AllReduce-512M-fp16 | 跨机 2×2 | 984 | 980 | 4 |
| 06 | cross-Reduce-8M-sum | 跨机 2×2 | 336 | 332 | 4 |
| 07 | cross-Reduce-512M-max | 跨机 2×2 | 656 | 652 | 4 |
| 08 | single-AllReduce-512M | 单机 1×8 | 10784 | 10728 | 56 |
| 09 | single-ReduceScatter-512M | 单机 1×8 | 5856 | 5800 | 56 |

**Partial Success 用例 syncIter=0 的错误示例（单机 #08）：**

```
[ErrorCode: 102] Cross-rank Record/Wait matching is stuck.
  firstBlockedWaitNode=[TaskWaitAICPU] node=11597, rank=7, stream=65, protocol=SDMA,
  notify={recordRank=5, waitRank=7, notifyId=579}, blockedWaitNodeCount=80
```

**syncIter=1 结果：**

```
[info] op[0] Checker Success
```

## 4. 错误模式分析

### 4.1 ErrorCode 102 — Record/Wait 匹配卡住

**出现位置：** `task_graph_generator_v3.cc` → `AddLocalNotifyEdges` / `AddInterRankNotifyEdges`

**两种变体：**

| 变体 | 出现场景 | 特征 |
|------|---------|------|
| **Local** (AddLocalNotifyEdges) | 跨机用例 syncIter=0 | `protocol=INVALID`, `blockedWaitNodeCount=2`, 同 rank 内 wait 无法匹配 |
| **Cross-rank** (AddInterRankNotifyEdges) | 单机用例 syncIter=0 | `protocol=SDMA`, `blockedWaitNodeCount=80`, 跨 rank wait 无法匹配 |

**分析：**
- `protocol=INVALID` 表明 wait 任务来源于 AICPU 提交（非 SDMA/RDMA），其对应的 record 任务可能由 `UDMA_OPC_SEND` WQE 产生，但 SEND 的 notify 部分未被完整解析为 NOTIFY_RECORD
- 单机用例 `protocol=SDMA` 且 blocked 数量大（80），可能是初始化阶段的 notify 序列在 Checker 的跨 rank 匹配中不完整

### 4.2 ErrorCode 409 — AllReduce 语义检查失败

**出现位置：** `allreduce_semantics_checker.cc` → `TaskCheckAllReduceSemantics`

```
AllReduce result range [0x0,0x40000) for rank 0
should combine inputs from 4 source ranks, but it actually combines 3.
```

**分析：**
- 4 ranks 的 AllReduce 结果应包含 4 个 rank 的输入数据
- Checker 追踪到只有 3 个 rank 的数据被合并，说明有 1 个 rank 的数据传输未被追踪到
- 该 rank 的数据可能通过 `UDMA_OPC_SEND` WQE 传输，而 SEND 的数据传输部分未被正确解析为 MEM_CPY/REDUCE 任务
- `0x40000` = 256KB（1M 数据 / 4 ranks = 256KB/rank），对应 Case #10 的 1M 数据量
- `0x200000` = 2MB（8M 数据 / 4 ranks = 2MB/rank），对应 Case #01 的 8M 数据量

### 4.3 完全失败 vs Partial Success 的区分规律

| 特征 | 完全失败 (#01,04,05,10) | Partial Success (#02,03,06,07,08,09) |
|------|------------------------|--------------------------------------|
| 算子 | AllReduce 小数据(1M/8M)、ReduceScatter(全部) | AllReduce 大数据(512M)、Reduce(全部) |
| 数据量 | 小（1M/8M）或 ReduceScatter | 大（512M）或 Reduce |
| syncIter=1 | 仍失败 | 通过 |
| NOTIFY_WAIT - RECORD diff | 24, 8, 0, 8 | 4, 4, 4, 4, 56, 56 |

**假设：**
- 小数据量和 ReduceScatter 算法可能使用更多的 SEND WQE 进行控制通信，导致 syncIter=1 也受影响
- 大数据量的 AllReduce 和 Reduce 算法在 syncIter=1（实际数据传输）阶段主要使用 WRITE/WRITE_REDUCE WQE，SEND WQE 集中在初始化阶段（syncIter=0），因此 syncIter=1 能通过

### 4.4 Case #05 特殊性

Case #05（ReduceScatter 512M）的 NOTIFY_WAIT 和 NOTIFY_RECORD 数量相等（496 vs 496），但 Checker 仍然失败。说明问题不仅在于 NOTIFY_RECORD 数量缺失，还在于 **record 和 wait 的匹配关系错误**——SEND WQE 产生的 NOTIFY_RECORD 可能使用了错误的 notifyId 或 streamId，导致无法与对应的 NOTIFY_WAIT 匹配。

## 5. SEND WQE 解析问题详解

### 5.1 UDMA SQE Opcode 枚举

```
定义位置: src/device_arm/proxy/udma_data_struct_stub.h

UDMA_OPC_SEND              = 0x0   ← 未处理（已临时修复）
UDMA_OPC_SEND_WITH_IMM     = 0x1   ← 未处理（已临时修复）
UDMA_OPC_SEND_WITH_INVALID = 0x2   ← 未处理（已临时修复）
UDMA_OPC_WRITE             = 0x3   ← 已处理
UDMA_OPC_WRITE_WITH_IMM    = 0x4   ← 未处理（已临时修复）
UDMA_OPC_WRITE_WITH_NOTIFY = 0x5   ← 已处理
UDMA_OPC_READ              = 0x6   ← 已处理
UDMA_OPC_CAS               = 0x7   ← 未处理
UDMA_OPC_FAA               = 0xb   ← 未处理
UDMA_OPC_NOP               = 0x11  ← 未处理
UDMA_OPC_INVALID           = 0x12  ← 未处理
```

### 5.2 SEND 与 WRITE 的 WQE 格式差异

当前代码将 SEND WQE 按 `UdmaSqeWrite` 结构解析，但 SEND 的语义不同：

| 字段 | WRITE WQE | SEND WQE (推测) |
|------|-----------|----------------|
| `comm.rmtAddrLow/High` | 远端设备内存地址 | 非 device 内存地址（UB 内部缓冲区？） |
| `u.sge.dataAddrLow/High` | 本地设备内存地址 | 可能不在 device 内存映射中 |
| `comm.rmtEid` | 远端 EID | 远端 EID（有效） |
| `comm.inlineEn` | inline 标志 | inline 标志 |
| `comm.udfFlag` | reduce 标志 | 可能含义不同 |

**`GetRankIdByDevAddr` 失败的地址示例：**
```
devAddr[70368752553792] = 0x4000_1000_0000   ← rmtAddr (SEND 的远端地址)
devAddr[70368752553752] = 0x4000_0FFF_FFD8   ← rmtAddr
devAddr[70368752553768] = 0x4000_1000_0018   ← locAddr (SEND 的本地地址)
```

这些地址在 `0x4000_xxxx_xxxx` 范围内，不在已注册的 device 虚拟内存区间中。

### 5.3 临时修复的效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 跨机 AICPU 测试 | 死锁（QEMU 进程卡住） | 测试完成（不再死锁） |
| mpirun errors | N/A（死锁） | 0 |
| `GetRankIdByDevAddr` errors | N/A | 0（SEND 用 EID 回退） |
| Checker 结果 | N/A | 6/10 Partial Success, 4/10 完全失败 |

## 6. 日志文件位置

### 6.1 测试脚本

```
/home/ytz/CANN/CheckerL2/hccl_vm_install/bin/run_write_reduce_tests.sh
```

### 6.2 各用例 mpirun 日志

```
/tmp/write_reduce_test/case_01_cross-2x2-AllReduce-8M.log
/tmp/write_reduce_test/case_02_cross-2x2-AllReduce-512M.log
/tmp/write_reduce_test/case_03_cross-2x2-AllReduce-512M-fp16.log
/tmp/write_reduce_test/case_04_cross-2x2-ReduceScatter-8M.log
/tmp/write_reduce_test/case_05_cross-2x2-ReduceScatter-512M.log
/tmp/write_reduce_test/case_06_cross-2x2-Reduce-8M.log
/tmp/write_reduce_test/case_07_cross-2x2-Reduce-512M-max.log
/tmp/write_reduce_test/case_08_single-1x8-AllReduce-512M.log
/tmp/write_reduce_test/case_09_single-1x8-ReduceScatter-512M.log
/tmp/write_reduce_test/case_10_cross-2x2-AllReduce-1M-edge.log
```

### 6.3 各用例 hccl-vm session 日志

```
/tmp/write_reduce_test/session_01.log ~ session_10.log
```

### 6.4 Checker 日志（仅保留最后一次运行 Case #10）

```
/home/ytz/CANN/CheckerL2/hccl_vm_install/logs/checker/checker_1241492_0.log
```

> 注意：Checker 日志每次运行会被覆盖，仅保留最后一次。如需分析特定用例，建议重跑该用例并保存日志。

### 6.5 修改的 hccl-vm 源码

```
src/device_arm/proxy/device_sqe_parse_stub.cc   (主要修改)
src/device_arm/proxy/device_sqe_parse_stub.h    (函数签名)
```

### 6.6 hccl-vm 代码仓

```
/home/ytz/CANN/CheckerL2/
```

## 7. 建议分析方向

1. **SEND WQE 格式**：与 hcomm 团队确认 `UDMA_OPC_SEND` WQE 的确切字段布局，特别是 `rmtAddr` 和 `locAddr` 在 SEND 语义下的含义，以及 notify 信号如何编码
2. **SEND notify 解析**：确认 SEND WQE 的 `inlineEn` 字段是否表示附带 notify，如果是，当前 `ParseDavidUBReadWriteSqe` 的 inline 分支（生成 NOTIFY_RECORD）是否正确覆盖了所有 SEND notify 场景
3. **syncIter=0 vs syncIter=1 差异**：对比 Partial Success 用例两个 syncIter 的任务列表，确认 syncIter=0 多出的任务是哪些类型、来自哪些 WQE
4. **Case #05 的匹配关系**：NOTIFY_WAIT 和 NOTIFY_RECORD 数量相等但匹配失败，需检查 notifyId/streamId/protocol 的匹配逻辑
5. **`protocol=INVALID` 的 wait 任务**：ErrorCode 102 中 `protocol=INVALID` 表明 wait 任务的 protocol 字段未设置，确认这是 AICPU 提交任务的正常行为还是解析遗漏

---

*报告日期: 2026-07-17*
*hccl-vm 代码仓: /home/ytz/CANN/CheckerL2/*
*HCCL 代码仓: /home/ytz/CANN/win/hccl_ratio/ (commit 27eaf29a)*

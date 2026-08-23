# HcommWriteOnThread 数据面执行流程

> 梳理上层调用 `HcommWriteOnThread` 进行跨卡 RDMA Write 数据传输时的完整执行路径。

---

## 一、总体架构

`HcommWriteOnThread` 是 HCOMM 数据面最核心的单边写原语。从 C API 入口到最终硬件下发，存在 **三条不同的执行路径**，根据设备类型和调用方式分发：

| 路径 | 入口文件 | 适用场景 | 传输层 | 硬件接口 |
|------|----------|----------|--------|----------|
| **A: Host/CPU** | `cpu_primitives_c_adpt.cc` | 非 A5 设备 (910B 等) Host 侧 | `TransportIbverbs` | HCCP 驱动 `dlRaSendWr` |
| **B: AICPU/A5** | `aicpu_ts_primitives_c_adpt.cc` | A5 (950) 设备侧 AICPU | `UbTransportLiteImpl` | SQE + Doorbell |
| **C: NBI** | `cpu_primitives_c_adpt.cc` | A5 Host 侧非阻塞 | `HostCpuRoceChannel` | libibverbs `ibv_post_send` |

```
                        HcommWriteOnThread()
                               │
                    ┌──────────┼──────────┐
                    │          │          │
              thread->IsDeviceA5()?      (NBI 路径)
                    │                    │
              ┌─────┴─────┐        HcommWriteNbiOnThread()
              │           │              │
            NO (A)     YES (B)     Channel::Write()
              │           │              │
        HcclRemoteWrite  BaseTransport  HostCpuRoceChannel
              │         LiteImpl::Write    ::Write()
        Transport::        │              │
        WriteAsync()  UbTransportLite  PostRdmaOp()
              │         Impl::Write       │
        TransportIbverbs    │         ibv_post_send()
        ::WriteCommon()  RmaConnLite     (libibverbs)
              │           ::Write()
        HrtRaSendWr()       │
              │         BuildUbDbSend
        dlRaSendWr()    Task() (Doorbell)
         (HCCP 驱动)
```

---

## 二、路径 A：Host/CPU 路径（非 A5 设备）

### 2.1 入口层：`cpu_primitives_c_adpt.cc`

文件：`src/base_comm/primitives/api_c_adpt/cpu_primitives_c_adpt.cc:248`

```cpp
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel,
                           void *dst, const void *src, uint64_t len)
{
    // 1. 参数校验 & 线程注册
    AddThreadWithTag(thread);
    Thread *const threadPtr = reinterpret_cast<Thread *>(thread);

    // 2. 构造 HcclBuf：本地 src / 远端 dst
    HcclBuf locBuf{const_cast<void *>(src), len, nullptr};
    HcclBuf rmtBuf{dst, len, nullptr};

    // 3. 从 Thread 获取 Stream（异步执行上下文）
    Stream *stream = GetStream(thread);

    // 4. 下发远端写
    HcclResult ret = HcclRemoteWrite(stream,
        reinterpret_cast<void *>(channel), &rmtBuf, &locBuf);
}
```

**角色说明**：
- `thread` — 执行线程句柄，内部持有 Stream 和 LaunchContext
- `channel` — 实际是 `Transport` 对象指针（在 A 路径中），被 opaque 为 `ChannelHandle` 传递
- `dst` — **远端**目标地址（RDMA Write 的目的端地址）
- `src` — **本地**源地址

### 2.2 原语适配层：`hccl_primitive_remote.cc`

文件：`src/legacy/ascend910/platform/comm_primitive/hccl_primitive_remote.cc:18`

```cpp
HcclResult HcclRemoteWrite(StreamHandle streamHandle,
    HcclMemTransport memTransport, HcclBuf *rmtBuf, HcclBuf *locBuf)
{
    Stream *stream = reinterpret_cast<Stream*>(streamHandle);
    struct Transport::Buffer localBuf(locBuf->addr, locBuf->len);
    struct Transport::Buffer remoteBuf(rmtBuf->addr, rmtBuf->len);

    // 关键：channel 被转换为 Transport 对象
    return reinterpret_cast<Transport*>(memTransport)
        ->WriteAsync(remoteBuf, localBuf, *stream);
}
```

**关键转换**：`ChannelHandle` → `Transport*`。在 Host ROCE 场景下，这个 Transport 对象的实际类型是 `TransportIbverbs`。

### 2.3 传输抽象层：`Transport::WriteAsync`

`Transport` 是 PIMPL 模式的外观类（`transport_pub.h:556`），核心实现在 `TransportBase` 子类中。

- `Transport::WriteAsync()` → `pimpl_->WriteAsync()`
- pimpl_ 对于 ROCE 链路是 `TransportIbverbs` 实例

### 2.4 ROCE 传输层：`TransportIbverbs::WriteAsync` → `WriteCommon`

文件：`src/legacy/ascend910/platform/resource/transport/host/transport_ibverbs.cc:2573`

```cpp
HcclResult TransportIbverbs::WriteAsync(
    struct Transport::Buffer &remoteBuf,
    struct Transport::Buffer &localBuf, Stream &stream)
{
    struct WrAuxInfo aux = {0};
    return WriteCommon(remoteBuf.addr, localBuf.addr,
        remoteBuf.size, stream, WqeType::WQE_TYPE_DATA, aux);
}
```

`WriteCommon` 是核心逻辑（`transport_ibverbs.cc:2510`）：

```
WriteCommon(remoteAddr, localAddr, length, stream, wqeType, aux)
  │
  ├─ 1. ConstructPayLoadWqe()  — 构造 RDMA WQE
  │     将大数据拆分为多段（每段 ≤ RDMA_SEND_MAX_SIZE）
  │     每段构造一个 WqeInfo（含 memList, dstAddr, length）
  │
  ├─ 2. GetActualQpNum(maxLength)  — 自适应多QP决策
  │     根据数据量和 qpThreshold 决定使用几个 QP
  │
  ├─ 3. TxSendDataAndNotify()  → RdmaSendAsync()
  │     多QP：按 QP 数量分配并发送
  │     单QP：合并 doorbell 或逐次发送
  │
  └─ 4. HrtRaSendWr(qpHandle, &wr, &opRsp)
        或 HrtRaSendWrlist(qpHandle, wr, opRsp, sendNum, &completeNum)
           │
           └─ 5. dlRaSendWr(handle, wr, opRsp)   ← HCCP 驱动动态加载
                  └─ [HCCP 闭源驱动层] → 硬件 QP/SQ
```

### 2.5 HCCP 适配器层：`adapter_hccp.cc`

文件：`src/legacy/ascend910/platform/common/adapter/adapter_hccp.cc:404`

```cpp
HcclResult HrtRaSendWr(QpHandle handle, struct SendWr *wr,
                       struct SendWrRsp *opRsp)
{
    while (true) {
        // 动态加载的 HCCP 驱动函数（dlopen/dlsym）
        ret = DlRaFunction::GetInstance().dlRaSendWr(handle, wr, opRsp);

        if (!ret) break;  // 成功

        if (ret == SOCK_ENOENT || ret == ROCE_EAGAIN ||
            (IsOpBase() && ret == ROCE_ENOMEM)) {
            // 可重试错误 → 轮询等待（含超时检测）
            if (timeout) return HCCL_E_ROCE_TRANSFER;
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            return HCCL_E_ROCE_TRANSFER;  // 不可恢复错误
        }
    }
}
```

**关键机制**：
- `DlRaFunction` 在运行时通过 `dlopen` 加载 HCCP 驱动 .so，`dlsym` 绑定 `RaSendWr`
- 支持三种可重试错误码：`SOCK_ENOENT`、`ROCE_EAGAIN`、`ROCE_ENOMEM`（单算子模式）
- 超时时间由 `GetExternalInputHcclLinkTimeOut()` 控制

### 2.6 数据结构链路

```
HcommWriteOnThread 参数:
  thread(ThreadHandle) → Thread → Stream (异步 SQ 上下文)
  channel(ChannelHandle) → Transport → TransportIbverbs
                            ├─ combineQpHandles_[]  (多个 QP Handle)
                            ├─ memMsg_[]            (本地 MR 注册信息)
                            └─ remoteMemMsg_[]      (远端 MR 信息)

调用链数据转换:
  void *src, *dst, len
    → HcclBuf {addr, len}
      → Transport::Buffer {addr, size}
        → WqeInfo {
            wqeData: SendWrlistDataExt {
              memList: {addr, len, key},   // 本地 MR
              dstAddr, dstKey              // 远端 MR
            }
          }
          → SendWr {bufList, dstAddr, op, sendFlag}
            → dlRaSendWr(QpHandle, SendWr, SendWrRsp)
```

---

## 三、路径 B：AICPU/A5 设备路径

### 3.1 入口层：`aicpu_ts_primitives_c_adpt.cc`

文件：`src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc:429`

```cpp
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel,
                           void *dst, const void *src, uint64_t len)
{
    AddThread(thread);
    Thread *const threadPtr = reinterpret_cast<Thread *>(thread);

    if (threadPtr->IsDeviceA5()) {
        // A5 路径：channel 实际是 BaseTransportLiteImpl 对象
        auto *const transportLitePtr =
            reinterpret_cast<Hccl::BaseTransportLiteImpl *>(channel);
        auto *const streamLitePtr =
            static_cast<Hccl::StreamLite *>(threadPtr->GetStreamLitePtr());

        // 构建本地 RMA Buffer（含 MR token）
        Hccl::RmaBufferLite locRmaBuf;
        transportLitePtr->BuildLocRmaBufferLite(
            reinterpret_cast<uintptr_t>(src), len, locRmaBuf);
        const Hccl::Buffer rmtBuf{reinterpret_cast<uintptr_t>(dst), len};

        transportLitePtr->Write(locRmaBuf, rmtBuf, *streamLitePtr);
    }
}
```

**与路径 A 的关键差异**：
- Channel 句柄是 `BaseTransportLiteImpl*`，而非 `Transport*`
- 使用 `StreamLite`（AICPU 侧轻量 Stream），非 Host `Stream`
- `BuildLocRmaBufferLite` 构建 RMA Buffer（含 UB token/地址映射）

### 3.2 UB 传输层：`UbTransportLiteImpl::Write`

文件：`src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc:491`

```cpp
void UbTransportLiteImpl::Write(const RmaBufferLite &loc,
                                const Buffer &rmt,
                                const StreamLite &stream)
{
    SqeConfigLite cfg;
    SetFenceConfig(cfg);  // 设置 fence/order 属性
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 1. 获取本地/远端 buffer slice
    auto locRmaBufSlicelite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSlicelite = GetRmtRmaBufSliceLite(rmt);

    // 2. 通过 RmaConnLite 构造 SQE（下发描述符）
    connVec[0]->Write(locRmaBufSlicelite, rmtRmaBufSlicelite, cfg,
                      stream, connOut);

    // 3. 敲 Doorbell 通知硬件
    BuildUbDbSendTask(stream, connVec[0]->GetUbJettyLiteId(), connOut.pi);

    // 4. Profiling 记录
    ProfilingProcess(..., DmaOp::HCCL_DMA_WRITE, taskId);
}
```

**执行步骤**：
1. **GetRmaBufSlicelite** — 将本地 `RmaBufferLite`（含 tokenId/tokenValue/addr/size）转换为 `RmaBufSliceLite`
2. **GetRmtRmaBufSliceLite** — 通过 `rmtBufferMap` 查找远端地址对应的 UB Token
3. **connVec[0]->Write()** — `RmaConnLite` 子类构造 SQE（Submission Queue Entry），填充源/目的地址、长度、token 等
4. **BuildUbDbSendTask** — 向 UB Jetty 写 Doorbell，通知硬件有新任务
5. **ProfilingProcess** — DFX profiling 记录

### 3.3 RmaConnLite：连接抽象层

文件：`src/legacy/ascend950/unified_platform/resource/connection/aicpu/rma_conn_lite.h:40`

```cpp
class RmaConnLite {
    virtual void Write(const RmaBufSliceLite &loc,
                       const RmtRmaBufSliceLite &rmt,
                       const SqeConfigLite &cfg,
                       const StreamLite &stream,
                       ConnLiteOperationOut &out);
protected:
    u32 qpVa_;            // QP 虚拟地址
    u64 dbAddr_;          // Doorbell 地址
    u64 sqVa_;            // SQ 虚拟地址
    u32 sqDepth_;         // SQ 深度
    u32 tpn_;             // 传输页号
};
```

子类（P2P/RDMA/UB/CCU）各自实现 SQE 填充逻辑，根据目标地址和 Token 构造硬件描述符。

### 3.4 数据结构链路（A5）

```
HcommWriteOnThread 参数:
  thread → Thread → StreamLite → RTSQ (运行时提交队列)
  channel → BaseTransportLiteImpl → UbTransportLiteImpl
              ├─ connVec[]         (RmaConnLite 连接数组)
              ├─ rmtBufferMap      (远端地址→UB Token 映射)
              ├─ locBufferMap      (本地地址→UB Token 映射)
              └─ connOut           (SQE PI 输出)

调用链数据转换:
  void *src, *dst, len
    → RmaBufferLite {addr, size, tokenId, tokenValue}
      → RmaBufSliceLite {addr, size, ubToken}
        → SQE [HW descriptor] → Doorbell → UB Jetty
```

---

## 四、路径 C：NBI 路径（HostCpuRoceChannel 直驱）

### 4.1 入口层

文件：`src/base_comm/primitives/api_c_adpt/cpu_primitives_c_adpt.cc:445`

```cpp
int32_t HcommWriteNbiOnThread(ThreadHandle thread, ChannelHandle channel,
                              void *dst, const void *src, uint64_t len)
{
    DevType devType;
    hrtGetDeviceType(devType);
    if (devType == DevType::DEV_TYPE_950) {
        // channel 直接作为 hcomm::Channel 使用
        auto *const channelPtr = reinterpret_cast<hcomm::Channel *>(channel);
        ret = channelPtr->Write(dst, src, len);
    }
}
```

**与路径 A/B 的本质差异**：
- `HcommWriteNbiOnThread` 是 Non-Blocking Immediate（NBI）版本
- 在 A5 Host 侧，channel 是 `HostCpuRoceChannel` 实例
- **绕过** HCCP 驱动，直接调用 libibverbs `ibv_post_send`

### 4.2 HostCpuRoceChannel::Write

文件：`src/base_comm/resources/endpoint_pairs/channels/host/host_cpu_roce_channel.cc:1213`

```cpp
HcclResult HostCpuRoceChannel::Write(void *dst, const void *src,
                                     const uint64_t len)
{
    // 按 maxMsgSize_ 切片循环
    uint64_t offset = 0;
    while (offset < len) {
        uint64_t chunkLen = std::min(len - offset, maxMsgSize_);
        PostRdmaOp("Write", IBV_WR_RDMA_WRITE,
                   src + offset, dst + offset, chunkLen);
        offset += chunkLen;
    }
}
```

### 4.3 PostRdmaOp：多 QP RDMA 下发

文件：`host_cpu_roce_channel.cc:1150`

```cpp
HcclResult HostCpuRoceChannel::PostRdmaOp(const char *caller,
    ibv_wr_opcode opcode, void *localAddr, const void *remoteAddr,
    const uint64_t len)
{
    // 1. 查找 Buffer 索引 → 获取本地/远端 MR
    FindLocalBuffer(localAddr, len, localIdx);
    FindRemoteBuffer(remoteAddr, len, rmtIdx);

    // 2. 多 QP 自适应负载均衡
    //    - 默认按 QP 数量均分数据
    //    - 如果每 QP 分到的数据量 < qpThreshold，自适应减少 QP 数
    uint32_t useQpNum = QpInfo.size();
    uint32_t tileLen = len / useQpNum;

    // 3. 对每个 QP 构造 ibv_send_wr 并 ibv_post_send
    for (uint32_t i = 0; i < useQpNum; i++) {
        struct ibv_send_wr wr{};
        struct ibv_sge sg;
        BuildRdmaWr(caller, opcode, localAddr + tileLen * i,
                    remoteAddr + tileLen * i, wrLen,
                    localIdx, rmtIdx, wr, sg);
        PostAndCheckSend(qpInfo[i].qp, i, caller, wr);  // 直接调 libibverbs
    }
}
```

### 4.4 PostAndCheckSend：libibverbs 调用

文件：`host_cpu_roce_channel.cc:1125`

```cpp
HcclResult PostAndCheckSend(struct ibv_qp *qp, uint32_t qpIdx,
                            const char *caller, struct ibv_send_wr &wr)
{
    struct ibv_send_wr *badWr = nullptr;
    s32 ret = ibv_post_send(qp, &wr, &badWr);  // 直通 libibverbs
    // ENOMEM → 返回 HCCL_E_AGAIN 要求调用方重试
    // 其他错误 → HCCL_E_NETWORK
}
```

**路径 C 特点**：
- 绕过 HCCP 抽象层，直接调用 `libibverbs` API
- 自己在用户态构造 `ibv_send_wr` + `ibv_sge`
- 适合 NBI（非阻塞）模式，延迟更低
- 多 QP 负载均衡：`qpThreshold` 自适应调整 QP 利用率

---

## 五、NBI 路径对比

| 维度 | `HcommWriteOnThread` (A/B) | `HcommWriteNbiOnThread` (C) |
|------|---------------------------|----------------------------|
| 阻塞模式 | 可能阻塞（依赖 HCCP 重试） | Non-Blocking Immediate |
| 传输层 | Transport / BaseTransportLiteImpl | Channel 直接驱动 |
| 硬件接口 | HCCP 驱动 / SQE+Doorbell | libibverbs |
| QP 管理 | Transport 内部管理 | Channel 直接管理 |
| 多 QP | lbMax + qpThreshold | qpThreshold 自适应 |
| 使用场景 | 集合通信算法 | 独立算子 / 单边通信 |

---

## 六、Stream 与异步执行模型

三种路径共用 `Stream` 模型实现异步下发：

```
路径 A: Thread → Stream (Host) → HCCP Driver SQ
路径 B: Thread → StreamLite → RTSQ → Doorbell → UB Jetty
路径 C: Thread → (不使用 Stream) → 直接 ibv_post_send
```

- **路径 A** 的 Stream 封装了 RDMA QP 的 SQ 操作，含 doorbell ring 逻辑
- **路径 B** 的 StreamLite 是 AICPU 侧的轻量级流，通过 RTSQ 和 Doorbell 机制与硬件交互
- **路径 C** 不使用 Stream，直接在调用线程上下文中执行 `ibv_post_send`

---

## 七、关键文件索引

| 文件 | 内容 | 路径标识 |
|------|------|----------|
| `include/hcomm_primitives.h:226` | `HcommWriteOnThread` 声明 | 公共 API |
| `src/base_comm/primitives/api_c_adpt/cpu_primitives_c_adpt.cc:248` | Host/CPU 路径入口 | A / C |
| `src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc:429` | AICPU/A5 路径入口 | B |
| `src/legacy/ascend910/platform/comm_primitive/hccl_primitive_remote.cc:18` | HcclRemoteWrite 原语适配 | A |
| `src/legacy/ascend910/pub_inc/transport_pub.h:556` | Transport 外观类 | A |
| `src/legacy/ascend910/platform/resource/transport/host/transport_ibverbs.cc:2573` | TransportIbverbs::WriteAsync | A |
| `src/legacy/ascend910/platform/resource/transport/host/transport_ibverbs.cc:2510` | TransportIbverbs::WriteCommon | A |
| `src/legacy/ascend910/platform/resource/transport/host/transport_ibverbs.cc:1391` | RdmaSendAsync → HrtRaSendWr | A |
| `src/legacy/ascend910/platform/common/adapter/adapter_hccp.cc:404` | HCCP 适配器 → dlRaSendWr | A |
| `src/legacy/ascend950/unified_platform/resource/transport/aicpu/base_transport_lite_impl.h:39` | BaseTransportLiteImpl 基类 | B |
| `src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.h:27` | UbTransportLiteImpl 声明 | B |
| `src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc:491` | UbTransportLiteImpl::Write | B |
| `src/legacy/ascend950/unified_platform/resource/connection/aicpu/rma_conn_lite.h:40` | RmaConnLite 连接抽象 | B |
| `src/base_comm/resources/endpoint_pairs/channels/host/host_cpu_roce_channel.cc:1213` | HostCpuRoceChannel::Write | C |
| `src/base_comm/resources/endpoint_pairs/channels/host/host_cpu_roce_channel.cc:1150` | PostRdmaOp + Multi-QP | C |
| `src/base_comm/resources/endpoint_pairs/channels/host/host_cpu_roce_channel.cc:1125` | PostAndCheckSend → ibv_post_send | C |
| `src/legacy/ascend910/platform/inc/adapter/adapter_hccp.h:215` | HrtRaSendWr 声明 | A |

---

## 八、完整调用链速查

### 路径 A（Host ROCE）：
```
HcommWriteOnThread
  → HcclRemoteWrite
    → Transport::WriteAsync
      → TransportIbverbs::WriteCommon
        → ConstructPayLoadWqe
        → TxSendDataAndNotify
          → RdmaSendAsync
            → TxWqeList
            → HrtRaSendWr
              → dlRaSendWr  [HCCP Driver]
```

### 路径 B（AICPU/A5）：
```
HcommWriteOnThread
  → BaseTransportLiteImpl::BuildLocRmaBufferLite
  → BaseTransportLiteImpl::Write
    → UbTransportLiteImpl::Write
      → GetRmaBufSlicelite / GetRmtRmaBufSliceLite
      → RmaConnLite::Write  (构造 SQE)
      → BuildUbDbSendTask    (Doorbell)
      → ProfilingProcess
```

### 路径 C（NBI Host ROCE）：
```
HcommWriteNbiOnThread
  → Channel::Write
    → HostCpuRoceChannel::Write
      → PostRdmaOp
        → FindLocalBuffer / FindRemoteBuffer
        → BuildRdmaWr
        → PostAndCheckSend
          → ibv_post_send  [libibverbs]
```

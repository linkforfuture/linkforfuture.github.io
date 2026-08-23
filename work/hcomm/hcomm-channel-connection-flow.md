# HCOMM Channel 建链流程详解

本文档梳理了从上层 `HcclChannelAcquire` 到底层 HCCP 驱动建立 RDMA 连接的完整链路，重点关注 RoCE 协议下 socket 监听/连接策略和 QP 建链过程。

> 基于代码版本：master 分支，以 A5（CommunicatorV2）流程为主线。

---

## 一、HcclChannelAcquire 整体流程

### 1.1 入口与分层

```text
上层算子
   ↓
HcclChannelAcquire(comm, engine, channelDescs, channelNum, channels)
   src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc:332
   ↓
MyRank::CreateChannels(engine, commTag, channelDescs, channelNum, channels)
   src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc:701
   ↓
BatchCreateSockets(...)      // 阶段一：建立 TCP 控制面 socket
BatchCreateChannels(...)     // 阶段二：创建 Channel 对象
BatchConnectChannels(...)    // 阶段三：轮询等待 Channel 状态 READY
```

`HcclChannelAcquire` 本身只做参数校验和协议属性填充（`ProcessHcclResPackReq` / `ProcessHcclChannelDesc`），真正的建链在 `MyRank` 中完成。

### 1.2 阶段一：`BatchCreateSockets`

```cpp
// my_rank.cc:416
BatchCreateSockets
  → BatchServerInitForChannels   // 对端监听
  → BatchGetSocketsForChannels   // 本端连接，得到 Hccl::Socket*
```

- 根据 `localEndpoint` / `remoteEndpoint` 的 IP、rank、locType 决定谁做 server、谁做 client（`QueryListenPort`，`my_rank.cc:292`）。
- 最终调用 `EndpointPair::ServerInit` / `GetConnectedSocket`（`endpoint_pair.cc:137/158`）。
- 对 device RoCE 场景，由 `SocketManager` 在指定端口上监听/连接，建立一条 **TCP 连接，仅用于交换 RDMA 元数据**（QP 号、PSN、GID、buffer rkey 等）。

### 1.3 阶段二：`BatchCreateChannels`

```cpp
// my_rank.cc:463
BatchCreateChannels
  → EndpointMgr::Get(endpointDesc)                // 拿到本地 EndpointHandle
  → HcommEndpointStartListen(epHandle, port)       // 本端 endpoint 监听
  → endpointMgr_->RegisterMemory(...)              // 注册通信内存为 MR
  → rankPair->GetEndpointPair(...)                 // 拿到 EndpointPair
  → endpointPair->CreateChannel(epHandle, engine, reuseIdx, &hcommDesc, &channelHandle)
        // endpoint_pair.cc:174
```

`EndpointPair::CreateChannel` 的核心调用：

```cpp
HcommCollectiveChannelCreate(endpointHandle, engine, channelDescs, 1, channels);
// → ChannelProcess::CreateChannelsLoop(endpointHandle, engine, channelDescs, channelNum, channels)
//   → Channel::CreateChannel(endpointHandle, engine, channelDesc, channelPtr)
//     // channel.cc:30，按 engine + protocol 分发到具体 Channel 类
```

### 1.4 阶段三：`BatchConnectChannels`

```cpp
// my_rank.cc:622
BatchConnectChannels
  → ChannelProcess::ChannelGetStatus(handles, num, statusList)
  → 循环轮询，直到所有 channel 状态为 READY
```

这是**异步建链**模型：创建对象后立即返回，上层通过 `GetStatus` 轮询等待 RDMA QP 真正就绪。

### 1.5 Channel 对象创建分发

`Channel::CreateChannel`（`channel.cc:30`）根据 `engine` + `channelDesc.remoteEndpoint.protocol` 选择具体实现：

| engine | protocol | 实际 Channel 类 |
|--------|----------|-----------------|
| `COMM_ENGINE_CPU` | `COMM_PROTOCOL_ROCE` | `HostCpuRoceChannel` |
| `COMM_ENGINE_AICPU` / `AICPU_TS` / `AIV` | `COMM_PROTOCOL_ROCE` | `AicpuTsRoceChannelV2` |
| `COMM_ENGINE_CPU` | `COMM_PROTOCOL_UBC_CTP/UBC_TP` | `HostCpuUrmaChannel` |
| `COMM_ENGINE_AICPU` | `COMM_PROTOCOL_PCIE` | `AicpuTsP2pChannel` |

---

## 二、RDMA 连接建立细节

### 2.1 HostCpuRoceChannel（CPU 引擎 + RoCE）

文件：`src/base_comm/resources/endpoint_pairs/channels/host/host_cpu_roce_channel.cc`

#### Init 阶段（`host_cpu_roce_channel.cc:242`）

依次调用：

1. **`ParseInputParam()`** — 从 `endpointHandle` 取出本地 `EndpointDesc` 和 `rdmaHandle_`。
2. **`BuildSocket()`** — 如果 `channelDesc.socket` 为空，则重建 TCP socket。
3. **`BuildConnection()`** — 创建 `HostRdmaConnection` 对象数组，数量由 `channelDesc.roceAttr.queueNum` 决定。
4. **`BuildNotify()`** — 分配本地 DPU notify ID。
5. **`BuildBuffer()`** — 收集本地 `RmaBuffer`（已注册内存 MR）。

#### 异步状态机：`GetStatus`（`host_cpu_roce_channel.cc:289`）

```text
RdmaStatus::INIT
   → CheckSocketStatus()          // TCP socket 是否 OK
RdmaStatus::SOCKET_OK
   → CreateQp()                   // 创建 QP/CQ
RdmaStatus::QP_CREATED
   → ExchangeData()               // 通过 TCP socket 交换 RDMA 元数据
RdmaStatus::DATA_EXCHANGE
   → ModifyQp()                   // 把 QP 切到 RTR/RTS
RdmaStatus::QP_MODIFIED
   → IbvPostRecv()                // 预投递 recv WQE
   → CONN_OK / channelStatus = READY
```

#### QP 创建：`HostRdmaConnection::CreateQp`

文件：`src/base_comm/resources/endpoint_pairs/channels/host/host_rdma_connection.cc:71`

```cpp
// 1. 创建完成通道
RaCreateCompChannel(rdmaHandle, &sendCompChannel_);
RaCreateCompChannel(rdmaHandle, &recvCompChannel_);

// 2. 创建 CQ + QP
Hccl::HrtRaCreateQpWithCq(rdmaHandle, -1, -1, sendCompChannel_, recvCompChannel_, qpInfo_, isHdcMode_);

// 3. 设置 QoS、超时、重传次数
RaSetQpAttrQos(qpHandle, &qosAttr);
RaSetQpAttrTimeout(qpHandle, retryInterval);
RaSetQpAttrRetryCnt(qpHandle, retryCnt);
```

#### 元数据交换：`ExchangeData`（`host_cpu_roce_channel.cc:369`）

通过已建立的 TCP socket 做同步收发：

**发送端打包：**
- `NotifyVecPack` — 本地 DPU notify ID 列表
- `BufferVecPack` — 本地 RMA buffer 的 addr/size/rkey（`ExchangeRdmaBufferDto`）
- `ConnVecPack` — 每个 connection 的 QP 信息（`ExchangeRdmaConnDto`：qpn、psn、gidIdx、gid）

**接收端解包：**
- `NotifyVecUnpack` — 对端 notify ID
- `RmtBufferVecUnpackProc` — 对端 RMA buffer 信息，构造 `RemoteRdmaRmaBuffer`
- `ConnVecUnpackProc` — 对端 QP 信息

#### QP 状态切换：`ModifyQp`（`host_rdma_connection.cc:200`）

拿到本地 QP 属性 + 对端 QP 属性后，调用：

```cpp
RaTypicalQpModify(qpHandle, &localQp, &rmtQp);
```

这是标准的 RDMA RC QP 建链最后一步：把 QP 从 `INIT` → `RTR` → `RTS`。

`localQp` 和 `rmtQp` 包含：
- `qpn` — QP 号
- `psn` — 包序列号
- `gidIdx` / `gid` — GID 索引和值
- `sl` / `tc` — Service Level / Traffic Class（QoS）
- `retryCnt` / `retryTime` — 重传次数 / 超时

#### 预投递 Recv WQE：`IbvPostRecv`（`host_cpu_roce_channel.cc:721`）

对每个 QP 投递 `SEND_RQE_COUNT` 个 recv WQE，之后 `channelStatus_ = READY`。

### 2.2 AicpuTsRoceChannelV2（AICPU/AIV 引擎 + RoCE，A5）

文件：`src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_roce_channel_v2.cc`

状态机与 Host CPU RoCE 一致：

```text
Init() → BuildSocket() → BuildConnection() → BuildNotify() → BuildBuffer() → BuildNotifyValueBuffer()
GetStatus()
   → CheckSocketStatus → CreateQp → ExchangeData → ModifyQp → READY
```

#### 差异点

- `BuildConnection()` 创建的是 `DevRdmaConnectionV2`（`dev_rdma_connection_v2.cc`），QP 创建调用：
  ```cpp
  Hccl::HrtRaNdaCqCreate(rdmaHandle, &ndaOps_, dmaMode_, &ndaCqInfo_, &cqHandle_);
  Hccl::HrtRaNdaQpCreate(rdmaHandle, &ndaOps_, dmaMode_, &ndaCqInfo_, &ndaQpInfo_, &qpHandle_);
  ```
  这里的 QP/CQ 是 device 侧 NDA（Non-DMA Access）队列，AICPU/AIV kernel 后续直接通过 doorbell/SQ/CQ 上下文下发 RDMA 操作。

- `BuildNotifyValueBuffer()` 创建 device 侧 notify value buffer（host 侧 MR 注册 + device 拷贝）。

- 建链完成后，AICPU 场景下 `ChannelProcess::ChannelKernelLaunchForComm` 把 QP/SQ/CQ/buffer/notify 上下文序列化打包，通过 AICPU kernel（`RunAicpuIndOpChannelInitV2`）下发到 device。

### 2.3 旧版兼容流程（ChannelManager::ChannelCommCreate）

如果 `hcclComm->IsCommunicatorV2()` 为 false，则走到：

```text
ChannelManager::ChannelCommCreate
   src/legacy/ascend910/framework/communicator/impl/independent_op/channel/channel_manager.cc:666
   ↓
BuildChannelRequests(...)         // 构造 TransportRequest
channelCallbacks_.indOpTransportAlloc(...)   // 分配 Transport（Link）
AicpuChannelInit(...)             // 对 AICPU 下 kernel
```

---

## 三、Socket 监听/连接策略

### 3.1 角色决定策略

决策分布在多个层面：

#### 3.1.1 `MyRank::QueryListenPort`（`my_rank.cc:292`）

- 从 `rankGraph_` 查询对端 rank 在该网口上的监听端口
- 比较本地 IP 与远端 IP：若 `localIpAddr < remoteIpAddr`，本端为 **server**
- 结果写入 `hcommDesc.role`（`HCOMM_SOCKET_ROLE_SERVER/CLIENT`）和 `hcommDesc.port`

#### 3.1.2 `SocketConfig` 构造函数（`socket_config.h`）

多种构造函数提供兜底策略：

| 场景 | 规则 |
|------|------|
| 同类型网卡（默认） | `localAddr < remoteAddr ? SERVER : CLIENT` |
| 带 rank id 版本 | `localRankId < remoteRankId ? SERVER : CLIENT` |
| host-device 混跑 | 强制 `myRank < rmtRank ? SERVER : CLIENT`（避免 IP 格式不一致导致错判） |

```cpp
// socket_config.h:40 - 按 IP 大小
role(link.GetLocalAddr() < link.GetRemoteAddr() ? SERVER : CLIENT)

// socket_config.h:72 - host nic ↔ device nic 混跑
role = myRank < rmtRank ? SERVER : CLIENT;
```

### 3.2 批量建链流程

#### 3.2.1 先监听：`BatchServerInitForChannels`（`my_rank.cc:344`）

```text
BatchServerInitForChannels
  → GetEndpointPairFromChannel → (rankPair, endpointPair)
  → 按 (rankPair, endpointPair) 去重，记录 reuseIdx
  → EndpointPair::ServerInit()
    → SocketManager::ServerListen(socketConfig)
      → PrepareLinkAndServerInit
        → ServerInit(PortData)  // 按 PortData 维护单例监听 socket
```

**Legacy `SocketManager::ServerInit`**（`socket_manager.cc:201`）：
- 从 `rankListenPortMap_` 获取该 rank + IP 的监听端口
- 若 `serverSocketMap` 中已有，且端口号改变过，则重新 `Listen(newPort)`
- 否则创建新 `Hccl::Socket`，调用 `socket->Listen(port)`

#### 3.2.2 后连接：`BatchGetSocketsForChannels`（`my_rank.cc:378`）

```text
BatchGetSocketsForChannels
  → QueryListenPort → listenPort + role
  → EndpointPair::GetConnectedSocket()
    → SocketManager::ConnectSockets(socketConfig)
      → AddWhiteList(remoteIp, tag)    // 为对端 IP + tag 开白名单
      → CreateConnectedSocket(socketConfig)
        → Hccl::Socket(...) → ConnectAsync()
```

### 3.3 两套 Socket 管理器并存

| 管理器 | 文件 | 用途 |
|--------|------|------|
| `Hccl::SocketManager`（legacy） | `socket_manager.cc` | device net 场景，被 `EndpointPair` 使用 |
| `hcomm::SocketMgr`（orion） | `socket_mgr.cc` | host net 场景，被 `HostCpuRoceChannel`、`SocketProcess` 使用 |

#### Legacy SocketManager 的特点

- `serverSocketMap`（static）：同一进程内，同一 `PortData` 只建一个监听 socket，所有复用该端口的 channel 共享。
- `connectedSocketMap`：按 `SocketConfig` 保存已创建的连接 socket。
- `rankListenPortMap_`：从 `RankGraph` 获得，通信域初始化时设置。

#### Orion SocketMgr 的特点

- `socketMap_`：`SocketConfig → unique_ptr<Hccl::Socket>`，支持查找/复用。
- `socketInUseMap_`：标记 socket 是否被占用，复用前需等待 release。
- `handle2WhiteListMap_`：记录每个 `SocketHandle` 加过的白名单。

### 3.4 Socket 复用策略

#### 3.4.1 Channel 维度

`MyRank::BatchCreateChannels` 中维护三维 map：

```cpp
reuseChannelIdxMap[RankPair][CommEngine][EndpointPair]
```

同 `(rankPair, engine, endpointPair)` 的多个 channel 按 `reuseIdx` 递增复用连接。

```cpp
// endpoint_pair.cc:174
if (channelHandles_[engine].size() <= reuseIdx) {
    // 新建 channel
    HcommCollectiveChannelCreate(...);
} else {
    channels[0] = channelHandles_[engine][reuseIdx]; // 复用
}
```

#### 3.4.2 不复用的场景

```cpp
// my_rank.cc:555
/* hostNIC -- DeviceNic（transport不复用link/Channel） */
if (localEndpointDesc.loc.locType != remoteEndpointDesc.loc.locType) {
    idx = UNREUSE_CHANNEL_IDX;
}
```

### 3.5 端口来源

- 优先从拓扑图 `rankIpPortMap_` 获取（`SocketManager::SetDeviceServerListenPortMap`）
- 若查询不到，回退到 `DEFAULT_VALUE_TCPPORT`
- 特定场景支持自动端口分配（`AUTO_LISTEN_PORT = 0` → HCCP 自动选择可用端口）

### 3.6 白名单机制

底层 HCCP socket 默认只接受白名单内的连接：

```cpp
RaSocketWhitelist wlistInfo;
wlistInfo.connLimit = 1;                                      // 每个对端 IP 只允许一条连接
wlistInfo.remoteIp = socketConfig.link.GetRemoteAddr();       // 对端 IP
wlistInfo.tag = socketConfig.GetHccpTag();                    // 匹配标识

HrtRaSocketWhiteListAdd(socketHandle, {wlistInfo});
```

`tag` 格式示例：`socketTag_localRank_remoteRank_localIp_remoteIp`，确保两端一致。

---

## 四、HCCP 调用详解

`Hccl::Socket` 对 HCCP 的调用分为四层：

```text
┌─ Hccl::Socket (socket.cc) ──────────────────────────────────┐
│ 状态机封装，异步/同步语义                                       │
├─ Adapter 层 (orion_adapter_hccp.cc) ────────────────────────┤
│ IpAddress→HccpIpAddr 转换，轮询 SOCK_EAGAIN，超时/错误处理      │
├─ RA Socket (ra_socket.c) ────────────────────────────────────┤
│ phyId/tag 合法性校验，RaInetPton IP转换                       │
├─ HCCP C API (hccp.h) ───────────────────────────────────────┤
│ RaSocketListenStart / RaSocketBatchConnect / RaGetSockets   │
├─ HCCP 驱动 (闭源, libhdc.so) ────────────────────────────────┤
│ TCP bind/listen/connect/accept，epoll，白名单+tag路由，心跳    │
└──────────────────────────────────────────────────────────────┘
```

### 4.1 netMode 的影响

| NicType | netMode | 含义 |
|---------|---------|------|
| `DEVICE_NIC_TYPE` | `HDC` / `NETWORK_OFFLINE` | 昇腾芯片内置 RoCE 网卡，走 HDC 通道 |
| `HOST_NIC_TYPE` | `PEER` / `NETWORK_PEER_ONLINE` | 标准 host 网卡（CX-N 等），走标准 TCP |
| `DEVICE_VNIC_TYPE` | `HDC` / `NETWORK_OFFLINE` | 虚拟网卡（P2P 场景），走 HDC |

### 4.2 Socket 初始化（SocketHandle 来源）

```cpp
// orion_adapter_hccp.cc:674
HrtRaSocketInit(netMode, raInterface)
  → RaSocketInit(mode, rdevInfo, &socketHandle)
```

`rdevInfo` 包含 `phyId`、`family`（AF_INET/AF_INET6）、`localIp`。

HCCP 驱动内部为指定 device + IP 分配网络资源，返回 `socketHandle`（不透明指针）。

### 4.3 Listen 调用链

#### 同步 Listen

```cpp
Socket::Listen()                                   // socket.cc:31
  → HrtRaSocketListenOneStart(param, netMode)      // orion_adapter_hccp.cc:425
    → HRaSocketListenStart(&listenInfo, 1, ...)    // orion_adapter_hccp.cc:336
      → RaSocketListenStart(conn, num)             // hccp.h:62
      → 轮询 SOCK_EAGAIN / 超时处理
```

错误码处理：
- `SOCK_EADDRINUSE` → 端口被占用，上报错误信息
- `SOCK_EADDRNOTAVAIL` → IP 不可用
- `SOCK_EAGAIN` → 速率限制，延时 1ms 重试

#### 异步 Listen

```cpp
Socket::ListenAsync()                              // socket.cc:417
  → reqHandle = RaSocketListenOneStartAsync(...)
    → RaSocketListenStartAsync(conn, num, reqHandle) // ra_socket.c:66
      → RaHdcSocketListenStartAsync(phyId, ...)

// 完成检测
Socket::GetAsyncStatus()
  → LISTEN_STARTING → CheckStartRequestResult()
    → HrtRaGetAsyncReqResult(reqHandle)
      → COMPLETED: isListening = true, status = LISTENING
      → NOT_COMPLETED: 继续等待
      → SOCK_E_AGAIN: 重新调用 ListenAsync()
```

### 4.4 Connect 调用链

#### 同步 Connect

```cpp
Socket::Connect()                                  // socket.cc:59
  → HrtRaSocketConnectOne(param)                  // orion_adapter_hccp.cc:264
    → SocketBatchConnect(&connInfo, 1)            // orion_adapter_hccp.cc:238
      → RaSocketBatchConnect(conn, num)           // hccp.h:28
      → 轮询 SOCK_EAGAIN / 超时处理
```

HCCP 驱动内部发起 TCP `connect()`（非阻塞语义），通过 **tag** 匹配 server 端白名单中的连接请求。

#### 异步 Connect

```cpp
Socket::ConnectAsync()                             // socket.cc:426
  → reqHandle = RaSocketConnectOneAsync(param)
    → RaSocketBatchConnectAsync(conn, num, reqHandle) // ra_socket.c:19
      → RaHdcSocketBatchConnectAsync(phyId, ...)
```

### 4.5 连接完成获取

#### 同步版本

```cpp
Socket::GetStatus()                                // socket.cc:81
  → HrtRaBlockGetOneSocket(role, param)
    → RaBlockGetSockets(role, &socketInfo, 1)
      → RaGetSockets(role, conn, num, &connectedNum) // hccp.h:86
      → 轮询直到 connectedNum == num 或超时
  → fdHandle = result.fdHandle
  → status: CONNECTED / TIMEOUT / CONNECTING
```

#### 异步版本

```cpp
Socket::GetAsyncStatus()                           // socket.cc:349
  → CONNECT_STARTING:
    → CheckStartRequestResult()
      → COMPLETED: GetOneSocket()
        → RaGetOneSocket(role, param)
          → CONNECTED: fdHandle赋值, status = OK
          → CONNECT_TIMEOUT
          → CONNECTING: 继续等待
      → SOCK_E_AGAIN: 重新 ConnectAsync()
```

### 4.6 关键数据结构

```cpp
// HCCP 初始化
struct rdev {
    uint32_t phyId;
    int family;                // AF_INET / AF_INET6
    union HccpIpAddr localIp;
};

// 监听参数
struct SocketListenInfoT {
    void *socketHandle;
    uint32_t port;
};

// 连接参数
struct SocketConnectInfoT {
    void *socketHandle;
    union HccpIpAddr remoteIp;
    uint32_t port;
    char tag[SOCK_CONN_TAG_SIZE];
};

// 获取已连接 socket
struct SocketInfoT {
    void *socketHandle;
    union HccpIpAddr remoteIp;
    void *fdHandle;            // OUTPUT: 连接建立后的 fd
    int status;                // 0=未连接, 1=已连接, 2=超时, 3=连接中
    char tag[SOCK_CONN_TAG_SIZE];
};
```

### 4.7 数据收发

| 接口 | 语义 | 关键行为 |
|------|------|----------|
| `HrtRaSocketBlockSend` | 阻塞发送 | 轮询 `RaSocketSend` 直到全部发完或超时 |
| `HrtRaSocketBlockRecv` | 阻塞接收 | 轮询 `RaSocketRecv` 直到全部收完或超时 |
| `HrtRaSocketNonBlockSendHeart` | 非阻塞发送（心跳） | `RaSocketSend` 单次调用，返回 `HCCL_E_AGAIN` 表示没发完 |
| `HrtRaSocketNonBlockRecvHeart` | 非阻塞接收（心跳） | `RaSocketRecv` 单次调用，返回 `HCCL_E_AGAIN` 表示暂无数据 |

HDC 模式下单次 send 最大 64KB（受 HDC 通道限制），PEER 模式下无此限制。

---

## 五、大规模集群建链优化策略

千卡集群场景下，每张卡可能需要与数百个 peer rank 建链，总的 TCP socket + RDMA QP 数量可达 O(N²) 级别。HCOMM 的建链优化从多个维度叠加，核心思路是：**减少重复建链、批量并行化、异步非阻塞、预初始化**。

---

### 5.1 Socket 层复用（减少重复建链）

#### 5.1.1 Server Socket 按 PortData 单例复用

`SocketManager::serverSocketMap` 是 **static 静态全局变量**（`socket_manager.cc:30`），同一进程内同一 `(IP, PortDeploymentType, LinkProtoType, portId)` 只建 **一个** 监听 socket：

```cpp
// socket_manager.cc:478
static std::unordered_map<PortData, shared_ptr<Socket>> &SocketManager::GetServerSocketMap() {
    static std::unordered_map<PortData, shared_ptr<Socket>> serverSocketMap;
    return serverSocketMap;
}
```

千卡场景下，同一物理网口上所有到不同 rank 的连接都复用这一个监听 socket，`connLimit` 控制同一个 remote IP 允许的连接数。

```cpp
// socket_manager.cc:211  — ServerSocketMap 中已存在则直接返回
auto serverSocketInMap = serverSocketMap.find(localPort);
if (serverSocketInMap != serverSocketMap.end()) {
    // ...检查是否需要更换端口
    return;  // 复用已有监听 socket
}
```

#### 5.1.2 Connected Socket 按 SocketConfig 复用

`connectedSocketMap` 按 `(remoteRank, localPort, remotePort, tag, listenPort)` 精确匹配，同一对 rank 之间多个 channel 复用同一条已连接 TCP socket：

```cpp
// socket_manager.cc:359
Socket *SocketManager::GetConnectedSocket(const SocketConfig &socketConfig) const {
    auto res = connectedSocketMap.find(socketConfig);
    if (res != connectedSocketMap.end()) {
        return res->second.get();  // 复用已有连接
    }
    return nullptr;
}
```

此外 `SocketMgr::socketInUseMap_` 配合 `condition_variable` 实现等待-唤醒机制：如果 socket 正被占用，后来者阻塞等待前一个使用者归还后再复用。

```cpp
// socket_mgr.cc:225
while (socketInUseMap_[socket] == true) {
    auto currentTime = std::chrono::steady_clock::now();
    if (currentTime >= timeoutPoint) {
        return HCCL_E_TIMEOUT;
    }
}
socketAvailableCv_.wait(lock);  // 或通过 PutSocket 的 notify_all 唤醒
```

#### 5.1.3 Channel 按 reuseIdx 复用

`EndpointPair::CreateChannel`（`endpoint_pair.cc:174`）中，同 `(engine, endpointPair)` 的 channel 按 `reuseIdx` 复用物理连接——千卡集群的同 node 内多个 sub-communicator 的 channel 共享底层 transport：

```cpp
// endpoint_pair.cc:177
if (channelHandles_.find(engine) == channelHandles_.end() ||
    channelHandles_[engine].size() <= reuseIdx) {
    // 真正新建
    HcommCollectiveChannelCreate(endpointHandle, engine, channelDescs, 1, channels);
    channelHandles_[engine].push_back(channels[0]);
} else {
    // 直接复用已存在的 channel
    channels[0] = channelHandles_[engine][reuseIdx];
}
```

`reuseIdx` 由 `MyRank::BatchCreateChannels`（`my_rank.cc:536`）中三维 map `reuseChannelIdxMap[RankPair][CommEngine][EndpointPair]` 递增管理。

#### 5.1.4 旧版（A2/A3）多 QP 复用

`HcclSocketManager`（`hccl_socket_manager.cc:234`）中 `socketsPerLink` 支持同一 rank 对创建多条 socket，但会优先查找已有连接复用：

```cpp
// hccl_socket_manager.cc:234
if (isSupportReuse) {
    GetSocketsByRankIP(commTag, remoteRank, remoteIp, socketsPerLink, ipSockets, gotLinkNum);
    socketsPerLink -= gotLinkNum;
    if (socketsPerLink == 0) return HCCL_SUCCESS;  // 全部复用，无需新建
}
```

---

### 5.2 Server 预初始化（减少建链时的延迟）

#### 5.2.1 ServerInitAll：通信域初始化时抢占所有端口

在 `rank_info_detect_client.cc:34`，通信域初始化阶段就调用：

```cpp
SocketManager::ServerInitAll(localRankTable.ranks[0]);
```

`ServerInitAll`（`socket_manager.cc:240`）遍历全拓扑图的所有 edge，为每个 `(deviceId, port, IP)` 组合提前创建 `Hccl::Socket` 并 `Listen`，配合 `PreemptPortManager` 从配置的端口范围（`HCCL_HOST_SOCKET_PORT_RANGE` / `HCCL_NPU_SOCKET_PORT_RANGE`）中抢占端口。

```cpp
// socket_manager.cc:288
for (auto &link : links) {
    PortData localPort{...};
    if (serverSocketMap.find(localPort) != serverSocketMap.end()) {
        listenPort = serverSocketMap[localPort]->GetListenPort();  // 复用老端口
    } else {
        // 首次执行启用新端口，走 PreemptPortManager 抢占
        PreemptPortManager::GetInstance(devLogicId).ListenPreempt(serverSocket, listenPortRanges, listenPort);
        serverSocketMap[localPort] = std::move(serverSocket);
    }
}
```

#### 5.2.2 PreemptPortManager：端口共享 + 引用计数

`PreemptPortManager`（`preempt_port_manager.cc`）以 IP 为粒度管理端口抢占，同一 IP 上只需抢占一个端口，多个 communicator/sub-communicator 通过 `Referenced` 引用计数共享：

```cpp
// preempt_port_manager.cc:89
if (portRef.find(ipAddr) != portRef.end()) {
    usePort = portRef[ipAddr].first;               // 复用已抢占端口
    listenSocket->Listen(usePort);
    portRef[ipAddr].second.Ref();                  // 引用计数 +1
    return;
}
// 否则轮询端口范围，找到第一个可用端口
for (auto &range : portRange) {
    for (u32 port = range.min; port <= range.max; ++port) {
        if (listenSocket->Listen(port)) {
            usePort = port;
            portRef[ipAddr].first = usePort;
            portRef[ipAddr].second.Ref();
            return;
        }
    }
}
```

**效果**：千卡集群通信域初始化时一次性完成所有 server listen，后续创建 channel 时不再需要重复 listen。

支持两种端口范围环境变量：
- `HCCL_HOST_SOCKET_PORT_RANGE`：host NIC 端口范围
- `HCCL_NPU_SOCKET_PORT_RANGE`：device NIC 端口范围

---

### 5.3 批量 + Server-First 顺序（避免死锁 + 提高并发）

#### 5.3.1 先全量 ServerInit，再全量 Connect

`MyRank::BatchCreateSockets`（`my_rank.cc:416`）严格遵循 **先全部监听、再全部连接** 的顺序：

```cpp
// my_rank.cc:416
BatchCreateSockets
  → BatchServerInitForChannels(..., reuseSocketIdxMap)  // 去重 → 统一 ServerInit
  → BatchGetSocketsForChannels(..., reuseSocketIdxMap)   // 统一 Connect
```

如果先 Connect 再 Listen，会形成类似死锁的依赖环（A 等 B listen，B 等 C listen，C 等 A listen）。Server-First 策略从根本上杜绝了这种跨 rank 的顺序依赖。

#### 5.3.2 批量加白名单

`SocketManager::BatchAddWhiteList`（`socket_manager.cc:148`）将一个 `PortData` 上所有对端 rank 的 IP + tag 打包，一次性加入白名单：

```cpp
// socket_manager.cc:148
for (const auto &link : links) {
    SocketRole role = link.GetLocalRankId() < link.GetRemoteRankId() ? SERVER : CLIENT;
    if (role == SocketRole::SERVER) {
        RaSocketWhitelist wlistInfo{};
        wlistInfo.connLimit = 1;
        wlistInfo.remoteIp = link.GetRemoteAddr();
        wlistInfo.tag = hccpSocketTag;
        wlistMap[link.GetLocalPort()].push_back(wlistInfo);  // 聚合到 PortData 粒度
    }
}
// 每个 PortData 调用一次 RaSocketWhiteListAdd，每次可传最多 MAX_NUM_OF_WHITE_LIST_NUM 条
for (auto &i : wlistMap) {
    AddWhiteList(i.first, i.second);
}
```

#### 5.3.3 批量交换一致性信息

`ExchangeInfoMgr::BatchExchangeFixedData`（`exchange_info_mgr.cc:118`）对所有 channel 的 socket 批量做异步收发：

```cpp
// exchange_info_mgr.cc:118
HcclResult BatchExchangeFixedData(...) {
    // 第一轮：SERVER 先 Recv、CLIENT 先 Send
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER)
            sockets[i]->RecvAsync(recvData + i * recvLen, recvLen);
        else
            sockets[i]->SendAsync(sendData, sendLen);
    }
    CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));

    // 第二轮：SERVER 再 Send、CLIENT 再 Recv
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER)
            sockets[i]->SendAsync(sendData, sendLen);
        else
            sockets[i]->RecvAsync(recvData + i * recvLen, recvLen);
    }
    CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));
}
```

两轮异步操作分别 `SendAsync`/`RecvAsync` 一次性全部下发给 HCCP 驱动，然后统一 `WaitAllAsyncComplete` 轮询。`WaitActiveAsyncComplete` 进一步优化了等待子集——只等待有实际数据需要交换的 socket。

---

### 5.4 异步非阻塞建链（提高并发吞吐）

#### 5.4.1 ConnectAsync + 轮询模型

`Hccl::Socket::ConnectAsync()` 不阻塞等待 TCP 握手完成，而是把请求提交给 HCCP 驱动后立即返回。HCCP 在内部通过 `reqHandle` 追踪完成状态：

```cpp
// socket.cc:426
void Socket::ConnectAsync() {
    if (role == SocketRole::SERVER || socketStatus == SocketStatus::OK) return;
    RaSocketConnectParam param(socketHandle, remoteIp, listenPort, tag);
    reqHandle = RaSocketConnectOneAsync(param);
    socketStatus = SocketStatus::CONNECT_STARTING;
    // 立即返回，不等待 TCP 三次握手
}
```

所有 socket 的 `ConnectAsync` 下发给 HCCP 后，HCCP 的 **`RaHdcThreadPool`**（`ra_adp_pool.c`）用 worker 线程池并行处理 TCP 握手，避免了每个连接一个线程的开销。

#### 5.4.2 建链 + 元数据交换合并

建链完成后不额外等待，直接在同一条 TCP 连接上用 `SendAsync`/`RecvAsync` 交换 QP 元数据——`HostCpuRoceChannel::ExchangeData()` 和 `AicpuTsRoceChannelV2::ExchangeData()` 中的同步收发在同一个 `GetStatus()` 的调用-轮询周期内完成。

#### 5.4.3 轮询合并：ChannelGetStatus 批量查询

`ChannelProcess::ConnectChannels` / `BatchConnectChannels` 对 `channelNum` 个 channel **一次性查询** `GetStatus`，不需要每个 channel 单独等待：

```cpp
// channel_process.cc:191
HcclResult ChannelProcess::ConnectChannels(ChannelHandle* targetChannels,
    uint32_t channelNum, CommEngine engine) {
    std::vector<int32_t> statusVec(channelNum, 0);
    int32_t* statusList = statusVec.data();

    while (true) {
        HcclResult ret = ChannelGetStatus(targetChannels, channelNum, statusList);
        // 批量查询所有 channel 状态
        if (ret == HCCL_E_AGAIN) continue;    // 任意 channel 未就绪则继续
        if (ret == HCCL_SUCCESS) break;       // 全部就绪
    }
}
```

#### 5.4.4 HCCP 驱动层线程池

`RaHdcPoolCreate`（`ra_adp_pool.c:110`）创建 `threadNum` 个 worker 线程 + `queueSize` 大小的任务队列，基于 pthread + condition variable 实现生产者-消费者模型：

```cpp
// ra_adp_pool.c:110
struct RaHdcThreadPool *RaHdcPoolCreate(unsigned int queueSize, unsigned int threadNum) {
    pool->taskQueue = calloc(queueSize, sizeof(struct RaHdcTask));
    pool->queueSize = queueSize;
    for (i = 0; i < threadNum; i++) {
        pthread_create(&pool->workerThreads[i], NULL, RaHdcWorkerThread, pool);
    }
}
```

所有 HCCP 异步操作（listen、connect、send、recv、qp_create）共享这个线程池，避免每个操作独立开线程的开销。

#### 5.4.5 `isWaitEstablished` 参数控制建链模式

`HcclSocketManager::CreateSockets`（`hccl_socket_manager.cc:332`）提供 `isWaitEstablished` 参数：

```cpp
// hccl_socket_manager.cc:332
HcclResult CreateSockets(..., bool isWaitEstablished) {
    // ...
    // 需要等待连接建立成功时，才会阻塞轮询
    if (isWaitEstablished) {
        CHK_RET(WaitLinksEstablishCompleted(SOCKET_ROLE_CLIENT, clientSocketsMap));
        CHK_RET(WaitLinksEstablishCompleted(SOCKET_ROLE_SERVER, serverSocketsMap));
    }
}
```

调用方可以设置为 `false` 实现"发起连接请求即返回"，后续异步轮询——在千卡场景下避免了串行阻塞等待。

---

### 5.5 多 QP 并行（单链带宽打满）

`HostCpuRoceChannel::BuildConnection()` 中，根据 `channelDesc.roceAttr.queueNum` 或 `lbMax_` 创建多条 `HostRdmaConnection`：

```cpp
// host_cpu_roce_channel.cc:205
for (u32 i = 0; i < loopTimes; i++) {
    conn = std::make_unique<HostRdmaConnection>(socket_, rdmaHandle_);
    conn->Init();
    Hccl::QpInfo& qpInfo = conn->GetQpInfo();
    qpInfo.lbValue = i % lbMax_;   // 每条 QP 绑定不同 load balance 值（不同硬件队列）
    connections_.emplace_back(std::move(conn));
}
```

实际 RDMA 数据传输时按 `qpThreshold` 自适应切分数据到多个 QP 上，流水线化并行下发 WQE：

```cpp
// host_cpu_roce_channel.cc:1063
// 计算每个 qp 需要发送的数据量
std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
uint32_t useQpNum = qpInfo.size();
uint32_t tileLen = tailLen / useQpNum;
// 若单 QP 数据量低于阈值，自适应减少 QP 数量
if (tileLen != 0 && tileLen < channelDesc_.roceAttr.qpThreshold) {
    useQpNum = (tailLen - 1) / channelDesc_.roceAttr.qpThreshold + 1;
}
```

---

### 5.6 Host-Device 混跑连接模式下不复用（安全隔离）

Host NIC 与 Device NIC 之间不复用 channel：

```cpp
// my_rank.cc:555
if (localEndpointDesc.loc.locType != remoteEndpointDesc.loc.locType) {
    idx = UNREUSE_CHANNEL_IDX;  // host ↔ device 不复用
}
```

防止混跑场景下错误地共享了不兼容的 transport。

---

### 5.7 优化效果总结

| 优化维度 | 机制 | 千卡集群收益 |
|----------|------|-------------|
| **减少建链次数** | Server socket 全局单例、Connected socket 复用、Channel reuseIdx | O(N²) 连接数降为 O(N) 物理 socket |
| **预初始化** | ServerInitAll + PreemptPortManager + 端口范围抢占 | 建链时零 listen 延迟 |
| **批量并行** | BatchServerInit → BatchConnect → BatchExchange → BatchGetStatus | 全量并发，无串行瓶颈 |
| **异步非阻塞** | ConnectAsync + HCCP RaHdcThreadPool 线程池 | 建链 + 数据交换流水线化 |
| **Server-First 顺序** | 先全部 listen 再全部 connect | 消除 SYN 风暴和跨 rank 顺序死锁 |
| **多 QP 打满带宽** | lbMax 负载均衡 + qpThreshold 自适应切分 | 单链带宽达线速 |
| **批量加白名单** | BatchAddWhiteList 每条 PortData 聚合 | O(M) 次 HCCP 调用而非 O(M×N) |
| **端口抢占共享** | 同 IP 多通信域引用计数共享 | 节省端口、避免多进程端口冲突 |
| **一致性信息批量交换** | Server-First 两轮异步 + WaitActiveAsyncComplete | 避免逐个 socket 同步收发 |

---

## 六、关键文件索引

| 文件 | 职责 |
|------|------|
| `src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc` | `HcclChannelAcquire` 入口 |
| `src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc` | A5 建链主逻辑：CreateChannels / BatchCreateSockets / BatchCreateChannels |
| `src/base_comm/resources/endpoint_pairs/endpoint_pair.cc` | EndpointPair 的 socket/channel 管理 |
| `src/base_comm/resources/endpoint_pairs/channels/channel.cc` | Channel 工厂，engine+protocol 分发 |
| `src/base_comm/resources/endpoint_pairs/channels/host/host_cpu_roce_channel.cc` | CPU 引擎 RoCE 建链 |
| `src/base_comm/resources/endpoint_pairs/channels/host/host_rdma_connection.cc` | 单条 RDMA QP 管理 |
| `src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_roce_channel_v2.cc` | AICPU/AIV 引擎 RoCE 建链（A5） |
| `src/base_comm/resources/endpoint_pairs/channels/aicpu/dev_rdma_connection_v2.cc` | Device 侧 NDA QP 管理 |
| `src/base_comm/resources/endpoint_pairs/channels/channel_process.cc` | Channel 生命周期管理 |
| `src/base_comm/resources/endpoint_pairs/sockets/socket_mgr.cc` | Orion SocketMgr（host net） |
| `src/base_comm/resources/endpoint_pairs/sockets/socket_process.cc` | SocketProcess（resource 层 socket 管理） |
| `src/legacy/ascend950/framework/resource_manager/socket/socket_manager.cc` | Legacy SocketManager（device net） |
| `src/legacy/ascend950/framework/resource_manager/socket/socket_config.h` | SocketConfig 角色/标签构造 |
| `src/legacy/ascend950/unified_platform/resource/socket/socket.cc` | `Hccl::Socket` 状态机实现 |
| `src/legacy/ascend950/unified_platform/external_system/orion_adapter_hccp.cc` | HCCP C API 的 C++ adapter 封装 |
| `src/legacy/ascend950/unified_platform/external_system/orion_adapter_hccp.h` | HCCP adapter 头文件 |
| `src/base_comm/resources/hccp/inc/network/hccp.h` | HCCP C API 声明 |
| `src/base_comm/resources/hccp/rdma_agent/client/async/ra_socket.c` | HCCP RA Socket 异步封装 |
| `src/legacy/ascend910/framework/communicator/impl/independent_op/channel/channel_manager.cc` | 旧版 ChannelManager 建链流程 |
| `src/legacy/ascend950/framework/topo/rank_info_detect/preempt_port_manager.cc` | 端口抢占管理器（ServerInitAll 依赖） |
| `src/legacy/ascend910/algorithm/impl/resource_manager/hccl_socket_manager.cc` | 旧版 A2/A3 HcclSocketManager 复用策略 |
| `src/base_comm/resources/hccp/rdma_agent/adapter/async/ra_adp_pool.c` | HCCP RaHdcThreadPool 工作线程池 |
| `src/coll_communicator_mgr/resource_mgr/local/my_rank/exchange_info_mgr.cc` | 批量交换一致性信息（BatchExchangeFixedData） |
| `src/legacy/ascend950/framework/resource_manager/transport/ub_memory_transport_mgr.cc` | UB Memory Transport 批量建链 |

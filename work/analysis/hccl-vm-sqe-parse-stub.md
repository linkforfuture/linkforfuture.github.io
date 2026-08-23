# hccl-vm device_sqe_parse_stub 代码改动说明

> 改动时间: 2026-07-17
> 改动仓库: `/home/ytz/CANN/CheckerL2/`
> 改动背景: 测试 HCCL `hccl_ratio` 仓提交 `27eaf29a`（write-reduce-with-notify）时，发现 AICPU 模式跨机用例因 `UDMA_OPC_SEND` WQE 未处理导致死锁和 Checker 失败，临时修复以使测试能跑通。

## 改动文件

| 文件 | 改动类型 |
|------|---------|
| `src/device_arm/proxy/device_sqe_parse_stub.cc` | 修改 |
| `src/device_arm/proxy/device_sqe_parse_stub.h` | 修改 |

## 改动详情

### 改动 1：新增 SEND 系列 opcode 处理

**文件:** `device_sqe_parse_stub.cc` — `ParseDavidUDMASqe` 函数的 switch 语句

**改动前:**

```cpp
switch (ubCommon->opcode) {
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE): {
        ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false);
        break;
    }
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY): {
        ParseDavidUBWriteWithNotifySqe(wqeAddr, streamId, jettyId);
        index++;
        break;
    }
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ): {
        ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, true);
        break;
    }
    default: {
        HCCL_VM_ERROR("not support opcode[{}].", static_cast<uint32_t>(ubCommon->opcode));
        return;
    }
}
```

**改动后:**

```cpp
switch (ubCommon->opcode) {
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_SEND):
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_SEND_WITH_IMM):
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_SEND_WITH_INVALID): {
        ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false, true);
        break;
    }
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE): {
        ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false);
        break;
    }
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_IMM):
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY): {
        ParseDavidUBWriteWithNotifySqe(wqeAddr, streamId, jettyId);
        index++;
        break;
    }
    case static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ): {
        ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, true);
        break;
    }
    default: {
        HCCL_VM_WARN("skip unsupported opcode[{}], continue parsing.", static_cast<uint32_t>(ubCommon->opcode));
        break;
    }
}
```

**具体变化:**

1. 新增 `UDMA_OPC_SEND`(0x0)、`UDMA_OPC_SEND_WITH_IMM`(0x1)、`UDMA_OPC_SEND_WITH_INVALID`(0x2) 三个 case，调用 `ParseDavidUBReadWriteSqe` 时传入 `isSend=true`
2. 新增 `UDMA_OPC_WRITE_WITH_IMM`(0x4) 归入 `WRITE_WITH_NOTIFY` 分支
3. default 分支：`HCCL_VM_ERROR` + `return` 改为 `HCCL_VM_WARN` + `break`

**原因:**

- 原代码 switch 仅处理 WRITE(0x3)、WRITE_WITH_NOTIFY(0x5)、READ(0x6) 三种 opcode
- 跨机 NHR 算法在 AICPU 模式下会产生 `UDMA_OPC_SEND`(0x0) WQE，落入 default 分支
- default 分支的 `return` 会**中断当前 jetty 队列中后续所有 WQE 的解析**，导致任务缺失和 Runner 死锁
- 改为 `break` 后跳过未知 opcode 但继续解析后续 WQE

---

### 改动 2：SEND WQE 的 rank 解析改用 EID 回退

**文件:** `device_sqe_parse_stub.cc` — `ParseDavidUBReadWriteSqe` 函数

**改动前:**

```cpp
void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead)
{
    // ...
    uint64_t srcOffset = isRead ? rmtAddr : locAddr;
    uint64_t dstOffset = isRead ? locAddr : rmtAddr;
    uint32_t srcRankId = GetRankIdByDevAddr(srcOffset);
    uint32_t dstRankId = GetRankIdByDevAddr(dstOffset);
    // ...
}
```

**改动后:**

```cpp
void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, bool isSend)
{
    // ...
    uint64_t srcOffset = isRead ? rmtAddr : locAddr;
    uint64_t dstOffset = isRead ? locAddr : rmtAddr;
    uint32_t srcRankId = isSend ? curRankId : GetRankIdByDevAddr(srcOffset);
    uint32_t dstRankId = isSend ? GetRmtRankIdByEid(ubWqe->comm.rmtEid[0]) : GetRankIdByDevAddr(dstOffset);
    // ...
}
```

**具体变化:**

- 函数签名新增 `bool isSend = false` 参数（默认 false，不影响原有 WRITE/READ 调用）
- 当 `isSend=true` 时：
  - `srcRankId` 直接用 `curRankId`（当前 rank，即发送方）
  - `dstRankId` 用 `GetRmtRankIdByEid(ubWqe->comm.rmtEid[0])`（通过远端 EID 查找接收方 rank）

**原因:**

- SEND WQE 的 `rmtAddr` 字段不是设备内存地址（在 `0x4000_xxxx_xxxx` 范围，不在已注册的 device 虚拟内存区间中）
- `GetRankIdByDevAddr` 查找失败时返回 0（误判为 rank 0），导致 src/dst rank 解析错误
- SEND WQE 的 `locAddr` 也可能不在设备内存映射中
- 因此对 SEND 不走地址解析，改用 `curRankId`（发送方确定）和 EID（接收方可从 WQE 的 `rmtEid` 字段获取）

---

### 改动 3：头文件函数签名同步

**文件:** `device_sqe_parse_stub.h`

**改动前:**

```cpp
void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead);
```

**改动后:**

```cpp
void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, bool isSend = false);
```

---

## UDMA SQE Opcode 枚举参考

```
定义位置: src/device_arm/proxy/udma_data_struct_stub.h

UDMA_OPC_SEND              = 0x0   ← 新增处理
UDMA_OPC_SEND_WITH_IMM     = 0x1   ← 新增处理
UDMA_OPC_SEND_WITH_INVALID = 0x2   ← 新增处理
UDMA_OPC_WRITE             = 0x3   ← 原有处理
UDMA_OPC_WRITE_WITH_IMM    = 0x4   ← 新增处理（归入 WRITE_WITH_NOTIFY）
UDMA_OPC_WRITE_WITH_NOTIFY = 0x5   ← 原有处理
UDMA_OPC_READ              = 0x6   ← 原有处理
UDMA_OPC_CAS               = 0x7   ← 未处理（default break 跳过）
UDMA_OPC_FAA               = 0xb   ← 未处理（default break 跳过）
UDMA_OPC_NOP               = 0x11  ← 未处理（default break 跳过）
UDMA_OPC_INVALID           = 0x12  ← 未处理（default break 跳过）
```

## 完整 diff

```diff
diff --git a/src/device_arm/proxy/device_sqe_parse_stub.cc b/src/device_arm/proxy/device_sqe_parse_stub.cc
index abb994b3..f424af5d 100644
--- a/src/device_arm/proxy/device_sqe_parse_stub.cc
+++ b/src/device_arm/proxy/device_sqe_parse_stub.cc
@@ -207,10 +207,17 @@ void ParseDavidUDMASqe(uint32_t streamId, void *sqeBuf)
         uint64_t wqeAddr = wqeBuffer + index * HCCL_WQE_SIZE;
         UdmaSqeCommon *ubCommon = reinterpret_cast<UdmaSqeCommon *>(wqeAddr);
         switch (ubCommon->opcode) {
+            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_SEND):
+            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_SEND_WITH_IMM):
+            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_SEND_WITH_INVALID): {
+                ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false, true);
+                break;
+            }
             case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE): {
                 ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false);
                 break;
             }
+            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_IMM):
             case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY): {
                 ParseDavidUBWriteWithNotifySqe(wqeAddr, streamId, jettyId);
                 index++;  // 占128字节两个WQE
@@ -221,14 +228,14 @@ void ParseDavidUDMASqe(uint32_t streamId, void *sqeBuf)
                 break;
             }
             default: {
-                HCCL_VM_ERROR("not support opcode[{}].", static_cast<uint32_t>(ubCommon->opcode));
-                return;
+                HCCL_VM_WARN("skip unsupported opcode[{}], continue parsing.", static_cast<uint32_t>(ubCommon->opcode));
+                break;
             }
         }
     }
 }
 
-void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead)
+void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, bool isSend)
 {
     UdmaSqeWrite *ubWqe = reinterpret_cast<UdmaSqeWrite *>(wqeAddr);
     HcclTaskMetaData taskMeta;
@@ -255,8 +262,8 @@ void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jett
     uint64_t rmtAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
     uint64_t srcOffset = isRead ? rmtAddr : locAddr;
     uint64_t dstOffset = isRead ? locAddr : rmtAddr;
-    uint32_t srcRankId = GetRankIdByDevAddr(srcOffset);
-    uint32_t dstRankId = GetRankIdByDevAddr(dstOffset);
+    uint32_t srcRankId = isSend ? curRankId : GetRankIdByDevAddr(srcOffset);
+    uint32_t dstRankId = isSend ? GetRmtRankIdByEid(ubWqe->comm.rmtEid[0]) : GetRankIdByDevAddr(dstOffset);
     taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
     taskMeta.taskData.transMem.srcOffset = srcOffset;
     taskMeta.taskData.transMem.dstOffset = dstOffset;
diff --git a/src/device_arm/proxy/device_sqe_parse_stub.h b/src/device_arm/proxy/device_sqe_parse_stub.h
index 3e2b5686..33f7b7fb 100644
--- a/src/device_arm/proxy/device_sqe_parse_stub.h
+++ b/src/device_arm/proxy/device_sqe_parse_stub.h
@@ -24,7 +24,7 @@ void ParseDavidNotifySqe(uint32_t streamId, void *sqeBuf, bool isPost);
 
 void ParseDavidUDMASqe(uint32_t streamId, void *sqeBuf);
 
-void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead);
+void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, bool isSend = false);
 
 void ParseDavidUBWriteWithNotifySqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId);
```

## 遗留问题

此修改为**临时权宜方案**，存在以下未解决问题：

1. **SEND WQE 的 `dstOffset` 仍使用 `rmtAddr`**：该地址不在设备内存映射中，Checker 的 buffer 范围追踪可能不准确
2. **SEND WQE 格式未确认**：当前按 `UdmaSqeWrite` 结构解析，但 SEND 的语义与 WRITE 不同，字段含义可能不同
3. **Checker 仍有失败用例**：4/10 用例完全失败，6/10 用例 Partial Success（syncIter=0 失败但 syncIter=1 通过），详见 `checker_failure_analysis.md`

完整修复需要与 hcomm 团队确认 `UDMA_OPC_SEND` WQE 的确切字段布局。

# HCCL-VM ReduceScatter 用例测试完整步骤

## 1. 环境准备

### 1.1 关键环境变量

```bash
# HCCL 源码路径（构建时需要）
export HCCL_CODE_HOME=/home/ytz/CANN/hccl

# CANN 工具链路径（已通过 set_env.sh 设置）
# export ASCEND_HOME_PATH=/home/ytz/CANN/Ascend/cann-9.1.0
# export ASCEND_TOOLKIT_HOME=/home/ytz/CANN/Ascend/cann-9.1.0

# 关键：禁用 hwloc 的 GL 和 OpenCL 组件，否则 mpirun 可能因硬件探测失败而报错
export HWLOC_COMPONENTS=-gl,-opencl
```

> **`HWLOC_COMPONENTS=-gl,-opencl` 是关键环境变量**。mpirun 启动时 hwloc 会探测硬件拓扑，在无 NPU 设备的虚拟环境中 GL 和 OpenCL 组件会探测失败导致异常，必须禁用。

### 1.2 确认依赖

```bash
# 确认 CANN 环境已加载
echo $ASCEND_HOME_PATH          # 应为 /home/ytz/CANN/Ascend/cann-9.1.0

# 确认测试二进制存在
ls ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test

# 确认 mpirun 可用
which mpirun                    # 应为 /usr/bin/mpirun

# 确认 hccl-vm 安装目录
ls /home/ytz/CANN/CheckerL2/hccl_vm_install/bin/hccl-vm
```

## 2. 构建 HCCL（修改算法代码后）

```bash
# 方式一：通过 build_pkg.sh 一键构建+安装+拷贝 device libs（需要 sudo）
export HCCL_CODE_HOME=/home/ytz/CANN/hccl
bash /home/ytz/CANN/CheckerL2/build_pkg.sh --install hccl

# 方式二：手动执行（无 sudo 时使用）
cd /home/ytz/CANN/hccl
bash build.sh --pkg --full                                              # 构建
./build_out/cann-hccl_9.1.0_linux-x86_64.run --full -q --pylocal \
    --install-path=/home/ytz/CANN/Ascend                                # 安装

# 拷贝 device libs 到 hccl_vm_install
SOURCE_DIR=/home/ytz/CANN/Ascend/cann/x86_64-linux/devlib/device
TARGET_DIR=/home/ytz/CANN/CheckerL2/hccl_vm_install/lib/aarch64
mkdir -p "$TARGET_DIR"
cp -fv "$SOURCE_DIR"/libascend_hal.so "$SOURCE_DIR"/libc_sec.so "$SOURCE_DIR"/libmmpa.so "$TARGET_DIR/"

# 解压 aicpu tar
HCCL_AICPU_TAR=$(find /home/ytz/CANN/hccl -name "aicpu_hccl.tar.gz" -type f | head -1)
tar -zxf "$HCCL_AICPU_TAR" -C "$TARGET_DIR" --strip-components=1
chmod -R 755 "$TARGET_DIR"
```

## 3. 清理环境

> **每次运行前必须清理**。hccl-vm 的 `envInit()` 使用 `sudo rm -fr` 清理 `/dev/shm`、`data/`、`logs/`，若 sudo 不可用则需手动清理。

```bash
# 清理共享内存（envInit 的 sudo 清理不可用时必须手动执行）
find /dev/shm -maxdepth 1 -type f -user $(whoami) -delete 2>/dev/null

# 清理 hccl-vm 数据和日志
rm -rf /home/ytz/CANN/CheckerL2/hccl_vm_install/data/*
rm -rf /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/*
```

## 4. 运行测试

### 4.1 关键说明

hccl-vm 的命令（`plugin install`、`mock-comm`、`plugin run @checker`）**必须在 `hccl-vm start` 创建的 `(hvm)$>` 交互式 shell 内执行**。直接在普通 shell 中运行这些命令会失败：
- 每个独立 `hccl-vm` 调用会执行 `envInit()` 清空所有数据
- `mock-comm` 依赖 `g_configClusterDir`，该变量只在 `start` 中设置
- mpirun 需要 `start` 设置的 `LD_PRELOAD`（代理库）才能拦截 HCCL 调用

### 4.2 通过管道执行（非交互式）

```bash
cd /home/ytz/CANN/CheckerL2/hccl_vm_install/bin

cat <<CMDS | ./hccl-vm start ascend950_cluster_1_server_hf > /tmp/hvm_session.log 2>&1
hccl-vm plugin install @runner
sleep 3
hccl-vm mock-comm 118
sleep 2
mpirun --allow-run-as-root --oversubscribe -np 8 \
    ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test \
    -b 512M -e 512M -d fp32 -o sum -w 0 -n 1 -c 1 \
    < /dev/null > /tmp/hccl_rs_8_fp32_512m_runner.log 2>&1
sleep 2
hccl-vm plugin run @checker
sleep 5
exit
CMDS
echo "EXIT CODE: $?"
```

### 4.3 命令说明

| 命令 | 作用 |
|------|------|
| `hccl-vm start ascend950_cluster_1_server_hf` | 启动仿真环境，fork 出 `(hvm)$>` bash，设置 `LD_PRELOAD` 代理库 |
| `hccl-vm plugin install @runner` | 安装 Runner 插件，拦截 HCCL 任务提交并模拟执行 |
| `hccl-vm mock-comm 118` | 选择通信域配置 `118.yaml`（1 server, 8 rank），生成 `topo.json` 和 `ranktable.json` |
| `mpirun ... reduce_scatter_test ...` | 运行 ReduceScatter 测试，`< /dev/null` 防止子进程吞掉管道中后续命令 |
| `hccl-vm plugin run @checker` | 运行 Checker V3，校验任务 DAG 的内存冲突和语义正确性 |
| `exit` | 退出 `(hvm)$>` shell，host 进程关闭并归档数据 |

### 4.4 mpirun 参数说明

| 参数 | 含义 |
|------|------|
| `--allow-run-as-root` | 允许 root 用户执行 mpirun |
| `--oversubscribe` | 允许过订阅（8 进程在同一机器上） |
| `-np 8` | 8 个进程（8 卡） |
| `-b 512M -e 512M` | 数据量从 512MB 到 512MB（固定） |
| `-d fp32` | 数据类型 fp32 |
| `-o sum` | reduce 操作为 sum |
| `-w 0` | warmup 次数 0 |
| `-n 1` | 执行 1 次 |
| `-c 1` | 检查结果 1 次 |

### 4.5 `< /dev/null` 的作用

mpirun 的 8 个子进程会继承父 shell 的 stdin（即管道）。若不加 `< /dev/null`，子进程会吞掉管道中的后续命令（`hccl-vm plugin run @checker` 和 `exit`），导致 checker 不执行。

## 5. 检查结果

### 5.1 mpirun 日志

```bash
# 检查 error
grep -c "\[error\]" /tmp/hccl_rs_8_fp32_512m_runner.log    # 应为 0

# 查看尾部（应为干净的 deinit 日志）
tail -n 20 /tmp/hccl_rs_8_fp32_512m_runner.log
```

### 5.2 Checker 日志

```bash
# 检查最终结果
grep -nE "Checker Success|Checker failed" \
    /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/checker/*.log

# 检查各阶段状态
grep -nE "stage finished" \
    /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/checker/*.log

# 检查内存冲突
grep -nE "parallelCandidatePairs" \
    /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/checker/*.log
```

### 5.3 预期输出（通过）

```
Checker Success              # 最终结果
parallelCandidatePairs=0     # 无并行内存冲突
```

### 5.4 预期输出（失败）

```
Checker failed
ErrorCode: 302               # 内存冲突
parallelCandidatePairs=1     # 存在并行写冲突
```

## 6. 拓扑配置说明

### 6.1 集群配置

`ascend950_cluster_1_server_hf`（路径: `config/cluster/ascend950_cluster_1_server_hf.yaml`）：
- 1 个超级节点，1 个服务器
- 服务器拓扑: `ascend950_server_topo_hf.yaml`
- 每服务器 16 个 device（HF 全互联），测试使用其中 8 个

### 6.2 通信域配置

`118`（路径: `config/topo_meta/118.yaml`）：
- 1 pod, 1 server, 8 rank
- `ranks: [0, 1, 2, 3, 4, 5, 6, 7]`

### 6.3 算法选择

| 数据量 | 每卡数据 | 选择算法 | 原因 |
|--------|---------|---------|------|
| 32MB | 4MB | `InsReduceScatterMesh1D` | `dataSize × ratio < 16MB` |
| 512MB | 64MB | `InsReduceScatterMesh1DMeshChunk` | `dataSize × ratio > 16MB`，触发子块流水线 |

阈值: `RS_AICPU_1D_MAX_DATA_SIZE = 16MB`（定义于 `reduce_scatter_auto_selector.cc`）

## 7. 完整一键脚本

```bash
#!/bin/bash
set -e

export HCCL_CODE_HOME=/home/ytz/CANN/hccl
export HWLOC_COMPONENTS=-gl,-opencl

HCCL_VM_BIN=/home/ytz/CANN/CheckerL2/hccl_vm_install/bin
CLUSTER=ascend950_cluster_1_server_hf
COMM_DOMAIN=118
DATA_SIZE=512M
LOG_FILE=/tmp/hccl_rs_8_fp32_${DATA_SIZE}_runner.log

# 1. 清理
find /dev/shm -maxdepth 1 -type f -user $(whoami) -delete 2>/dev/null
rm -rf /home/ytz/CANN/CheckerL2/hccl_vm_install/data/* /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/*

# 2. 运行
cd "$HCCL_VM_BIN"
cat <<CMDS | ./hccl-vm start ${CLUSTER} > /tmp/hvm_session.log 2>&1
hccl-vm plugin install @runner
sleep 3
hccl-vm mock-comm ${COMM_DOMAIN}
sleep 2
mpirun --allow-run-as-root --oversubscribe -np 8 \
    \${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test \
    -b ${DATA_SIZE} -e ${DATA_SIZE} -d fp32 -o sum -w 0 -n 1 -c 1 \
    < /dev/null > ${LOG_FILE} 2>&1
sleep 2
hccl-vm plugin run @checker
sleep 5
exit
CMDS

# 3. 检查结果
echo "=== mpirun errors: $(grep -c '\[error\]' ${LOG_FILE}) ==="
grep -nE "Checker Success|Checker failed" \
    /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/checker/*.log
grep -nE "parallelCandidatePairs" \
    /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/checker/*.log
```

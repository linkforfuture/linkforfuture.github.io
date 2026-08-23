# CheckerL2

执行 ReduceScatter 用例的完整顺序如下，包含 runner：

cd /home/ytz/CANN/CheckerL2/hccl_vm_install/bin

source /home/ytz/CANN/Ascend/cann-9.1.0/set_env.sh

export HWLOC_COMPONENTS=-gl,-opencl

  ./hccl-vm start ascend950_cluster_4_server_normal.yaml

  进入 (hvm) 环境后依次执行：

安装实际数据执行插件；需在 mock-comm 和 mpirun 前完成

  hccl-vm plugin install @runner

  # 注入 8 卡拓扑（118.yaml）
  hccl-vm mock-comm 118

  # fp32、32MB、8 卡、SUM、执行一次
  mpirun --allow-run-as-root --oversubscribe -np 8 \
    ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test \
    -b 32M -e 32M -d fp32 -o sum -w 0 -n 1 -c 1 \
    > /tmp/hccl_rs_8_fp32_32m_runner.log 2>&1

  # 校验任务图、内存冲突和语义
  hccl-vm plugin run @checker

  查看结果：

  tail -n 80 /tmp/hccl_rs_8_fp32_32m_runner.log

  rg -n "failed|error|ERROR|HCCL_E|Checker Success|Checker Failed" \
    /tmp/hccl_rs_8_fp32_32m_runner.log \
    /home/ytz/CANN/CheckerL2/hccl_vm_install/logs/checker

  最后退出并归档本次运行日志：

  exit

  关键顺序是：启动 VM -> 安装 @runner -> mock-comm 118 -> mpirun -> @checker。
  HWLOC_COMPONENTS=-gl,-opencl 用于规避当前机器上 OpenMPI 初始化 hwloc 图形/OpenCL 组件时卡
  住的问题。



用 OpenMPI 路径编译 CANN 里的 hccl_test：

  cd /home/ytz/CANN/Ascend/cann-9.1.0/tools/hccl_test
  source /home/ytz/CANN/Ascend/cann-9.1.0/set_env.sh
  make clean
  make MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi ASCEND_DIR=${ASCEND_HOME_PATH}
- [← 返回主页](https://linkforfuture.github.io/)

- 首页
  - [关于本站](README.md)

- 需求文档（SRS/SD）
  - [AllGather NHR 减少 LocalCopy](srs-sd/allgather-nhr-reduce-localcopy-srs-sd.md)
  - [Parallel 数据切分比例配置](srs-sd/parallel-split-ratio-srs-sd.md)
  - [WriteReduceWithNotify 需求](srs-sd/write-reduce-with-notify-srs-sd.md)
  - [任务超时时间控制](srs-sd/task-timeout-control-srs-sd.md)
  - [TopoMatch 三级降两级](srs-sd/topo-match-multilevel-3to2-srs-sd.md)
  - [TopoParse Match](srs-sd/topo-parse-match-srs-sd.md)

- 设计 / 方案
  - [ParallelExecutor 多维切分系统设计](design/parallel-executor-split-design.md)
  - [ParallelExecutor 数据切分设计文档](design/parallel-executor-split-design-doc.md)
  - [WriteReduceWithNotify 改造设计](design/write-reduce-with-notify-refactor.md)
  - [aclrtGetDevice 逻辑设备ID转换](design/aclrt-get-device-id-conversion.md)
  - [HCCL 多维度切分比例算法](design/hccl-alg-multiple-dimension-split-ratio.md)
  - [AICPU 64bit Prod Reduce 设计](design/aicpu-64bit-prod-reduce-design.md)
  - [AllGather NHR 末步优化](design/allgather-nhr-last-step-optimization.md)
  - [HCCL 架构入口分析](design/hccl-architecture-entry.md)

- 拓扑（Topo）
  - [Link Binding 实现计划](topo/link-binding-impl-plan.md)
  - [Link Binding 重构设计](topo/link-binding-refactor-design.md)
  - [物理层级设计](topo/physical-levels-design.md)
  - [TopoMatch 实现计划](topo/topo-match-impl-plan.md)
  - [TopoMatch 重构设计](topo/topo-match-refactor-design.md)
  - [TopoParse 实现计划](topo/topo-parse-impl-plan.md)
  - [TopoParse 重构设计](topo/topo-parse-refactor-design.md)
  - [TopoParse 虚拟设计](topo/topo-parse-virtual-design.md)
  - [TopoMatch 当前行为分析](topo/topo-match-current-behavior.md)
  - [跨 Pod 拓扑分析](topo/cross-pod-topology-analysis.md)
  - [设备形态因子分析](topo/device-form-factor.md)

- 分析报告
  - [MeshChunk 难复现同步问题](analysis/meshchunk-sync-issue.md)
  - [AllGather NHR 优化分析](analysis/allgather-nhr-optimization-analysis.md)
  - [ParallelExecutor 切分审查报告](analysis/parallel-executor-split-review.md)
  - [AICPU 执行超时分析](analysis/aicpu-exec-timeout-analysis.md)
  - [AllGather NHR 空channels崩溃](analysis/allgather-nhr-rank1-empty-channels.md)
  - [Checker Barrier 分析](analysis/checker-barrier-analysis.md)
  - [AllGather NHR 回归测试计划](analysis/allgather-nhr-regression-test-plan.md)
  - [Checker L2 执行分析](analysis/checker-l2-execution.md)
  - [Checker 失败分析](analysis/checker-failure-analysis.md)
  - [HCCL VM SQE Parse Stub 改动](analysis/hccl-vm-sqe-parse-stub.md)
  - [HCCL VM 测试指南](analysis/hccl-vm-test-guide.md)
  - [MeshChunk Barrier 问题](analysis/meshchunk-barrier-issue.md)
  - [RS 512MB 总结](analysis/rs-512mb-summary.md)

- HCOMM 分析
  - [HCOMM Channel 连接流程](hcomm/hcomm-channel-connection-flow.md)
  - [HCOMM Write 数据流](hcomm/hcomm-write-data-flow.md)

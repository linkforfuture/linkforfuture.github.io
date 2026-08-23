# ReduceScatter 512MB 测试结果（Checker 通过）

## 测试配置

- 算法: `InsTempReduceScatterMesh1DMeshChunk`（已修复步间屏障顺序：PostSync→PreSync）
- 拓扑: ascend950_cluster_1_server_hf, 8 rank
- 数据: 512MB, fp32, reduce op=sum, 1 iteration
- 日期: 2026-07-14
- 构建: `build_pkg.sh --install hccl`（通过 build.sh --pkg --full + .run 安装 + device libs 拷贝）

## mpirun 结果

| 指标 | 值 |
|------|-----|
| 总行数 | 228,544 |
| `[error]` | 0 |
| `[warning]` | 7,503 |

mpirun 侧无 error，测试正常运行并完成。

## Checker V3 结果

| 阶段 | op[0] | op[1] |
|------|-------|-------|
| GenGraph | success (6672 nodes, 28ms) | success (26ms) |
| SingleTaskCheck | success (816 data task nodes, 9ms) | success (8ms) |
| MemConflict | **success** (0 parallel conflicts, 53ms) | **success** (0 parallel conflicts, 54ms) |
| SemanticCheck | success (80 semantics, 5.3GB, 15ms) | success (80 semantics, 15ms) |
| **最终结果** | **Checker Success** | **Checker Success** |

### 关键指标对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| parallelCandidatePairs | 1 | **0** |
| orderedCandidatePairs | 2 | **7600** |
| SemanticCheck | 跳过 | **success** |
| 最终结果 | Checker failed | **Checker Success** |

### MemConflict 统计

```
nodeCount=6673, dataTaskNodeCount=816, processedDataTaskNodeCount=816
accessIntervalCount=1632, memoryBucketCount=24
overlapCandidatePairs=7600, orderedCandidatePairs=7600, parallelCandidatePairs=0
```

所有 7600 对重叠内存访问均有序，无并行冲突。

### SemanticCheck 统计

```
handledNodeCount=6673, rankCount=8
normalSemanticCount=80, normalSemanticBytes=5309988864 (≈4.94GB)
```

## 结论

算法步间屏障修复（`PostSync→PreSync`）经 `build_pkg.sh` 完整构建安装后，checker 全部通过：
- MemConflict: 0 个并行冲突（此前为 1）
- SemanticCheck: 80 个语义检查全部成功
- 两个 op 均为 **Checker Success**

## 文件清单

| 文件 | 说明 |
|------|------|
| `hccl_rs_8_fp32_512m_runner.log` | mpirun 完整日志 |
| `checker.log` | checker 完整日志 |
| `hvm_session.log` | hvm 会话日志 |
| `rs_512mb_summary.md` | 本总结 |

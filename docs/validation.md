# 验证证据与边界

项目当前把证据分成三条线：

1. **v0.26 端到端 GPU 实验**：用于证明 HOL blocking、bypass 收益与 fairness / throughput 反例；
2. **v0.29 Trace 工程验证**：用于证明当前 Scheduler / MRV2 Trace 的 schema、热路径接入和跨进程 correlation contract；
3. **upstream fairness / revocable-backfill**：用于验证 Scheduler 语义边界与窄范围策略 prototype。

这三类证据不能互相替代。尤其是 **v0.26 GPU 性能结果不能直接作为 v0.29 Trace 的性能结论**。

## v0.29 Trace 当前验证

当前实现分支：[`port/scheduler-trace-v029`](https://github.com/Xiaoda11/vllm/tree/port/scheduler-trace-v029)。

截至 2026-09-14：

| 验证层级 | 内容 | 结果 |
|---|---|---:|
| Isolated trace CPU suite | writer/schema、Scheduler event、MRV2 batch、TP shard、静态 no-sync contract | **22 passed** |
| Real Scheduler / IPC CPU suite | real Scheduler、KV allocation failure、preemption、MessageQueue、WorkerProc RPC、sync KV-load mock path | **8 passed** |
| Ruff / typos | migration checkpoint 的定向检查 | 通过 |
| Full pre-commit | markdownlint 环境安装受 npm `EALLOWGIT` 阻塞 | 未完整完成 |
| Real GPU engine | Scheduler ↔ MRV2 live join、TP/PP/PCP、多 worker 文件 | 待验证 |
| v0.29 Trace overhead | trace-off/on real-model GPU 对照 | 待验证 |

### 已验证的关键 contract

- Trace disabled 时普通 Scheduler path 不附加 trace-only correlation 属性；
- real WAITING → RUNNING prefill 时可以记录 KV allocation 与 Scheduler step；
- step correlation ID 可以穿过真实 vLLM `MessageQueue` 的跨进程序列化；
- 同一个 RPC tuple 能进入生产 `WorkerProc._execute_worker_rpc()` dispatch path 并保持 scheduled-token map 与 step ID；
- real KV pressure 下可以观测 `RUNNING -> PREEMPTED`、preempted request ID 与 block 变化；
- mock synchronous KV connector 下，`SchedulerOutput.has_sync_kv_loads` 与 trace field 一致；
- MRV2 event 区分 `scheduler_logical`、`runner_effective_unpadded`、`model_input_after_padding`；
- TP sampler shard 的 persistent-row ownership 与 per-rank logit split 有 CPU contract test；
- trace event 参数路径有静态约束，避免新增 `.item()`、`.cpu()`、`.tolist()`、`synchronize()` 形式的 GPU readback / sync。

这些结果证明当前 v0.29 Trace 的 **CPU 语义、真实 Scheduler 接入和 tested IPC / RPC transport contract**，但不等于完整 GPU engine 已验证。

## v0.26 已完成的行为与性能验证

v0.26 是当前唯一具有完整单卡 GPU benchmark 证据的版本。

| 层级 | 验证内容 | 结果 |
|---|---|---|
| Trace / workload / analyzer lab tests | writer、schema、工作负载、分析脚本 | 52 项通过 |
| MRV2 跨层校验 | 44 个 forward step 的 token 对齐 | 无不一致 |
| strict / bypass benchmark | HOL、TTFT、makespan、throughput、公平性 | 完成 |
| Scheduler-aligned profiler | batch composition 变化 | 完成 |
| Nsight Compute | targeted kernel counter | 完成 |

### v0.26 关键行为证据

无界 bypass 三请求重复 Gate：

- C TTFT median：`33.250 s -> 0.182 s`；

短请求 burst 反例：

- blocked head first scheduled step：`517 -> 587`；
- blocked head TTFT：约 `+10%`；

bounded 长生命周期反例：

- 新增 3 次 preemption；
- makespan `+1.21%`；
- throughput `-1.20%`；
- TTFT Jain `-10.64%`。

因此结论不是“bypass 越多越好”，而是：

> **retry-first 不等于保护 blocked head 的 future KV-admission opportunity。**

这条 fairness invariant 后来被收缩成 deterministic Scheduler reproduction，并进入 upstream PR #33499 的设计讨论。

## Revocable-backfill prototype 验证

当前 prototype 分支：[`test/pr33499-revocable-backfill-regression`](https://github.com/Xiaoda11/vllm/tree/test/pr33499-revocable-backfill-regression)。

focused CPU regression 使用 4 个可分配 KV blocks 构造：

- incumbent 持有部分 KV；
- older heavy request 需要全部可分配 blocks，因此被 blocked；
- younger backfill 可以先被准入；
- incumbent 释放后，只有撤销 tracked backfill 才能让 heavy 立即满足准入条件。

测试同时约束：

- 只有“确实 bypass 过该 protected head”的 request 才能成为 reclaim candidate；
- ordinary running request 不得被推断成 revocation victim；
- reclaim 不足以使 protected head 可准入时，不应抢占 backfill；
- backfill 完成 / 退出后 tracking 必须清理。

prototype 当前刻意限制在 narrow local KV-only FCFS path，并加上 single concurrent batch / single KV group 等 guard。它是策略方向验证，不是可直接宣称 production-ready 的通用 Scheduler policy。

## 总体证据边界

当前可以较强地声称：

- v0.26 实验真实复现了 KV-pressure 下的 HOL blocking，并量化了 bypass 的收益与反例；
- fairness 问题已被压缩成 deterministic Scheduler semantic case；
- v0.29 Trace 已完成比 v0.28 更深入的 CPU contract / real Scheduler / IPC / WorkerProc dispatch 验证；
- 当前工程明确区分 Scheduler decision、runner trimming、CUDA Graph padding 和 TP shard ownership。

当前不能声称：

- v0.29 已做完整 real-model GPU end-to-end；
- v0.29 Trace overhead 已有可靠性能数据；
- real TP/PP/PCP / Ray / KV network transport 已全部覆盖；
- revocable backfill 是 upstream 已接受或 production-safe 的最终方案。

更详细的当前迁移证据见 [`v029_migration.md`](v029_migration.md)，upstream fairness 过程见 [`upstream_fairness_review.md`](upstream_fairness_review.md)。

# Source of Truth / 展示仓库同步规则

本展示仓库只负责解释设计、实验、验证证据和 upstream 协作过程；**实现状态以 `Xiaoda11/vllm` 中对应分支为唯一事实源**。当源码分支前进时，展示仓库不得继续引用旧版本状态作为“当前实现”。

## 当前事实源（2026-09-16）

| 工程线 | 唯一事实源 | 当前分支头 | 展示仓库允许声称的范围 |
|---|---|---|---|
| v0.29 Scheduler / MRV2 Trace | `Xiaoda11/vllm:port/scheduler-trace-v029` | `bb03a6e3702c88803a1074825ec4d84ba139dfb0` | Scheduler trace、MRV2 batch trace、TP sampler-shard trace、process-local MRV2 JSONL、Scheduler→MessageQueue→WorkerProc correlation；isolated CPU 22 passed + real Scheduler/IPC CPU 8 passed；不宣称 real-model GPU E2E / overhead 已验证 |
| fairness / revocable-backfill prototype | `Xiaoda11/vllm:test/pr33499-revocable-backfill-regression` | `71cba372641121f3a87387e678e310eb0bc2d9fa` | deterministic fairness boundary、tracked backfill reclaim prototype、focused CPU regression contracts；不宣称 production-ready 或 upstream 已接受 |
| v0.26 GPU 实验基线 | `Xiaoda11/vllm:exp/mrv2-scheduler-trace` 及展示仓库历史结果 | 历史固定实验线 | HOL blocking、strict/bypass/bounded 的 GPU benchmark 与 profiler 证据；不能外推成 v0.29 性能结论 |

## v0.29 Trace 必须与源码保持一致的关键事实

当前 `port/scheduler-trace-v029` 的设计/迁移记录明确包含：

- opt-in background JSONL writer，Trace 默认关闭；
- Scheduler `scheduler_step` 事件：双预算、queue/request snapshot、KV block mapping diff、allocation failure、preemption、finished IDs；
- v0.29 KV connector：`has_sync_kv_loads` 与 offered block state；
- `model_runner_batch`：区分 `scheduler_logical`、`runner_effective_unpadded`、`model_input_after_padding`；
- `sampler_batch_shard`：TP batch-sharded sampling 的 persistent-row ownership 与 per-rank split；
- 每个 MRV2 worker 使用 process-local JSONL，并记录 global `worker_rank`；
- Scheduler correlation ID 只在 Trace 开启时附加，且以 per-`InputBatch` 方式穿过 worker 路径；
- isolated trace CPU suite：22 passed；real Scheduler / IPC CPU suite：8 passed；
- 未完成 real-model GPU engine、真实 TP/PP/PCP E2E、真实 KV network transport 和 v0.29 overhead 对照。

## Revocable-backfill 必须与源码保持一致的关键事实

当前 `test/pr33499-revocable-backfill-regression` 只验证一个窄策略原型：

- blocked FCFS head 被保护；
- 只有确实在该 head 被 KV-blocked 时绕过并准入的 younger request 才进入 tracking；
- 只有当 reclaim 某 tracked backfill 的 KV 足以使 protected head 立即满足准入条件时，才允许通过正常 preemption 路径撤销它；
- ordinary running request 不能被推断成 revocation victim；
- reclaim 不足时不能抢占；
- finished/removed backfill tracking 必须清理；
- prototype 加有 narrow guards（local KV-only、single concurrent batch、single KV group 等），因此不是通用 production policy。

核心 fairness invariant 仍然是：

> **retry-first != protecting the blocked head's future KV-admission opportunity.**

## 展示仓库更新规则

以后每次 `Xiaoda11/vllm` 的上述分支发生实质更新，应先核对源码/测试，再更新展示仓库。至少同步以下项目：

1. 分支头 commit；
2. 当前实现范围；
3. 已通过的测试层级与数量；
4. 新增或删除的事件/schema 字段；
5. remaining gates / 尚未验证项；
6. fairness prototype 的 guard、测试契约和结论边界。

如果展示仓库与源码发生冲突，**以 `Xiaoda11/vllm` 当前分支代码和其分支内验证记录为准**，展示仓库应被视为需要更新，而不是反过来解释源码。

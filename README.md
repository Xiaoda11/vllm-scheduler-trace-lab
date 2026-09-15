# vLLM Scheduler Trace Lab

面向 LLM Serving 的 Scheduler 可观测性、KV-pressure HOL blocking 与 fairness 实验项目。

**本仓库是展示/研究仓库，不是实现事实源。当前实现状态始终以 `Xiaoda11/vllm` 对应源码分支为准。** 详细同步规则见 [`docs/source_of_truth.md`](docs/source_of_truth.md)。

## 当前源码事实源（2026-09-16）

| 工程线 | 源码分支 | 当前分支头 | 当前证据边界 |
|---|---|---|---|
| **v0.29 Scheduler / MRV2 Trace** | [`port/scheduler-trace-v029`](https://github.com/Xiaoda11/vllm/tree/port/scheduler-trace-v029) | `bb03a6e3702c88803a1074825ec4d84ba139dfb0` | isolated CPU **22 passed**；real Scheduler / IPC CPU **8 passed**；real-model GPU E2E / overhead 待验证 |
| **Fairness / revocable-backfill prototype** | [`test/pr33499-revocable-backfill-regression`](https://github.com/Xiaoda11/vllm/tree/test/pr33499-revocable-backfill-regression) | `71cba372641121f3a87387e678e310eb0bc2d9fa` | deterministic fairness case + narrow reclaim prototype；不是 production-ready policy |
| **v0.26 GPU 实验基线** | 历史 `exp/mrv2-scheduler-trace` 实验线 | 历史基线 | HOL / bypass / fairness / profiling 有单卡 GPU 证据；不能当作 v0.29 性能结果 |

> **核心 fairness invariant：retry-first != protecting the blocked head's future KV-admission opportunity.**

队首请求每个 iteration 先重试，并不意味着它未来的 KV 准入机会被保护。此前已绕过它、进入 running 的 younger request 仍可能长期持有 KV，从而继续阻塞 protected head。

## 一条主故事，而不是三个互不相关的项目

```text
1. 先发现问题
   v0.26 strict FCFS + KV pressure
   -> blocked head 申请 KV 失败
   -> 后面的短请求其实放得下，但 scheduler 停止扫描
   -> HOL blocking

2. 先建立观测能力
   Scheduler Trace + MRV2 Trace
   -> queue / request / KV / token budget
   -> SchedulerOutput
   -> Model Runner batch
   -> 按 step 对齐

3. 再做策略实验
   strict -> bypass -> bounded bypass
   -> younger TTFT 大幅改善
   -> 但 blocked head 更晚 / preemption / fairness 退化
   -> naive bypass 不能作为最终答案

4. 把问题缩成确定性语义
   upstream PR #33499 fairness review
   -> queue retry priority != future KV-admission protection
   -> deterministic Scheduler testcase

5. 继续验证更窄的策略方向
   revocable backfill
   -> 只追踪真正绕过 protected head 的 younger work
   -> reclaim 足以解锁 head 时才撤销
   -> ordinary running request 不能被误伤

并行工程线：
v0.26 Trace -> v0.28 migration -> v0.29 Scheduler / MRV2 / IPC / TP trace
```

## v0.29 Trace 当前实现

### Scheduler side

`scheduler_step` 记录：

- running / waiting / skipped-waiting before / after；
- request 状态与 block table snapshot；
- per-request scheduled tokens；
- `token_budget` 与 `input_budget`；
- KV usage / free blocks / request block mapping diff；
- prefix hit、allocation failure、preemption、finished request；
- v0.29 KV connector `has_sync_kv_loads` 与 offered block state。

Trace 默认关闭。开启后只使用已有 CPU/Python metadata 做 side-band instrumentation，不为了 trace 主动读取 GPU tensor，也不增加显式 CUDA synchronization。

### Scheduler -> Worker correlation

Trace 开启时，SchedulerOutput 附加 step correlation ID，并在 tested CPU path 中穿过：

```text
SchedulerOutput
  -> MessageQueue cross-process serialization
  -> WorkerProc._execute_worker_rpc()
  -> worker.execute_model(...)
```

correlation ID 使用 per-`InputBatch` 保存，避免 overlapping batch 时 runner-global slot 被覆盖。

### MRV2 / Model Runner side

`model_runner_batch` 在 v0.29 明确区分：

```text
scheduler_logical
  -> CPU-visible trimming / adaptive verification
runner_effective_unpadded
  -> CUDA Graph padding
model_input_after_padding
```

因此 runner trimming 和 graph padding 不会被误解释成 Scheduler 决策。

此外，TP batch-sharded sampling 增加 `sampler_batch_shard` event，记录 persistent-row ownership、local request shard、per-rank request count 与 logit split；不把 GPU-only gather/sort plan tensors 拉回 CPU。

每个 MRV2 worker 使用独立 JSONL：

```text
<trace-stem>.mrv2.pid<PID>.jsonl
```

事件携带 global `worker_rank`，用于后续按 step / worker 做语义合并。

详细设计见 [`docs/trace_design.md`](docs/trace_design.md)，完整 v0.29 迁移与验证见 [`docs/v029_migration.md`](docs/v029_migration.md)。

## v0.26：从 HOL 收益到 fairness 反例

历史单卡实验得到过三类关键结果：

| 实验 | 观察 | 含义 |
|---|---:|---|
| 三请求无界 bypass | C TTFT median `33.250 s -> 0.182 s` | HOL 局部收益明确 |
| younger burst | blocked head first step `517 -> 587`，TTFT 约 `+10%` | retry-first 不是 starvation bound |
| bounded 长生命周期反例 | `+3` preemption，makespan `+1.21%`，throughput `-1.20%`，TTFT Jain `-10.64%` | admission count 不能限制 KV 生命周期 |

因此项目没有把 bypass 包装成最终方案，而是继续追问：**如何允许闲置 capacity 被 younger work 使用，同时不永久消耗 blocked head 的未来准入机会？**

## Upstream fairness review

围绕 vLLM PR #33499，把 wall-clock 现象缩成 deterministic Scheduler case：

```text
4 allocatable KV blocks

incumbent holds 2
heavy older request needs 4 -> blocked
backfill younger request needs 1 -> admitted

incumbent exits -> 3 free
heavy retried first but still needs 4
backfill still owns 1

=> retry-first, but heavy still cannot enter
```

这直接证明：

```text
queue retry priority
!=
future KV-admission protection
```

完整过程见 [`docs/upstream_fairness_review.md`](docs/upstream_fairness_review.md)。

## Revocable-backfill prototype

当前源码原型只处理窄范围 FCFS / local-KV 场景：

1. blocked head 进入 protected tracking；
2. 记录哪些 younger requests **确实在它被 KV-blocked 时被准入**；
3. 未来若 `free KV + tracked backfill KV` 足以让 protected head 立即满足准入，则可以通过正常 preemption path 撤销该 backfill；
4. ordinary running request 不能被推断成 reclaim victim；
5. reclaim 不足以解锁 head 时不能抢占；
6. backfill 完成或退出后 tracking 必须清理。

源码目前还加有 single concurrent batch、single KV group 等 guard，所以应描述成 **fairness invariant + narrow revocable-backfill prototype**，而不是“解决了 vLLM starvation”。

## 当前验证边界

当前可以声称：

- v0.26 已有真实单卡实验，复现 HOL 并量化 bypass 收益与 fairness 反例；
- fairness 问题已缩成 deterministic Scheduler semantic case；
- v0.29 Trace 已验证 Scheduler schema/热路径、MRV2 event、TP shard contract、真实 Scheduler CPU 行为、MessageQueue 跨进程 transport 和 WorkerProc RPC dispatch；
- 当前明确区分 Scheduler decision、runner trimming 与 CUDA Graph padding。

当前不能声称：

- v0.29 已做完整 real-model GPU end-to-end；
- v0.29 Trace overhead 已有可靠 GPU 对照；
- real TP/PP/PCP、Ray 或真实 KV network transport 全部验证；
- revocable backfill 已被 upstream 接受或是 production-safe 最终方案。

## 面试 / Code Review 阅读顺序

1. 本 README：先讲问题与研究路径；
2. [`docs/trace_design.md`](docs/trace_design.md)：讲 v0.29 数据流；
3. [`docs/v029_migration.md`](docs/v029_migration.md)：讲 migration、CPU validation 和 remaining gates；
4. [`docs/upstream_fairness_review.md`](docs/upstream_fairness_review.md)：讲 fairness invariant；
5. [`docs/validation.md`](docs/validation.md)：核对证据强弱；
6. 再进入 `Xiaoda11/vllm` 当前源码分支看实现。

源码重点：

- [`trace.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/trace.py)
- [`trace_event.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/trace_event.py)
- [`scheduler.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/scheduler.py)
- [`model_runner_trace_event.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/model_runner_trace_event.py)
- [`GPUModelRunner`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/worker/gpu/model_runner.py)
- [`revocable-backfill regression`](https://github.com/Xiaoda11/vllm/blob/test/pr33499-revocable-backfill-regression/tests/v1/core/test_scheduler_revocable_backfill_regression.py)

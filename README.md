# vLLM Scheduler Trace Lab

面向 LLM Serving 的调度可观测性、KV-pressure fairness 与策略实验项目。

项目从 vLLM v0.26 的 Scheduler / MRV2 跨层 Trace 与 HOL blocking 实验开始，经历 v0.28 迁移后，当前 Trace 主线已经迁移到 **vLLM v0.29**；同时围绕 upstream PR #33499 将 fairness boundary 收缩成 deterministic Scheduler case，并继续验证 bounded / revocable-backfill 方向。

> **当前核心结论：retry-first ≠ protecting the blocked head's future KV-admission opportunity。**
>
> 队首请求每轮优先重试，并不代表它未来的 KV 准入机会受到保护；此前已准入的 younger request 仍可能跨 iteration 持有 KV，使 blocked head 长时间无法进入。

展示仓库负责设计、实验、验证证据与 upstream 协作记录；当前实现与测试维护在 [`Xiaoda11/vllm`](https://github.com/Xiaoda11/vllm)。

## 当前状态（2026-09-16）

| 阶段 | 当前状态 | 证据边界 |
|---|---|---|
| **v0.29 Scheduler Trace** | Scheduler + MRV2 + TP sampler-shard trace 已迁移 | isolated CPU **22 passed**；real Scheduler / IPC CPU **8 passed** |
| **v0.29 跨进程 correlation** | step ID 已验证穿过真实 MessageQueue 与 `WorkerProc._execute_worker_rpc()` | 尚未完成完整 real-model worker lifecycle |
| **v0.29 KV connector trace** | sync KV-load 标记与 offered block state 已接入 | mock connector 验证；非真实网络 transfer |
| **v0.29 多 worker 设计** | process-local MRV2 JSONL + `worker_rank` | contract-tested；真实 TP/PP/PCP GPU 待验证 |
| **v0.26 实验基线** | HOL、bypass、fairness、profiling 均有单卡 GPU 数据 | 旧版性能结果不能直接代表 v0.29 |
| **Upstream fairness review** | PR #33499 deterministic reproduction + semantic review | upstream 设计讨论，不宣称 merge ownership |
| **Revocable backfill prototype** | narrow FCFS / local-KV reclaim prototype + focused CPU regression | 仍是实验策略，不是 production-ready policy |

当前 v0.29 Trace 分支：[`port/scheduler-trace-v029`](https://github.com/Xiaoda11/vllm/tree/port/scheduler-trace-v029)，当前记录的分支头为 `bb03a6e3702c88803a1074825ec4d84ba139dfb0`。

Revocable-backfill 实验分支：[`test/pr33499-revocable-backfill-regression`](https://github.com/Xiaoda11/vllm/tree/test/pr33499-revocable-backfill-regression)。

## 项目主线

```text
v0.26: 先建立 Scheduler <-> MRV2 Trace
              ↓
       复现 KV-pressure HOL blocking
              ↓
       strict vs bypass 实验
              ↓
       发现 fairness / preemption 反例
              ↓
upstream PR #33499 fairness review
              ↓
 deterministic Scheduler reduction
              ↓
 bounded / revocable-backfill direction

同时：
v0.26 Trace -> v0.28 migration -> v0.29 Scheduler/MRV2/IPC/TP trace
```

## v0.29 Trace：当前工程重点

### 1. Scheduler step trace

Scheduler 侧记录：

- running / waiting / skipped-waiting before / after；
- request 状态与 block table；
- per-request scheduled tokens；
- `token_budget` 与 `input_budget`；
- KV usage、free blocks、block mapping diff；
- prefix-cache hit；
- allocation failure、preemption、finished request；
- v0.29 `has_sync_kv_loads` 与 connector offered block state。

Trace 默认关闭。开启时使用已有 CPU/Python metadata 构造事件，不为了观测主动读取 GPU tensor 内容，也不引入显式 CUDA synchronization。

### 2. MRV2 batch trace

v0.29 不再把“Scheduler token 数”和“真正 model input token 数”当作同一个量。

当前 `model_runner_batch` 明确区分：

```text
scheduler_logical
    ↓ CPU-visible trimming / adaptive verification
runner_effective_unpadded
    ↓ CUDA Graph padding
model_input_after_padding
```

这使 Trace 可以解释 batch shape 是在哪一层发生变化，而不是把 runner / graph 行为错误归因给 Scheduler。

### 3. Scheduler → Worker correlation

Trace step ID 随 `SchedulerOutput` 穿过 vLLM MessageQueue / WorkerProc RPC，并保存在对应 `InputBatch` 上。

CPU CI 已验证：

```text
SchedulerOutput
    ↓ pickle / MessageQueue cross-process
WorkerProc._execute_worker_rpc()
    ↓
worker.execute_model(...)
```

correlation ID 和 scheduled-token map 在 tested path 中保持一致。

### 4. TP batch-sharded sampling

新增 `sampler_batch_shard` event，记录：

- global / local request layout；
- persistent rows；
- `persistent_row % tp_size` ownership；
- per-rank request count；
- per-rank logit split。

不会为了 Trace 把 GPU-only gather / sort plan tensor 拉回 CPU。

### 5. 多 worker JSONL

每个 MRV2 worker 使用独立输出文件：

```text
<trace-stem>.mrv2.pid<PID>.jsonl
```

事件同时包含 global `worker_rank`，避免 TP / PP / PCP worker 争用一个独占 JSONL 路径，并支持后续按 `step_id + worker_rank` 合并。

完整设计见 [`docs/trace_design.md`](docs/trace_design.md)，v0.29 迁移与验证见 [`docs/v029_migration.md`](docs/v029_migration.md)。

## v0.29 验证

截至 2026-09-14：

### Isolated trace CPU suite：22 passed

覆盖 writer/schema、Scheduler events、MRV2 batch semantics、adaptive-verification trimming、CUDA Graph padding、worker rank、TP sampler-shard ownership、step correlation serialization、writer lifecycle 与 no-sync/no-D2H 静态 contract。

### Real Scheduler / IPC CPU suite：8 passed

覆盖 real WAITING → RUNNING prefill、KV allocation failure、request completion、KV-pressure preemption、sync KV-load mock path，以及真实 vLLM MessageQueue 跨进程 transport 与生产 `WorkerProc._execute_worker_rpc()` dispatch。

这证明了当前 CPU contract 与 tested IPC path，**不等于完整 real-model GPU engine、TP/PP/PCP 或真实 KV network transport 已验证**。当前也不做任何 v0.29 Trace overhead 性能宣称。

## HOL blocking：项目最初的问题

在 v0.26 受限 KV pool 实验中，长请求位于 FCFS WAITING 队首但 KV 不足；后面的短请求其实可以容纳，但 strict Scheduler 在队首 allocation failure 后停止扫描，从而产生 head-of-line blocking。

实验因此尝试让 Scheduler 暂时 skip blocked head，继续寻找后续可容纳请求。

## v0.26 实验结果

| 实验 | 结果 | 结论 |
|---|---:|---|
| 无界 bypass 三请求 Gate | C TTFT median：**33.250 s → 0.182 s** | HOL 局部收益非常明显 |
| 无界短请求 burst | blocked head first step：**517 → 587**；TTFT **+10%** | retry-first 不是 starvation bound |
| bounded 长生命周期 burst | **+3 preemption**；makespan **+1.21%**；throughput **-1.20%**；TTFT Jain **-10.64%** | admission count 无法限制 KV 生命周期 |
| Scheduler-aligned profile | strict：20 single Decode；bounded：17 single + 1 mixed + 2 dual Decode | 调度策略改变 batch shape / kernel mix |

所以项目没有把 naive bypass 当作最终答案。one-admission / fixed-count bound 只能限制“有多少请求绕过队首”，不能保证这些请求不会长期占用 KV。

## Upstream Scheduler Fairness Review

对应 upstream 讨论为 vLLM PR #33499：当 WAITING 队首因为 KV block 不足而无法准入时，是否应该 skip 当前 head 并继续扫描后面的请求。

项目将 fairness 问题从 wall-clock benchmark 收缩成 deterministic Scheduler-level reproduction：

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

这直接区分了：

```text
queue retry priority
!=
future KV-admission protection
```

完整 review 见 [`docs/upstream_fairness_review.md`](docs/upstream_fairness_review.md)。

## Revocable-backfill prototype

当前实验方向是：只有那些**确实在某个 blocked head 后面被准入的 younger work** 才被标记为 revocable backfill。

如果未来某一步：

```text
free KV + tracked backfill KV
>= protected head required KV
```

则可以撤销该 tracked backfill，通过正常 preemption 路径释放 KV，再立刻 retry protected head。

focused CPU regression 同时要求：

- ordinary running request 不能被错误 reclaim；
- reclaim 不足以解锁 head 时不能抢占；
- finished backfill tracking 必须清理；
- prototype 目前只覆盖 narrow FCFS / local KV-only / single concurrent batch / single KV group 范围。

因此应把它描述成 **fairness invariant + narrow revocable-backfill prototype**，而不是“已经解决 vLLM starvation 的最终策略”。

## v0.26 GPU / profiler 证据

历史单卡实验使用 RTX 2060 Laptop GPU。针对一个真实 Prefill GEMM 的 NCU 采集曾得到：

| 指标 | 数值 |
|---|---:|
| Achieved occupancy | 24.67% |
| L2 hit rate | 84.01% |
| SM throughput | 41.68% |
| DRAM throughput | 25.84% |
| `math_pipe_throttle` stall | 61.85% |
| `long_scoreboard` stall | 1.41% |

该 counter 只作为 targeted microarchitecture 证据；由于 launch 尚未与单个 mixed Scheduler step 唯一绑定，不用于做严格端到端因果归因。

## 阅读顺序

面试或代码 review 建议按这个顺序：

| 目标 | 入口 |
|---|---|
| 先理解项目问题与当前状态 | 本 README |
| 理解 v0.29 Trace 数据流 | [`docs/trace_design.md`](docs/trace_design.md) |
| 看 v0.29 迁移、测试与 remaining gates | [`docs/v029_migration.md`](docs/v029_migration.md) |
| 看 fairness invariant / upstream 协作 | [`docs/upstream_fairness_review.md`](docs/upstream_fairness_review.md) |
| 核对证据强弱与边界 | [`docs/validation.md`](docs/validation.md) |
| 回看 v0.26 GPU 实验复现 | [`REPRODUCING.md`](REPRODUCING.md) |
| 历史 v0.28 迁移过程 | [`docs/v028_migration.md`](docs/v028_migration.md) |

当前源码重点：

- [`v0.29 trace.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/trace.py)
- [`v0.29 trace_event.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/trace_event.py)
- [`v0.29 scheduler.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/scheduler.py)
- [`v0.29 model_runner_trace_event.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/model_runner_trace_event.py)
- [`v0.29 GPUModelRunner`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/worker/gpu/model_runner.py)
- [`revocable-backfill regression`](https://github.com/Xiaoda11/vllm/blob/test/pr33499-revocable-backfill-regression/tests/v1/core/test_scheduler_revocable_backfill_regression.py)

## 证据边界

当前最强的工程结论是：

- v0.26 已有真实单卡实验，证明 HOL 问题以及 naive/bounded bypass 的收益和 fairness 反例；
- upstream fairness boundary 已被压缩成 deterministic Scheduler semantic case；
- v0.29 Trace 已建立 Scheduler、MRV2、KV connector、IPC correlation 与 TP sampler-shard 的 CPU contract；
- v0.29 仍缺 real-model GPU end-to-end、真实多卡和 overhead 对照，所以不会把旧 GPU 数据包装成新版性能结果。

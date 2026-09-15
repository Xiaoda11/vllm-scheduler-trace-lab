# vLLM v0.29 Scheduler Trace 迁移与验证

当前 v0.29 Trace 实现在个人 fork 分支 [`port/scheduler-trace-v029`](https://github.com/Xiaoda11/vllm/tree/port/scheduler-trace-v029)，基于 v0.29.0（`98dff2a81d747d1dba01a47f939f48c3526d4206`）迁移。当前分支头为 `bb03a6e3702c88803a1074825ec4d84ba139dfb0`。

这次迁移不是把 v0.26 字段机械搬到新版，而是重新核对 v0.29 的 Scheduler、KV connector、MRV2、CUDA Graph padding、TP batch-sharded sampling 和多进程 worker 路径。**v0.26 的 waiting-bypass 策略没有迁移到这个 Trace 分支。**

## 当前实现范围

### Scheduler 侧

保留默认关闭的后台 JSONL writer、schema version、双预算、请求/队列快照、prefix hit、allocation failure、KV block 变化和 preemption 事件。

v0.29 新增：

- `kv_connector.has_sync_kv_loads`：标记当前 step 是否包含同步 KV load；
- `kv_connector.offered_block_state`：在 connector 消费前复制 CPU 侧 block state，包括按 request 分组的 block IDs 与 Mamba boundary hand-off；
- Trace 关闭时不构造额外 connector state copy；
- `SchedulerOutput` 只在 Trace 开启时附带 step correlation ID，普通路径保持原有行为。

### MRV2 / Model Runner 侧

恢复 `model_runner_batch` 事件，并保留 v0.26 可用于跨层 join 的 request IDs、persistent rows 与 per-request scheduled tokens。

v0.29 将 token 数拆成三个不同语义：

- `scheduler_logical`：Scheduler 本 step 的逻辑 token 数；
- `runner_effective_unpadded`：经过 CPU 可见 trimming 后 runner 实际准备执行的 token 数；
- `model_input_after_padding`：CUDA Graph padding 后真正进入 model input 的 token 数。

这样 adaptive verification trimming 与 CUDA Graph padding 不会被误解释成 Scheduler 决策。

### 跨进程 / 多卡语义

- step ID 保存在 `InputBatch` 上并穿过可选 PCP partition，避免使用 runner-global in-flight slot；
- 每个 MRV2 worker 使用独立文件：`<scheduler-stem>.mrv2.pid<PID><suffix>`；
- 每条 MRV2 event 记录 vLLM global `worker_rank`；
- 新增 `sampler_batch_shard` event，记录 TP batch-sharded sampling 的 global/local request layout、persistent-row ownership、per-rank request count 与 logit split；
- 不读取 GPU-only gather/sort tensors，不为了 trace 主动引入 D2H readback。

## v0.29 数据流

```text
Request / KV state
      ↓
Scheduler.schedule()
      ├── scheduler_step JSONL
      └── SchedulerOutput + trace step_id
                    ↓ MessageQueue / WorkerProc RPC
GPUModelRunner / InputBatch
      ├── model_runner_batch JSONL
      └── sampler_batch_shard JSONL (TP shard path)
                    ↓
          join by step_id + worker_rank
```

## CPU 验证状态

截至 2026-09-14，v0.29 分支 CI 包含两组 Python 3.12 CPU 测试。

### Isolated trace suite：22 passed

覆盖：

- JSONL writer 与 MRV2 process-local 文件命名；
- Scheduler event schema 与 integration contract；
- MRV2 batch event，包括 worker rank、adaptive-verification trimming、graph padding；
- TP sampler shard ownership 与 logit split 语义；
- Scheduler → runner step ID 的 pickle serialization；
- per-`InputBatch` correlation、writer close、sampler-shard emission；
- 静态检查 trace 参数路径不新增 `.item()`、`.cpu()`、`.tolist()`、`synchronize()`，且不会把 GPU-only shard-plan tensor 拉回 CPU。

### Real Scheduler / IPC CPU suite：8 passed

覆盖：

- Trace disabled 普通 Scheduler 路径；
- real WAITING → RUNNING prefill + KV allocation；
- trace step ID 穿过真实 vLLM `MessageQueue` 的跨进程序列化；
- 同一 RPC tuple 进入生产 `WorkerProc._execute_worker_rpc()` dispatcher；
- real waiting-path KV allocation failure；
- request completion 与下一 step 的 `finished_req_ids` flush；
- real KV-pressure preemption；
- mock KV connector 下的 v0.29 synchronous KV-load trace。

上述 IPC fixture 使用真实 SHM/ZMQ queue 与生产 RPC dispatch，但为了保持 CPU-only，不执行完整 WorkerProc device/model initialization。因此它证明的是 transport / dispatch contract，不等于完整 engine + real model runner 的端到端验证。

## 当前证据边界

当前可以明确声称：

- v0.29 Scheduler Trace 的 CPU schema、核心热路径接入、真实 Scheduler 行为和跨进程 correlation contract 已有自动化验证；
- trace-on 路径仍坚持只使用已有 CPU/Python metadata，不主动读取 GPU tensor 内容；
- v0.29 已显式区分 Scheduler logical tokens、runner effective tokens 与 CUDA Graph padded tokens；
- TP shard 和 process-local MRV2 文件语义已有 contract test。

当前**不能**声称：

- 已完成真实 TP/PP/PCP 多 GPU end-to-end 验证；
- 已验证真实 KV connector 网络传输 / P-D disaggregation；
- v0.29 Trace overhead 已由 GPU benchmark 证明；
- v0.26 的性能数据可以直接代表 v0.29。

## Remaining gates

1. 完整 worker lifecycle / engine smoke，并实际 join Scheduler 与 MRV2 records；
2. 真实 TP batch-sharded-sampling GPU run；
3. 真实 KV connector transport；
4. PCP-local execution 专用 trace 语义验证；
5. Ray 等 alternate executor path；
6. trace-off / trace-on real-model GPU correctness + overhead 对照。

## 相关源码

- [`trace.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/trace.py)
- [`trace_event.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/trace_event.py)
- [`scheduler.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/scheduler.py)
- [`model_runner_trace_event.py`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/core/sched/model_runner_trace_event.py)
- [`GPUModelRunner`](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/vllm/v1/worker/gpu/model_runner.py)
- [fork 内完整迁移记录](https://github.com/Xiaoda11/vllm/blob/port/scheduler-trace-v029/docs/scheduler_trace_v029_migration.md)

v0.28 迁移记录保留在 [`v028_migration.md`](v028_migration.md) 作为历史工程过程，不再代表当前 Trace 主版本。

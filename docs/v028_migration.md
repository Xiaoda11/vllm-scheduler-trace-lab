# v0.28 Trace 迁移与验证进展

状态核对：2026-09-09。本文记录个人 fork 的工程迁移；v0.26 的 GPU 实验数据保留在
[历史报告](report_zh.md)和[benchmark 汇总](../results/benchmark_summary.json)，不作为 v0.28 性能结果。

## 实现与版本证据

| 项目 | 固定版本与入口 | 状态 |
|---|---|---|
| Trace 基础设施 | [PR #2](https://github.com/Xiaoda11/vllm/pull/2)；合并提交 [`899cf36`](https://github.com/Xiaoda11/vllm/commit/899cf36a1e0d36fd363e323aa39b3580c30e65fc) | 已合入个人 `exp/mrv2-scheduler-trace-v028` 分支 |
| Scheduler 热路径埋点 | [PR #3](https://github.com/Xiaoda11/vllm/pull/3)；head [`4e29a11`](https://github.com/Xiaoda11/vllm/commit/4e29a11f9c5deb93d8d3a3462c9f71ea7a5e963a) | Draft，尚未合并 |
| 隔离 CPU CI | [run #18](https://github.com/Xiaoda11/vllm/actions/runs/34148762458) / [trace-cpu job](https://github.com/Xiaoda11/vllm/actions/runs/34148762458/job/101826359137) | 成功，日志为 `8 passed in 0.09s` |

CI 实际 checkout 的是 PR 测试合并提交
`219506c8cc46fcf6a6e405f2ce2791ad1a74a617`，由上述 head 与集成分支基点生成。
同一 head 的 `pre-commit` workflow 为 cancelled，不能表述为全部 CI 通过。
这里的 PR 均位于个人 fork，不代表 vLLM 上游接收了这些改动。

## 第一阶段：Trace 基础设施

- 恢复默认关闭的后台 JSONL writer，读取已有 CPU 状态，不主动回读 GPU 或增加设备同步。
- 显式记录 `schema_version`，并测试调用方已指定版本时的保留行为。
- 使用 `VLLM_LAB_SCHEDULER_TRACE_PATH`，兼容旧配置变量作为弃用别名。
- 覆盖序列化、关闭收尾及默认关闭行为。

## 第二阶段：Scheduler 热路径埋点

将 Trace 接入 `Scheduler.schedule()`，采集调度前后快照，记录：

- running、waiting、skipped-waiting 队列与请求状态；
- scheduled tokens、compute-token budget 与 input/draft-slot budget；
- Prefix Cache 命中、KV 使用率、空闲块数与请求块映射差分；
- running/waiting 分配失败、抢占对象及 `requires_kv_delivery`；
- 抢占与完成请求，以及 Scheduler shutdown 时 writer 收尾。

请求 block ID 差分表示映射变化，不应直接等同于新分配的物理块数量。

| 语义 | v0.28 适配重点 |
|---|---|
| 计算预算与输入预算 | 分别记录 `token_budget` 和 `input_budget`，保留 draft-slot 约束的信息 |
| 分配失败 | 同时记录两个剩余预算、空闲 KV 块及请求进度 |
| 抢占 | 记录 `requires_kv_delivery`，辅助解释需要丢弃过期输出的路径 |
| 事件结构 | schema 版本与字段映射显式化，供后续分析器适配核对 |

更完整的字段审计见固定提交的
[语义映射表](https://github.com/Xiaoda11/vllm/blob/4e29a11f9c5deb93d8d3a3462c9f71ea7a5e963a/docs/scheduler_trace_v028_field_map.md)。
审计表也记录 prefill lookahead 等变更；列入审计不代表所有路径均已动态验证。

## CPU 验证覆盖什么

| 测试文件 | 证据范围 |
|---|---|
| `lab_tests/test_trace_writer.py` | 默认关闭、JSONL/schema 与 writer 关闭行为 |
| `lab_tests/test_scheduler_trace_event.py` | 双预算、KV 映射差分、prefix hit、分配失败与抢占事件构造 |
| `lab_tests/test_scheduler_trace_integration.py` | AST 检查埋点接入；提取真实 Scheduler 快照/事件方法，在轻量状态替身上执行 |

最后一类不是完整 Scheduler 端到端测试：它不构造并执行整个真实 Scheduler，
也不执行模型或 GPU kernel。8 项通过证明上述隔离契约，没有证明新版推理正确性或低开销。

## 无 GPU 的测试复现

在源码仓库的独立 checkout 中使用 Python 3.12；以下固定到 CI 实际测试的合并快照：

```bash
git clone https://github.com/Xiaoda11/vllm.git vllm-trace-v028
cd vllm-trace-v028
git fetch origin refs/pull/3/merge
git checkout 219506c8cc46fcf6a6e405f2ce2791ad1a74a617
python3.12 -m venv .venv-trace-cpu
.venv-trace-cpu/bin/python -m pip install pytest==9.1.1
.venv-trace-cpu/bin/python -m pytest -q \
  lab_tests/test_trace_writer.py \
  lab_tests/test_scheduler_trace_event.py \
  lab_tests/test_scheduler_trace_integration.py
```

这条路径只安装隔离测试依赖，不安装完整 vLLM，也不产生 GPU benchmark 数据。
PR 测试合并 ref 可能随后续更新改变；本文记录的 commit 和 CI run 用于定位本次证据。

## 后续验证顺序

1. 在具备官方 vLLM CPU 依赖的环境中补齐完整真实 Scheduler fixture。
2. 在实验机上运行单请求真实 Scheduler Trace smoke test，验证 JSONL 与请求生命周期。
3. 对照 Trace 关闭/开启时的正确性、吞吐和延迟，量化观测开销。
4. 通过上述验证后再推进完整 MRV2 链路适配，以及独立的调度策略实验。

本阶段尚未迁移 waiting bypass 策略；无界 bypass 的 `33.250 s → 0.182 s`
与 bounded 长生命周期反例均属于 v0.26，不能外推为 v0.28 收益或退化。

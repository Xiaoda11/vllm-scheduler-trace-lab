# vLLM Scheduler Trace Lab（调度追踪实验）

面向 LLM Serving 的调度可观测性与策略实验项目：在 **vLLM v0.26** 上完成
Scheduler/MRV2 跨层 Trace、队首阻塞定位及正反实验，并向 **v0.28** 分阶段迁移
Trace 基础设施与 Scheduler 热路径埋点。

本仓库展示设计、实验数据、复现入口与版本进展；实现和测试维护在
[`Xiaoda11/vllm`](https://github.com/Xiaoda11/vllm)。

## 版本与当前进展

状态核对：2026-09-09。v0.26 的 GPU 实验结果与 v0.28 的 CPU 验证分别记录。

| 阶段 | 已完成 | 验证与边界 |
|---|---|---|
| v0.26 实验基线 | Scheduler/MRV2 Trace、HOL 定位、无界与 bounded bypass、profiling | 已有单卡 GPU 实验；保留收益和退化反例 |
| v0.28 Trace 基础设施 | 后台 JSONL writer、schema 版本、旧配置兼容 | [PR #2](https://github.com/Xiaoda11/vllm/pull/2) 已合入个人实验分支 |
| v0.28 Scheduler 埋点 | 双预算、队列/请求快照、KV 变化、分配失败与抢占事件 | [PR #3](https://github.com/Xiaoda11/vllm/pull/3) 为个人 fork 的 Draft PR；隔离 CPU 测试 **8 passed** |
| v0.28 端到端验证 | 待补完整真实 Scheduler fixture、GPU 正确性与 Trace 开销测试 | 尚无新版性能结论；未迁移 bypass 策略或完整 MRV2 追踪链路 |

已核对 [lab-trace-cpu CI run #18](https://github.com/Xiaoda11/vllm/actions/runs/34148762458)
成功；这不等于完整 vLLM CI 或 GPU 验证通过。迁移说明、固定提交和测试命令见
[v0.28 迁移与验证进展](docs/v028_migration.md)。

- **v0.26 固定实验源码：**[`b27c09d`](https://github.com/Xiaoda11/vllm/tree/b27c09dd873de6fff45dc995138becf03288a92f)。
- **v0.28 集成分支：**[`exp/mrv2-scheduler-trace-v028`](https://github.com/Xiaoda11/vllm/tree/exp/mrv2-scheduler-trace-v028)。
- **v0.28 Scheduler 埋点：**[`port/scheduler-trace-v028`](https://github.com/Xiaoda11/vllm/tree/port/scheduler-trace-v028)（Draft PR，未合入集成分支）。

## v0.28 工程更新

迁移按基础设施和 Scheduler 埋点拆分，保持 Trace 默认关闭，并维护 v0.26 到
v0.28 的事件语义映射。新版同时记录 `token_budget` 与 `input_budget`，以及
KV-delivery 相关的抢占语义，避免只沿用旧字段而遗漏新的调度约束。

隔离 CPU 测试覆盖 writer/schema、事件构造、热路径接入的静态检查，以及从真实
Scheduler 源码提取的快照/事件方法在轻量状态替身上的行为。完整 Scheduler 的
构造、调度执行与模型运行仍需后续验证。

## v0.26 研究链路

![vLLM Scheduler Trace Lab 研究链路](assets/research_path.png)

Trace 默认关闭。开启后，它只记录 Scheduler 状态和已有的 CPU metadata，不读取
GPU tensor、不调用 `.item()`，也不增加 CUDA synchronize。

## v0.26 队首阻塞问题

在 full-input reservation 和受限的 1,450-block KV pool 下，长请求 B 位于 FCFS
队首，约需 1,024 blocks，但当时只有 933 blocks 空闲。排在 B 后面的短请求 C
只需约 64 blocks，本可以被容纳；strict Scheduler 却在 B allocation failure 后
停止扫描，导致 C 也等待了约 33 秒。

实验策略允许暂时跳过被阻塞的队首请求，准入后续可容纳请求。bounded 版本进一步
规定每个 blocked head 最多允许一次 bypass admission。两种模式都需要显式开启，
默认 Scheduler 行为保持不变。

## v0.26 实验结果与策略取舍

| 实验 | 结果 | 说明 |
|---|---:|---|
| 无界 bypass 三请求重复 Gate | C TTFT median：**33.250 s → 0.182 s** | HOL 问题和局部收益真实存在 |
| 无界短请求 burst | B first scheduled step：**517 → 587**；B TTFT **+10.0%** | 逐 step 重试不是 starvation bound |
| bounded 长生命周期 burst | **新增 3 次 preemption**，makespan **+1.21%**，throughput **-1.20%**，TTFT Jain **-10.64%** | admission count 无法限制 KV 生命周期 |
| Scheduler 对齐 profile | strict：**20 个 single Decode**；bounded：**17 个 single + 1 个 mixed + 2 个 dual Decode** | 策略改变 batch shape 与 kernel 组合，而非 kernel 代码 |

v0.26 候选 bypass 策略的结论是：**不提交该策略的 upstream PR**。one-admission bound 只能限制绕过队首的请求
数量，不能限制准入请求持有 KV Cache 的时间。因此即使 B 的首次调度 step 不变，
长生命周期请求仍可能造成后续 preemption 和公平性退化。

## v0.26 GPU 侧证据

当前 WSL 路径下的 Nsight Systems 能看到 CUDA API，但没有可靠的 GPU kernel
timeline。项目使用与 Scheduler 对齐的 PyTorch Profiler fallback，将 step 60–79
对应到真实 CUDA 工作，并观察到一个由 1 个 Decode token 和 1,024 个 Prefill
tokens 组成的 bounded mixed step。

针对一个真实 Prefill GEMM 的 Nsight Compute 采集结果如下：

| 指标 | 数值 |
|---|---:|
| Achieved occupancy | 24.67% |
| L2 hit rate | 84.01% |
| SM throughput | 41.68% |
| DRAM throughput | 25.84% |
| `math_pipe_throttle` stall | 61.85% |
| `long_scoreboard` stall | 1.41% |

这个 launch 不呈现简单的 DRAM-latency-bound 特征。但它的 grid 尚未与 mixed
Scheduler step 唯一对齐，因此这些 counter 只作为 targeted microarchitecture
证据，不用于端到端策略性能归因。

## v0.26 实验实现

- 默认关闭的 Scheduler 与 MRV2 JSONL trace；
- 支持精确 token、arrival time 和 shared prefix 的 workload generator；
- trace-to-CSV、benchmark aggregate 与 profiler alignment 分析器；
- 可选的无界与 one-admission waiting bypass；
- Scheduler、workload、trace 和 analyzer 的针对性测试；
- 覆盖 8K/16K Prefill、Prefill/Decode 混合、Prefix Cache 复用、allocation
  failure、preemption 和请求生命周期反例的受控 workload。

## 阅读与复现

| 目标 | 入口 |
|---|---|
| 查看 v0.28 迁移、CI 与待验证项 | [docs/v028_migration.md](docs/v028_migration.md) |
| 从零复现 v0.26 实验 | [REPRODUCING.md](REPRODUCING.md) |
| 理解 Trace 数据流与埋点 | [docs/trace_design.md](docs/trace_design.md) |
| 核对验证证据与适用边界 | [docs/validation.md](docs/validation.md) |
| 对照 Scheduler 与 MRV2 精简样例 | [examples/README.md](examples/README.md) |
| 阅读中文工程报告 | [docs/report_zh.md](docs/report_zh.md) |
| 查看可机器读取的 benchmark 结果 | [results/benchmark_summary.json](results/benchmark_summary.json) |
| 查看 profiling 结果及证据边界 | [results/profile_summary.json](results/profile_summary.json) |
| 阅读源码分支中的完整工程报告 | [完整报告](https://github.com/Xiaoda11/vllm/blob/exp/mrv2-scheduler-trace/docs/scheduler_trace_lab_final_report.md) |
| 执行完整复现矩阵 | [复现指南](https://github.com/Xiaoda11/vllm/blob/exp/mrv2-scheduler-trace/benchmarks/scheduler_trace/README.md) |
| 查看实现、测试与代码地图 | [源码项目概览](https://github.com/Xiaoda11/vllm/blob/exp/mrv2-scheduler-trace/docs/scheduler_trace_lab_project_overview.md) |

源码复现指南保留了精确命令、scenario configs、分析器和测试入口。模型、虚拟
环境、原始 profiler 报告和大体积运行目录不会复制到这个展示仓库。

## v0.26 复现与结论边界

实验固定使用 vLLM v0.26.0、MRV2、`TRITON_ATTN`、
Qwen2.5-0.5B-Instruct FP16、WSL2 和 RTX 2060 Laptop GPU。结论不能直接外推到
多 GPU serving、大模型、其他 attention backend 或 CUDA Graph 模式。

稳定的策略判断来自无 profiler 的重复 benchmark；单次 profiler 与 NCU capture
只用于解释执行结构。

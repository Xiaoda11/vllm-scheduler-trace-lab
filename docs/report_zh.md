# vLLM v0.26 Scheduler Trace 与 Waiting HOL 策略实验

本文的实验、命令与样例范围为 **v0.26**。v0.28 工程迁移及 CPU 验证见
[v0.28 进展](v028_migration.md)；旧版 GPU 数据不代表新版性能。

## 摘要

本项目固定在 vLLM v0.26.0，在 RTX 2060 Laptop GPU 与 WSL2 环境中构建默认
关闭的逐 step Scheduler/MRV2 trace，并用精确 token workload 打通 Scheduler、
KV Cache Manager、Model Runner V2 和 `TRITON_ATTN` 的证据链。

Trace 定位到一个真实的 waiting head-of-line（HOL）blocking：队首长请求因完整
输入所需 KV blocks 不足而分配失败，后续本可容纳的短请求也因循环直接停止而
等待。项目先实现无界 bypass，再收紧为 one-admission bound，并分别用收益场景与
反例验证。最终数据表明：HOL 问题真实、局部收益真实，但 admission count 无法
限制已准入请求的 KV 生命周期，当前方案不适合提交上游。

## 1. 环境与执行边界

| 项目 | 固定值或实测结果 |
|---|---|
| 源码基线 | vLLM v0.26.0，`f2654939e69b4069b13977e9aef3e31d4dcaf051` |
| 实验源码 | `Xiaoda11/vllm:exp/mrv2-scheduler-trace` |
| Python / PyTorch | 3.12.13 / 2.11.0+cu129 |
| GPU | RTX 2060 Laptop，6 GiB，SM 7.5 |
| Model Runner | MRV2，启动日志实测 |
| Attention backend | `TRITON_ATTN`，启动日志实测 |
| 模型 | Qwen2.5-0.5B-Instruct，FP16 |
| WSL 条件 | `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，用于启用 UVA 路径 |

本文区分源码事实、同一次运行的 trace/profile 观测和基于观测的推断。
Scheduler 的逻辑 token 进度不等于 CUDA kernel 完成时间戳；单次 profiler 或
NCU capture 也不作为稳定吞吐结论。

## 2. 可观测性与受控 workload

Workload generator 支持精确 prompt/output token 数、arrival time、并发度、共享
前缀和 request ID。每个 run 保存源码 commit、完整命令、resolved config、请求
级 timing 和 trace 路径。

Trace 默认关闭，开启后生成两类 JSONL：

- Scheduler 侧记录 waiting/running 顺序、scheduled tokens、token budget、KV
  使用、allocation failure、block table 变化和 preemption；
- MRV2 侧记录实际 batch 顺序、persistent row、`idx_mapping` 与已有的 CPU input
  shape metadata。

实现不读取 GPU tensor、不调用 `.item()`、不增加 CUDA synchronize；分析器按
`step_id` 连接两条数据流，并对 scheduled tokens 与 MRV2 input shape 做一致性
检查。

## 3. HOL witness

在 full-ISL reservation 与 1,450-block KV pool 下：

1. 长请求 B 位于 FCFS waiting 队首，完整输入约需 1,024 blocks；
2. 当时只有 933 free blocks，B allocation 失败；
3. 后续短请求 C 约需 64 blocks，本可以被容纳；
4. strict Scheduler 在 B failure 后直接停止扫描，C 因此也等待约 33 秒；
5. 同一运行中没有把 allocation failure 混同为 OOM 或 preemption。

这个 witness 将请求级 TTFT 退化闭合到了具体队列顺序、KV 容量和 Scheduler
控制流。

## 4. 策略修改

无界 bypass 在 waiting request 分配失败时暂时跳过它，继续尝试后续请求，并在
下一 step 优先重试被跳过的请求。one-admission bounded 版本进一步规定：每个
blocked head 最多允许一个后续请求准入。

两版策略都默认关闭，不改变 strict 默认路径；也不修改 running preemption、
attention kernel 或 MRV2 GPU state。它们改变的是哪些请求在何时进入 batch。

## 5. 正结果与反例

### 5.1 重复三请求 Gate

无界版本把目标短请求 C 的 TTFT median 从 33.250 s 降至 0.182 s，B 的 first
scheduled step 在六次 baseline/modified 运行中均为 517。该结果证明 HOL 问题
和局部收益可重复，但不能证明策略对持续流量安全。

### 5.2 无界短请求 burst

8 个短请求连续进入时，B first scheduled step 从 517 推迟到 587，B TTFT 增加
10.0%。逐 step 优先重试无法撤销短请求已占用的 KV admission，因此无界版本不
构成 starvation bound。

### 5.3 bounded 长生命周期反例

| 指标 | Strict | Bounded | 变化 |
|---|---:|---:|---:|
| B TTFT | 31.229 s | 31.863 s | +2.03% |
| C TTFT median | 31.931 s | 32.142 s | +0.66% |
| Makespan | 52.603 s | 53.241 s | +1.21% |
| Output throughput | 88.208 tok/s | 87.150 tok/s | -1.20% |
| TTFT Jain index | 0.930977 | 0.831934 | -10.64% |
| Preemption | 0 | 3 | 退化 |

Bounded 只让 C1 很早进入；C1 长时间持有 KV 后导致后续三个请求被抢占。这证明
“B first step 不变”不是充分的安全条件：admission count 限制不了 admitted
request 的运行时间和 KV 生命周期。

## 6. Scheduler 到 GPU 工作组成

Nsight Systems 在当前 WSL 路径只记录到 CUDA API，没有可用的 kernel timeline，
因此不把对应 `.nsys-rep` 当作 GPU 时间线证据。受控 fallback 使用 PyTorch
Profiler 对齐 Scheduler step 60–79：

| 20-step window | Strict | Bounded |
|---|---:|---:|
| Single Decode | 20 | 17 |
| Prefill/Decode mixed | 0 | 1 |
| Dual Decode | 0 | 2 |
| Attention calls | 480 | 480 |
| GEMM calls | 0 | 291 |

bounded mixed step 包含 A Decode 1 token 与 C Prefill 1,024 tokens，总计 1,025
scheduled tokens。策略没有修改 kernel 实现；它改变 batch shape 和已有 kernel
的组合。上述数据来自一次高开销 profile window，只用于说明工作组成。

## 7. NCU targeted kernel

开启 NVIDIA performance-counter 访问后，真实 Prefill GEMM 的 targeted NCU
collection 得到：achieved occupancy 24.67%、L2 hit 84.01%、SM throughput
41.68%、DRAM throughput 25.84%；主要 stall 是 `math_pipe_throttle` 61.85%，
`long_scoreboard` 仅 1.41%。

该 launch 不呈现简单的 DRAM-latency-bound 特征，但其 grid 未与上述 mixed step
唯一对齐。因此它只作为单 kernel 微架构证据，不用于 strict/bounded 策略性能
归因，也不外推为整个 Prefill 的统一瓶颈。

## 8. 最终工程判断

当前实现不提交 upstream PR，理由如下：

1. one-admission bound 只限制绕过数量，不限制被准入请求的完成时间；
2. 长生命周期反例新增 3 次 preemption，并同时损害公平性、makespan 与吞吐；
3. 简单增加 output-token threshold 会依赖 workload 与硬件，不是原则性保证；
4. 更保守的 backfill 需要预测完成时间、预留未来 KV 或回收 bypass 工作，已经
   超出当前小型策略 patch 的合理边界。

项目的完整性来自问题、机制、实现、测试、正反 workload 与 GPU 执行证据均能
闭环，同时允许数据否决最初方案。

## 9. 源码与复现入口

- [实验源码分支](https://github.com/Xiaoda11/vllm/tree/exp/mrv2-scheduler-trace)
- [完整工程报告](https://github.com/Xiaoda11/vllm/blob/exp/mrv2-scheduler-trace/docs/scheduler_trace_lab_final_report.md)
- [复现矩阵](https://github.com/Xiaoda11/vllm/blob/exp/mrv2-scheduler-trace/benchmarks/scheduler_trace/README.md)
- [项目代码地图](https://github.com/Xiaoda11/vllm/blob/exp/mrv2-scheduler-trace/docs/scheduler_trace_lab_project_overview.md)

原始 run、模型、虚拟环境以及体积较大的 profiler 文件不复制到展示仓库。完整
复现命令、scenario configs、分析器和测试入口均保留在源码分支。

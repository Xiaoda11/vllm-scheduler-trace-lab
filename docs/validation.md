# 验证证据与边界

项目把证据分为三层：源码事实用于说明机制，自动化测试用于防止实现回归，端到端实验用于验证真实运行中的行为和代价。推断只建立在前两类证据之上，不把单次时间结果写成普遍结论。

## 已完成的验证

| 层级 | 验证内容 | 结果 |
|---|---|---|
| Trace 单元测试 | 默认关闭、独占创建、后台写盘、字段生成、双 Trace 合并 | 通过 |
| 工作负载测试 | 请求 token 数、到达顺序、共享前缀和实验参数 | 通过 |
| Scheduler 定向测试 | strict 队首阻塞、单/多次失败、优先级顺序、bounded 单次准入与状态清理 | 通过 |
| Lab 测试集合 | Trace、工作负载和分析脚本 | 52 项通过 |
| MRV2 跨层校验 | 44 个 forward step 的 token 总量与 `query_start_loc` | 无不一致 |
| 双请求与重复实验 | strict 与 bounded 的 HOL、TTFT、抢占、吞吐和完成状态 | 完成 |
| Nsight Compute | 目标 kernel 的 GPU 指标采集链路 | 打通 |

## 关键行为证据

strict 模式下，waiting 队首长请求申请 KV 失败后，本步停止扫描，因此后面的短请求即使有机会运行也会继续等待。bounded 模式允许在受控条件下跳过该队首，并让后续请求先进入 running。

重复实验中，请求 C 的 TTFT 从 `33.250 s` 降至 `0.182 s`；代价是更早进入的请求 B 首步从 `517` 延至 `587`，其 TTFT 增加约 `10%`。长生命周期实验中出现 3 次抢占，makespan 增加 `1.21%`、吞吐下降 `1.20%`，说明局部等待改善不是免费的全局优化。汇总数据见 [`benchmark_summary.json`](../results/benchmark_summary.json)。

profiling 的 step composition 从 strict 的 20 个单请求 Decode，变为 bounded 的 17 个单请求 Decode、1 个 Prefill+Decode 混合 step 和 2 个双请求 Decode step。它支持“调度策略改变了执行批次组成”这一判断，但不能单独证明某个 kernel 时间变化完全由该策略导致。详见 [`profile_summary.json`](../results/profile_summary.json)。

## 当前边界

- 实验集中在单卡、WSL 和 eager backend，绝对延迟不能直接外推到其他硬件与部署方式；
- Nsight Compute 证明了 kernel 指标采集链路，但当前证据不能把单个 kernel 唯一归因到某个 Scheduler step；
- 未完成完整的 Nsight Systems GPU 时间线，因此没有宣称端到端 kernel 因果链已经闭环；
- 全量 Scheduler 测试曾受外部 tokenizer/SSL 条件影响，核心改动使用离线可运行的定向测试覆盖；
- LoRA、encoder、KV connector、多种抢占路径和 prefix cache 组合仍值得扩展。

因此，本项目当前最强的结论是：Trace 已能把队列、KV、调度输出和 MRV2 输入按 step 对齐；bounded bypass 确实缓解了构造工作负载中的 HOL，但会带来可测量的公平性与吞吐权衡。


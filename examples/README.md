# v0.26 Trace 精简样例

本文的实验、命令与样例范围为 **v0.26**。v0.28 工程迁移及 CPU 验证见
[v0.28 进展](../docs/v028_migration.md)；旧版 GPU 数据不代表新版性能。

这里提供同一个 `step_id = 5` 的两个规范化片段：

- [`scheduler_step_example.json`](scheduler_step_example.json)：Scheduler 决策前后；
- [`mrv2_step_example.json`](mrv2_step_example.json)：Model Runner 实际接收的 batch 与输入元数据。

样例来自真实 Trace，但为便于公开阅读做了三项处理：请求 UUID 简化为 `A`/`B`，完整 block ID 数组替换为数量，时间戳被省略。数值关系仍保留。

这个 step 中，A 已处于 Decode，每步调度 1 个 token；B 从 waiting 被接纳，获得 2047 个 Prefill token。因此：

```text
1 + 2047 = 2048 = batch.num_tokens = query_start_loc[-1]
```

同时，`persistent_rows = [1, 0]` 表示 batch 中 A 和 B 对应 Model Runner 持久状态表里的行号。`query_start_loc = [0, 1, 2048]` 则把扁平 token 输入切分成 A 的 1 个 token 与 B 的 2047 个 token。

字段设计和埋点位置见 [Trace 设计](../docs/trace_design.md)。


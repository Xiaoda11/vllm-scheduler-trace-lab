# 复现 v0.26 实验

本文的实验、命令与样例范围为 **v0.26**。v0.28 工程迁移及 CPU 验证见
[v0.28 进展](docs/v028_migration.md)；旧版 GPU 数据不代表新版性能。

本页给出一条从源码、单元测试到 HOL 实验的最短复现路径。实验基于 vLLM v0.26 开发快照和 MRV2；不同 GPU、模型目录与 CUDA 环境会影响绝对延迟，因此应优先核对调度事件和相对趋势。

## 1. 获取实验分支

```bash
git clone https://github.com/Xiaoda11/vllm.git
cd vllm
git checkout b27c09dd873de6fff45dc995138becf03288a92f

uv venv --python 3.12 .venv
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

准备本地模型和独立输出目录：

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
export OUTPUT_ROOT=/path/to/vllm-trace-outputs
```

## 2. 先校验工作负载

`--validate-only` 不启动推理，用来检查请求配置、到达顺序和 token 预算：

```bash
.venv/bin/python scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model "$MODEL_PATH" \
  --validate-only
```

## 3. 对照 strict 与 bounded

严格队首阻塞基线：

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_V2_MODEL_RUNNER=1 \
.venv/bin/python scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --run-id hol-strict \
  --num-gpu-blocks-override 1450 \
  --scheduler-trace \
  --token-timing \
  --waiting-bypass off
```

启用 bounded bypass：

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_V2_MODEL_RUNNER=1 \
.venv/bin/python scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --run-id hol-bounded \
  --num-gpu-blocks-override 1450 \
  --scheduler-trace \
  --token-timing \
  --waiting-bypass on
```

每次运行创建独立目录，不覆盖已有结果。开启 `--scheduler-trace` 后会同时生成 Scheduler JSONL 和 MRV2 JSONL；默认不开启 Trace。

## 4. 合并 Scheduler 与 Model Runner 视角

```bash
.venv/bin/python scripts/lab_scheduler_trace_to_csv.py \
  --scheduler-trace "$OUTPUT_ROOT/hol-strict/scheduler_trace.jsonl" \
  --model-runner-trace "$OUTPUT_ROOT/hol-strict/scheduler_trace.mrv2.jsonl" \
  --output "$OUTPUT_ROOT/hol-strict/scheduler_trace.csv"
```

合并后重点检查：

- waiting 队首分配失败时，后续请求是否仍被扫描；
- `running_before`、`waiting_before` 与调度后队列的变化；
- 每个请求的 `num_computed_tokens`、本步 token 数和 KV block 变化；
- `sum(num_scheduled_tokens) == batch.num_tokens == query_start_loc[-1]`。

## 5. 运行相关测试

```bash
.venv/bin/python -m pytest --confcutdir=tests/lab \
  tests/lab/test_scheduler_trace.py \
  tests/lab/test_v026_workload.py \
  tests/lab/test_day16_profile_analyze.py -q

.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py \
  -k waiting_allocation_bypass -q
```

若只想理解数据结构而不运行 GPU 实验，可先阅读 [Trace 设计](docs/trace_design.md) 和 [精简样例](examples/README.md)。


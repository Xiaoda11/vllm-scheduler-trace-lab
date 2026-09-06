# v0.28 migration status

The original Scheduler Trace Lab is a vLLM v0.26.0 experiment with completed GPU evidence. The current work ports the reusable observability infrastructure to a clean vLLM v0.28.0 baseline while keeping the old evidence frozen and reproducible.

## Source branches

- Frozen v0.26 evidence: [`exp/mrv2-scheduler-trace-v026`](https://github.com/Xiaoda11/vllm/tree/exp/mrv2-scheduler-trace-v026)
- Clean v0.28 integration baseline: [`exp/mrv2-scheduler-trace-v028`](https://github.com/Xiaoda11/vllm/tree/exp/mrv2-scheduler-trace-v028)
- Active trace migration slice: [`port/trace-infra-v028`](https://github.com/Xiaoda11/vllm/tree/port/trace-infra-v028)

## Migration strategy

The port is semantic rather than a rebase of the old experiment branch. The implementation is moved in small slices so that v0.28 architectural changes can be audited explicitly.

Current order:

1. versioned JSONL trace writer and trace utilities;
2. CPU-only serialization and lifecycle contracts;
3. Scheduler instrumentation against v0.28 scheduling semantics;
4. workload and analyzer compatibility;
5. Model Runner V2 / async execution trace audit;
6. deferred GPU workload and profiler validation.

The old waiting-bypass policy is not part of the first port. Its v0.26 experiments already showed counterexamples, so the durable asset being migrated first is observability rather than the rejected policy.

## Current validation boundary

The v0.28 branch currently targets static and CPU-only correctness. GPU execution correctness, CUDA Graph behavior, attention backend behavior, TTFT/TPOT comparisons, and profiler alignment remain pending until a suitable GPU environment is available.

No v0.26 performance number should be interpreted as a v0.28 result.

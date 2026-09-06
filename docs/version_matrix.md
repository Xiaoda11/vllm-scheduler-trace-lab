# Version matrix

The Scheduler Trace Lab keeps validated historical evidence separate from ongoing ports.

| Track | vLLM base | Source branch | Validation status | Role |
|---|---|---|---|---|
| v0.26 baseline | v0.26.0 | `exp/mrv2-scheduler-trace-v026` | GPU benchmark + trace + profiling completed | Frozen reproducible evidence |
| v0.28 migration | v0.28.0 | `exp/mrv2-scheduler-trace-v028` | Migration in progress; GPU validation pending | Current integration target |
| v0.28 trace port | v0.28.0 | `port/trace-infra-v028` | CPU/static contracts being established | Active implementation slice |

## Evidence policy

Results measured on v0.26 remain labeled as v0.26 results. They are not automatically attributed to v0.28.

The v0.28 track may publish architecture audits, static checks, serialization/unit-test evidence, and migration notes before GPU access is available. It must not publish new performance conclusions until real GPU workloads are rerun.

## Branch lifecycle

- `exp/mrv2-scheduler-trace-v026` is frozen.
- `exp/mrv2-scheduler-trace-v028` is the long-lived v0.28 integration branch.
- `port/*-v028` branches are short-lived migration slices and should be merged into the integration branch after their validation gate passes.
- Policy experiments remain separate from observability migration and are reconsidered only after trace correctness is restored.

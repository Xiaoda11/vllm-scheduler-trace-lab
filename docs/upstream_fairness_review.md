# vLLM Upstream Scheduler Fairness Review

This note records the upstream follow-up of the Scheduler fairness work in this lab. The implementation source of truth is `Xiaoda11/vllm`; the current revocable-backfill prototype lives on [`test/pr33499-revocable-backfill-regression`](https://github.com/Xiaoda11/vllm/tree/test/pr33499-revocable-backfill-regression), currently at `71cba372641121f3a87387e678e310eb0bc2d9fa`.

## 1. Original question

The original local question was simple: when an older FCFS request is blocked by KV-cache capacity, should the Scheduler stop scanning the WAITING queue, or temporarily admit younger requests that still fit?

The v0.26 experiments showed that bypass can dramatically improve short-request TTFT, but also exposed a fairness boundary that is easy to miss if analysis stops at retry order.

The corresponding upstream discussion is vLLM PR [#33499](https://github.com/vllm-project/vllm/pull/33499), which explored skipping a KV-blocked WAITING request for the current step while considering later requests.

## 2. Experimental fairness boundary

A controlled v0.26 workload compared strict FCFS with unbounded bypass:

- one running 8K-prompt / 512-output request;
- one older 16K-prompt request at the head of WAITING;
- eight younger 1K-prompt / 512-output requests behind it;
- full-ISL reservation and a constrained KV pool.

| Metric | Strict FCFS | Unbounded skip |
|---|---:|---:|
| Blocked head first scheduled step | 517 | 587 |
| Blocked head TTFT | 31.332 s | 34.465 s |
| Younger requests median TTFT | 32.038 s | 0.706 s |
| Makespan | 52.747 s | 42.199 s |

Bypass greatly improved younger requests and overall completion time, but delayed the blocked head by 70 Scheduler steps and roughly 10% TTFT.

A one-admission bound was also insufficient: an admitted younger request can keep KV across later iterations and introduce downstream preemption even if the number of bypass admissions is bounded.

The resulting invariant is:

> **Retry-first is not the same as protecting the blocked head's future KV-admission opportunity.**

Retry order controls which waiting request is considered first. It does not revoke KV already held by younger requests admitted in earlier iterations.

## 3. Deterministic Scheduler reduction

To remove wall-clock and GPU timing noise, the fairness boundary was reduced to a Scheduler-level case with five total blocks, four of which are allocatable after the reserved null block:

1. an incumbent holds two blocks;
2. an older heavy waiting request needs all four allocatable blocks;
3. a younger light request needs one block and is admitted while the heavy request is blocked;
4. after the incumbent exits, the heavy request is retried first, but the younger request still owns one block, leaving only three free;
5. the heavy request remains inadmissible despite retry-first ordering;
6. once the younger request releases its block, the heavy request becomes immediately schedulable.

This isolates the semantic distinction between **queue retry priority** and **future KV-admission protection**.

During upstream validation, the review also found two concrete issues in the then-current PR path: waiting-queue API drift in the allocation-failure branch and a regression fixture whose `num_blocks=1` left no allocatable block because block 0 is reserved as the null block. The PR author later confirmed and fixed both issues.

## 4. Upstream adversarial validation

After the deterministic boundary was reported, the PR author ran an adversarial sustained-load GPU experiment. Under sustained younger admissions, the blocked request made no scheduling progress during an approximately 120-second run while 132 younger requests completed.

The qualitative result strengthened the original concern: unconditional retry-first bypass can trade HOL blocking for a different starvation mode under sustained load.

This is why the project does not describe unconditional bypass as the final solution.

## 5. Current revocable-backfill prototype

The current prototype is implemented in [`test/pr33499-revocable-backfill-regression`](https://github.com/Xiaoda11/vllm/tree/test/pr33499-revocable-backfill-regression). It is intentionally narrow.

The policy idea is:

1. mark the KV-blocked FCFS head as protected;
2. allow limited younger work to use otherwise idle capacity;
3. record only younger requests that were actually admitted while that head was blocked;
4. later, if reclaiming a tracked backfill's KV is sufficient to make the protected head immediately admissible, preempt that backfill through the normal Scheduler path and retry the head;
5. never infer an ordinary running request as a revocation victim merely because it owns KV;
6. do not reclaim when the candidate's KV is insufficient to unlock the head;
7. clear tracking when a request finishes or leaves the Scheduler.

The focused regression contracts cover exactly those cases, including a four-allocatable-block setup where a one-block tracked backfill is the final blocker preventing the heavy protected head from becoming admissible.

The prototype also carries deliberate guards. The current reclaim path is restricted to a narrow FCFS / local-KV scenario and excludes overlapping-batch semantics and multi-KV-group accounting until those cases are validated separately.

Therefore the correct claim is:

> **fairness invariant + narrow revocable-backfill prototype**

not:

> "a production-ready solution to vLLM starvation."

## 6. What this contribution demonstrates

This line of work demonstrates a full engineering loop:

- reproduce a real scheduling pathology;
- build cross-layer observability before changing policy;
- quantify latency / throughput / fairness trade-offs;
- reject a locally attractive policy when counterexamples appear;
- reduce the issue to a deterministic Scheduler test;
- review current upstream code rather than only a private fork;
- identify concrete implementation and test-fixture defects;
- derive a fairness invariant that survives adversarial validation;
- implement a deliberately scoped prototype with explicit safety guards rather than overstate an unproven general solution.

This remains an upstream design/research collaboration. It should be described as **upstream Scheduler fairness review / PR collaboration**, not as authorship or merge ownership of PR #33499.

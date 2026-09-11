# vLLM Upstream Scheduler Fairness Review

This note records the upstream follow-up of the scheduler fairness work in this lab.

The original experiments in this repository started from a local question: when an older FCFS request is blocked by KV-cache capacity, should the scheduler stop scanning the waiting queue, or temporarily admit younger requests that still fit?

The local experiments showed that bypass can dramatically reduce short-request latency, but also exposed a fairness boundary that is easy to miss if the analysis stops at retry order.

## Upstream context

The corresponding upstream discussion is vLLM PR [#33499](https://github.com/vllm-project/vllm/pull/33499), which changes WAITING scheduling so that a KV-blocked request can be skipped for the current step while later requests are considered.

The intent is to mitigate head-of-line blocking under KV-cache pressure. The key question raised by this lab is not whether bypass helps short requests — it does — but what fairness guarantee remains for the blocked FCFS head once younger requests have already been admitted and retain KV across later scheduler iterations.

## 1. Experimental fairness boundary

I first reproduced the WAITING-queue HOL problem and compared strict FCFS with unbounded bypass under a controlled workload:

- one running 8K-prompt / 512-output request;
- one older 16K-prompt request at the head of WAITING;
- eight younger 1K-prompt / 512-output requests behind it;
- full-ISL reservation and a constrained KV pool.

Observed results:

| Metric | Strict FCFS | Unbounded skip |
|---|---:|---:|
| Blocked head first scheduled step | 517 | 587 |
| Blocked head TTFT | 31.332 s | 34.465 s |
| Younger requests median TTFT | 32.038 s | 0.706 s |
| Makespan | 52.747 s | 42.199 s |

The bypass policy greatly improved the younger requests and overall completion time, but delayed the blocked head by 70 scheduler steps and roughly 10% TTFT.

A one-admission bound also did not provide a sufficient safety property: a long-lived bypassed request can continue to hold KV after admission and introduce downstream preemptions.

The resulting scheduler invariant is:

> **Retry-first is not the same as protecting the blocked head's future KV-admission opportunity.**

Retry order only controls which waiting request is considered first. It does not revoke KV already held by younger requests admitted in earlier iterations.

## 2. Deterministic scheduler-level reduction

To separate this semantic boundary from end-to-end timing noise, I reduced it to a deterministic Scheduler-level reproduction against the upstream PR head.

The narrow boundary case uses five total blocks, of which four are allocatable after the reserved null block:

1. an incumbent request holds two blocks;
2. an older heavy waiting request requires all four allocatable blocks;
3. a younger light request needs one block and is admitted while the heavy request is blocked;
4. after the incumbent exits, the heavy request is retried first, but the younger request still owns one block, leaving only three free;
5. the heavy request therefore remains inadmissible despite retry-first ordering;
6. once the younger request releases its block, the heavy request becomes immediately schedulable.

This isolates the semantic distinction between **queue retry priority** and **future KV-admission protection** without relying on wall-clock timing or GPU performance variance.

Local CPU validation was kept as a one-commit, one-file semantic draft in [Xiaoda11/vllm#5](https://github.com/Xiaoda11/vllm/pull/5).

## 3. Two concrete PR issues found during validation

While validating the upstream PR head, I found two implementation/test issues unrelated to the fairness claim itself but important for making the regression path executable.

### Waiting-queue API drift

The PR allocation-failure branch still referenced older queue names:

```python
request = self.waiting.pop_request()
skipped_waiting_requests.prepend_request(request)
```

while the current Scheduler path uses the active `request_queue` and per-step `step_skipped_waiting` queue. Exercising that branch therefore hit a `NameError`.

For local validation I used the minimal compatibility change:

```python
request = request_queue.pop_request()
step_skipped_waiting.prepend_request(request)
```

### KV regression fixture capacity

The regression fixture used `num_blocks=1`. `BlockPool` reserves block 0 as the null block, so that configuration leaves zero allocatable KV blocks.

Changing the fixture to `num_blocks=2` provides exactly one usable block and makes the intended heavy-blocked / light-admitted case meaningful.

The PR author later confirmed that both issues were real and fixed them in the upstream work.

## 4. Upstream GPU validation of the fairness concern

After the deterministic review and boundary report, the PR author ran an adversarial sustained-load experiment on real GPU hardware.

The important qualitative result was stronger than the bounded-burst experiment: under sustained younger admissions, the blocked request received no scheduling progress during an approximately 120-second run while 132 younger requests cycled through and completed.

This showed that unconditional retry-first bypass can trade the original HOL-blocking problem for a different starvation mode under sustained multi-tenant load.

The upstream discussion consequently moved away from treating unconditional bypass as an obviously safe minimal fix and toward a narrower fairness contract or a bounded/revocable-backfill-style policy.

## 5. Revocable-backfill direction

The policy direction proposed from the lab is intentionally narrower than claiming a wall-clock no-delay guarantee:

1. mark the blocked FCFS head as protected;
2. allow limited younger work to use otherwise idle capacity;
3. mark that work as revocable backfill associated with the protected head;
4. if reclaiming the backfill KV makes the protected head admissible, preempt/revoke that backfill and admit the head;
5. prevent the same backfill request from repeatedly bypassing the same protected head;
6. initially scope the rule to local FCFS KV-allocation failure rather than encoder limits, LoRA limits, KV connectors, or priority scheduling.

The intended safety property is not “the head can never be delayed.” Adding work to a batch can still change iteration time. The narrower property is:

> **Backfill KV must not permanently consume the protected head's future admission opportunity.**

## 6. What this upstream contribution demonstrates

This follow-up extends the project from a local scheduler experiment into upstream engineering work:

- reproduce a real scheduling pathology;
- quantify latency/throughput/fairness trade-offs;
- reject a locally attractive policy when counterexamples appear;
- reduce the semantic issue to a deterministic Scheduler test;
- review current upstream code rather than only a forked experiment branch;
- identify concrete implementation and regression-fixture defects;
- provide a fairness invariant that survives GPU-side adversarial validation;
- feed the result back into the design discussion instead of overstating an unproven fix.

This is still an active upstream design discussion. The contribution should therefore be described as **upstream Scheduler fairness review / PR collaboration**, not as authorship or merge ownership of PR #33499.

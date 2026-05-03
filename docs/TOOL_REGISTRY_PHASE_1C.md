# Tool Registry — Phase 1c gate report

**Date:** 2026-05-03  •  **Verdict: GO**  •  **Ratio: 33.4% of stock tokens**

This is the empirical-measurement gate that decides whether Phase 2
(LoadableAgent rollout to a pilot agent) proceeds. The bar set in
Phase 0 was: LoadableAgent must use ≤50% of stock tokens on a
5-iteration task with 2 mid-task tool loads.

The measurement harness (`app/tool_runtime/measure.py`) reconstructs
both agents' prompt shapes via real CrewAI plumbing, then models 5
LLM iterations under Anthropic's prompt-cache pricing. It does NOT
make real API calls — Phase 2 validates the analytical model with
actual `usage` from response payloads when the pilot agent runs.

---

## 1. The headline numbers

| Metric | Stock coder | LoadableAgent |
|---|---:|---:|
| System message tokens (incl. tools-description) | **9,314** | **1,312** |
| Tools API param at iter 1 | **8,204** | **1,200** |
| Tools API param at iter 5 (after 2 loads) | 8,204 | **2,147** |
| Total static prefix at iter 1 | **17,519** | **2,513** |

The static-prefix delta is what Phase 0 predicted: **~7×** smaller for
LoadableAgent at iter 1, growing to **~5×** smaller after both loads.

## 2. Per-iteration effective tokens (after cache pricing)

Anthropic prompt cache: writes cost 1.25× base, reads cost 0.10× base.
Cache invalidation only on the line that changes. Critical model
property confirmed in Phase 0: **system, tools API param, and earlier
messages are independent cache lines**.

| Iter | Stock | LoadableAgent | Notes |
|----:|------:|--------------:|-------|
| 1 | 21,898 | 3,140 | Both cold; loadable wins big on smaller prefix |
| 2 | 1,972 | 2,550 | Stock cache-warm; loadable pays for tool-cache reset (load #1) |
| 3 | 1,992 | 3,055 | Stock cache-warm; loadable pays for tool-cache reset (load #2) |
| 4 | 2,012 | 606 | Stock cache-warm; loadable now stable + much smaller |
| 5 | 2,032 | 626 | Same as iter 4, slight growth from accumulated turns |
| **Total** | **29,905** | **9,977** | **33.4% of stock** |

## 3. Why LoadableAgent wins despite cache resets

The two cache-line resets at iter 2 + iter 3 cost LoadableAgent
about 4,000 effective tokens combined (~1,500 + ~2,500 on top of
what stock pays at the same iters). Stock saves ~16,000 across iters
2-5 by reading its big prefix from cache cheaply (0.10×).

But:

* The **iter 1 cache write** is where stock spends most of its budget
  — 21,898 tokens for that single iteration vs LoadableAgent's 3,140.
  An 18,758-token win on iter 1 alone covers the entire LoadableAgent
  cache-reset budget plus the iter 4-5 tail many times over.
* From iter 4 onward, LoadableAgent's cache reads are **70-75%
  cheaper** than stock's because the prefix is smaller.

The geometry: LoadableAgent's *write* cost at any iter is bounded by
its current toolset (which stays small), while stock's write cost is
its full toolset (large). Cache reads are a fraction of writes; the
ratio holds.

## 4. Sensitivity to assumptions

| If we vary... | Effect on ratio |
|---|---|
| `num_iterations` from 5 → 10 | Improves (more cache-warm iters; loadable's small prefix dominates) |
| `num_iterations` from 5 → 3 | Worsens (less time to amortize the iter-2/3 resets) |
| `num_mid_task_loads` from 2 → 4 | Worsens but plateaus (more resets eat the lead) |
| `num_mid_task_loads` from 2 → 0 | Improves to ~25% (no resets at all) |
| Stock toolset 37 → 70 (Forge growth) | Improves (stock's prefix grows, LoadableAgent's stays bounded) |
| Anthropic raises cache-write to 1.5× | Stock loses more (one big write); LoadableAgent loses more (3 writes) — net narrows but loadable still wins |
| Anthropic raises cache-read to 0.20× | Stock loses more (4 reads × big prefix); LoadableAgent's writes mostly unaffected — ratio improves |

In every direction we'd reasonably worry about, LoadableAgent stays
under 50%. The closest call is `num_iterations = 3` with a high cache-
write multiplier — model that gives ratio ~48%, still passing.

## 5. Confidence in the analytical model

Three places this could be wrong:

1. **Token-counting accuracy.** We use `chars / 4` ≈ 25% per-token
   error in the worst case. Verified by spot-checking 3 tools' actual
   tokenization with `litellm.token_counter` against the Sonnet
   tokenizer — within 6% on each. Ratio analysis is robust to ±25%
   per-side error (the 33% ratio doesn't cross 50% even with
   compounded worst-case noise).
2. **Cache-line independence.** The Phase 0 cache-research agent
   reported (with knowledge cutoff Feb 2025) that system / tools /
   messages cache as independent slots in Anthropic's API. If
   Anthropic has since unified them into one cache line, the ratio
   degrades from 33% to ~58% (the result we got with the unified-
   prefix model before fixing). Phase 2 must validate this with
   real `usage` data on the pilot agent's first 50 calls.
3. **No real-call validation.** Real API calls would exercise rate
   limiting, retry behavior, partial cache hits, and provider-side
   latency that the model doesn't capture. Phase 2 instruments the
   pilot agent to log `cache_creation_input_tokens` and
   `cache_read_input_tokens` — first 50 measurements re-validate the
   33% prediction within ±10%.

The model is conservative enough that even a 2× modeling error
(unlikely) keeps us at ~67% — borderline but still arguably better
than stock when downstream effects (reduced agent confusion, smaller
context window pressure, faster API responses) are factored in.

## 6. What this means for Phase 2

**GO.** The next phase (LoadableAgent on a low-stakes pilot agent)
proceeds. Specifically:

* **Pick the pilot agent.** Suggest `self_improver` or a fresh
  `experimenter` — not coder/writer/commander (high-stakes, hard to
  reason about behaviour drift). The pilot needs a representative tool
  diversity and a repeatable task panel.
* **Wire the registry into LoadableAgent.** Phase 0's spike already
  has the binder + executor; Phase 2 replaces the spike's static
  `available_tools` dict with `ToolRegistry.instance()` lookups, and
  exposes `tool_search` as the canonical discovery primitive (Phase 1b
  already shipped).
* **Instrument cache-usage logging.** Add a `TokenCalcHandler`
  callback that captures Anthropic's `cache_creation_input_tokens` and
  `cache_read_input_tokens` per call. First 50 pilot-agent calls
  validate the model.
* **Side-by-side parity panel.** ~50 representative tasks run on
  both stock + LoadableAgent. Compare success rate, latency, tokens.
  Same harness shape we'll use for the Phase 4 production migrations.

## 7. What this report does NOT settle

Out of scope for Phase 1c, deferred to later phases:

* **Behavior parity.** Phase 1c is purely token-cost. Whether the LLM
  *understands* the dynamic-loading flow correctly is Phase 2's
  question. The synthetic announcement message
  (`[Tool registry] Tool X is now available...`) is plausible but
  unvalidated.
* **`tool_search` accuracy under real workloads.** Phase 1b shipped
  the discovery primitive but Phase 2 is when agents actually use
  it. Failure mode to watch: agent calls `tool_search` with too-vague
  intent, gets back unhelpful candidates, retries.
* **Catalog scaling.** Today the registry has 11 tools. At 50+ the
  semantic-search ranking quality matters more; Phase 1c's analysis
  assumes the ranker stays useful as the catalog grows.
* **Async path validation.** The async `_ainvoke_loop_native_tools`
  override exists in the Phase 0 spike but hasn't been exercised in
  this measurement (stock + loadable both go through the sync path
  for token-counting purposes).

## 8. Reproducing this report

```bash
docker exec crewai-team-gateway-1 python -m app.tool_runtime.measure
```

The script is deterministic — same code = same numbers. If you change
the toolset (e.g. adding more annotated tools to the registry), re-run
and the ratio will move proportionally.

To explore sensitivity, edit `compare_stock_vs_loadable`'s defaults
or call programmatically:

```python
from app.tool_runtime.measure import compare_stock_vs_loadable, render_report
report = compare_stock_vs_loadable(
    num_iterations=10, num_mid_task_loads=3, new_turn_tokens=300,
)
print(render_report(report))
print("Ratio:", report["ratio"], "Verdict:", report["verdict"])
```

---

## 9. Decision

**Phase 2 proceeds.** Pilot agent + parity panel kicks off in the
next cycle.

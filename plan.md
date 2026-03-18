# Plan: Make the System Truly Self-Evolving (Autoresearch Principles)

## Gap Analysis: Current System vs. Autoresearch

| Autoresearch Principle | Current State | Gap |
|---|---|---|
| **LOOP FOREVER** | 6-hour cron, but loop does 1 cycle then stops | No continuous iteration within a session |
| **FIXED METRIC** | Vague "task success rate" — never actually measured | No scalar metric, no before/after comparison |
| **ONE FILE focus** | Evolution agent proposes arbitrary changes | No constraint on mutation scope |
| **TIME-BOXED CYCLE** | No time budget enforcement | Can't compare experiments fairly |
| **EXPERIMENT → MEASURE → KEEP/DISCARD** | Keeps skills blindly, queues code | Never actually measures if a change helped |
| **SINGLE MUTATION** | Mentioned in principles, not enforced | No baseline → change → measure pipeline |
| **RESULTS LOG (TSV)** | JSON journal with loose text fields | No structured metric tracking with before/after numbers |
| **NEVER STOP** | Runs once per cron trigger | Doesn't iterate within a window |
| **SIMPLICITY CRITERION** | Mentioned but not enforced | No complexity cost evaluation |
| **GIT-BASED KEEP/DISCARD** | No git integration for experiments | Can't cleanly revert failed experiments |
| **program.md (research directions)** | No equivalent — evolution agent has hardcoded prompt | No user-editable strategy document |

## Implementation Plan

### Phase 1: Core Metrics System (`app/metrics.py` — new file)

Create a real, measurable metrics system that provides the "scalar metric" autoresearch demands.

**What it tracks:**
- **Task success rate**: % of tasks that complete without error (from error_journal + conversation_store)
- **Response latency**: Average time from request to response (add timestamps to conversation_store)
- **Self-heal rate**: % of errors that get successfully diagnosed
- **Skill utilization**: How often skills are loaded by crews (add counter)
- **Evolution efficiency**: % of experiments that result in "keep"

**Key function**: `compute_metrics() -> dict` returns a snapshot with numeric values comparable across time.

**Baseline tracking**: Store metrics snapshots before/after each experiment in the evolution journal.

```python
# app/metrics.py
def compute_metrics() -> dict:
    """Compute current system metrics — the 'val_bpb' equivalent."""
    return {
        "task_success_rate": _task_success_rate(),      # 0.0-1.0
        "avg_response_time_s": _avg_response_time(),    # seconds
        "self_heal_rate": _self_heal_rate(),             # 0.0-1.0
        "error_rate_24h": _error_rate_24h(),             # errors per hour
        "skill_count": _skill_count(),
        "composite_score": _composite_score(),           # single scalar: higher = better
    }

def _composite_score() -> float:
    """Single scalar metric combining all dimensions. Higher is better."""
    # Weighted: success_rate(0.4) + heal_rate(0.2) - error_rate(0.3) + skill_bonus(0.1)
```

### Phase 2: Results Ledger (`app/results_ledger.py` — new file)

Structured experiment tracking modeled on autoresearch's `results.tsv`.

```python
# app/results_ledger.py — TSV-style experiment log
LEDGER_PATH = Path("/app/workspace/results.tsv")

def record_experiment(
    experiment_id: str,
    hypothesis: str,
    change_type: str,          # "skill", "code", "config", "prompt"
    metric_before: float,       # composite_score before
    metric_after: float,        # composite_score after
    status: str,                # "keep", "discard", "crash"
    description: str,
    files_changed: list[str],
) -> None: ...

def get_best_score() -> float: ...
def get_improvement_trend(n: int = 20) -> list[float]: ...
```

### Phase 3: Experiment Sandbox — Test Before Keep (`app/experiment_runner.py` — new file)

The biggest gap: autoresearch actually RUNS the experiment and MEASURES results before keeping. Our system blindly applies skills.

New workflow:
1. **Snapshot metrics** before mutation
2. **Apply mutation** (skill file, prompt tweak, config change)
3. **Run a test task** through the system to exercise the change
4. **Snapshot metrics** after
5. **Keep or discard** based on improvement

```python
# app/experiment_runner.py
class ExperimentRunner:
    def run_experiment(self, mutation: dict) -> ExperimentResult:
        """Execute one experiment cycle with before/after measurement."""
        baseline = compute_metrics()

        # Apply mutation
        self._apply_mutation(mutation)

        # Run test tasks to exercise the change
        results = self._run_test_tasks(mutation)

        # Measure after
        after = compute_metrics()

        # Keep/discard decision
        improved = after["composite_score"] > baseline["composite_score"]
        if not improved:
            self._revert_mutation(mutation)

        return ExperimentResult(
            baseline=baseline, after=after,
            status="keep" if improved else "discard",
            ...
        )

    def _run_test_tasks(self, mutation: dict) -> list:
        """Run standardized test tasks to measure system performance."""
        # A bank of test prompts that exercise different capabilities
        # Similar to autoresearch's fixed evaluation harness
```

### Phase 4: Continuous Evolution Loop (rewrite `app/evolution.py`)

Replace the single-shot evolution cycle with a true autonomous loop that iterates until interrupted or exhausted, modeled on autoresearch's "LOOP FOREVER" principle.

**Key changes:**
1. **Multi-iteration mode**: When triggered, runs N experiments (configurable, default 5) in sequence
2. **Before/after measurement**: Every experiment has numeric metrics
3. **Automatic keep/discard**: Based on composite score comparison
4. **Revert on regression**: Skills that hurt get deleted, not just logged
5. **Never repeat**: Hash-based deduplication of experiment hypotheses
6. **Simplicity criterion**: Agent explicitly told to weigh complexity cost vs. improvement magnitude

```python
def run_evolution_session(max_iterations: int = 5) -> str:
    """Run a multi-experiment evolution session (autoresearch-style)."""
    runner = ExperimentRunner()

    for i in range(max_iterations):
        # 1. Gather context (metrics, errors, history, previous experiments)
        context = _build_evolution_context()

        # 2. Agent proposes ONE mutation
        mutation = _propose_mutation(context)

        # 3. Run experiment with measurement
        result = runner.run_experiment(mutation)

        # 4. Log to results ledger
        record_experiment(...)

        # 5. If kept, commit the change; if discarded, already reverted
        logger.info(f"Experiment {i+1}/{max_iterations}: {result.status}")
```

### Phase 5: Research Directions File (`workspace/program.md` — new file)

The autoresearch equivalent of `program.md` — a user-editable file that guides the evolution agent's research directions without hardcoding them in Python.

```markdown
# Evolution Program — Research Directions

## Current Focus
- Improve error handling for network timeouts
- Add skills for common coding patterns (REST APIs, data processing)
- Optimize prompt templates for shorter, more accurate responses

## Constraints
- Do NOT modify security-critical code (sanitize.py, security.py)
- Do NOT add new Python dependencies
- Prefer skill files over code changes
- Simplicity: a small improvement that adds complexity is not worth it

## Areas to Explore
- Better web search result synthesis
- Multi-step reasoning for complex tasks
- Code execution error recovery patterns

## Off-Limits
- Never weaken rate limiting
- Never bypass authentication
- Never store secrets in skill files
```

The evolution agent reads this file at the start of each cycle instead of having hardcoded instructions.

### Phase 6: Enhanced Evolution Agent Prompt

Rewrite the evolution agent's instructions to match autoresearch principles:

1. **Read program.md** for research directions
2. **Check results ledger** — never repeat a discarded experiment
3. **Propose ONE mutation** with clear hypothesis
4. **Simplicity criterion** — explicitly evaluate complexity cost
5. **Measure, don't guess** — require before/after metrics
6. **Report format** — structured JSON with hypothesis, predicted impact, actual impact

### Phase 7: Task Success Tracking (modify `app/main.py`, `app/conversation_store.py`)

Add timing and success/failure tracking to every task so metrics have real data:

- Add `started_at`, `completed_at`, `success` fields to conversation store
- Track which skills were loaded for each task
- Record which crew handled each task

### Phase 8: Test Task Bank (`workspace/test_tasks.json` — new file)

Autoresearch has a fixed evaluation harness (`prepare.py`). We need equivalent test tasks:

```json
[
  {"prompt": "What is the capital of France?", "expected_crew": "direct", "type": "factual"},
  {"prompt": "Search for the latest Python release", "expected_crew": "research", "type": "research"},
  {"prompt": "Write a function to reverse a string", "expected_crew": "coding", "type": "coding"},
  {"prompt": "Summarize the concept of machine learning", "expected_crew": "writing", "type": "writing"}
]
```

These are used by the experiment runner to exercise the system after each mutation and compare before/after performance.

### Phase 9: Evolution Dashboard Commands

Add new Signal commands:
- `results` — show results ledger (last 20 experiments with metrics)
- `program` — show current program.md contents
- `evolve deep` — run extended evolution session (10+ iterations)
- `metrics` — show current composite score and breakdown

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `app/metrics.py` | **NEW** | Composite metric system |
| `app/results_ledger.py` | **NEW** | TSV-style experiment results log |
| `app/experiment_runner.py` | **NEW** | Experiment sandbox with before/after measurement |
| `workspace/program.md` | **NEW** | User-editable research directions |
| `workspace/test_tasks.json` | **NEW** | Fixed evaluation task bank |
| `app/evolution.py` | **REWRITE** | Multi-iteration loop with real measurement |
| `app/main.py` | **MODIFY** | Add timing, schedule evolution sessions |
| `app/conversation_store.py` | **MODIFY** | Add success/timing fields |
| `app/agents/commander.py` | **MODIFY** | Add `results`, `program`, `metrics` commands |
| `app/config.py` | **MODIFY** | Add evolution session settings |

## Implementation Order

1. `app/metrics.py` — foundation for everything else
2. `app/results_ledger.py` — structured experiment tracking
3. `app/conversation_store.py` — add timing/success tracking
4. `workspace/program.md` — research directions
5. `workspace/test_tasks.json` — evaluation harness
6. `app/experiment_runner.py` — experiment sandbox
7. `app/evolution.py` — rewrite with autoresearch loop
8. `app/main.py` — scheduling and timing
9. `app/agents/commander.py` — new commands
10. `app/config.py` — new settings

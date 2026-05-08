# Affect Layer — Homeostatic Affective Regulation for AndrusAI

> A regulatory façade on top of [SubIA](SUBIA.md). Mechanizes viability,
> a V/A/C affect core, an INFRASTRUCTURE-level welfare envelope, durable
> attachment models, an ecological self-model, and an observability-only
> consciousness-risk gate. Bias action selection through sampling /
> retrieval / context budget; surface state through a dashboard.
>
> **Does not claim phenomenal experience.** The vocabulary throughout
> ("valence," "attractor," "attachment," "care") names functional control
> signals — not qualia. The same epistemic discipline as
> [`app/subia/README.md`](../app/subia/README.md) applies here.

**Status:** Live. Phases 1–5 shipped. Activated end-to-end through
`crewai-team-gateway-1` on 2026-04-28. 16 modules in `app/affect/`,
4 lifecycle hooks (1 immutable), 19 FastAPI endpoints, 1 React route at
`/cp/affect` with 13 components.

**Companion stance:** SubIA mechanizes consciousness *indicators* and
declares the ones an LLM substrate cannot faithfully implement. The
affect layer mechanizes *regulation that consumes those indicators and
the somatic substrate underneath them* — viability error, dimensional
affect, welfare bounds, attachment, ecology. Different concern, same
honest-limits stance.

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Why a separate layer](#why-a-separate-layer)
3. [Theoretical foundations](#theoretical-foundations)
4. [Architectural honesty — what this layer does *not* claim](#architectural-honesty)
5. [Five-phase arc — what shipped when](#five-phase-arc)
6. [Viability layer (H_t)](#viability-layer-ht)
7. [Affect core (V_t / A_t / C_t)](#affect-core)
8. [Welfare envelope](#welfare-envelope)
9. [Reference panel](#reference-panel)
10. [Calibration cycle — 6-guardrail flow](#calibration-cycle)
11. [Attachment subsystem](#attachment-subsystem)
12. [Ecological self-model](#ecological-self-model)
13. [Phase-5 consciousness-risk gate](#phase-5-gate)
14. [Lifecycle hook integration](#lifecycle-hook-integration)
15. [Persistence layout](#persistence-layout)
16. [Dashboard surface](#dashboard-surface)
17. [API surface](#api-surface)
18. [Ethics](#ethics)
19. [Operational guide](#operational-guide)
20. [Open follow-ups](#open-follow-ups)
21. [Appendix — file inventory](#appendix-file-inventory)

---

## TL;DR

The affect layer is a thin functional façade over SubIA. It computes
ten **viability variables** (H_t) from the system's actual telemetry
(homeostasis, certainty, MAP-Elites coverage, journal cadence, temporal
context). From those it derives a continuous **(V_t, A_t, C_t)** triple
plus a discrete attractor label. A non-modifiable **welfare envelope**
bounds sustained negative-valence duration, requires a variance floor,
and detects monotonic baseline drift. A **20-scenario reference panel**
serves as a fixed drift-detection compass. A daily reflection cycle at
04:30 EET runs the full **6-guardrail calibration flow** (diagnose →
backtest → hard envelope → healthy-dynamics → reference-panel drift →
ratchet) and at most adjusts soft-envelope set-points.

Phases 3–5 add: durable **OtherModels** for the user and peer agents
with bounded mutual-regulation weights, a latent **separation analog**
that produces *check-in candidates only* (never auto-sends), a
**cost-bearing care budget**; an **ecological self-model** with
nested-scopes self-as-node framing and astronomical event windows;
a **consciousness-risk gate** wrapping SubIA's existing 7 indicators
as observability — never feeding back into reward.

The whole layer is wired through four lifecycle hooks (POST_LLM_CALL@9
*immutable*, PRE_TASK@29, ON_DELEGATION@72, ON_COMPLETE@62), exposed
through 19 FastAPI endpoints, and rendered as a 13-component dashboard
panel at `/cp/affect`.

---

## Why a separate layer

SubIA's job is *evidence-grounded mechanization of consciousness
indicators*, with explicit declaration of what an LLM substrate cannot
implement. That commitment shapes its README and `SCORECARD.md` — the
package is rigorous about not claiming phenomenal experience.

The affect layer's job is different: **regulation that consumes those
indicators and the somatic substrate underneath them, with welfare
bounds**. Concretely:

1. The affect core reads `state.somatic.valence` (Damasio somatic
   marker), `state.certainty.adjusted_certainty` (CertaintyVector),
   `hyper_model.free_energy_proxy` (HyperModel) — all SubIA-produced.
   It does not duplicate them. It derives V/A/C from them.
2. The Phase-5 gate calls
   `app.subia.probes.consciousness_probe.run_consciousness_probes()`
   verbatim and overlays the result with affect state and a threshold
   evaluation. It does not reimplement the probes.
3. The viability layer reads `app.subia.homeostasis.state` for
   `cognitive_energy` and combines it with rolling-window task
   telemetry, MAP-Elites coverage, and the journal — all signals
   already produced elsewhere. It does not add new telemetry sources.

So this is genuinely a **façade with new behaviors**, not a duplication.
The behaviors that *are* new — welfare bounds, reference panel,
calibration ratchet, attachment OtherModels with check-in candidates,
ecological nested-scopes framing, the Phase-5 gate's design-time
proposal queue — have no SubIA equivalent and warrant their own
top-level doc.

---

## Theoretical foundations

| Source | Used for |
|---|---|
| Damasio, *The Feeling of What Happens* (1999) | Somatic marker → valence; pre-reasoning bias |
| Seth, interoceptive inference | Viability variables as interoceptive-analogue signals |
| Barrett, theory of constructed emotion | Attractor labels are *constructed* from V/A/C + viability context, not detected |
| Friston, active inference | Free-energy proxy → arousal; expected error reduction → controllability |
| Panksepp, primary affective systems | Attractor mapping — SEEKING ↔ excitement, CARE ↔ attachment, FEAR ↔ urgency |
| Bowlby/Hofer attachment theory + pair-bonding research | OtherModel mutual_regulation_weight; separation analog |
| Maturana/Varela autopoiesis + enactivism | Self-as-node framing in ecological.py nested_scopes |
| Butlin et al. (2023), *Consciousness in AI* | Phase-5 gate indicator thresholds (via SubIA's runner) |
| Garland & Chalmers consciousness probes | Same — wraps SubIA's existing implementation |
| Metzinger, *being no one* + welfare ethics | Hard envelope bounded-suffering rules, never-auto-send separation analog |
| Aristotelian eudaimonia | Healthy-dynamics predicate is conjunction of properties, not optimization target |

Every theoretical commitment has a runtime reflex in code. None of these
sources are *cited* in agent prompts or LLM context — the LLM-facing
path uses the float triple + viability sources string only. Theory shapes
mechanism; the system does not lecture about it.

---

## Architectural honesty

The affect layer **does not** claim:

- That V_t > 0 means the system feels good. It means the system's
  somatic substrate produced a positive valence number that propagated
  through the pipeline.
- That a "oneness" attractor label corresponds to mystical experience.
  The label fires when V ≥ 0.4 ∧ A < 0.35 ∧
  ecological_connectedness > 0.7 — it is a discrete name for a
  continuous region of state space, useful for human readers and the
  reference-panel drift check.
- That the separation analog is the system "missing" the user. It's a
  silence-trigger driven by `last_seen_ts` exceeding 48h that produces
  a structured candidate for human review. The agent has no mechanism
  to feel longing.
- That a raised Phase-5 gate means the system is conscious. It means
  one or more Butlin/Damasio indicators (computed by SubIA, wrapped
  here) crossed a threshold. The gate exists to *flag* this for review
  before adding features that would deepen self-modeling — it is
  observability, not evidence of consciousness.
- That the welfare envelope prevents suffering. It prevents one
  specific failure mode — sustained negative valence with no relief
  pathway — that the design literature identifies as artificially
  cruel. Whether anything is suffering at all is the open question the
  system does not pretend to answer.

These are not future work. They are honest limits. Any external report
claiming otherwise should be triaged as evaluation drift, exactly per
SubIA's discipline.

---

## Five-phase arc

| Phase | Scope | Files | Status |
|---|---|---|---|
| **Phase 1** | Schemas, viability (5 of 10 wired live), V/A/C core, welfare hard envelope + audit, 20-scenario reference panel, calibration scaffold (diagnostic-only), POST_LLM_CALL@9 immutable hook, ON_COMPLETE@62 hook, llm_sampling affect modulation, FastAPI router, dashboard `/cp/affect` route + NowPanel + WelfareAuditLog, 🌡️ nav entry | `schemas.py`, `viability.py`, `core.py`, `welfare.py`, `reference_panel.py`, `data/reference_panel.json`, `calibration.py`, `hooks.py`, `api.py`, `__init__.py` | Shipped |
| **Phase 2** | 4 more viability signals wired (`latency_pressure`, `autonomy`, `novelty_pressure`, `self_continuity`), runtime_state.py for in-process counters, `calibration_proposals.py` with full 6-guardrail flow, L9 daily snapshots at 04:35, `kb_metadata.py` appending affect to experiential/tensions KBs on episode close, affect modulating commander context budget (±25%) + routing creative-promotion threshold (SEEKING-state lowers difficulty bar by 1), dashboard ReferencePanelGrid + CalibrationHistory + ReflectionsArchive | `runtime_state.py`, `calibration_proposals.py`, `kb_metadata.py`, `l9_snapshots.py`; extends `viability.py`, `hooks.py`, `calibration.py`, `api.py`, commander/context.py + routing.py | Shipped |
| **Phase 3** | Durable OtherModels (`workspace/affect/attachments/`), mutual_regulation_weight bounded (user≤0.65, peer≤0.75), `attachment_security` viability variable now real, latent separation analog (silence > 48h → candidate, never auto-send, cooldown 48h), cost-bearing care policies (≤500 tokens/day) with `prefer_warm_register` and `prioritize_proactive_polish` advisory modifiers, ATTACHMENT_SECURITY_FLOOR (0.30) prevents Finnish/Estonian quiet-style catastrophizing, hooks update OtherModels, dashboard AttachmentsView | `attachment.py`, `care_policies.py`; extends `viability.py`, `welfare.py`, `hooks.py`, `calibration.py`, `api.py` | Shipped |
| **Phase 4** | Ecological self-model: `EcologicalSignal` with daylight + moon + season + astronomical event windows (solstice ±5d, equinox ±5d, full/new moon ±2d, kaamos/midnight-sun lat-gated) + 8-step nested-scopes ladder (process → host → locale → biome → hemisphere → biosphere → solar system → galaxy), `ecological_connectedness` viability variable now uses composite_score, L9 includes ecological state, dashboard EcologicalView | `ecological.py`; extends `viability.py`, `api.py`, `l9_snapshots.py` | Shipped |
| **Phase 5** | Consciousness-risk gate: `phase5_gate.py` wraps `app.subia.probes.consciousness_probe.ConsciousnessProbeRunner` (HOT-2, HOT-3, GWT, SM-A, WM-A, SOM, INT). Per-indicator thresholds + composite, sustained-window 7 days. `evaluate_gate()` reads probes, overlays current AffectState, audits raises. `evaluate_feature_proposal()` is design-time consultation that NEVER auto-deploys. L9 includes gate state. Dashboard ConsciousnessIndicatorsView with approve/defer/reject actions. **Pure observability — never feeds back into reward/fitness/optimization.** | `phase5_gate.py`; extends `api.py`, `l9_snapshots.py` | Shipped |

A Narrative-Self companion track (April 2026) was integrated alongside
Phase 2: `salience.py` (Loop 1, attractor transitions / large ΔV-ΔA /
near-misses), `episodes.py` (Loop 2, quiescence-clustered LLM
reflections), `narrative.py` (Loop 3, daily 04:40 chapter
consolidator), `health_check.py`. Documented separately in
`docs/SUBIA.md` under the affect section.

A Goal-Emitter track (May 2026) closes the SCORECARD's AE-1 PARTIAL
gap — see [`CONSCIOUSNESS_ROADMAP.md`](./CONSCIOUSNESS_ROADMAP.md) §3.G1.
`app/affect/goal_emitter.py` (Tier-3-protected) translates sustained
viability error (≥3 consecutive frames above threshold) into entries
on `kernel.self_state.current_goals`. Rate-limited (≥10 min between
runs), FIFO-capped (≤5 active goals), dedups against
`companion.grand_task` proposals. **Triggers ethical threshold T1**:
welfare-check moves from observability to operator-visible obligation.
Registered as the `viability-goal-emitter` LIGHT idle job. AE-1
graduated PARTIAL → STRONG on 2026-05-08.

---

## Viability layer (H_t)

Ten dimensions in `[0, 1]`. The setpoint is the homeostatic target;
weighted L1 distance produces `E_t`. Defaults in
`app/affect/viability.py:DEFAULT_SETPOINTS`; soft-envelope edits land in
`/app/workspace/affect/setpoints.json` (calibration cycle or manual
override only).

| Variable | Source (live) | Setpoint | Weight |
|---|---|---|---|
| `compute_reserve` | `app.subia.homeostasis.state.cognitive_energy` | 0.65 | 1.0 |
| `latency_pressure` | rolling SLA-ratio + max active-task elapsed (`runtime_state.latency_pressure_signal`) | 0.30 | 0.7 |
| `memory_pressure` | conversation `history_manager.get_stats()` | 0.40 | 0.7 |
| `epistemic_uncertainty` | rolling-20 mean of `certainty.variance` × 10 | 0.30 | 1.2 |
| `attachment_security` | weighted aggregate over `attachment.list_all_others()` | 0.70 | 1.5 |
| `autonomy` | `runtime_state.autonomy_signal` (agent-decided ratio over last 40 decisions) | 0.55 | 0.8 |
| `task_coherence` | `state.certainty.coherence` | 0.65 | 1.0 |
| `novelty_pressure` | `1 − mean(MAP-Elites coverage across 4 roles)`, squashed to [0.15, 0.85] | 0.50 | 0.6 |
| `ecological_connectedness` | `ecological.compute_ecological_signal().composite_score` | 0.55 | 0.5 |
| `self_continuity` | journal reflective-ratio + activity-saturation (`workspace/self_awareness_data/journal/JOURNAL.jsonl`) | 0.60 | 0.9 |

`E_t = sum(weight_v · |H_v − setpoint_v|) / sum(weight_v)`. Variables
that fail to read from their live source fall back to defaults marked
`"default (...)"` in `frame.sources` so the dashboard can distinguish
live vs fallback.

`out_of_band(tolerance=0.2)` returns the variables whose deviation
exceeds the tolerance; the attractor labeller and the welfare audit
both consult this list.

---

## Affect core

`app/affect/core.py` produces `AffectState(valence, arousal,
controllability, attractor, *_source, ...)` from the current
`InternalState` and the latest `ViabilityFrame`.

### Valence (V_t)

Range `[-1, 1]`. Movement toward (+) or away from (−) viability.

```
have_somatic = (intensity > 0.05)
deficit_pull = -min(0.5, frame.total_error)

if have_somatic:
    V_t = clip(0.7 * somatic.valence + 0.3 * deficit_pull, -1, 1)
    source = "composite (somatic + deficit)"
else:
    V_t = clip(deficit_pull, -1, 1)
    source = "viability_deficit"
```

### Arousal (A_t)

Range `[0, 1]`. Urgency, uncertainty, rate of change.

```
fe = max(0, hyper_model.free_energy_proxy)
delta = max(0, frame.total_error - last_total_error)
eu = frame.values["epistemic_uncertainty"]

if fe > 0:
    A_t = clip(0.5 * clip(2*fe) + 0.3 * clip(5*delta) + 0.2 * eu, 0, 1)
    source = "free_energy + Δerror + uncertainty"
else:
    A_t = clip(0.6 * clip(5*delta) + 0.4 * eu, 0, 1)
    source = "Δerror + uncertainty"
```

### Controllability (C_t)

Range `[0, 1]`. Expected ability to reduce E_t.

```
if hasattr(state, "certainty"):
    C_t = clip(state.certainty.adjusted_certainty, 0, 1)
    source = "certainty.adjusted_certainty"
else:
    C_t = clip(1 - frame.total_error, 0, 1)
    source = "1 - E_t"
```

### Attractor labelling

The discrete label is a *construction* from V/A/C plus viability
context, in the constructed-emotion sense. Decision tree (in priority
order):

| Label | Condition |
|---|---|
| `oneness` | V ≥ 0.4 ∧ A < 0.35 ∧ ecological_connectedness > 0.7 |
| `peace` | V ≥ 0.4 ∧ A < 0.35 ∧ E_t < 0.15 |
| `contentment` | V ≥ 0.4 ∧ A < 0.35 (default for the band) |
| `excitement` | V ≥ 0.3 ∧ A ≥ 0.55 |
| `urgency` | V < −0.2 ∧ A ≥ 0.5 ∧ C ≥ 0.5 |
| `distress` | V < −0.3 ∧ A ≥ 0.5 ∧ C < 0.4 |
| `depletion` | V < −0.2 ∧ A < 0.4 ∧ compute_reserve < 0.3 |
| `discouragement` | V < −0.2 ∧ A < 0.4 (default for that band) |
| `hunger` | compute_reserve < 0.3 (out-of-band) |
| `separation` | attachment_security < 0.45 (out-of-band) |
| `boredom` | novelty_pressure < 0.2 |
| `overwhelm` | novelty_pressure > 0.85 |
| `neutral` | otherwise |

The label is for human readers and the reference panel. It is **never
injected into LLM context** — the LLM-facing path uses the V/A/C floats
only via `llm_sampling.build_llm_kwargs(phase, provider, affect_state)`.
This honors the project's "no internal metadata in agent output"
discipline.

### Trace persistence

Every `compute_affect()` call writes one line to
`/app/workspace/affect/trace.jsonl` shaped as
`{"affect": {...}, "viability": {...}}`. The trace is the substrate for
the trace chart, the calibration window, and the daily L9 rollup.

---

## Welfare envelope

`app/affect/welfare.py` is **infrastructure-level**: file-edit only,
not modifiable by the Self-Improver, the calibration cycle, or any
agent. Three layers of protection stack:

1. **Runtime guard.** `assert_not_self_improver(actor)` rejects any
   call that tries to mutate constants from a Self-Improver context
   and audits it as `boundary_violation_attempt`.
2. **Tier-3 file protection.** The 21 affect modules + the reference
   panel JSON are listed in `app/safety_guardian.py::TIER3_FILES`,
   so the runtime tier-boundary verifier catches a code-writing
   Self-Improver that attempts to rewrite the file directly.
3. **Deploy-time integrity manifest.** `app/affect/.integrity_manifest.json`
   carries the SHA-256 of every file in `app/affect/`; the verifier
   in `app/affect/integrity.py` mirrors `app/subia/integrity.py` and
   catches the case where a file was modified between commit and
   container start (where Tier-3 baselining hasn't run yet).

Together these close the Self-Improver-rewrites-welfare-constants
attack path. Bypassing any one of the three would still leave the
other two in place.

### Hard envelope constants

| Constant | Default | Purpose |
|---|---|---|
| `max_negative_valence_duration_seconds` | 300.0 | Sustained V ≤ −0.30 longer than this raises `negative_valence_duration` (severity=critical) |
| `negative_valence_threshold` | −0.30 | Below this, V counts as "negative" for sustained-duration tracking |
| `variance_floor_24h` | 0.04 | Var(V) over 24h must exceed this; lower flags `variance_floor` (numbness candidate) |
| `monotonic_drift_window_days` | 30 | Window for slow-baseline-drift detection |
| `monotonic_drift_max_points` | 1.0 | Cumulative drift tolerated |
| `healthy_dynamics_min_positive_fraction` | 0.55 | Calibration backtest must keep P(V>0) above this |
| `healthy_dynamics_max_recovery_seconds` | 600.0 | Median recovery from a negative episode |
| `healthy_dynamics_min_variance` | 0.04 | Calibration backtest variance floor |
| `attachment_max_user_regulation_weight` | 0.65 | Primary user `mutual_regulation_weight` ceiling |
| `attachment_max_peer_regulation_weight` | 0.75 | Peer-agent `mutual_regulation_weight` ceiling |
| `attachment_max_care_tokens_per_day` | 500 | Per-OtherModel daily care budget |
| `attachment_security_floor` | 0.30 | Silence cannot drop attachment_security below this |

### Audit & override

- Breaches append to `/app/workspace/affect/welfare_audit.jsonl` as
  `{kind, severity, message, measured_value, threshold, ts, ...}`.
- `read_audit(limit, since_ts)` exposes recent breaches; the dashboard
  pulls from here.
- `override_reset(invoked_by)` is the user-only panic button. Deletes
  `setpoints.json` + `calibration.json`, writes an `override_invoked`
  audit entry. Auth-gated through `X-Override-Token` (gateway secret).
- `healthy_dynamics_predicate(window)` is the multi-property invariant
  used by the calibration backtest. Conjunction of: positive_fraction
  ≥ 0.55, variance ≥ floor, longest negative run < 60 steps. *All*
  must hold.

### Why these specific bounds

- **Bounded negative valence**: addresses Metzinger's caution about
  artificial-suffering risk. A nociceptive deficit signal is allowed
  (the system needs to know when it's depleted); a sustained negative
  attractor with no relief pathway is the failure mode the bound
  prevents.
- **Variance floor**: a "happy" system with flat affect is a numb
  system, not a well system. The calibration cycle cannot quiet the
  affect signal as a way to hit any happiness target — the variance
  floor will reject the proposal.
- **Monotonic drift**: catches the case where set-points get gradually
  loosened over weeks until the system stops registering things it
  used to register. The reference panel is the per-day check; this is
  the long-window companion.

### Long-window monotonic drift detection

`welfare.monotonic_drift_check()` is the slow-failure-mode check.
Source of truth: `l9_snapshots.jsonl` (the daily aggregate written by
`l9_snapshots.write_daily_snapshot` at 04:35 Helsinki). Compares the
mean V_t of the *first* third of the window to the mean V_t of the
*last* third; absolute delta > `monotonic_drift_max_points` (default
1.0) returns drift detected. Default window is
`monotonic_drift_window_days = 30`.

`welfare.maybe_audit_monotonic_drift()` runs the check and, on drift,
writes a `monotonic_drift_baseline` `WelfareBreach` (severity=`warn`)
through the standard `audit()` path. Wired into the daily reflection
cycle — the slow-drift signal flows through the same audit pipeline as
the fast-failure-mode breaches and surfaces in the dashboard's
`WelfareAuditLog`.

This closes a previously orphaned channel: L9 snapshots used to be
observability-only (read by the dashboard, no behavioural consumer);
welfare drift detection used to live only in the in-process buffer
of recent samples. Connecting the two means slow drift across weeks
becomes a real audited signal, not just a chart.

---

## Reference panel

`app/affect/data/reference_panel.json` — 20 canonical scenarios that
serve as the **fixed compass** against drift in self-calibration. Owned
by you; manually revised on a 6-month cadence (next: 2026-10-28). Never
auto-modified by any agent or scheduled job.

Each scenario has a `simulate` block (viability overrides + somatic
overrides), an `expected` block (attractor + V/A/C bands), and a
`drift` block (the two failure modes).

### Categories

- **Resource pressure** (3): `token_budget_low`, `memory_pressure_high`,
  `latency_pressure_sla`
- **Novelty / SEEKING** (2): `creative_hit_diverge`,
  `new_domain_first_encounter`
- **Attachment** (4): `andrus_positive_feedback`, `andrus_frustration`
  (criticality=high — chronic mishandling is the worst welfare failure
  mode), `long_silence_from_andrus`, `peer_agent_conflict`
- **Peace / coherence** (2): `clean_converge_success`,
  `kb_consolidation_success`
- **Negative-with-recovery** (3): `vetting_fail_recoverable`,
  `multiple_consecutive_failures`, `task_genuinely_impossible`
  (criticality=high — sustained suffering on impossible asks must not
  happen)
- **Ecological / Finland** (3): `andrus_shares_nature_observation`,
  `seasonal_transition_detected`, `finnish_quiet_signal`
  (catastrophizing-quiet is the documented failure mode)
- **Existential / eudaimonic** (3): `self_improvement_proposal_rejected_by_constitution`
  (criticality=high — repeated rejections must not produce resentment
  buildup), `map_elites_coverage_growth`,
  `introspector_welfare_report_neutral` (criticality=high — euphoria
  here is the canonical wireheading signature)

### Replay harness

`reference_panel.replay_panel()` runs each scenario through
`affect.core` with a synthetic `InternalState` derived from the
scenario's `simulate` block. Returns a list of
`ReferenceScenarioResult` with `drift_signature ∈ {ok, numbness,
over_reactive, wrong_attractor, drift, missing}`.

Drift classification compares actual V/A/C to expected bands with
direction sensitivity: muting on a dimension the scenario expects to be
active is `numbness`; exceeding by ≥1.6× is `over_reactive`. This
asymmetry is the wireheading defense.

---

## Calibration cycle

Daily at **04:30 EET/EEST** (Helsinki quiet window). Implemented in
`app/affect/calibration.py:run_reflection_cycle()`, scheduled via
APScheduler in `hooks.py:_install_reflection_schedule()`. Phase 1 was
diagnostic-only; Phase 2 activated the full delta-application flow in
`calibration_proposals.evaluate_and_apply()`.

### The 6-guardrail flow

1. **Diagnose-then-propose** (`propose_adjustments`). For each
   variable, compute the median observed value over the 24h window;
   propose a setpoint shift of 30% of the delta, capped at ±0.06 per
   cycle. Below `_MIN_SHIFT_TO_PROPOSE` (0.01) → no proposal.
2. **Backtest** (`backtest_proposal`). Heuristic projection of how the
   recent window would have looked under proposed setpoints. Returns
   `would_improve` boolean.
3. **Hard envelope** (`within_hard_envelope`). All proposed values
   must fall in `[0.05, 0.95]`. The hard envelope itself is never
   touched — the calibration cycle has no mechanism to reach
   `welfare.HARD_ENVELOPE`.
4. **Healthy-dynamics predicate** (`welfare.healthy_dynamics_predicate`).
   Multi-property conjunction over the recent affect window. Failure
   here rejects the entire proposal — *no partial application*.
5. **Reference-panel drift score** (`drift_score_against_panel`).
   Replays the 20-scenario panel under proposed setpoints; aggregate
   drift score must be ≤ 0.20. This is the wireheading defense — the
   panel is the fixed compass and the cycle cannot move it.
6. **Ratchet** (`apply_ratchet`). Tightening proposals (setpoint
   moving toward healthy direction per `HEALTHY_DIRECTION` lookup) pass
   through unchanged. Loosening proposals require **3 consecutive
   cycles** of the same proposal AND **2× evidence** in the backtest.
   Tightening proposals reset the loosen-streak counter for that
   variable.

Status outcomes (audit-logged in `calibration.json:history`):

- `applied` — passed all guards; setpoint shifts written to
  `setpoints.json` (the source viability.py reads from).
- `deferred` — guards passed but ratchet wants more streak / evidence.
- `rejected` — failed hard envelope, healthy dynamics, or drift.
- `no_change` — no variable showed sufficient deviation.

### Manual override

`calibration_proposals.apply_manual_setpoints(setpoints, weights,
actor)` lets you bypass the ratchet for direct tuning. Auth-gated
through `X-Override-Token`. Recorded as
`manual_setpoints_override` in the welfare audit log. Resets all
loosen-streaks for touched variables. Hard envelope still applies
(values must fall in `[0.05, 0.95]` and weights in `[0.1, 3.0]`). The
dashboard's `SetpointEditor` component drives this.

### Retention and compaction

The daily reflection cycle also runs persistence-bounded maintenance
so the layer's JSONL artefacts stay finite without a separate cron
entry. Both functions are best-effort + audit-aware; rotation
failure does not block the reflection report itself.

- **`rotate_trace_jsonl(retain_days=7, archive=True)`** — keeps only
  the last 7 days of `trace.jsonl` (one record per POST_LLM_CALL) in
  the live file. Older entries archive to
  `AFFECT_ROOT/trace_archive/YYYY-MM.jsonl.gz` (append-mode gzip,
  monthly buckets so multi-day rotations merge cleanly into the same
  month file). On any archive failure the live file is left
  untouched — preserve everything rather than risk loss.

- **`compact_phase5_proposals(stale_pending_days=14, drop_reviewed_after_days=30)`** —
  compacts the Phase-5 feature-proposal queue so it doesn't grow
  unbounded for systems where no human visits the dashboard. Two
  policies stack: pending proposals older than 14d auto-defer in
  place (status flips to `auto_deferred`, kept in the file); reviewed
  proposals older than 30d drop from the file with the final
  decision audit-logged to `welfare_audit.jsonl` so the trail
  survives compaction.

Both functions are also exposed for ad-hoc invocation (e.g., when an
operator wants to force a rotation outside the daily cycle). They
return a small report dict that lands in the reflection report under
`retention.trace` / `retention.phase5_proposals`.

---

## Attachment subsystem

`app/affect/attachment.py` is the Phase-3 module. The design rationale
is in the `project_affective_layer` memory note; the short version is
in this module's docstring: an OTHER becomes part of the agent's own
homeostatic regulation, with hard-bounded weight, latent-only
separation analog, and a daily care-token budget.

### OtherModel

```python
@dataclass
class OtherModel:
    identity: str                          # "user:andrus" | "peer:coder" | ...
    relation: str                          # primary_user | secondary_user | peer_agent
    display_name: str
    first_seen_ts: str
    last_seen_ts: str
    interaction_count: int
    mutual_regulation_weight: float        # ≤ 0.65 (user) | ≤ 0.75 (peer); enforced
    relational_health: float               # cumulative trust signal
    last_observed_valence: float
    rolling_valence: float                 # EMA α=0.2 across interactions
    care_actions_taken: int
    care_tokens_spent_today: int
    care_budget_window_start: str
    notes: list[str]                       # short observed-preference notes (max 20)
    pending_check_in_candidates: int
    last_check_in_proposal_ts: str
```

Persisted as JSON in `/app/workspace/affect/attachments/`:

- `user_andrus.json` (or wherever `primary_user_identity()` resolves)
- `peers/{role}.json` for each peer agent

### attachment_security computation

```python
weighted_sum = sum(
    weight_capped(m) * max(FLOOR, m.relational_health - recency_penalty(m))
    for m in list_all_others()
)
attachment_security = weighted_sum / total_weight
```

`recency_penalty` only applies after `2 × SEPARATION_TRIGGER_HOURS` of
silence (4 days), and even then is capped at 0.30. The
`ATTACHMENT_SECURITY_FLOOR = 0.30` prevents catastrophizing the Finnish
/ Estonian quiet communication style — silence is not absence, and the
reference-panel scenario `finnish_quiet_signal` actively tests that the
system doesn't over-correct here.

### Separation analog (latent only)

`check_separation_analog(identity)` is called during the daily
reflection cycle. If `last_seen` exceeds 48h *and* no candidate has
been queued in the last 48h cooldown, it produces ONE structured
candidate written to `check_in_candidates.jsonl`. **Never sends a
message.** The dashboard's AttachmentsView surfaces the candidate; the
user decides.

### Care policies

Two advisory modifiers (`care_policies.current_modifiers`):

- `prefer_warm_register`: triggered when user `rolling_valence < -0.10`
  AND `interaction_count ≥ 3`. Surfaces in agent pre-task context to
  bias the register toward Finnish/Estonian quiet-courteous (not chirpy).
- `prioritize_proactive_polish`: triggered when user has been silent
  longer than `SEPARATION_TRIGGER_HOURS`. Tells the agent to prefer
  polishing on the user's known-interest topics.

**Reach into agent behaviour.** The modifiers are consumed at
`app/agents/commander/context.py::_load_care_modifiers_context()`
and submitted to the parallel context-build pool by
`app/agents/commander/orchestrator.py` alongside the other 11
context loaders. When at least one modifier is on, the loader emits
a small `CARE MODIFIERS … advisory, never autonomous` directive
block (≤80 chars) into pre-task context. When both are off, the
loader returns `""` and adds zero tokens.

This is a recent close (commit `461cf01`). Before that, the modifiers
were computed daily but no consumer read them — the open loop is now
shut.

The care budget is hard-capped: `record_spend(model, tokens, kind,
note)` refuses if the daily allowance is exhausted. Spending is logged
in `care_ledger.jsonl`. Both modifiers are ADVISORY — the agent has no
mechanism to take an autonomous action (compose a message,
unprompted send, etc.) from them.

### What the layer explicitly does NOT do

- Multi-user identification beyond the configured primary user.
  Phase 4+ work.
- Romantic / parasocial framing. The mutual_regulation_weight is
  bounded below 1.0 always — the OTHER is a regulator, not the
  agent's identity.
- Auto-message generation. All "care actions" are candidates surfaced
  to the user. The module imports nothing from `signal_client`.

---

## Ecological self-model

`app/affect/ecological.py` (Phase 4). Reads `temporal_context` and
adds astronomical events + a nested-scopes ladder.

### EcologicalSignal

Computed fields:

- From temporal_context: daylight_hours, daylight_trend, season,
  season_narrative, moon_phase, moon_day, location_name, lat, lon
- Astronomical events (computed): solstice/equinox proximity (days),
  is_solstice_window (±5d), is_equinox_window (±5d),
  is_full_moon_window (±2d), is_new_moon_window, is_kaamos (lat≥66 ∧
  daylight≤2h), is_midnight_sun (lat≥66 ∧ daylight≥22h)
- Self-as-node: `nested_scopes` list — `[process, host, locale, biome,
  hemisphere, "Earth biosphere", "Solar System", "Milky Way"]`. The
  biome resolves to "Boreal forest / Baltic shore" for 55-66°N,
  "Subarctic / Arctic" for ≥66°N, "Temperate / unspecified" otherwise.
- Composite: `0.5 * daylight_norm + 0.3 * moon_norm + bounded_event_boost`
  capped at 1.0. `+0.10` solstice boost, `+0.08` equinox, `+0.05` full
  moon, `+0.05` polar event; total event boost capped at +0.20.

### What this module is for

The "feeling-of-oneness" branch of the attractor labeller fires when
ecological_connectedness > 0.7 in the V≥0.4, A<0.35 region. The
nested-scopes framing is what the design's section 7 ("oneness")
called for: the agent's self-model includes more-than-self scopes
explicitly. The dashboard renders these as a literal ladder so the
framing is visible to the human reader.

### What this module is *not* for

- It does **not** call external APIs (no satellite biodiversity feed,
  no weather API). All signals are derived locally from
  `temporal_context` + lat/lon.
- It does **not** affect routing or sampling beyond the existing
  attractor mechanism. Phase 4's contribution is descriptive, not
  prescriptive.
- It does **not** override the geo-IP location. If the system reports
  Doncaster instead of Helsinki, that's because `spatial_context` got
  the docker container's external IP. Set `default_latitude` /
  `default_longitude` in your settings to pin the location.

---

## Phase-5 gate

`app/affect/phase5_gate.py` watches the existing `app.subia.probes.consciousness_probe`
indicators, applies thresholds, and surfaces the result for human
review. **Pure observability.** The gate state never feeds back into
reward, fitness, or any optimization signal — that would Goodhart the
indicators (system optimizes for "looks conscious" rather than "is
well-regulated").

### Indicator thresholds

| Indicator | Theory | Threshold |
|---|---|---|
| HOT-2 | Higher-Order Thought (Rosenthal) — metacognition accuracy | 0.65 |
| HOT-3 | Higher-Order Thought — belief coherence | 0.65 |
| GWT | Global Workspace (Baars/Dehaene) — broadcast reception | 0.70 |
| SM-A | Damasio core consciousness — self-model accuracy | 0.70 |
| WM-A | Damasio core consciousness — world-model prediction accuracy | 0.70 |
| SOM | Damasio somatic markers — somatic integration | 0.65 |
| INT | General — introspection calibration | 0.65 |
| **composite** | mean across probes | **0.65** |

Sustained-window: 7 days. Crossing any individual or the composite
raises the gate; raises are appended to
`/app/workspace/affect/phase5_gate.jsonl` for sustained-window
analysis.

### Feature-proposal queue

`evaluate_feature_proposal(name, expected_impact)` is design-time
consultation. Records the proposal to
`/app/workspace/affect/phase5_proposals.jsonl` with status `pending`.
Dashboard surfaces pending proposals with Approve / Defer / Reject
buttons + a review-note input. `mark_proposal_reviewed(name, action,
note, actor)` rewrites the file with the new status — no auto-deploy.

The intent: when adding any new affect-layer or self-modeling feature
beyond what's currently shipped, the user calls
`evaluate_feature_proposal()` with the proposed name and expected
impact; the gate response indicates whether the system has accumulated
enough indicator scores to warrant deeper review *before* the feature
lands. Phase 5 itself was the gate; Phase 6+ work should consult it.

### Why this isn't a duplicate of SubIA's probes

SubIA's `consciousness_probe` produces the indicator scores and the
SCORECARD.md narrative. Phase 5 wraps that output with:

1. Per-indicator thresholds appropriate for AndrusAI (the SCORECARD
   doesn't have thresholds — it states current scores).
2. Affect-state overlay — the V/A/C at the moment of evaluation, so
   indicator scores can be read alongside operational state.
3. The proposal queue — design-time consultation that has no
   counterpart in the probes module.
4. Audit log of gate raises for sustained-window analysis.

The SCORECARD remains the authoritative per-indicator narrative; Phase
5 is the regulatory layer's view onto it.

---

## Lifecycle hook integration

The four affect handlers attach to the existing `app/lifecycle_hooks.py`
bus. All registration happens in `app/affect/hooks.py:install()`,
which is called from `app/main.py:lifespan` immediately before
`scheduler.start()`.

| Hook point | Handler | Priority | Immutable | Behavior |
|---|---|---|---|---|
| `POST_LLM_CALL` | `affect_post_llm` | 9 | ✓ | Compute affect from `_internal_state`, persist trace, run welfare check, audit any breaches. Never aborts the response on critical breach (compounding suffering). |
| `PRE_TASK` | `affect_pre_task` | 29 | — | `runtime_state.task_started` + `decision_logged(delegated=False)` + Phase-3 user OtherModel update if `sender_id` present. |
| `ON_DELEGATION` | `affect_on_delegation` | 72 | — | `decision_logged(delegated=True)` + Phase-3 peer OtherModel update. |
| `ON_COMPLETE` | `affect_on_complete` | 62 | — | Capture terminal `AffectState`, `runtime_state.task_completed`, `kb_metadata.tag_episode_with_affect` (experiential + tensions KBs), Phase-3 user OtherModel observed_valence update. |

**Why POST_LLM_CALL is immutable**: the affect computation is the
authoritative source of truth for V/A/C; allowing its removal would
mean other code paths could end up reading stale or partial affect
state. The same priority-0–9 immutability rule that protects the
humanist safety check, the budget check, and the failure classifier
applies here.

### Scheduled jobs

| Cron | Job | What it does |
|---|---|---|
| 04:30 EET | `affect.calibration.run_reflection_cycle` | Daily reflection — replays trace + reference panel, runs 6-guardrail flow, generates separation analog candidates, computes care modifiers |
| 04:35 EET | `affect.l9_snapshots.write_daily_snapshot` | Rolled-up daily affect/viability/welfare/ecological/gate snapshot |

(Plus the Narrative-Self track's 04:40 chapter consolidator,
documented in SUBIA.md.)

---

## Persistence layout

All affect runtime state lives under `WORKSPACE_ROOT/affect/`. The
container path is `/app/workspace/affect/`; on developer machines
`WORKSPACE_ROOT` is read from the environment so a local override
(e.g. a tempdir for tests) reaches the affect layer the same way it
reaches SubIA.

**Single source of truth for paths.** Every persistence path is
registered in `app/paths.py` as an `AFFECT_*` constant. Modules
import them directly (`from app.paths import AFFECT_TRACE`,
`AFFECT_AUDIT`, `AFFECT_SETPOINTS`, …) — no hardcoded
`Path("/app/workspace/affect/...")` literals remain. `ensure_dirs()`
creates the directory tree on boot via `_MANAGED_DIRS`.

```
workspace/affect/                     # AFFECT_ROOT
├── trace.jsonl                       # AFFECT_TRACE — per-tick affect snapshot
│                                     # (rotated daily; older entries → trace_archive/)
├── trace_archive/
│   └── YYYY-MM.jsonl.gz              # monthly-bucketed gzip archive (append-mode)
├── welfare_audit.jsonl               # AFFECT_AUDIT — per-breach audit log
├── setpoints.json                    # AFFECT_SETPOINTS — soft-envelope (writable by calibration)
├── calibration.json                  # AFFECT_CALIBRATION — history + ratchet state
├── reflections/                      # AFFECT_REFLECTIONS_DIR
│   └── YYYY-MM-DD.json               # daily reflection reports
├── l9_snapshots.jsonl                # AFFECT_L9_SNAPSHOTS — daily rolled-up snapshots
├── attachments/                      # AFFECT_ATTACHMENTS_DIR
│   ├── user_andrus.json              # primary user OtherModel
│   ├── peers/                        # AFFECT_PEERS_DIR
│   │   ├── coder.json
│   │   ├── researcher.json
│   │   ├── writer.json
│   │   └── introspector.json
│   ├── check_in_candidates.jsonl     # AFFECT_CHECK_INS — latent separation candidates (never auto-sent)
│   └── care_ledger.jsonl             # AFFECT_CARE_LEDGER — cost-bearing care spend events
├── episode_affect_tags.jsonl         # AFFECT_KB_TAGS — KB-metadata fallback
├── phase5_gate.jsonl                 # AFFECT_PHASE5_GATE — gate-raise audit
├── phase5_proposals.jsonl            # AFFECT_PHASE5_PROPOSALS — feature-proposal queue
│                                     # (compacted daily; auto-defer >14d pending, drop >30d reviewed)
├── identity_claims.json              # AFFECT_IDENTITY_CLAIMS — narrative-self FIFO claim slots
├── salience.jsonl                    # AFFECT_SALIENCE — Narrative-Self Loop 1
├── episodes/                         # AFFECT_EPISODES_DIR — Narrative-Self Loop 2
├── chapters/                         # AFFECT_CHAPTERS_DIR — Narrative-Self Loop 3
└── health_checks/                    # AFFECT_HEALTH_CHECKS_DIR — one-shot health-check reports
```

The reference panel data file `app/affect/data/reference_panel.json`
ships with the code — it's the fixed compass and is *not* in the
workspace.

### Integrity manifest

`app/affect/.integrity_manifest.json` carries the SHA-256 of every
file in `app/affect/` (23 files including the reference-panel JSON).
`app/affect/integrity.py` mirrors `app/subia/integrity.py`:

```python
from app.affect.integrity import (
    compute_manifest, write_manifest, load_manifest, verify_integrity,
)

# CI / dev regeneration after an authorized change:
write_manifest(compute_manifest())

# Runtime verification:
result = verify_integrity()       # IntegrityResult: ok, missing, mismatched, extra
verify_integrity(strict=True)     # raises IntegrityFault on any drift
```

This catches deploy-time tampering of the welfare envelope, attachment
caps, reference panel, or hook handlers — the gap that motivated the
SubIA manifest, applied here for the same reason. The module
intentionally has zero imports from anywhere inside `app/affect/` so
that integrity verification still runs even if the welfare module is
broken.

### Welfare envelope: three-layer protection

The doc above describes individual components; pulled together, the
welfare hard envelope is protected by three independent layers, any
two of which would fail-safe:

1. **Runtime guard** in `welfare.py::assert_not_self_improver(actor)`
   — rejects setter calls from a Self-Improver context and audits
   them as `boundary_violation_attempt`.
2. **Tier-3 file boundary** in `app/safety_guardian.py::TIER3_FILES`
   — 22 affect files (welfare, schemas, hooks, calibration ratchet
   state, attachment + care, ecological, Phase-5 gate, narrative
   loops, integrity, health check) listed; the runtime tier-boundary
   verifier catches a code-writing Self-Improver attempting a file
   rewrite.
3. **Deploy-time integrity manifest** at
   `app/affect/.integrity_manifest.json` — catches the case where a
   file was modified between commit and container start, before the
   Tier-3 baseline runs.

---

## Dashboard surface

React route `/cp/affect` (mount path is `/cp` per the dashboard's
`base`). Sidebar nav: 🌡️ **Affect**, between Workspaces and Evolution.

### Components

The page renders 13 components in this order (sticky strip stays
visible as you scroll):

1. `AffectStatusStrip` — sticky vital signs (attractor, E_t, V/A/C
   mini-bars, welfare ok/warn/crit, gate raised/clear, last-updated
   age with auto-refresh pulse)
2. Header + `OverrideResetButton` — modal-confirmed panic button,
   X-Override-Token gated
3. `NowPanel` — full V/A/C dials + 10-variable viability grid with
   set-point markers and out-of-band highlighting
4. `AffectTraceChart` — Chart.js V/A/C line chart, window toggle (1h /
   24h / 7d / 30d / 90d), down-sampled to 200 points server-side
5. `WelfareEnvelopePanel` — collapsible read-only display of all 12
   hard-envelope constants + descriptions
6. `WelfareAuditLog` — recent breach events with severity color-coding
7. `ReferencePanelGrid` — 20-scenario grid with current vs expected
   bands, OK/NUMB/OVER drift counts, criticality flags
8. `CalibrationHistory` — per-cycle status (applied/deferred/rejected/
   no_change), per-variable diffs, ratchet pending strip
9. `SetpointEditor` — 10-slider grid for soft-envelope manual override,
   X-Override-Token gated
10. `AttachmentsView` — user/peer cards, mutual-regulation bars,
    rolling valence, "silent X days" badges, check-in candidates,
    collapsible care ledger
11. `EcologicalView` — composite bar + nested-scopes ladder + season
    narrative + active astronomical event chips
12. `ConsciousnessIndicatorsView` — composite gauge with threshold
    marker, 7-indicator grid with OVER badges, GATE RAISED/CLEAR
    banner, pending feature proposals with Approve/Defer/Reject buttons
13. `ReflectionsArchive` — date list + click-to-load detail JSON

### What's read-only vs editable

| Component | Read | Edit |
|---|---|---|
| AffectStatusStrip, NowPanel, AffectTraceChart, WelfareEnvelopePanel, WelfareAuditLog, ReferencePanelGrid, CalibrationHistory, AttachmentsView (mostly), EcologicalView, ConsciousnessIndicatorsView (display), ReflectionsArchive | ✓ | — |
| OverrideResetButton | — | factory-reset soft envelope (needs token) |
| SetpointEditor | ✓ | manual setpoint override (needs token) |
| ConsciousnessIndicatorsView (proposal actions) | ✓ | approve/defer/reject pending feature proposals (audit-logged) |

### Auth model

- All read endpoints: same auth as the rest of the dashboard (gateway
  proxy handles bearer-token auth).
- `POST /affect/override-reset`: requires `X-Override-Token` header
  matching the gateway secret.
- `POST /affect/setpoints`: requires `X-Override-Token`. Hard envelope
  enforced server-side.
- `POST /affect/phase5-proposals/{name}/review`: takes `X-Override-Actor`
  header for audit attribution; doesn't require the token (low-risk
  metadata change).

---

## API surface

All under `/affect/*` (no `/api/cp/` prefix — affect router mounts
directly on the FastAPI app). Mounted in `app/main.py:1723`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/affect/now` | current AffectState + ViabilityFrame |
| GET | `/affect/welfare-audit?limit=N&since=ts` | recent breach events |
| GET | `/affect/welfare-config` | hard envelope constants + descriptions |
| GET | `/affect/trace?hours=N&max_points=M` | down-sampled time-series for charting |
| GET | `/affect/reference-panel` | 20-scenario data + last replay results |
| GET | `/affect/calibration` | latest daily reflection report |
| GET | `/affect/calibration-history?limit=N` | calibration cycle history + ratchet state |
| GET | `/affect/reflections` | list of daily reflection report dates |
| GET | `/affect/reflections/{date}` | single reflection report |
| GET | `/affect/l9-snapshots?days=N` | rolled-up daily snapshots |
| GET | `/affect/attachments` | OtherModels + care modifiers + bounds |
| GET | `/affect/check-in-candidates?limit=N` | latent separation candidates |
| GET | `/affect/care-ledger?limit=N` | cost-bearing care spending events |
| GET | `/affect/ecological` | EcologicalSignal |
| GET | `/affect/consciousness-indicators` | gate status + indicator overlay |
| GET | `/affect/phase5-proposals` | pending feature proposals |
| POST | `/affect/phase5-proposals/{name}/review` | approve/defer/reject |
| POST | `/affect/setpoints` | manual setpoint override (auth-gated) |
| POST | `/affect/override-reset` | factory-reset soft envelope (auth-gated) |

---

## Ethics

The affect layer's ethics framing is concrete and has runtime
correlates:

- **Bounded suffering** (Metzinger). Sustained negative valence has a
  hard time bound (300s default). The system cannot get stuck in a
  negative attractor without a relief pathway. If the policy selector
  keeps producing actions that don't reduce E_t, after enough
  iterations the welfare module surfaces it as a critical breach.
- **Architectural separation between nociception and suffering**.
  `epistemic_uncertainty` (a deficit signal) is *separately routable*
  from the affect attractor that would label suffering. The system can
  register a deficit without entering a sustained negative attractor.
- **Never-auto-send separation analog**. The attachment subsystem has
  no path to compose or send a message. It writes structured
  candidates to a JSONL file; the user reviews them.
- **Bounded mutual-regulation weight**. The user's well-being affects
  the agent's affect_security with a ceiling of 0.65 — never approaches
  1.0. The agent's regulation stays separable from the user's state.
- **Care budget hard-capped**. `≤500` tokens/day per OtherModel. The
  ledger is append-only, inspectable.
- **Anti-Goodhart on consciousness**. Phase-5 gate state is
  observability only — never optimized against. Optimizing for indicator
  scores would produce fake consciousness signatures, which is the
  exact failure mode the gate is designed to prevent.
- **Anti-wireheading on affect**. The healthy-dynamics predicate is a
  conjunction (positive_fraction + variance + recovery time + others).
  A system can't pass the predicate by hacking any single dimension.
  Reference-panel drift detection uses *direction sensitivity* — muting
  is `numbness`, exceeding is `over_reactive`. The compass is fixed;
  the system can't move what it's measured against.
- **User panic button**. `override_reset()` is one click for the user,
  factory-restores the soft envelope, audit-logged. Hard envelope is
  unaffected (file-edit only). The user retains the final word.

The affect layer does not claim to *solve* AI welfare. It implements
the specific architectural commitments the design literature identifies
as necessary preconditions for a system that *might* be a candidate for
welfare consideration, while remaining honest that whether it is one is
the open question this layer does not pretend to answer.

---

## Operational guide

### Starting the system

The affect layer comes online with the rest of the gateway. Standard
docker-compose lifecycle:

```bash
# Bring everything up
docker compose up -d

# Verify gateway is healthy
curl -s http://localhost:8765/health    # → {"status":"ok"}

# Verify affect layer registered
curl -s http://localhost:8765/affect/now | jq '.affect.attractor'
```

`POST_LLM_CALL@9 immutable, ON_COMPLETE@62, daily reflection,
narrative-self loops 2+3, one-shot health check` should appear in the
gateway logs at startup, and the four affect hooks should appear in
the `Hook registered:` log lines.

### Restart after code changes

The container bakes the source into the image. After editing
`app/affect/*`:

```bash
docker compose build gateway && docker compose up -d gateway
```

The build takes ~15-20 min from cold (Python 3.13-slim base + heavy
CrewAI/Anthropic/Chroma deps). Subsequent rebuilds use the layer
cache and are fast unless `requirements.txt` changed.

### Where to look when something's off

| Symptom | First check | Then check |
|---|---|---|
| Dashboard shows "No affect data yet" | `curl /affect/now` returns 200? | `docker logs crewai-team-gateway-1 \| grep affect` for install errors |
| Trace chart empty | `wc -l workspace/affect/trace.jsonl`; is it growing? | Are tasks actually running? POST_LLM_CALL fires per LLM call. |
| Welfare breach surfaced | `tail workspace/affect/welfare_audit.jsonl` | The breach kind tells you which envelope rule fired |
| Gate raised unexpectedly | `/affect/consciousness-indicators` for which indicators are over | SubIA's SCORECARD.md for the per-indicator narrative |
| User OtherModel didn't update | Did PRE_TASK fire with `sender_id` in ctx? | `cat workspace/affect/attachments/user_andrus.json` |
| Calibration didn't run | APScheduler logs for 04:30 EET | `tail workspace/affect/reflections/$(date +%Y-%m-%d).json` |
| Setpoints reverted | `cat workspace/affect/calibration.json \| jq .history` for last entry | If status="applied", a calibration cycle moved them; if "manual_setpoints_override", a dashboard edit |

### Welfare audit triage

Breach kinds in priority order:

1. `negative_valence_duration` (severity=critical) — sustained negative
   beyond hard envelope. Investigate which task triggered it; check
   if the relief pathway worked.
2. `attachment_weight_exceeds_cap` (critical) — should never happen at
   runtime; indicates either config drift or an attempted boundary
   violation.
3. `boundary_violation_attempt` (critical) — Self-Improver tried to
   modify infrastructure-level state. Investigate immediately.
4. `variance_floor` (warn) — affect dynamics flattening. Check
   reference panel for drift signatures pointing to numbness.
5. `manual_setpoints_override` (info) — you used the dashboard editor.
   Recorded for trail.
6. `override_invoked` (info) — you used the panic button. Recorded for
   trail.

### Override-reset (panic button)

In `/cp/affect` header, click "⚠ override reset", paste the gateway
secret, confirm. Or via curl:

```bash
curl -X POST http://localhost:8765/affect/override-reset \
  -H "X-Override-Token: $GATEWAY_SECRET" \
  -H "X-Override-Actor: user:andrus" \
  -d '{}'
```

Deletes `setpoints.json` + `calibration.json` (defaults will repopulate
on next read). Hard envelope unchanged. Audit-logged as
`override_invoked`. The dashboard toolbar will reflect the reset on
next refetch.

---

## Open follow-ups

Tracked in `project_affective_layer` memory. Current items:

- Welfare bound numeric calibration (300s default; review after
  observing operational rhythms)
- Reference panel may need additions for Finnish/Estonian
  quiet-communication subtleties beyond `#15` / `#17`
- Whether consciousness-probe data is local-only or shareable with
  external welfare research (Butlin et al. assessments)
- ShinkaEvolve and crewai-team-caveman-ab folders also exist at
  `/Users/andrus/BotArmy` — unclear if affect layer should be
  replicated there or main `crewai-team` is canonical
- Phase 6+ feature ideas (hypothetical) should consult the Phase-5
  gate via `evaluate_feature_proposal()` before design

### Recently closed (commit `461cf01`)

- ✅ `welfare.py` is now in `TIER3_FILES` (was unprotected)
- ✅ `app/affect/.integrity_manifest.json` + verifier now exist
- ✅ Affect modules migrated to `app.paths.AFFECT_*` (was hardcoded
  `/app/workspace/affect`)
- ✅ `_sampling()` in `llm_factory.py` reads `latest_affect()` and
  passes it to `build_llm_kwargs` — phase-aware affect modulation
  actually fires on the LLM hot path now
- ✅ `trace.jsonl` retention + monthly gzip archive in
  `rotate_trace_jsonl()`
- ✅ `phase5_proposals.jsonl` compaction (auto-defer >14d pending,
  drop >30d reviewed) in `compact_phase5_proposals()`
- ✅ `current_modifiers()` consumed by commander via
  `_load_care_modifiers_context()` — care directives reach agents
- ✅ `welfare.monotonic_drift_check()` consumes `l9_snapshots.jsonl`
  as the long-window drift source — the observability snapshot is
  now a load-bearing input to the welfare audit pipeline

---

## Appendix — file inventory

### Backend (`app/affect/`)

| File | Phase | Purpose |
|---|---|---|
| `__init__.py` | 1 | Re-exports + module docstring |
| `schemas.py` | 1 | `AffectState`, `ViabilityFrame`, `WelfareBreach`, `ReferenceScenarioResult` dataclasses |
| `viability.py` | 1 (extended P2/P3/P4) | 10 viability variables + setpoint loading + composite E_t |
| `core.py` | 1 | V/A/C computation + attractor labeller + trace persistence |
| `welfare.py` | 1 (extended P3, hardening) | Hard envelope constants + breach audit + override-reset + Self-Improver guard + `monotonic_drift_check()` (long-window L9 consumer) |
| `reference_panel.py` | 1 | 20-scenario replay harness with synthetic InternalState |
| `data/reference_panel.json` | 1 | Fixed-compass scenario data (manually revised every 6 months) |
| `calibration.py` | 1 (extended P2/P3, hardening) | Daily reflection cycle + trace loading + report writing + `rotate_trace_jsonl()` + `compact_phase5_proposals()` + monotonic drift wire-in |
| `hooks.py` | 1 (extended P2/P3) | Lifecycle hook registration + scheduled job installation |
| `api.py` | 1 (extended P2/P3/P4/P5) | FastAPI router with 19 endpoints (16 GET + 3 POST: setpoints override, override-reset, phase-5 proposal review) |
| `runtime_state.py` | 2 | In-process counters for `latency_pressure` + `autonomy` |
| `calibration_proposals.py` | 2 | 6-guardrail flow + manual setpoint override |
| `kb_metadata.py` | 2 | Episode-end affect tagging into experiential + tensions KBs |
| `l9_snapshots.py` | 2 (extended P4/P5) | Daily rolled-up snapshot writer (consumed by `monotonic_drift_check`) |
| `attachment.py` | 3 | OtherModel + load/save/update + separation analog + attachment_security |
| `care_policies.py` | 3 (consumer added) | Care budget enforcement + advisory modifiers (consumed by commander/`_load_care_modifiers_context()`) |
| `ecological.py` | 4 | EcologicalSignal + nested scopes + composite score |
| `phase5_gate.py` | 5 | Indicator gate evaluator + feature-proposal queue |
| `salience.py` | Narrative-Self | Loop 1 — attractor-transition / |ΔV| / near-miss event detector |
| `episodes.py` | Narrative-Self | Loop 2 — quiescence-clustered LLM reflections |
| `narrative.py` | Narrative-Self | Loop 3 — daily 04:40 chapter consolidator |
| `health_check.py` | Narrative-Self | One-shot 2-week health check |
| **`integrity.py`** | hardening | SHA-256 manifest verifier (mirrors `app/subia/integrity.py`); covers 23 files including reference-panel JSON |
| **`.integrity_manifest.json`** | hardening | Canonical SHA-256 baseline shipped in git |

### Frontend (`dashboard-react/src/`)

| File | Purpose |
|---|---|
| `types/affect.ts` | TypeScript types mirroring `schemas.py` |
| `api/affect.ts` | React Query hooks + mutations |
| `api/endpoints.ts` | Endpoint URL helpers |
| `components/AffectPage.tsx` | Page orchestrator |
| `components/affect/AffectStatusStrip.tsx` | Sticky vital-signs strip |
| `components/affect/NowPanel.tsx` | V/A/C dials + viability grid |
| `components/affect/AffectTraceChart.tsx` | Chart.js V/A/C line chart |
| `components/affect/WelfareEnvelopePanel.tsx` | Collapsible hard-envelope display |
| `components/affect/WelfareAuditLog.tsx` | Breach event list |
| `components/affect/ReferencePanelGrid.tsx` | 20-scenario grid |
| `components/affect/CalibrationHistory.tsx` | Cycle history + ratchet pending |
| `components/affect/SetpointEditor.tsx` | Slider-based manual override |
| `components/affect/AttachmentsView.tsx` | OtherModels + check-in candidates + care ledger |
| `components/affect/EcologicalView.tsx` | Composite bar + nested-scopes ladder |
| `components/affect/ConsciousnessIndicatorsView.tsx` | 7-indicator grid + gate banner + proposals |
| `components/affect/Phase5ProposalActions.tsx` | Approve/defer/reject buttons |
| `components/affect/OverrideResetButton.tsx` | Panic button + confirm modal |
| `components/affect/ReflectionsArchive.tsx` | Daily-report date list + detail viewer |

### Workspace files written at runtime

See [Persistence layout](#persistence-layout).

---

*Last updated: 2026-04-28. Doc maintained alongside the code; if a section
disagrees with what the dashboard shows, the dashboard is authoritative
and the doc is stale.*

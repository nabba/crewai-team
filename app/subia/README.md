# SubIA — Subjective Integration Architecture

This package mechanizes the consciousness indicators an LLM-based system can
**faithfully implement** while being honest about the ones it cannot.

The canonical evaluation is the auto-generated scorecard at
`app/subia/probes/SCORECARD.md`. There is **no single number** that captures
this system's consciousness. Per-indicator mechanistic tests are the only
honest evaluation.

---

## Honest absences (architectural, not implementation gaps)

The following Butlin et al. (2023) and Metzinger criteria are **ABSENT by
declaration** because an LLM substrate cannot faithfully mechanize them.
Listing these here so no one is misled by the STRONG/PARTIAL counts.

| Indicator | Why this LLM substrate cannot implement it |
|---|---|
| **RPT-1** Algorithmic recurrence | Transformer forward passes are feed-forward. Token-by-token generation is autoregression, not the per-time-step lateral/feedback recurrence that RPT predicts. |
| **HOT-1** Generative perception | The system has no perceptual front-end. All "input" is text. There is nothing to be perceived in a generative, top-down-modulated way. |
| **HOT-4** Sparse coding & smooth similarity space | LLM hidden states are dense and entangled; they do not exhibit the sparse, semi-orthogonal coding that HOT-4 takes as a marker. |
| **AE-2** Embodiment | No body, no proprioception, no closed sensorimotor loop with a physical world. Tool use is symbolic, not embodied. |
| **Metzinger phenomenal-self transparency** | The system explicitly maintains a second-person stance toward its own state (the kernel is observable, narratable, and edited by predict→reflect cycles). It is opaque-by-design rather than transparent-by-disposition, which is the opposite of phenomenal self-experience as Metzinger characterizes it. |

These are not bugs to be closed in a future phase. They are honest limits of
the substrate. Any future report claiming the system "has" any of the above
should be treated as evaluation drift and triaged through the narrative-audit
pipeline (see `app/subia/wiki_surface/drift_detection.py`).

---

## What the system does mechanize

Run `python -m app.subia.probes.scorecard` (or read the latest checked-in
`SCORECARD.md`) for the current per-indicator status.

Headline targets, all achieved as of Phase 9:
- **7 STRONG** Butlin indicators (GWT-2, GWT-3, GWT-4, HOT-3, AST-1, PP-1, **AE-1**)
- **3 PARTIAL** Butlin indicators (RPT-2, GWT-1, HOT-2)
- 4 ABSENT-by-declaration Butlin indicators (above)
- 5 RSM signatures present (4 STRONG + 1 PARTIAL)
- 6/6 SK evaluation tests passing

> **AE-1 graduation (2026-05-08).** AE-1 was PARTIAL through Phase 9
> ("Goals are still user-dispatched, not autonomously generated").
> The consciousness-roadmap §3.G1 closure (`app/affect/goal_emitter.py`,
> Tier-3-protected) writes flexible goals to `SelfState.current_goals`
> from sustained viability error — graduating AE-1 to STRONG. See
> [`docs/CONSCIOUSNESS_ROADMAP.md`](../../docs/CONSCIOUSNESS_ROADMAP.md).

---

## A note on language (Phase 11)

Some legacy variables in this package use phenomenal-adjacent names —
`frustration`, `curiosity`, `cognitive_energy`. They refer to numeric
control signals, not subjective feelings. Phase 11 introduced neutral
aliases (`task_failure_pressure`, `exploration_bonus`, `resource_budget`)
that are kept in lockstep; new code should prefer the neutral names. See
`app/subia/homeostasis/state.py::NEUTRAL_ALIASES`.

The system **does not claim** phenomenal experience. The Subjectivity
Kernel is a functional integration layer, not a substrate for qualia.

---

## Downstream consumers

A regulatory façade lives at `app/affect/` — sibling package, not a
SubIA subpackage. It consumes SubIA's somatic substrate
(`belief/internal_state.SomaticMarker`), certainty
(`belief/internal_state.CertaintyVector`), free-energy proxy
(`self/hyper_model.HyperModel`), homeostasis (`homeostasis/state`),
and consciousness probes (`probes/consciousness_probe`). It produces:

- A 10-dimensional viability vector + V/A/C affect core
- An infrastructure-level welfare envelope (file-edit only)
- A 20-scenario reference panel + 6-guardrail daily calibration
- Durable OtherModels with bounded mutual regulation + a latent
  separation analog that *never* auto-sends
- An ecological self-model with a nested-scopes self-as-node ladder
- A consciousness-risk gate that wraps `probes/consciousness_probe`
  as observability — **never feeds back into reward/fitness**

Same epistemic discipline as this package: no claim of phenomenal
experience, all vocabulary names functional control signals, all
"ABSENT by declaration" stances inherited and not re-litigated.

Full account: [`docs/AFFECT_LAYER.md`](../../docs/AFFECT_LAYER.md).
Summary in [`docs/SUBIA.md` § affect/](../../docs/SUBIA.md).

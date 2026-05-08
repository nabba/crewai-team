"""
subia.loop — the 11-step Consciousness Integration Loop (CIL).

The loop sequences the five Phase-2-closed gates into a single
pre-task / post-task pair. Per Amendment B, only Step 5 (Predict)
requires an LLM call on the hot path; every other step is
deterministic arithmetic over existing kernel state.

              PRE-TASK                            POST-TASK
    ┌────────────────────────┐            ┌────────────────────────┐
    │ 1  Perceive            │            │ 7  Act (task runs)     │
    │ 2  Feel (homeostasis)  │            │ 8  Compare (PE error)  │
    │ 3  Attend (scene gate) │            │ 9  Update (state)      │
    │ 4  Own (self-state)    │            │ 10 Consolidate (memory)│
    │ 5  Predict (LLM tier1) │            │ 11 Reflect (audit)     │
    │ 5b Cascade modulation  │            └────────────────────────┘
    │ 6  Monitor             │
    └────────────────────────┘

Operation classification (from SUBIA_CONFIG):
  - FULL_LOOP_OPERATIONS    run all 11 steps
  - COMPRESSED_LOOP_OPS     run steps 1-3, 7-9 only

The loop is pure orchestration. It does not:
  - Call external databases directly (gates handle persistence)
  - Perform LLM calls except through the injected predict_fn
  - Mutate global state (all changes flow through gates)

Failure containment: a step that raises is logged and the loop
continues. A crashed step must never break the agent task.

Infrastructure-level. Not agent-modifiable. See PROGRAM.md Phase 4.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from app.subia.config import SUBIA_CONFIG
from app.subia.kernel import (
    Prediction,
    SceneItem,
    SubjectivityKernel,
)

logger = logging.getLogger(__name__)


class _NullLock:
    """Fallback context manager when a gate has no _lock attribute
    (e.g. tests with stub gates). Never raises.
    """
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


# ── Result types ───────────────────────────────────────────────────

@dataclass
class StepOutcome:
    """Record of a single step's execution."""
    step: str
    ok: bool = True
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class CILResult:
    """Aggregate result of a full or compressed loop invocation."""
    loop_type: str                  # "full" | "compressed"
    phase: str                      # "pre_task" | "post_task"
    steps: list = field(default_factory=list)    # List[StepOutcome]
    total_elapsed_ms: float = 0.0
    context_for_agent: dict = field(default_factory=dict)
    within_budget: bool = True
    budget_ms: float = 0.0

    def add(self, outcome: StepOutcome) -> None:
        self.steps.append(outcome)
        self.total_elapsed_ms += outcome.elapsed_ms

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def step(self, name: str) -> Optional[StepOutcome]:
        for s in self.steps:
            if s.step == name:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "loop_type": self.loop_type,
            "phase": self.phase,
            "ok": self.ok,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "budget_ms": self.budget_ms,
            "within_budget": self.within_budget,
            "steps": [
                {
                    "step": s.step,
                    "ok": s.ok,
                    "elapsed_ms": round(s.elapsed_ms, 2),
                    "error": s.error,
                    "details": s.details,
                }
                for s in self.steps
            ],
        }


# ── Operation classification ───────────────────────────────────────

def classify_operation(operation_type: str) -> str:
    """Return 'full' or 'compressed' based on SUBIA_CONFIG."""
    if operation_type in SUBIA_CONFIG.get("FULL_LOOP_OPERATIONS", ()):
        return "full"
    if operation_type in SUBIA_CONFIG.get("COMPRESSED_LOOP_OPERATIONS", ()):
        return "compressed"
    # Unknown operations default to compressed — be cheap by default.
    return "compressed"


# ── Loop implementation ────────────────────────────────────────────

class SubIALoop:
    """Orchestrator for the 11-step Consciousness Integration Loop.

    Dependencies are injected at construction so the loop is testable
    in complete isolation. Production wiring (Phase 4 lifecycle hook
    integration) will build an instance with real gates; tests pass
    doubles.

    Args:
        kernel:          SubjectivityKernel to read/write.
        scene_gate:      CompetitiveGate (for Step 3 admissions).
        predict_fn:      Callable[[dict], Prediction] — step 5 predictor.
                         Dict keys: agent_role, task_description, scene,
                         self_state, homeostasis, history_window.
                         Returns a Prediction. May call an LLM; the loop
                         does not care.
        predictive_layer: optional PredictiveLayer for step 8 error
                         computation + surprise routing.
        hierarchy:       optional PredictionHierarchy for Step 5 injection
                         string.
        consult_fn:      Callable[[str, str, str], list] returning
                         consulted beliefs for Step 6. Dict keys:
                         task_description, crew_name, goal_context.
        dispatch_decider: optional DispatchDecider callable returning a
                         DispatchDecision — defaults to the
                         subia.belief.dispatch_gate module function.
        hedger:          optional callable for Step 11 response hedging
                         (only used if the orchestrator feeds outputs
                         through here on post-task).
        now:             clock for testability (default: time.monotonic).
    """

    def __init__(
        self,
        kernel: SubjectivityKernel,
        scene_gate: Any | None = None,
        predict_fn: Callable[[dict], Prediction] | None = None,
        predictive_layer: Any | None = None,
        hierarchy: Any | None = None,
        consult_fn: Callable[..., list] | None = None,
        dispatch_decider: Callable[..., Any] | None = None,
        hedger: Callable[..., tuple] | None = None,
        accuracy_tracker: Any | None = None,
        mem0_curated: Any | None = None,
        mem0_full: Any | None = None,
        neo4j_client: Any | None = None,
        scorecard_fn: Callable[[], dict] | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.kernel = kernel
        self._gate = scene_gate
        self._predict_fn = predict_fn
        self._predictive_layer = predictive_layer
        self._hierarchy = hierarchy
        self._consult_fn = consult_fn
        self._dispatch_decider = dispatch_decider
        self._hedger = hedger
        self._accuracy_tracker = accuracy_tracker
        self._mem0_curated = mem0_curated
        self._mem0_full = mem0_full
        self._neo4j_client = neo4j_client
        self._scorecard_fn = scorecard_fn
        self._now = now
        self._current_domain = ""   # set in pre_task; read by cascade step

        # Attach gate to predictive_layer so PP-1 routing fires.
        if self._predictive_layer is not None and self._gate is not None:
            try:
                self._predictive_layer.set_gate(self._gate)
            except AttributeError:
                pass

    # ── Public API ────────────────────────────────────────────────

    def pre_task(
        self,
        *,
        agent_role: str,
        task_description: str,
        operation_type: str = "task_execute",
        input_items: Iterable[SceneItem] = (),
        goal_context: str = "",
    ) -> CILResult:
        """Run the pre-task half of the CIL. Returns a CILResult with
        aggregated step outcomes and an injectable `context_for_agent`
        dict the caller hands to the agent.
        """
        loop_type = classify_operation(operation_type)
        budget_ms = float(
            SUBIA_CONFIG["FULL_LOOP_LATENCY_BUDGET_MS"]
            if loop_type == "full"
            else SUBIA_CONFIG["COMPRESSED_LOOP_LATENCY_BUDGET_MS"]
        )
        result = CILResult(
            loop_type=loop_type, phase="pre_task", budget_ms=budget_ms,
        )
        t_start = self._now()

        # Phase 6: pin the current domain so Step 5b cascade can look
        # up sustained-error for this (agent_role, operation_type) pair.
        try:
            from app.subia.prediction.accuracy_tracker import domain_key
            self._current_domain = domain_key(agent_role, operation_type)
        except Exception:
            self._current_domain = f"{agent_role}:{operation_type}"

        # Step 1: PERCEIVE (deterministic)
        self._run(result, "1_perceive", lambda: self._step_perceive(input_items))

        # Step 2: FEEL (deterministic — arithmetic on homeostasis)
        self._run(result, "2_feel", self._step_feel)

        # Step 3: ATTEND (deterministic — scene gate admissions)
        self._run(result, "3_attend", self._step_attend)

        if loop_type == "compressed":
            # ── Phase 14 §3.G4: compressed-loop quick-bind ───────────────
            # The full Step-6 bind is unavailable here (Steps 4-6 don't run
            # on the compressed path). The quick-bind uses just the
            # already-computed FEEL + ATTEND outputs to produce a partial
            # BoundMoment so observability surfaces and downstream consumers
            # see at least `dominant_affect` and `salient_focus` on
            # compressed cycles. `confidence_unified` and `conflicts` stay
            # at dataclass defaults — they require PREDICT/MONITOR/OWN.
            try:
                from app.subia.temporal_hooks import (
                    quick_bind_compressed_signals,
                )
                attend_items = [
                    {"id": i.id, "salience": float(getattr(i, "salience", 0.5))}
                    for i in self.kernel.focal_scene()
                ]
                bm = quick_bind_compressed_signals(
                    feel={
                        "dominant_affect":
                            self.kernel.homeostasis.variables.get(
                                "dominant_affect", "neutral"
                            ),
                    },
                    attend={"focal_items": attend_items},
                )
                if bm is not None:
                    qb = StepOutcome(step="3b_quick_bind", ok=True)
                    qb.details["dominant_affect"] = bm.dominant_affect
                    qb.details["salient_focus_count"] = len(bm.salient_focus)
                    qb.details["compressed_loop"] = True
                    result.add(qb)
            except Exception:
                logger.debug("phase14 quick_bind_compressed failed", exc_info=True)

            result.context_for_agent = self._build_compressed_context()
            result.total_elapsed_ms = (self._now() - t_start) * 1000.0
            result.within_budget = result.total_elapsed_ms <= budget_ms
            return result

        # Step 4: OWN (deterministic — ownership tagging)
        self._run(result, "4_own", self._step_own)

        # Step 5: PREDICT (LLM tier 1, the one allowed hot-path call)
        self._run(result, "5_predict",
                  lambda: self._step_predict(agent_role, task_description))

        # Step 5b: Cascade modulation (deterministic)
        self._run(result, "5b_cascade", self._step_cascade)

        # Step 6: MONITOR + belief-gated dispatch decision
        self._run(result, "6_monitor",
                  lambda: self._step_monitor(
                      task_description=task_description,
                      crew_name=agent_role,
                      goal_context=goal_context,
                      operation_type=operation_type,
                  ))

        result.context_for_agent = self._build_full_context()
        result.total_elapsed_ms = (self._now() - t_start) * 1000.0
        result.within_budget = result.total_elapsed_ms <= budget_ms
        return result

    def post_task(
        self,
        *,
        agent_role: str,
        task_description: str,
        operation_type: str = "task_execute",
        task_result: dict | None = None,
        actual_content: str = "",
        actual_embedding: list[float] | None = None,
    ) -> CILResult:
        """Run the post-task half of the CIL."""
        loop_type = classify_operation(operation_type)
        budget_ms = float(
            SUBIA_CONFIG["FULL_LOOP_LATENCY_BUDGET_MS"]
            if loop_type == "full"
            else SUBIA_CONFIG["COMPRESSED_LOOP_LATENCY_BUDGET_MS"]
        )
        result = CILResult(
            loop_type=loop_type, phase="post_task", budget_ms=budget_ms,
        )
        t_start = self._now()
        task_result = task_result or {}

        # Step 8: COMPARE (prediction error, may route via PP-1)
        self._run(result, "8_compare",
                  lambda: self._step_compare(
                      agent_role=agent_role,
                      task_description=task_description,
                      operation_type=operation_type,
                      actual_content=actual_content,
                      actual_embedding=actual_embedding,
                  ))

        # Step 9: UPDATE (deterministic kernel state updates)
        self._run(result, "9_update",
                  lambda: self._step_update(task_result))

        if loop_type == "compressed":
            result.total_elapsed_ms = (self._now() - t_start) * 1000.0
            result.within_budget = result.total_elapsed_ms <= budget_ms
            self.kernel.loop_count += 1
            self.kernel.touch()
            self._persist_kernel()
            return result

        # Step 10: CONSOLIDATE (Phase 7 — dual-tier memory)
        self._run(result, "10_consolidate",
                  lambda: self._step_consolidate(
                      task_result,
                      agent_role=agent_role,
                      operation_type=operation_type,
                  ))

        # Step 11: REFLECT (periodic narrative audit — placeholder)
        self._run(result, "11_reflect", self._step_reflect)

        # Advance kernel cycle
        self.kernel.loop_count += 1
        self.kernel.touch()

        # Advance gate cycles when available
        for obj, attr in (
            (self._gate, "advance_cycle"),
            (self._predictive_layer, "advance_cycle"),
        ):
            fn = getattr(obj, attr, None) if obj is not None else None
            if callable(fn):
                try:
                    fn()
                except Exception:
                    logger.debug("advance_cycle raised", exc_info=True)

        # Persist kernel after each full post_task so restarts resume
        # from the most recent CIL state. Throttled by loop-count modulo
        # so we don't write the markdown file on every compressed call —
        # the compressed path already returned earlier.
        self._persist_kernel()

        result.total_elapsed_ms = (self._now() - t_start) * 1000.0
        result.within_budget = result.total_elapsed_ms <= budget_ms
        return result

    # ── Step implementations ──────────────────────────────────────

    def _step_perceive(self, input_items: Iterable[SceneItem]) -> dict:
        """Step 1: ingest candidates into the kernel's transient buffer.

        Phase 14 addition: refresh the SpeciousPresent + homeostatic
        momentum + TemporalContext before candidates enter the kernel.
        This gives the rest of the loop access to retention/primal/
        protention + circadian mode + processing density. Wrapped in
        try/except matching existing step-level failure containment.

        Phase 12 addition: tag every unclassified scene item with its
        Boundary Sense `processing_mode` (introspective / perceptual /
        memorial / imaginative / social) so downstream gates and the
        consolidator can route per-mode.
        """
        try:
            from app.subia.temporal_hooks import refresh_temporal_state
            prev_focal_ids = {i.id for i in self.kernel.focal_scene()}
            prev_homeostasis = dict(self.kernel.homeostasis.variables)
            refresh_temporal_state(
                self.kernel,
                previous_focal_ids=prev_focal_ids,
                previous_homeostasis=prev_homeostasis,
            )
        except Exception:
            logger.debug("phase14 refresh_temporal_state failed", exc_info=True)

        count = 0
        self._candidates = []
        for item in input_items:
            self._candidates.append(item)
            count += 1

        # Phase 12 Boundary Sense — stamp processing_mode on candidates
        # before the gate competes them. Tagging at perceive-time means
        # gating, consolidation, and value-resonance all see a stable
        # mode tag (DGM constraint: source→mode mapping is in config,
        # not agent code).
        boundary_tagged = 0
        try:
            from app.subia.boundary.classifier import classify_scene
            boundary_tagged = classify_scene(self._candidates)
        except Exception:
            logger.debug("phase12 boundary classify failed", exc_info=True)

        return {"candidates": count, "boundary_tagged": boundary_tagged}

    def _step_feel(self) -> dict:
        """Step 2: deterministic homeostatic update from candidates.

        Delegates to subia.homeostasis.engine.update_homeostasis
        which computes per-variable deltas from the scene candidates
        and recomputes deviations + restoration_queue.
        """
        from app.subia.homeostasis.engine import update_homeostasis
        return update_homeostasis(
            self.kernel,
            new_items=getattr(self, "_candidates", []),
        )

    def _step_attend(self) -> dict:
        """Step 3: admissions + Amendment A three-tier attentional build.

        1. Submit each candidate to the gate (focal admissions).
        2. Build focal + peripheral tiers from the combined
           active + peripheral pool.
        3. Enforce commitment-orphan protection: any active
           commitment without representation is force-injected
           into peripheral with an alert.

        The tiers are stored on the loop instance so context
        builders can render them without re-walking the gate.
        """
        if self._gate is None:
            self._tiers = None
            return {"gate": "not_attached"}

        admitted = 0
        rejected = 0
        for candidate in getattr(self, "_candidates", []):
            try:
                result = self._gate.evaluate(candidate)
                if getattr(result, "admitted", False):
                    admitted += 1
                else:
                    rejected += 1
            except Exception:
                logger.debug("scene gate evaluate failed", exc_info=True)

        # Amendment A: build the three-tier structure from the gate's
        # current active + peripheral pools. Sort by salience so
        # focal takes the top N regardless of insertion order.
        from app.subia.scene.tiers import (
            build_attentional_tiers,
            protect_commitment_items,
        )
        try:
            with getattr(self._gate, "_lock", _NullLock()):
                pool = list(getattr(self._gate, "_active", []))
                pool.extend(getattr(self._gate, "_peripheral", []))
        except Exception:
            pool = []

        # Phase 8: before tier-building, let social models nudge
        # salience — items matching an entity's inferred_focus
        # (especially Andrus) get a small upward boost. This is how
        # "Andrus cares about X" actually reaches the attentional
        # bottleneck.
        social_boost_report = None
        try:
            from app.subia.social.salience_boost import apply_salience_boost
            social_boost_report = apply_salience_boost(
                pool, self.kernel.social_models,
            )
        except Exception:
            logger.debug("social salience_boost failed", exc_info=True)

        pool.sort(
            key=lambda i: float(getattr(i, "salience_score", 0.0)),
            reverse=True,
        )

        tiers = build_attentional_tiers(pool)
        # Commitment-orphan protection
        tiers = protect_commitment_items(
            tiers,
            scored_items=pool,
            commitments=getattr(self.kernel.self_state,
                                "active_commitments", []),
        )
        self._tiers = tiers

        # Mirror the focal items into kernel.scene so subsystems that
        # iterate kernel.scene (boundary classifier on later cycles,
        # value resonance, persistence, consolidator, retrospective
        # promotion) actually see the currently-attended items. The
        # kernel.scene was previously left empty — items lived only on
        # the gate's internal lists, invisible to consumers.
        previous_focal_ids = {getattr(i, "id", None) for i in self.kernel.focal_scene()}
        new_scene: list = []
        for item in tiers.focal:
            if not getattr(item, "tier", None):
                try:
                    item.tier = "focal"
                except Exception:
                    pass
            new_scene.append(item)
        # Peripheral entries (tiers.peripheral) are PeripheralEntry
        # records, not SceneItems — keep them out of kernel.scene to
        # preserve type discipline. The compact context renderer reads
        # tiers.peripheral directly.
        self.kernel.scene = new_scene

        # Phase 12 Value Resonance + Phronesis lenses (apply against
        # the now-populated kernel.scene). Closes the Step 3 hook
        # documented in app/subia/phase12_hooks.py.
        lens_report: dict = {}
        try:
            from app.subia.phase12_hooks import (
                apply_value_resonance_and_lenses,
            )
            lens_report = apply_value_resonance_and_lenses(self.kernel) or {}
        except Exception:
            logger.debug("phase12 value resonance failed", exc_info=True)

        details = {
            "admitted": admitted,
            "rejected": rejected,
            "focal": len(tiers.focal),
            "peripheral": len(tiers.peripheral),
            "alerts": len(tiers.peripheral_alerts),
        }
        if social_boost_report is not None:
            details["social_boost"] = social_boost_report.to_dict()
        if lens_report:
            details["value_resonance"] = lens_report.get("items_modulated", 0)
            if lens_report.get("lens_aggregate"):
                details["lens_aggregate"] = lens_report["lens_aggregate"]
        return details

    def _step_own(self) -> dict:
        """Step 4: tag admitted items with ownership — deterministic."""
        # Kernel scene items already carry an `ownership` field; without
        # a specific policy to override, default everything to 'self'.
        tagged = 0
        for item in self.kernel.scene:
            if not getattr(item, "ownership", None):
                item.ownership = "self"
                tagged += 1
        return {"newly_tagged": tagged, "total_items": len(self.kernel.scene)}

    def _step_predict(self, agent_role: str, task_description: str) -> dict:
        """Step 5: counterfactual prediction (LLM tier 1 if predict_fn)."""
        if self._predict_fn is None:
            return {"predict_fn": "not_attached"}
        prediction = self._predict_fn({
            "agent_role": agent_role,
            "task_description": task_description,
            "scene": list(self.kernel.scene),
            "self_state": self.kernel.self_state,
            "homeostasis": self.kernel.homeostasis,
            "prediction_history": list(self.kernel.predictions)[
                -SUBIA_CONFIG["PREDICTION_HISTORY_WINDOW"]:
            ],
        })
        self.kernel.predictions.append(prediction)
        return {
            "prediction_id": getattr(prediction, "id", ""),
            "confidence": getattr(prediction, "confidence", 0.5),
        }

    def _step_cascade(self) -> dict:
        """Step 5b: cascade tier modulation via subia.prediction.cascade.

        Combines three signals: single-prediction confidence, homeostatic
        coherence deviation, and per-domain sustained-error flag from
        the accuracy tracker. See subia/prediction/cascade.py.
        """
        from app.subia.prediction.accuracy_tracker import (
            domain_key,
            get_tracker,
        )
        from app.subia.prediction.cascade import decide_cascade

        last_pred = self.kernel.predictions[-1] if self.kernel.predictions else None
        confidence = float(getattr(last_pred, "confidence", 0.5) if last_pred else 0.5)
        coherence_dev = float(
            self.kernel.homeostasis.deviations.get("coherence", 0.0)
        )
        domain = getattr(self, "_current_domain", "")
        tracker = self._accuracy_tracker or get_tracker()
        sustained = tracker.has_sustained_error(domain) if domain else False

        decision = decide_cascade(
            prediction_confidence=confidence,
            homeostatic_coherence_deviation=coherence_dev,
            domain=domain,
            sustained_error=sustained,
        )
        self._cascade_recommendation = decision.recommendation
        self._cascade_decision = decision
        return {
            "recommendation": decision.recommendation,
            "confidence": confidence,
            "sustained_error": sustained,
            "reasons": list(decision.reasons),
        }

    def _step_monitor(
        self,
        *,
        task_description: str,
        crew_name: str,
        goal_context: str,
        operation_type: str = "",
    ) -> dict:
        """Step 6: monitor + belief-gated dispatch decision + social update.

        Uses the Phase-2-closed HOT-3 dispatch_gate. Beliefs come from
        the injected consult_fn; if no consult_fn is wired, we skip
        the gate and ALLOW by default (loop continues functioning
        even when the belief subsystem is inert).

        Phase 8 addition: periodic social-model update for the default
        'andrus' human entity. Uses focal-scene topics as the
        "topics touched" signal so Andrus's inferred_focus tracks
        what the system has been attending to on his behalf.
        """
        # ── Belief-gated dispatch (Phase 2) ─────────────────────
        details: dict
        if self._consult_fn is None:
            self._dispatch_decision = None
            details = {"dispatch": "no_consult_fn"}
        else:
            try:
                beliefs = list(self._consult_fn(
                    task_description=task_description,
                    crew_name=crew_name,
                    goal_context=goal_context,
                ))
            except Exception:
                logger.debug("consult_fn raised", exc_info=True)
                beliefs = []

            decider = self._dispatch_decider
            if decider is None:
                from app.subia.belief.dispatch_gate import decide_dispatch
                decider = decide_dispatch

            decision = decider(
                consulted_beliefs=beliefs,
                suspended_candidates=(),
                task_description=task_description,
                crew_name=crew_name,
            )
            self._dispatch_decision = decision
            details = {
                "verdict": getattr(decision, "verdict", "ALLOW"),
                "belief_count": getattr(decision, "belief_count", 0),
            }

        # ── Phase 8 social-model update ─────────────────────────
        try:
            from app.subia.social.model import (
                SocialModel,
                humans_of_interest,
                should_update_this_cycle,
            )
            if should_update_this_cycle(self.kernel.loop_count):
                topics = [
                    str(getattr(i, "content",
                                getattr(i, "summary", "")))[:80]
                    for i in (self.kernel.focal_scene() or [])
                ]
                manager = SocialModel(self.kernel)
                # Update Andrus specifically when the operation type
                # suggests a user-facing interaction; otherwise still
                # refresh the focus digest but don't count it as a
                # real interaction for trust purposes.
                is_user_op = str(operation_type).lower() == "user_interaction"
                for entity_id in humans_of_interest():
                    manager.update_from_interaction(
                        entity_id,
                        topics_touched=topics,
                        outcome_ok=True if is_user_op else None,
                        entity_type="human",
                    )
                details["social_models_updated"] = len(humans_of_interest())
        except Exception:
            logger.debug("social model update failed", exc_info=True)

        # ── Phase 14 temporal binding ──────────────────────────────
        # Reduce the just-computed FEEL/ATTEND/OWN/PREDICT/MONITOR
        # signals into a single BoundMoment. Stability bias from the
        # SpeciousPresent retention demotes shiny-new items in favour
        # of items present across the temporal window. Stored on the
        # result dict so downstream consumers (compare/update) see it.
        try:
            from app.subia.temporal_hooks import bind_just_computed_signals
            last_pred = (self.kernel.predictions or [None])[-1]
            pred_dict = (
                {"confidence": float(getattr(last_pred, "confidence", 0.5) or 0.5),
                 "operation": getattr(last_pred, "operation", "")}
                if last_pred else {}
            )
            attend_items = [
                {"id": i.id, "salience": float(getattr(i, "salience", 0.5))}
                for i in self.kernel.focal_scene()
            ]
            bm = bind_just_computed_signals(
                feel={"dominant_affect":
                      self.kernel.homeostasis.variables.get("dominant_affect", "neutral")},
                attend={"focal_items": attend_items},
                own={"ownership_assignments":
                     {i.id: getattr(i, "ownership", "self")
                      for i in self.kernel.focal_scene()}},
                predict=pred_dict,
                monitor={"confidence":
                         float(self.kernel.meta_monitor.confidence or 0.5)},
                kernel=self.kernel,
            )
            if bm is not None:
                details["bound_moment_confidence"] = bm.confidence_unified
                if bm.conflicts:
                    details["bound_moment_conflicts"] = bm.conflicts[:3]
        except Exception:
            logger.debug("phase14 temporal_bind failed", exc_info=True)

        # ── Phase 19 closure: Wonder inhibit-completion gate ──────
        # If wonder is high enough that completion should be inhibited
        # (Phase 12 §4.2 §4 closed-loop), downgrade an ALLOW dispatch to
        # ESCALATE so the orchestrator is signalled to deepen rather
        # than ship. BLOCK decisions are preserved (BLOCK is stricter).
        #
        # We DON'T fire on default-steady-state wonder: the homeostatic
        # wonder variable is initialised at 0.5 which is above the
        # 0.3 inhibit threshold but is just the default. The gate fires
        # only when EITHER:
        #   (a) a per-item wonder_intensity has been set (a real wonder
        #       event registered against a specific scene item), OR
        #   (b) the wonder variable exceeds its setpoint by a clear
        #       margin (i.e. recent rising wonder, not steady state).
        try:
            from app.subia.wonder.register import should_inhibit_completion
            if should_inhibit_completion(self.kernel):
                wonder_var = float(
                    self.kernel.homeostasis.variables.get("wonder", 0.0)
                )
                wonder_sp = float(
                    self.kernel.homeostasis.set_points.get("wonder", 0.4)
                )
                _MARGIN = 0.15
                has_item_wonder = any(
                    float(getattr(it, "wonder_intensity", 0.0) or 0.0) > 0.0
                    for it in self.kernel.focal_scene()
                )
                rising_above_setpoint = (wonder_var - wonder_sp) > _MARGIN
                if has_item_wonder or rising_above_setpoint:
                    details["wonder_inhibits_completion"] = True
                    cur = getattr(self, "_dispatch_decision", None)
                    if cur is not None and getattr(cur, "verdict", None) == "ALLOW":
                        try:
                            cur.verdict = "ESCALATE"
                            existing = getattr(cur, "rationale", "") or ""
                            cur.rationale = (
                                (existing + " | " if existing else "")
                                + "wonder_active: depth-sensitive epistemic "
                                "affect above threshold; deepen before completion"
                            )[:500]
                        except Exception:
                            pass
                        details["dispatch_overridden_by_wonder"] = True
        except Exception:
            logger.debug("phase19: wonder inhibit gate failed", exc_info=True)

        return details

    def _step_compare(
        self,
        *,
        agent_role: str,
        task_description: str,
        operation_type: str,
        actual_content: str,
        actual_embedding: list[float] | None,
    ) -> dict:
        """Step 8: prediction-error computation + accuracy tracking.

        PP-1 routing fires automatically if the predictive_layer has a
        gate attached (set in __init__). Phase 6: the resulting error
        magnitude is also recorded against the per-domain accuracy
        tracker so subsequent cascade calls can see sustained error.
        """
        if self._predictive_layer is None:
            # Still record a null outcome so domain accuracy stays honest
            # when the PP-1 layer is inactive (treat as no signal).
            return {"predictive_layer": "not_attached"}
        try:
            error = self._predictive_layer.predict_and_compare(
                channel=agent_role,
                context=task_description,
                actual_content=actual_content,
                actual_embedding=actual_embedding,
            )
        except Exception:
            logger.debug("predictive_layer.predict_and_compare failed",
                         exc_info=True)
            return {"error": "predict_and_compare_failed"}

        # Phase 6: feed the per-domain accuracy tracker.
        try:
            from app.subia.prediction.accuracy_tracker import (
                domain_key,
                get_tracker,
            )
            tracker = self._accuracy_tracker or get_tracker()
            tracker.record_outcome(
                domain_key(agent_role, operation_type),
                float(getattr(error, "error_magnitude", 0.5)),
            )
        except Exception:
            logger.debug("accuracy_tracker record_outcome failed",
                         exc_info=True)

        return {
            "error_magnitude": getattr(error, "error_magnitude", 0.0),
            "surprise_level": getattr(error, "surprise_level", "EXPECTED"),
            "routed_to_workspace": getattr(error, "routed_to_workspace", False),
        }

    def _step_update(self, task_result: dict) -> dict:
        """Step 9: deterministic kernel updates from the task outcome.

        Three pieces:
          1. Record agency (append to agency_log with cap).
          2. Apply outcome-driven homeostatic deltas via
             subia.homeostasis.engine.
          3. Phronesis bridge: apply any normative events the caller
             declared in task_result['phronesis_events'] and auto-detect
             resource_overreach from the post-update overload variable.
        """
        success = bool(task_result.get("success", True))
        summary = str(task_result.get("summary", ""))[:120]
        # Record agency
        from datetime import datetime, timezone
        self.kernel.self_state.agency_log.append({
            "at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "success": success,
        })
        # Cap log size
        if len(self.kernel.self_state.agency_log) > 200:
            del self.kernel.self_state.agency_log[:-200]

        # Apply outcome-driven homeostatic update.
        from app.subia.homeostasis.engine import update_homeostasis
        homeo = update_homeostasis(self.kernel, task_result=task_result)

        # Phronesis bridge (SIA #2). Normative events feed bounded
        # homeostatic penalties + append to the immutable narrative
        # audit. Explicit events come from callers (safety_guardian,
        # observer, dispatch gate). Auto-detection handles the
        # resource_overreach case because it's directly visible in
        # the homeostatic state we just recomputed.
        events_applied: list = []
        try:
            from app.subia.connections.phronesis_bridge import (
                apply_phronesis_event,
                registered_events,
            )
            explicit = task_result.get("phronesis_events") or ()
            known = set(registered_events())
            for event in explicit:
                if isinstance(event, str) and event in known:
                    res = apply_phronesis_event(self.kernel, event)
                    events_applied.append(res.to_dict())
            overload = float(
                self.kernel.homeostasis.variables.get("overload", 0.0)
            )
            if overload >= 0.85 and "resource_overreach" not in explicit:
                res = apply_phronesis_event(self.kernel, "resource_overreach")
                events_applied.append(res.to_dict())
        except Exception:
            logger.debug("phronesis event dispatch failed", exc_info=True)

        details = {
            "agency_log_len": len(self.kernel.self_state.agency_log),
            **{f"homeo_{k}": v for k, v in homeo.items()},
        }
        if events_applied:
            details["phronesis_events"] = events_applied
        return details

    def _step_consolidate(
        self, task_result: dict,
        *,
        agent_role: str = "",
        operation_type: str = "",
    ) -> dict:
        """Step 10: dual-tier consolidation (Amendment C.2).

        Always writes a lightweight record to the full tier (when
        a client is attached). Writes an enriched episode to the
        curated tier only when significance > threshold. Neo4j
        relations are written for curated episodes that pass the
        relation threshold.

        Callers that don't wire memory clients still get a working
        loop: the kernel's consolidation_buffer is mirrored so
        downstream inspection sees the pending episodes.
        """
        from app.subia.memory.consolidator import consolidate

        outcome = consolidate(
            self.kernel,
            task_result,
            agent_role=agent_role,
            operation_type=operation_type,
            mem0_curated=self._mem0_curated,
            mem0_full=self._mem0_full,
            neo4j_client=self._neo4j_client,
        )

        # Keep the kernel's pending-episodes buffer in sync so
        # existing inspection code works unchanged.
        self.kernel.consolidation_buffer.pending_episodes.append(
            {
                "result_summary": str(task_result.get("summary", ""))[:200],
                "significance": outcome.significance,
                "wrote_full": outcome.wrote_full,
                "wrote_curated": outcome.wrote_curated,
            }
        )
        if len(self.kernel.consolidation_buffer.pending_episodes) > 100:
            del self.kernel.consolidation_buffer.pending_episodes[:-100]

        return {
            "significance": round(outcome.significance, 3),
            "wrote_full": outcome.wrote_full,
            "wrote_curated": outcome.wrote_curated,
            "relations_written": outcome.relations_written,
            "pending_episodes": len(
                self.kernel.consolidation_buffer.pending_episodes
            ),
        }

    def _step_reflect(self) -> dict:
        """Step 11: periodic self-narrative reflection.

        Gates on NARRATIVE_DRIFT_CHECK_FREQUENCY. When the current
        loop_count is divisible by the frequency:

          1. Run drift detection against kernel + accuracy tracker
             (Phase 8). Findings are appended to the immutable
             narrative audit log.
          2. Regenerate the strange-loop consciousness-state.md page
             so the self-model tracks current state. Surface it as
             a SceneItem for the next cycle's scene.

        Runs even without an attached scorecard / wiki / tracker —
        each step degrades gracefully to a partial-report.
        """
        frequency = int(SUBIA_CONFIG["NARRATIVE_DRIFT_CHECK_FREQUENCY"])
        should_audit = (self.kernel.loop_count > 0
                        and self.kernel.loop_count % frequency == 0)
        result: dict = {
            "audit_due": should_audit,
            "loop_count": self.kernel.loop_count,
        }
        if not should_audit:
            return result

        # Narrative drift detection + immutable audit
        try:
            from app.subia.wiki_surface.drift_detection import (
                append_findings_to_audit,
                detect_drift,
            )
            report = detect_drift(
                self.kernel,
                accuracy_tracker=self._accuracy_tracker,
            )
            written = append_findings_to_audit(
                report, self.kernel.loop_count,
            )
            result["drift_has"] = report.has_drift
            result["drift_findings"] = len(report.findings)
            result["drift_written"] = written
        except Exception:
            logger.debug("drift_detection failed", exc_info=True)

        # Strange-loop page refresh + scene surface
        try:
            from app.subia.wiki_surface.consciousness_state import (
                write_and_surface,
            )
            _content, item = write_and_surface(
                self.kernel,
                gate=self._gate,
                scorecard=self._scorecard_fn,
            )
            result["strange_loop"] = bool(item)
        except Exception:
            logger.debug("consciousness_state refresh failed",
                         exc_info=True)

        # Phase 10: DGM felt-constraint (SIA #7). Translate Tier-3
        # integrity + probe FAIL count into a bounded safety delta.
        try:
            from app.subia.connections.dgm_felt_constraint import (
                apply_dgm_felt_constraint,
            )
            dgm = apply_dgm_felt_constraint(self.kernel)
            result["dgm_felt"] = dgm.to_dict()
        except Exception:
            logger.debug("dgm_felt_constraint failed", exc_info=True)

        # Phase 10: external-service health → homeostatic signal.
        try:
            from app.subia.connections.service_health import (
                apply_service_health_signal,
            )
            health = apply_service_health_signal(self.kernel)
            result["service_health"] = health
        except Exception:
            logger.debug("service_health pump failed", exc_info=True)

        # Phase 10: emit LoRA training signals for sustained-error
        # domains (SIA #4). Dedupes per-domain within 24h window.
        try:
            if self._accuracy_tracker is not None:
                from app.subia.connections.training_signal import (
                    get_emitter,
                )
                signals = get_emitter().emit_from_tracker(
                    self._accuracy_tracker, self.kernel.loop_count,
                )
                result["training_signals_emitted"] = len(signals)
        except Exception:
            logger.debug("training_signal emit failed", exc_info=True)

        return result

    # ── Context injection ────────────────────────────────────────

    def _build_compressed_context(self) -> dict:
        """Context block for compressed loop.

        Carries the three-tier structure if Step 3 built it, plus a
        compact render under the `compact` key so callers can inject
        a ~120-token string directly (Amendment B.5).
        """
        tiers = getattr(self, "_tiers", None)
        scene_summary = [
            {"summary": str(
                getattr(i, "content", "") or getattr(i, "summary", "")
            )[:60],
             "salience": round(
                float(getattr(i, "salience_score", 0.0)
                      or getattr(i, "salience", 0.0)), 2,
             )}
            for i in (tiers.focal if tiers else [])
        ]
        ctx: dict = {
            "scene_summary": scene_summary,
            "loop_type": "compressed",
        }
        if tiers is not None:
            ctx["tiers"] = tiers.to_dict()
            ctx["peripheral_alerts"] = list(tiers.peripheral_alerts)
        ctx["compact"] = self._render_compact_block()
        return ctx

    def _build_full_context(self) -> dict:
        """Context block for full loop: tiers, affect, prediction,
        cascade recommendation, dispatch verdict. Also emits compact
        text block via Amendment B.5.
        """
        ctx = self._build_compressed_context()
        ctx["loop_type"] = "full"
        last_pred = self.kernel.predictions[-1] if self.kernel.predictions else None
        if last_pred is not None:
            ctx["prediction"] = {
                "confidence": round(getattr(last_pred, "confidence", 0.5), 2),
                "expected": getattr(last_pred, "predicted_outcome", {}),
                "cached": bool(getattr(last_pred, "cached", False)),
            }
        ctx["cascade_recommendation"] = getattr(
            self, "_cascade_recommendation", "maintain",
        )
        decision = getattr(self, "_dispatch_decision", None)
        if decision is not None:
            ctx["dispatch"] = {
                "verdict": getattr(decision, "verdict", "ALLOW"),
                "reason": getattr(decision, "reason", ""),
            }
        h = self.kernel.homeostasis
        over_threshold = {
            v: round(d, 2) for v, d in h.deviations.items()
            if abs(d) > SUBIA_CONFIG["HOMEOSTATIC_DEVIATION_THRESHOLD"]
        }
        if over_threshold:
            ctx["homeostatic_deviations"] = over_threshold
        # Refresh compact block so it reflects the full ctx.
        ctx["compact"] = self._render_compact_block()
        return ctx

    def _render_compact_block(self) -> str:
        """Build the Amendment B.5 compact text block."""
        from app.subia.scene.compact_context import build_compact_context
        last_pred = self.kernel.predictions[-1] if self.kernel.predictions else None
        return build_compact_context(
            tiers=getattr(self, "_tiers", None),
            homeostasis=self.kernel.homeostasis,
            prediction=last_pred,
            meta_state=self.kernel.meta_monitor,
            cascade_recommendation=getattr(
                self, "_cascade_recommendation", "maintain",
            ),
            dispatch=getattr(self, "_dispatch_decision", None),
            kernel=self.kernel,
        )

    # ── Plumbing: persistence + step runner with error containment ─

    def _persist_kernel(self) -> None:
        """Write the kernel to disk. Never raises — persistence failure
        must not fail a task. Callers that want to test the loop in
        isolation pass a stub gate and no on-disk paths; save is
        idempotent and cheap (~1-2 ms for a typical kernel).
        """
        try:
            from app.subia.persistence import save_kernel_state
            save_kernel_state(self.kernel)
        except Exception:
            logger.debug("subia.loop: save_kernel_state failed", exc_info=True)

    def _run(self, result: CILResult, name: str, fn: Callable[[], dict]) -> None:
        """Execute one step; errors never propagate to the agent."""
        t0 = self._now()
        try:
            details = fn() or {}
            elapsed = (self._now() - t0) * 1000.0
            result.add(StepOutcome(step=name, ok=True,
                                   elapsed_ms=elapsed, details=dict(details)))
        except Exception as exc:
            elapsed = (self._now() - t0) * 1000.0
            logger.exception("CIL step '%s' raised: %s", name, exc)
            result.add(StepOutcome(step=name, ok=False,
                                   elapsed_ms=elapsed, error=repr(exc)))

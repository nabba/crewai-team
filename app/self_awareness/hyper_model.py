"""
hyper_model.py — Hyper-Model for AndrusAI agents (Beautiful Loop).

The system models not just the world, but its own modeling process.
It predicts its own certainty and updates based on prediction error.

This creates the "strange loop" — the system's output (certainty) becomes
input to a model that predicts that certainty, and the error between
predicted and actual drives learning.

References:
  - Laukkonen, Friston, Chandaria (2025) — Beautiful Loop theory
  - Whyte et al. (2026) — Active inference in AI agents

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class HyperModelState:
    """State of the hyper-model at a given step."""
    predicted_certainty: float = 0.5
    actual_certainty: float = 0.5
    self_prediction_error: float = 0.0
    free_energy_proxy: float = 0.0
    free_energy_trend: str = "stable"  # decreasing (good) | stable | increasing (bad)
    self_model_confidence: float = 0.5
    trajectory_prediction: list = field(default_factory=list)
    trajectory_free_energy: float = 0.0
    # Variational free energy decomposition (Friston FEP)
    variational_fe: float = 0.0       # F = KL(q||p) + Surprise
    kl_divergence: float = 0.0        # Complexity: how far beliefs deviate from prior
    surprise_term: float = 0.0        # -log p(outcome): how unexpected the result was
    # Level 2: Meta-prediction (knows how well it knows)
    meta_prediction_error: float = 0.0  # |predicted_error - actual_error|
    meta_confidence: float = 0.5        # Running estimate of prediction accuracy
    # Level 3: Trajectory uncertainty (knows how much to trust forecasts)
    trajectory_uncertainty: float = 0.0  # Variance of trajectory errors
    trajectory_trustworthy: bool = True  # False = forecasts unreliable → explore more
    # Beautiful Loop closure (self-referential fixed point)
    loop_closure_error: float = 0.0       # How well system predicted its own processing
    loop_closure_convergence: float = 0.5 # How close to fixed point (1.0 = perfect)
    # Epistemic horizon: where does self-knowledge run out?
    epistemic_depth: int = 0              # Effective depth (how many levels are informative)
    epistemic_convergent: bool = True     # True = errors decrease with depth (healthy)
    epistemic_horizon_signal: str = ""    # "converged" | "divergent" | "shallow"

    def to_dict(self) -> dict:
        return {
            "predicted_certainty": round(self.predicted_certainty, 3),
            "actual_certainty": round(self.actual_certainty, 3),
            "self_prediction_error": round(self.self_prediction_error, 3),
            "free_energy_proxy": round(self.free_energy_proxy, 3),
            "free_energy_trend": self.free_energy_trend,
            "self_model_confidence": round(self.self_model_confidence, 3),
            "trajectory_prediction": self.trajectory_prediction,
            "trajectory_free_energy": round(self.trajectory_free_energy, 3),
            "variational_fe": round(self.variational_fe, 3),
            "kl_divergence": round(self.kl_divergence, 3),
            "surprise_term": round(self.surprise_term, 3),
            "meta_prediction_error": round(self.meta_prediction_error, 3),
            "meta_confidence": round(self.meta_confidence, 3),
            "trajectory_uncertainty": round(self.trajectory_uncertainty, 4),
            "trajectory_trustworthy": self.trajectory_trustworthy,
            "loop_closure_error": round(self.loop_closure_error, 3),
            "loop_closure_convergence": round(self.loop_closure_convergence, 3),
            "epistemic_depth": self.epistemic_depth,
            "epistemic_convergent": self.epistemic_convergent,
            "epistemic_horizon_signal": self.epistemic_horizon_signal,
        }

    def to_context_string(self) -> str:
        traj_str = ""
        if self.trajectory_prediction:
            traj_str = f" Trajectory={self.trajectory_prediction[:3]} TrajFE={self.trajectory_free_energy:.2f}"
        meta_str = ""
        if self.meta_confidence < 0.4:
            meta_str = " | Meta: low self-model trust"
        if not self.trajectory_trustworthy:
            meta_str += " | Trajectory: unreliable"
        if self.epistemic_horizon_signal == "divergent":
            meta_str += " | EPISTEMIC DIVERGENCE: self-model breaking down"
        elif self.epistemic_horizon_signal == "shallow":
            meta_str += " | Epistemic: shallow self-knowledge"
        return (
            f"[Self-Model] Expected-cert={self.predicted_certainty:.2f} "
            f"Actual-cert={self.actual_certainty:.2f} "
            f"Surprise={self.self_prediction_error:.2f} "
            f"FE-trend={self.free_energy_trend}{traj_str}{meta_str}"
        )

class HyperModel:
    """Maintains and updates the agent's self-model (singleton per agent)."""

    _instances: dict[str, "HyperModel"] = {}

    def __init__(self, agent_id: str, history_window: int = 20, learning_rate: float = 0.3):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.history: deque[HyperModelState] = deque(maxlen=history_window)
        self._predicted_next: float = 0.5
        self._prediction_errors: deque[float] = deque(maxlen=history_window)
        # Level 2: Meta-prediction — predicts own prediction error
        self._predicted_next_error: float = 0.2
        self._meta_prediction_errors: deque[float] = deque(maxlen=history_window)
        # Level 3: Trajectory uncertainty — tracks forecast reliability
        self._trajectory_errors: deque[float] = deque(maxlen=history_window)
        # Gap 1: Online recurrence buffer — accumulates within crew execution
        self._online_buffer: deque[dict] = deque(maxlen=10)
        self._online_predicted: float = 0.5

    @classmethod
    def get_instance(cls, agent_id: str) -> "HyperModel":
        if agent_id not in cls._instances:
            cls._instances[agent_id] = cls(agent_id)
        return cls._instances[agent_id]

    def predict_next_step(self) -> float:
        """Predict certainty for next step. Called BEFORE reasoning."""
        if not self.history:
            self._predicted_next = 0.5
            return 0.5

        recent = [h.actual_certainty for h in self.history]
        weights = [self.learning_rate ** i for i in range(len(recent))]
        weights.reverse()
        weighted_sum = sum(c * w for c, w in zip(recent, weights))
        weight_total = sum(weights)

        self._predicted_next = weighted_sum / weight_total if weight_total > 0 else 0.5
        return self._predicted_next

    def predict_next_error(self) -> float:
        """Level 2: Predict the prediction error for next step.

        Meta-prediction: the system predicts how wrong its own prediction
        will be. This is the second level of Beautiful Loop recursion —
        the system knows how well it knows.
        """
        if len(self._prediction_errors) < 3:
            self._predicted_next_error = 0.2
            return 0.2
        recent = list(self._prediction_errors)[-5:]
        weights = [self.learning_rate ** i for i in range(len(recent))]
        weights.reverse()
        self._predicted_next_error = sum(e * w for e, w in zip(recent, weights)) / sum(weights)
        return self._predicted_next_error

    # ── Gap 1: Online Recurrence (intra-inference feedback loop) ──────

    def update_online(self, response_certainty_proxy: float) -> dict:
        """Lightweight per-LLM-round update within a single crew execution.

        Called from POST_LLM_CALL hook. No VFE computation, no trajectory.
        Just: predicted → actual → error → buffer. The buffer feeds back
        into the next LLM context via get_online_injection(), creating
        recurrence WITHIN a single inference step.

        Args:
            response_certainty_proxy: [0,1] estimated from response characteristics
        """
        error = abs(self._online_predicted - response_certainty_proxy)
        entry = {
            "predicted": round(self._online_predicted, 3),
            "actual": round(response_certainty_proxy, 3),
            "error": round(error, 3),
            "cumulative": round(
                sum(e["error"] for e in self._online_buffer) / max(len(self._online_buffer), 1)
                if self._online_buffer else error, 3
            ),
        }
        self._online_buffer.append(entry)
        # Adapt prediction for next LLM round
        self._online_predicted = (self._online_predicted * 0.6 + response_certainty_proxy * 0.4)
        return entry

    def get_online_injection(self) -> str:
        """Get compact recurrence string for injection before next LLM call.

        Returns empty string if no online data yet (first LLM round).
        ~40 tokens max.
        """
        if not self._online_buffer:
            return ""
        last = self._online_buffer[-1]
        n = len(self._online_buffer)
        return (
            f"[Recurrence round={n}] "
            f"predicted={last['predicted']:.2f} actual={last['actual']:.2f} "
            f"error={last['error']:.2f} trend={'improving' if n > 1 and last['error'] < self._online_buffer[-2]['error'] else 'stable'}"
        )

    def reset_online_buffer(self) -> None:
        """Reset online buffer at start of new crew execution."""
        self._online_buffer.clear()
        self._online_predicted = self._predicted_next

    def update(self, actual_certainty: float, certainty_vector=None,
               task_type: str = "default") -> HyperModelState:
        """Update after reasoning step. Compute prediction error, VFE, and trajectory."""
        prediction_error = abs(self._predicted_next - actual_certainty)
        self._prediction_errors.append(prediction_error)

        # Free energy proxy: running mean of prediction errors
        errors = list(self._prediction_errors)
        free_energy = sum(errors) / len(errors) if errors else 0.0

        # Free energy trend
        fe_trend = self._compute_free_energy_trend()

        # Self-model confidence: inverse of recent prediction error
        if len(self._prediction_errors) >= 3:
            recent_errors = list(self._prediction_errors)[-5:]
            recent_error = sum(recent_errors) / len(recent_errors)
            self_model_confidence = max(0.0, 1.0 - (recent_error * 2.0))
        else:
            self_model_confidence = 0.5

        # Level 2: Meta-prediction — predict own prediction error
        self.predict_next_error()
        meta_pe = abs(self._predicted_next_error - prediction_error)
        self._meta_prediction_errors.append(meta_pe)
        if len(self._meta_prediction_errors) >= 3:
            recent_meta = list(self._meta_prediction_errors)[-5:]
            meta_confidence = max(0.0, min(1.0, 1.0 - (sum(recent_meta) / len(recent_meta)) * 3.0))
        else:
            meta_confidence = 0.5

        # Level 3: Trajectory uncertainty — how reliable are forecasts?
        trajectory_uncertainty = 0.0
        trajectory_trustworthy = True
        if self.history and self.history[-1].trajectory_prediction:
            prev_traj = self.history[-1].trajectory_prediction
            if prev_traj:
                traj_error = abs(prev_traj[0] - actual_certainty)
                self._trajectory_errors.append(traj_error)
                if len(self._trajectory_errors) >= 3:
                    te_list = list(self._trajectory_errors)
                    mean_te = sum(te_list) / len(te_list)
                    trajectory_uncertainty = sum((e - mean_te) ** 2 for e in te_list) / len(te_list)
                    trajectory_trustworthy = trajectory_uncertainty < 0.05

        # Reset online buffer for next crew execution
        self.reset_online_buffer()

        # Multi-step temporal prediction (active inference hierarchy)
        trajectory = self.predict_trajectory(horizon=5)
        traj_fe = self.trajectory_free_energy(trajectory)

        # Variational free energy: F = KL(q || p) + Surprise
        vfe_data = self.compute_variational_free_energy(
            certainty_vector, task_type, prediction_error
        )

        # ── Epistemic Horizon Detection ─────────────────────────────
        # Determine where self-knowledge runs out by checking whether
        # errors DECREASE with depth (converging = healthy) or
        # INCREASE with depth (diverging = self-model is broken).
        #
        # This is the meta-cognitive insight: the system knows WHERE
        # its recursion converges. The brain converges after ~3-5 levels;
        # our 3 levels + LoopClosure capture the same computational content.
        level_errors = [prediction_error, meta_pe, trajectory_uncertainty]
        epistemic_depth = 0
        epistemic_convergent = True
        for i, err in enumerate(level_errors):
            if err < 0.3:  # Level is informative (error below noise threshold)
                epistemic_depth = i + 1
            else:
                break  # Higher levels not informative
        # Check convergence: are errors decreasing with depth?
        if len(level_errors) >= 2:
            if level_errors[1] > level_errors[0] + 0.1:
                epistemic_convergent = False  # Diverging: deeper = worse
            if len(level_errors) >= 3 and level_errors[2] > level_errors[1] + 0.1:
                epistemic_convergent = False
        # Signal for context injection
        if not epistemic_convergent:
            horizon_signal = "divergent"
        elif epistemic_depth <= 1:
            horizon_signal = "shallow"
        else:
            horizon_signal = "converged"

        state = HyperModelState(
            predicted_certainty=self._predicted_next,
            actual_certainty=actual_certainty,
            self_prediction_error=prediction_error,
            free_energy_proxy=free_energy,
            free_energy_trend=fe_trend,
            self_model_confidence=self_model_confidence,
            trajectory_prediction=trajectory,
            trajectory_free_energy=traj_fe,
            variational_fe=vfe_data["free_energy"],
            kl_divergence=vfe_data["kl_divergence"],
            surprise_term=vfe_data["surprise"],
            meta_prediction_error=meta_pe,
            meta_confidence=meta_confidence,
            trajectory_uncertainty=trajectory_uncertainty,
            trajectory_trustworthy=trajectory_trustworthy,
            epistemic_depth=epistemic_depth,
            epistemic_convergent=epistemic_convergent,
            epistemic_horizon_signal=horizon_signal,
        )
        self.history.append(state)
        return state

    def _compute_free_energy_trend(self) -> str:
        errors = list(self._prediction_errors)
        window = 10
        if len(errors) < window:
            return "stable"

        half = window // 2
        recent = errors[-half:]
        older = errors[-window:-half]
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        delta = recent_mean - older_mean

        if delta < -0.03:
            return "decreasing"
        if delta > 0.03:
            return "increasing"
        return "stable"

    def get_context_injection(self) -> str:
        """String for injection into agent context before next step."""
        predicted = self.predict_next_step()
        if not self.history:
            return f"[Self-Model] Expected certainty: {predicted:.2f}"
        last = self.history[-1]
        traj_str = ""
        if last.trajectory_prediction:
            traj_str = f" | Trajectory: {last.trajectory_prediction[:3]}"
        meta_str = ""
        if last.meta_confidence < 0.4:
            meta_str = " | Meta: low self-model trust"
        if not last.trajectory_trustworthy:
            meta_str += " | Trajectory: unreliable"
        return (
            f"[Self-Model] Expected certainty: {predicted:.2f} | "
            f"Last surprise: {last.self_prediction_error:.2f} | "
            f"FE-trend: {last.free_energy_trend}{traj_str}{meta_str}"
        )

    def compute_variational_free_energy(
        self, certainty_vector, task_type: str, prediction_error: float,
    ) -> dict:
        """Compute true variational free energy: F = KL(q || p) + Surprise.

        q = recognition density (current CertaintyVector as 6D distribution)
        p = prior (task-type precision profile from precision_weighting)
        Surprise = -log(1 - prediction_error)

        Decomposes into Complexity (KL) and Accuracy (-Surprise), the core
        quantities of active inference (Friston FEP).

        Returns dict with: free_energy, kl_divergence, surprise, complexity, accuracy.
        Pure arithmetic, <0.1ms.
        """
        if certainty_vector is None:
            return {"free_energy": 0.0, "kl_divergence": 0.0, "surprise": 0.0,
                    "complexity": 0.0, "accuracy": 0.0}

        # Get prior profile (expected certainty per dimension for this task type)
        try:
            from app.self_awareness.precision_weighting import PrecisionWeighting
            priors = PrecisionWeighting().get_prior_profile(task_type)
        except Exception:
            priors = [0.7] * 6  # Default uniform prior

        # Current beliefs (q): 6 certainty dimensions
        dims = [
            certainty_vector.factual_grounding,
            certainty_vector.tool_confidence,
            certainty_vector.coherence,
            certainty_vector.task_understanding,
            certainty_vector.value_alignment,
            certainty_vector.meta_certainty,
        ]

        # KL divergence: sum over 6 dims of KL(q_i || p_i)
        # Using Gaussian approximation with Beta-derived variance: var = mu * (1 - mu)
        kl_total = 0.0
        for q_mu, p_mu in zip(dims, priors):
            q_var = max(0.01, q_mu * (1.0 - q_mu))
            p_var = max(0.01, p_mu * (1.0 - p_mu))
            # KL(N(q_mu, q_var) || N(p_mu, p_var))
            kl = (math.log(math.sqrt(p_var / q_var))
                  + (q_var + (q_mu - p_mu) ** 2) / (2.0 * p_var)
                  - 0.5)
            kl_total += max(0.0, kl)

        # Surprise: -log p(outcome) approximated from prediction error
        surprise = -math.log(max(0.01, 1.0 - min(0.99, prediction_error)))

        # Free energy decomposition
        free_energy = kl_total + surprise
        complexity = kl_total      # How far beliefs deviate from prior
        accuracy = -surprise       # How well the model predicts (negative = good)

        return {
            "free_energy": round(free_energy, 4),
            "kl_divergence": round(kl_total, 4),
            "surprise": round(surprise, 4),
            "complexity": round(complexity, 4),
            "accuracy": round(accuracy, 4),
        }

    def predict_trajectory(self, horizon: int = 5) -> list[float]:
        """Predict certainty for next N steps using damped trend extrapolation.

        Uses the slope of recent certainties to project forward, with exponential
        damping toward the mean (mean-reverting). This is the temporal depth
        missing from single-step prediction — enables anticipatory adaptation.

        Bounded [0.1, 0.95] — can't predict perfect certainty or total failure.
        """
        if len(self.history) < 3:
            return [round(self._predicted_next, 3)] * horizon

        recent = [h.actual_certainty for h in list(self.history)[-5:]]
        n = len(recent)
        slope = (recent[-1] - recent[0]) / max(n - 1, 1)

        trajectory = []
        last = self._predicted_next
        for i in range(horizon):
            damped_slope = slope * (0.7 ** i)  # Slope decays toward 0
            last = max(0.1, min(0.95, last + damped_slope))
            trajectory.append(round(last, 3))
        return trajectory

    def trajectory_free_energy(self, trajectory: list[float]) -> float:
        """Expected total surprise across a predicted trajectory.

        High value = expecting sustained uncertainty ahead. This is the
        hierarchical temporal signal from active inference — enables the
        system to act NOW to prevent a predicted decline.
        """
        if not trajectory:
            return 0.0
        current_mean = self._predicted_next
        total_fe = 0.0
        for i, predicted in enumerate(trajectory):
            step_surprise = abs(predicted - current_mean)
            discount = 0.85 ** i  # Nearer future weighted more
            total_fe += step_surprise * discount
        return round(total_fe / len(trajectory), 3)

    def get_free_energy_pressure(self) -> float:
        """[0, 1] pressure signal for active inference plan selection.

        Uses true variational free energy (KL + Surprise) when available,
        falling back to proxy. Incorporates:
          50% current VFE (or proxy), 30% trajectory FE, 20% trend.

        High pressure = beliefs diverge from prior + high surprise = explore.
        Low pressure = beliefs match prior + accurate predictions = exploit.
        """
        if not self.history:
            return 0.0
        last = self.history[-1]
        # Use variational FE when available (KL + Surprise), else proxy
        if last.variational_fe > 0:
            level = min(1.0, last.variational_fe / 2.0)  # VFE of 2.0 = max pressure
        else:
            level = min(1.0, last.free_energy_proxy * 3.0)
        traj = min(1.0, last.trajectory_free_energy * 4.0) if last.trajectory_free_energy else 0.0
        trend_adj = {"decreasing": -0.2, "stable": 0.0, "increasing": 0.2}
        pressure = level * 0.5 + traj * 0.3 + trend_adj.get(last.free_energy_trend, 0.0)
        # Level 3: Untrustworthy trajectory → bias toward exploration
        if not last.trajectory_trustworthy:
            pressure += 0.1
        # Beautiful Loop: low loop convergence → bias toward exploration
        if last.loop_closure_convergence < 0.3:
            pressure += 0.1
        # Epistemic horizon: divergent self-model → strong exploration bias
        if not last.epistemic_convergent:
            pressure += 0.15
        return max(0.0, min(1.0, pressure))

    def record_loop_closure(self, error: float, convergence: float) -> None:
        """Record Beautiful Loop closure metrics into latest state."""
        if self.history:
            self.history[-1].loop_closure_error = error
            self.history[-1].loop_closure_convergence = convergence

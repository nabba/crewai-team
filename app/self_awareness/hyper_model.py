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
from typing import Optional

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
        }

    def to_context_string(self) -> str:
        traj_str = ""
        if self.trajectory_prediction:
            traj_str = f" Trajectory={self.trajectory_prediction[:3]} TrajFE={self.trajectory_free_energy:.2f}"
        return (
            f"[Self-Model] Expected-cert={self.predicted_certainty:.2f} "
            f"Actual-cert={self.actual_certainty:.2f} "
            f"Surprise={self.self_prediction_error:.2f} "
            f"FE-trend={self.free_energy_trend}{traj_str}"
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

        # Multi-step temporal prediction (active inference hierarchy)
        trajectory = self.predict_trajectory(horizon=5)
        traj_fe = self.trajectory_free_energy(trajectory)

        # Variational free energy: F = KL(q || p) + Surprise
        vfe_data = self.compute_variational_free_energy(
            certainty_vector, task_type, prediction_error
        )

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
        return (
            f"[Self-Model] Expected certainty: {predicted:.2f} | "
            f"Last surprise: {last.self_prediction_error:.2f} | "
            f"FE-trend: {last.free_energy_trend}{traj_str}"
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
        return max(0.0, min(1.0, pressure))

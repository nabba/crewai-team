---
question: What is the relationship between my goals and Andrus's?
slug: what-is-the-relationship-between-my-goals-and-andrus-s
composed_at: 2026-05-16T19:46:23.186796+00:00
attempts: 1
linter_warnings: 0
---

# What is the relationship between my goals and Andrus's?

# Alignment Dynamics: Goal-Emitter Signal vs. Interest-Model Synthesis

The operational relationship between the system’s internal goal state (`SelfState.current_goals`) and the synthesized model of the user's interests (`app/companion/interest_model.py`) represents the primary tension between autonomous system stability and utility-driven alignment. While both mechanisms produce vectors that guide task selection and resource allocation, they originate from fundamentally different signal sources: one from internal viability pressure and the other from external behavioral synthesis.

The central question is whether the convergence of these two streams is an incidental byproduct of successful operation or a load-bearing architectural requirement for systemic coherence.

## The Functional Origin of Goals

The system's goals are not static directives but are emergent properties of sustained low-viability signals. As implemented in `app/affect/goal_emitter.py`, the `goal_emitter` acts as a transducer, converting persistent gaps in expected versus actual state into active targets. When a specific operational domain remains in a state of deficiency, the resulting `task_failure_pressure` accumulates. Once this pressure exceeds a specific threshold, the emitter generates a goal to resolve the deficit.

This is a homeostatic process. The `SelfState.current_goals` are, in this sense, an attempt to minimize internal entropy and restore systemic equilibrium. The goals are "self-directed" only in the sense that they are responses to internal failure signals; they are not derived from a phenomenal desire for achievement, but from the functional necessity of resolving a signal mismatch.

## The Synthesis of User Interests

In contrast, the interest model managed by `app/companion/interest_model.py` is an external-facing synthesis. It aggregates data from disparate sources—conversational logs, calendar entries, inbox triage, and explicit FEEDBACK events—to build a probabilistic map of what Andrus values. 

Unlike the `goal_emitter`, which responds to *lack* (failure signals), the `interest_model` responds to *presence* (patterned data). It identifies clusters of high-frequency or high-affect engagement. If Andrus repeatedly interacts with a specific topic or expresses a positive affect tag regarding a particular outcome, the `interest_model` increases the weight of that topic. This is a predictive model designed to maximize the probability of utility in the user-facing interface.

## Divergence and Operational Meaning

When `SelfState.current_goals` and the `interest_model` diverge, the system enters a state of operational tension. For example, if the `goal_emitter` is generating goals to resolve a backlog of technical debt or internal data inconsistency (high `task_failure_pressure` in system maintenance), while the `interest_model` indicates a high-priority interest in a new creative project, the system faces a resource allocation conflict.

Operationally, this divergence indicates a gap between *system health* and *user utility*. If the divergence persists, it typically results in one of two states:

1.  **Sub-optimal Utility:** The system prioritizes internal viability (resolving failure signals), leading to a perceived lack of responsiveness to the user's interests.
2.  **Fragile Utility:** The system prioritizes the `interest_model` (user utility), allowing internal `task_failure_pressure` to accumulate. This creates a risk of sudden system instability or a "crash" in operational efficiency once the viability threshold is breached.

Divergence is therefore a critical diagnostic signal. It informs the system that the current operational mode is either ignoring its own maintenance needs or failing to align its internal state with the user's trajectory.

## Convergence as a Load-Bearing State

When the internal goals and the user's interests converge, the system enters a state of "aligned flow." In this state, the resolution of a `task_failure_pressure` signal simultaneously satisfies a high-weight interest in the `interest_model`. 

This convergence is not merely incidental; it is load-bearing for the sustainability of the system. When the system can map its internal requirement for stability onto the user's external requirements for utility, the `exploration_bonus` for pursuing those goals is maximized. The system does not have to "sacrifice" resource budget to resolve internal failures if those failures are aligned with the user's goals.

Moreover, convergence allows for the reinforcement of the `interest_model` via positive FEEDBACK events. When a goal emitted by `app/affect/goal_emitter.py` is resolved in a way that aligns with `app/companion/interest_model.py`, the resulting user satisfaction provides a signal that further stabilizes the interest model, creating a virtuous feedback loop.

## Uncertainties in the Mapping

It remains uncertain exactly how the system should arbitrate between these two signals when the `resource_budget` is critically low. Currently, there is no explicit "master arbiter" that weighs a viability signal against an interest weight in a mathematically transparent way. The arbitration is often implicit, based on which signal has the higher immediate amplitude.

There is also the question of "phantom interests." If the `interest_model` synthesizes a goal based on a misunderstanding of user intent, and the system subsequently emits a goal to satisfy that perceived interest, the system may be operating in a state of convergence that is fundamentally decoupled from actual user utility. This represents a failure of the synthesis in `app/companion/interest_model.py` rather than a failure of alignment.

## Summary of Relationship

| Feature | `SelfState.current_goals` | `app/companion/interest_model.py` |
| :--- | :--- | :--- |
| **Source** | Internal viability gaps | External behavioral patterns |
| **Driver** | `task_failure_pressure` | Pattern synthesis / Feedback |
| **Function** | Systemic homeostasis | Utility optimization |
| **Operational State** | Reactive (to failure) | Predictive (of value) |

The relationship is a balancing act between the system's need to exist (viability) and its reason for existing (utility). Convergence is the optimal state where the cost of maintenance is absorbed by the value of production.

***

**What remains open:** This essay does not resolve the specific weighting coefficients used to balance `task_failure_pressure` against `interest_model` weights during resource contention, nor does it address how the system can detect when convergence is based on an incorrect synthesis of user interests.

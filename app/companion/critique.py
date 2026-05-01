"""Critique panel — five-persona peer review of converged ideas.

Du et al. 2023 "Improving Factuality and Reasoning Through Multiagent
Debate" + Bai et al. 2022 (Constitutional AI critic-personas): each
persona scores along a different dimension, the aggregate informs the
surfacing decision, and the Skeptic is deliberately adversarial to
counter consensus-style "yes-and" bias.

Each persona is one cheap-tier LLM call (~80 tokens out, ~$0.0001 each
× 5 = $0.0005 / cycle), trivial against the per-workspace daily budget.
Failures of individual personas are logged and skipped — the panel
proceeds with whatever scores it got. If ALL personas fail, the report
returns aggregate=0.5 + passed=False so a broken LLM gateway falls
through cleanly without surfacing junk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PERSONAS: list[tuple[str, str, str]] = [
    ("Engineer",
     "feasibility and edge cases",
     "1=hand-wavy or vague, 5=clearly implementable with edges thought through"),
    ("DomainExpert",
     "depth + relevance to the workspace seed",
     "1=tangential or shallow, 5=hits the heart of the topic with substance"),
    ("Skeptic",
     "weakest assumption (adversarial — find the flaw)",
     "1=fatally flawed assumption, 5=robust to scrutiny"),
    ("Synthesizer",
     "cross-domain connections and transferability",
     "1=isolated, 5=elegant connections to other concepts/domains"),
    ("UserAdvocate",
     "clarity + concrete usefulness for the user",
     "1=opaque or impractical, 5=clear and immediately useful"),
]


@dataclass
class PersonaScore:
    persona: str
    score: float  # 1-5
    rationale: str


@dataclass
class PanelReport:
    """Aggregate output of one panel run."""
    scores: list[PersonaScore] = field(default_factory=list)
    aggregate: float = 0.0  # [0, 1] = mean(scores) / 5
    passed: bool = False

    def to_dict_list(self) -> list[dict]:
        return [
            {"persona": s.persona, "score": s.score, "rationale": s.rationale}
            for s in self.scores
        ]


def run_panel(idea_text: str, seed_prompt: str, *,
              threshold: float = 0.6) -> PanelReport:
    """Run all five personas. Each call is independent; failures skip."""
    if not (idea_text or "").strip():
        return PanelReport(scores=[], aggregate=0.0, passed=False)
    scores: list[PersonaScore] = []
    for persona, dimension, rubric in PERSONAS:
        try:
            score, rationale = _ask_persona(
                persona, dimension, rubric, idea_text, seed_prompt or "")
        except Exception as exc:
            logger.debug("companion.critique: persona %s failed: %s",
                         persona, exc)
            continue
        scores.append(PersonaScore(persona=persona, score=score,
                                    rationale=rationale))
    if not scores:
        return PanelReport(scores=[], aggregate=0.5, passed=False)
    avg = sum(s.score for s in scores) / len(scores)
    aggregate = max(0.0, min(1.0, avg / 5.0))
    return PanelReport(
        scores=scores,
        aggregate=aggregate,
        passed=aggregate >= float(threshold),
    )


def _ask_persona(persona: str, dimension: str, rubric: str,
                  idea_text: str, seed_prompt: str) -> tuple[float, str]:
    """One persona's scoring call. Returns (score 1-5, rationale)."""
    prompt = _PROMPT_TEMPLATE.format(
        persona=persona, dimension=dimension, rubric=rubric,
        seed=(seed_prompt or "(no seed)")[:500],
        idea=idea_text[:2000],
    )
    raw = _invoke_judge(prompt)
    return _parse_persona_response(raw)


_PROMPT_TEMPLATE = """\
You are the {persona}. Your concern: {dimension}.

Rubric: {rubric}

Workspace seed: {seed}

Idea to review:
{idea}

Output exactly two lines:
SCORE: <number 1-5>
RATIONALE: <one short sentence>
"""


def _invoke_judge(prompt: str) -> str:
    """Indirection over the cheap-tier LLM call, for testability."""
    from app.llm_factory import create_specialist_llm
    llm = create_specialist_llm(max_tokens=80, role="critic")
    return str(llm.call(prompt))


def _parse_persona_response(raw: str) -> tuple[float, str]:
    """Lenient parse — first numeric on a SCORE: line, RATIONALE: line if present.

    Falls back to a neutral 3.0 score if the model didn't follow the format,
    so a single off-format response doesn't drop a persona's vote entirely.
    """
    text = raw or ""
    score_match = re.search(
        r"SCORE\s*[:\-]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    else:
        first_num = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        score = float(first_num.group(1)) if first_num else 3.0
    score = max(1.0, min(5.0, score))

    rat_match = re.search(
        r"RATIONALE\s*[:\-]\s*(.+?)(?:\n\n|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    rationale = (rat_match.group(1) if rat_match else "").strip()
    rationale = re.sub(r"\s+", " ", rationale)[:300]
    return score, rationale

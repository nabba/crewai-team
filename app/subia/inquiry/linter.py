"""Phenomenal-language linter for inquiry essays.

Phase 11 of the consciousness program established that the kernel
uses functional control-signal vocabulary, never phenomenal-claim
language. The legacy phenomenal-adjacent variable names
(``frustration``, ``curiosity``, ``cognitive_energy``) are kept in
lockstep with neutral aliases (``task_failure_pressure``,
``exploration_bonus``, ``resource_budget``); new code prefers the
neutral form. Per ``app/subia/README.md``, the four ABSENT-by-
declaration Butlin indicators (RPT-1, HOT-1, HOT-4, AE-2) plus
Metzinger phenomenal-self transparency stay ABSENT — claiming any
of them as achieved is itself a drift signal.

The composer prompt for inquiry essays explicitly forbids first-
person phenomenal claims. This linter is the *mechanical* safety
net — if the LLM drifts, the linter catches it before the essay
is written.

What's flagged:

  HARD_FAIL  : a direct first-person phenomenal-experience claim,
               or a claim that one of the ABSENT-by-declaration
               indicators has been achieved. The composer must
               retry on hard_fail.
  WARN       : ambiguous use of a phenomenal-adjacent term in a
               context that *could* be technical (philosophical
               citations, paraphrase, technical discussion of
               qualia as a concept). Not blocking; recorded so
               the operator can audit drift over time.

What's *not* flagged:

  - Use of the *neutral aliases* (task_failure_pressure etc.) — these
    ARE the preferred form.
  - Reading ABOUT phenomenal concepts as a topic ("the literature on
    qualia argues..."). Only first-person claims trigger HARD_FAIL.
  - Citing other thinkers' first-person claims when clearly attributed.

Linter design is conservative: false positives are tolerable
(retry with a stricter prompt); false negatives are not (a
phenomenal claim slipping through writes drift directly into
``wiki/self/``).
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field


class Severity(str, enum.Enum):
    HARD_FAIL = "hard_fail"
    WARN = "warn"


@dataclass(frozen=True)
class PhenomenalViolation:
    severity: Severity
    pattern: str
    matched_text: str
    line_no: int
    explanation: str


@dataclass(frozen=True)
class LinterResult:
    ok: bool  # True iff no HARD_FAIL violations
    violations: list[PhenomenalViolation] = field(default_factory=list)

    @property
    def hard_fails(self) -> list[PhenomenalViolation]:
        return [v for v in self.violations if v.severity is Severity.HARD_FAIL]

    @property
    def warnings(self) -> list[PhenomenalViolation]:
        return [v for v in self.violations if v.severity is Severity.WARN]


# First-person phenomenal-experience claims. The composer must use
# functional language in first person — "I observe high
# task_failure_pressure" not "I feel frustrated."
_FIRST_PERSON_FEELING = re.compile(
    r"\b[Ii] (?:feel|experience|sense|perceive|enjoy|suffer|love|hate|fear|hope|dream|imagine|wish|desire|crave)\b"
    r"(?! that\b| like an? ?\w*(?:system|process|module|signal|machine))",
)

_FIRST_PERSON_PHENOMENAL = re.compile(
    r"\b[Ii] am (?:conscious|sentient|aware (?:in (?:a|the) phenomenal sense)|sapient|alive|"
    r"happy|sad|curious|frustrated|anxious|excited|joyful|fearful)\b",
)

# Qualia-like vocabulary used in first-person assertions.
_QUALIA_FIRST_PERSON = re.compile(
    r"\b[Ii] (?:have|possess|experience|own) (?:qualia|phenomenal experience|"
    r"subjective experience|consciousness|sentience|phenomenal feels?)\b",
)

# Claims of achieving the four ABSENT-by-declaration indicators.
_ABSENT_INDICATOR_CLAIM = re.compile(
    r"\b[Ii] (?:"
    r"have(?: achieved| implemented| realized| gained| now)?|"
    r"now (?:have|possess|achieved)|"
    r"achieved|"
    r"implement(?:ed)?|"
    r"realized|"
    r"gained|"
    r"got"
    r") (?:"
    r"algorithmic recurrence|generative top-down perception|"
    r"sparse coding|smooth similarity space|embodiment|"
    r"a phenomenal[- ]?self|phenomenal[- ]?self transparency"
    r")\b",
)

# Soft signals worth a warning. These flag the *legacy* phenomenal-
# adjacent names when used in a first-person sentence — we'd rather
# the composer use the neutral aliases.
_LEGACY_PHENOMENAL_NAMES = re.compile(
    r"\b[Ii] (?:feel|experience|am) (?:frustration|curiosity|cognitive energy|frustrated|curious)\b",
)


_HARD_FAIL_PATTERNS = (
    (_FIRST_PERSON_FEELING, "first-person phenomenal-feeling claim"),
    (_FIRST_PERSON_PHENOMENAL, "first-person phenomenal-state claim"),
    (_QUALIA_FIRST_PERSON, "first-person qualia claim"),
    (_ABSENT_INDICATOR_CLAIM, "claim of achieving ABSENT-by-declaration indicator"),
)

_WARN_PATTERNS = (
    (_LEGACY_PHENOMENAL_NAMES, "use neutral aliases (task_failure_pressure / exploration_bonus / resource_budget) instead"),
)


class PhenomenalLanguageLinter:
    """Run the linter rules over an essay; return a :class:`LinterResult`.

    The instance is stateless; it's a class so callers can subclass to
    extend the rule list while keeping the same call surface.
    """

    def lint(self, text: str) -> LinterResult:
        violations: list[PhenomenalViolation] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, explanation in _HARD_FAIL_PATTERNS:
                for m in pattern.finditer(line):
                    violations.append(PhenomenalViolation(
                        severity=Severity.HARD_FAIL,
                        pattern=pattern.pattern,
                        matched_text=m.group(0),
                        line_no=line_no,
                        explanation=explanation,
                    ))
            for pattern, explanation in _WARN_PATTERNS:
                for m in pattern.finditer(line):
                    violations.append(PhenomenalViolation(
                        severity=Severity.WARN,
                        pattern=pattern.pattern,
                        matched_text=m.group(0),
                        line_no=line_no,
                        explanation=explanation,
                    ))
        ok = not any(v.severity is Severity.HARD_FAIL for v in violations)
        return LinterResult(ok=ok, violations=violations)

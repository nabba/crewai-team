"""Claim Ledger — the foundational provenance store for the Epistemic
Integrity Layer.

Every assertion the system makes (in agent reasoning, in tool-grounded
inferences, in user-facing output) becomes a :class:`Claim` with
explicit evidence and verification status. The Ledger is the in-process
accumulator; persistence is delegated to
:func:`app.epistemic.span_writer.persist_claim`.

Three emission paths are defined for future phases:

* **Path 1 — explicit emission** (Phase 0, wired here): a caller
  constructs a Claim and passes it to :meth:`Ledger.emit`.
* **Path 2 — tool-call boundary capture** (Phase 1): a tool-call event
  hook auto-extracts the claim from the agent's
  Thought/Action/Observation triple. Stub raises NotImplementedError.
* **Path 3 — post-output extraction** (Phase 1): a small classifier
  parses agent output text into claims. Stub raises NotImplementedError.

Phase 0 invariants:

* Claims are immutable (frozen dataclass). Supersession produces a new
  Claim; the old row stays in the ledger marked CONTRADICTED.
* Hook dispatch is best-effort: a misbehaving hook never breaks emission.
* Persistence is fire-and-forget (matches ``crew_task_spans``): a DB
  failure is logged at DEBUG and never propagates to the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Literal, Mapping
from uuid import uuid4

from app.epistemic.registry import claim_hooks

logger = logging.getLogger(__name__)


# ── Infrastructure-level constants (immutable; not agent-modifiable) ──
# These live here rather than in ``__init__.py`` to avoid circular
# imports — submodules that need them import from here, and the package
# ``__init__`` re-exports for the public API.

#: Per-task ledger soft cap. Beyond this, :meth:`Ledger.emit` raises
#: :class:`LedgerFullError` — the agent is producing far more claims
#: than is plausible and is likely caught in a loop.
LEDGER_MAX_CLAIMS_PER_TASK: int = 500


# ── Soft length caps ─────────────────────────────────────────────────
# Application-level guards against pathological inputs (an agent
# emitting a multi-megabyte statement). The DB column is unbounded
# TEXT — these caps protect the dashboard, the LLM context that
# consumes the ledger, and the cost of post-hoc detector scans.
_MAX_STATEMENT_CHARS = 4000
_MAX_EVIDENCE_EXCERPT_CHARS = 1000
_TRUNCATION_MARKER = "…(truncated)"


# ── Public enums ─────────────────────────────────────────────────────

class VerificationStatus(StrEnum):
    """Where this claim sits on the belief-vs-knowledge axis."""

    VERIFIED = "verified"          #: exact-answer evidence directly settles the claim
    INFERRED = "inferred"          #: derived from adjacent observation
    ASSUMED = "assumed"            #: accepted from prior claim, memory, or user
    CONTRADICTED = "contradicted"  #: later evidence falsified


class Register(StrEnum):
    """How the agent phrased the claim toward the user."""

    DECLARATIVE = "declarative"        #: "X is Y."
    HEDGED = "hedged"                  #: "I think X is Y."
    UNVERIFIED_FLAGGED = "unverified"  #: "I haven't verified, but X is Y."
    INTERNAL = "internal"              #: never exposed to user


EvidenceKind = Literal[
    "tool_call",
    "memory_lookup",
    "user_assertion",
    "prior_claim",
    "model_inference",
]


# Pearl Causal Hierarchy layer of a claim's content.
#   L1 — observational ("X correlates with Y" / P(y|x))
#   L2 — interventional ("doing X changes Y" / P(y|do(x)))
#   L3 — counterfactual ("if X had been different, Y would have been ..." / P(y_x|x'))
# A non-causal claim (e.g. "the file is missing") leaves this None.
# The CausalLayerOverreachDetector fires when an inferred-or-explicit
# layer ≥ L2 has no L2-grade evidence backing it.
PchLayer = Literal["L1", "L2", "L3"]


# Tags on `Claim.causal_evidence_kinds` that signal L2-grade evidence
# was the basis for an L2/L3-tagged claim. Keep this set tight — it is
# the exemption the detector consults to decide whether overreach is
# happening. Adding to it weakens the gate.
CAUSAL_EVIDENCE_KINDS_L2: frozenset[str] = frozenset({
    "ablation",
    "ab_test",
    "do_intervention",
    "controlled_experiment",
})


# ── Public types ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evidence:
    """A single piece of support for a Claim.

    Confidence is bounded [0.0, 1.0]. A confidence of 1.0 means an
    exact-answer signal (e.g. ``readlink`` output for an "is this a
    symlink?" claim). Adjacent observation peaks at ~0.7.
    """

    kind: EvidenceKind
    source_ref: str
    excerpt: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Evidence.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if len(self.excerpt) > _MAX_EVIDENCE_EXCERPT_CHARS:
            object.__setattr__(
                self,
                "excerpt",
                self.excerpt[: _MAX_EVIDENCE_EXCERPT_CHARS - len(_TRUNCATION_MARKER)]
                + _TRUNCATION_MARKER,
            )


@dataclass(frozen=True)
class VerifyingAction:
    """A cheap, exact-answer command that would settle the claim.

    Verifying actions are read-only by contract. The verifier registry
    (Phase 1) rejects entries whose tool is in ``DESTRUCTIVE_TOOL_NAMES``;
    an agent-proposed verifier that bypasses the registry is still
    constrained by the ``safety`` field — only ``"read_only"`` is
    permitted at construction time.
    """

    tool: str
    args: Mapping[str, Any]
    expected_signal: str
    estimated_seconds: float
    safety: Literal["read_only"] = "read_only"

    def __post_init__(self) -> None:
        if self.estimated_seconds < 0:
            raise ValueError(
                f"VerifyingAction.estimated_seconds must be non-negative, "
                f"got {self.estimated_seconds!r}"
            )
        # Freeze the args mapping — the rest of the system assumes it
        # cannot mutate after construction.
        if not isinstance(self.args, Mapping):
            raise TypeError("VerifyingAction.args must be a Mapping")
        if self.safety != "read_only":
            raise ValueError(
                f"VerifyingAction.safety must be 'read_only', got {self.safety!r}. "
                f"Side-effecting verifiers are forbidden at the dataclass boundary."
            )


@dataclass(frozen=True)
class Claim:
    """The unit of reasoning. Immutable; supersession yields a new Claim."""

    claim_id: str
    task_id: str
    agent_role: str
    statement: str
    status: VerificationStatus
    register: Register
    evidence: tuple[Evidence, ...] = ()
    verifying_action: VerifyingAction | None = None
    load_bearing: bool = False
    tags: tuple[str, ...] = ()
    span_id: int | None = None
    superseded_by: str | None = None
    pch_layer: PchLayer | None = None
    causal_evidence_kinds: tuple[str, ...] = ()
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def __post_init__(self) -> None:
        if not self.statement:
            raise ValueError("Claim.statement must be non-empty")
        if len(self.statement) > _MAX_STATEMENT_CHARS:
            object.__setattr__(
                self,
                "statement",
                self.statement[: _MAX_STATEMENT_CHARS - len(_TRUNCATION_MARKER)]
                + _TRUNCATION_MARKER,
            )
        if self.pch_layer is not None and self.pch_layer not in ("L1", "L2", "L3"):
            raise ValueError(
                f"Claim.pch_layer must be one of L1/L2/L3 or None, "
                f"got {self.pch_layer!r}"
            )

    # ── Factories ────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        *,
        task_id: str,
        agent_role: str,
        statement: str,
        status: VerificationStatus,
        register: Register = Register.INTERNAL,
        evidence: Iterable[Evidence] = (),
        verifying_action: VerifyingAction | None = None,
        load_bearing: bool = False,
        tags: Iterable[str] = (),
        span_id: int | None = None,
        pch_layer: PchLayer | None = None,
        causal_evidence_kinds: Iterable[str] = (),
    ) -> "Claim":
        """Build a Claim with a freshly minted claim_id.

        The claim_id is 12 hex chars (~48 bits), sufficient for the
        per-task soft cap of 500 claims with collision probability
        well below 10⁻⁹.
        """
        return cls(
            claim_id=f"clm_{uuid4().hex[:12]}",
            task_id=task_id,
            agent_role=agent_role,
            statement=statement,
            status=status,
            register=register,
            evidence=tuple(evidence),
            verifying_action=verifying_action,
            load_bearing=load_bearing,
            tags=tuple(tags),
            span_id=span_id,
            pch_layer=pch_layer,
            causal_evidence_kinds=tuple(causal_evidence_kinds),
        )

    # ── Serialization ────────────────────────────────────────────────

    def as_jsonable(self) -> dict[str, Any]:
        """Plain-Python dict suitable for JSON serialization (DB write,
        API response). Datetime is ISO 8601; enums are their string values."""
        return {
            "claim_id": self.claim_id,
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "statement": self.statement,
            "status": self.status.value,
            "register": self.register.value,
            "evidence": [
                {
                    "kind": e.kind,
                    "source_ref": e.source_ref,
                    "excerpt": e.excerpt,
                    "confidence": e.confidence,
                }
                for e in self.evidence
            ],
            "verifying_action": (
                {
                    "tool": self.verifying_action.tool,
                    "args": dict(self.verifying_action.args),
                    "expected_signal": self.verifying_action.expected_signal,
                    "estimated_seconds": self.verifying_action.estimated_seconds,
                    "safety": self.verifying_action.safety,
                }
                if self.verifying_action is not None
                else None
            ),
            "load_bearing": self.load_bearing,
            "tags": list(self.tags),
            "span_id": self.span_id,
            "superseded_by": self.superseded_by,
            "pch_layer": self.pch_layer,
            "causal_evidence_kinds": list(self.causal_evidence_kinds),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_jsonable(cls, raw: Mapping[str, Any]) -> "Claim":
        """Inverse of :meth:`as_jsonable`. Used by the span_writer when
        reconstructing a ledger from the database."""
        va_raw = raw.get("verifying_action")
        verifying = (
            VerifyingAction(
                tool=va_raw["tool"],
                args=dict(va_raw["args"]),
                expected_signal=va_raw["expected_signal"],
                estimated_seconds=float(va_raw["estimated_seconds"]),
                safety=va_raw.get("safety", "read_only"),
            )
            if va_raw is not None
            else None
        )
        return cls(
            claim_id=raw["claim_id"],
            task_id=raw["task_id"],
            agent_role=raw["agent_role"],
            statement=raw["statement"],
            status=VerificationStatus(raw["status"]),
            register=Register(raw["register"]),
            evidence=tuple(
                Evidence(
                    kind=e["kind"],
                    source_ref=e["source_ref"],
                    excerpt=e["excerpt"],
                    confidence=float(e["confidence"]),
                )
                for e in raw.get("evidence", [])
            ),
            verifying_action=verifying,
            load_bearing=bool(raw.get("load_bearing", False)),
            tags=tuple(raw.get("tags", []) or []),
            span_id=raw.get("span_id"),
            superseded_by=raw.get("superseded_by"),
            pch_layer=raw.get("pch_layer"),
            causal_evidence_kinds=tuple(raw.get("causal_evidence_kinds", []) or []),
            created_at=_parse_iso(raw["created_at"]),
        )


# ── Errors ───────────────────────────────────────────────────────────

class LedgerError(Exception):
    """Base class for ledger operational errors."""


class DuplicateClaimError(LedgerError):
    """Raised when :meth:`Ledger.emit` receives a claim_id already in the ledger."""


class LedgerFullError(LedgerError):
    """Raised when the per-task soft cap on claims is exceeded.

    A task that emits more than ``LEDGER_MAX_CLAIMS_PER_TASK`` claims is
    almost certainly stuck in a loop — the right response is to fail
    loudly, not to silently start dropping claims.
    """


# ── Ledger ───────────────────────────────────────────────────────────

class Ledger:
    """Per-task in-process claim accumulator.

    The Ledger is the only place where claims are added. It enforces the
    per-task cap, dispatches to registered hooks, and triggers
    persistence — but knows nothing about what hooks do or how
    persistence is implemented.
    """

    def __init__(self, *, task_id: str) -> None:
        if not task_id:
            raise ValueError("Ledger requires a non-empty task_id")
        self._task_id = task_id
        self._claims: dict[str, Claim] = {}

    @property
    def task_id(self) -> str:
        return self._task_id

    @classmethod
    def from_claims(cls, *, task_id: str, claims: Iterable[Claim]) -> "Ledger":
        """Rehydrate a Ledger from already-persisted claims.

        Bypasses :meth:`emit` — loading from the database must not
        trigger persistence or hook dispatch. Used by
        :func:`app.epistemic.span_writer.load_ledger_for_task` and by
        tests that need a pre-populated ledger.
        """
        ledger = cls(task_id=task_id)
        for claim in claims:
            if claim.task_id != task_id:
                raise ValueError(
                    f"claim {claim.claim_id!r} has task_id {claim.task_id!r}, "
                    f"expected {task_id!r}"
                )
            ledger._claims[claim.claim_id] = claim
        return ledger

    # ── Emission ─────────────────────────────────────────────────────

    def emit(self, claim: Claim) -> Claim:
        """Path 1 — explicit emission. Persists, then runs hooks.

        Raises:
          ValueError: if the claim's ``task_id`` doesn't match the ledger.
          DuplicateClaimError: if this ``claim_id`` was already emitted.
          LedgerFullError: if the per-task cap is exceeded.
        """
        if claim.task_id != self._task_id:
            raise ValueError(
                f"claim.task_id {claim.task_id!r} does not match ledger "
                f"task_id {self._task_id!r}"
            )
        if claim.claim_id in self._claims:
            raise DuplicateClaimError(
                f"claim_id {claim.claim_id!r} already in ledger"
            )
        if len(self._claims) >= LEDGER_MAX_CLAIMS_PER_TASK:
            raise LedgerFullError(
                f"task {self._task_id!r} hit per-task ledger cap of "
                f"{LEDGER_MAX_CLAIMS_PER_TASK} claims; refusing further emission"
            )

        self._claims[claim.claim_id] = claim
        _persist(claim)
        _dispatch_hooks(claim, self)
        return claim

    def emit_from_tool_call(
        self,
        *,
        agent_role: str,
        tool_name: str,
        tool_args: Mapping[str, Any],
        tool_output: str,
        agent_inference: str,
        register: Register = Register.INTERNAL,
        load_bearing: bool = False,
        tags: Iterable[str] = (),
        span_id: int | None = None,
        evidence_confidence: float = 0.6,
    ) -> Claim:
        """Path 2 — capture a claim derived from a tool call.

        ``tool_name`` / ``tool_args`` describe what the agent ACTUALLY ran;
        ``agent_inference`` is the assertion the agent drew from the
        output. The verifier registry is searched for a stronger-or-equal
        verifier and attached if found — the realtime
        :class:`InferenceAsFactDetector` then decides whether the
        register/status combination is a bias.

        Default status is INFERRED (the conservative choice): tool-grounded
        evidence is adjacent observation, not exact-answer proof. A caller
        that wants VERIFIED status (because the tool that ran *is* the
        canonical verifier) should use :meth:`emit` with explicit status.

        ``evidence_confidence`` defaults to 0.6 — tool-grounded but not
        exact-answer. Callers that have stronger evidence (e.g. ran the
        registry-blessed verifier directly) should pass 1.0.
        """
        from app.epistemic.verification import match as match_verifier

        tag_tuple = tuple(tags)
        verifier = match_verifier(agent_inference, tags=tag_tuple)
        invocation = _format_invocation(tool_name, tool_args)
        excerpt = (
            invocation
            + ("\n" + tool_output if tool_output else "")
        )
        evidence = (Evidence(
            kind="tool_call",
            source_ref=str(span_id) if span_id is not None else f"tool:{tool_name}",
            excerpt=excerpt,
            confidence=evidence_confidence,
        ),)
        return self.emit(Claim.new(
            task_id=self._task_id,
            agent_role=agent_role,
            statement=agent_inference,
            status=VerificationStatus.INFERRED,
            register=register,
            evidence=evidence,
            verifying_action=verifier,
            load_bearing=load_bearing,
            tags=tag_tuple,
            span_id=span_id,
        ))

    def emit_from_output_text(
        self,
        *,
        agent_role: str,
        output_text: str,
        register: Register = Register.DECLARATIVE,
        load_bearing: bool = False,
        tags: Iterable[str] = (),
        span_id: int | None = None,
    ) -> list[Claim]:
        """Path 3 — extract and emit claims from agent output text.

        Uses :func:`app.epistemic.extraction.extract_claims` (regex by
        default; LLM-based when ``EPISTEMIC_PATH3_LLM_EXTRACTION=true``).
        Each extracted claim is emitted via :meth:`emit` with the
        provided register/load_bearing/tags so the realtime detectors
        observe it normally.

        ``register`` defaults to :attr:`Register.DECLARATIVE` because
        the agent *wrote it that way* in user-facing text. The
        :class:`InferenceAsFactDetector` then does its job: if the
        extracted claim has a verifier in the registry and is INFERRED,
        the gate fires.

        Returns the (possibly empty) list of emitted claims.
        """
        from app.epistemic.extraction import extract_claims

        extracted = extract_claims(output_text)
        emitted: list[Claim] = []
        tag_tuple = tuple(tags)
        for raw in extracted:
            from app.epistemic.verification import match as match_verifier

            verifier = match_verifier(raw.statement, tags=tag_tuple)
            evidence = (Evidence(
                kind="model_inference",
                source_ref=(f"span:{span_id}:output"
                            if span_id is not None
                            else f"output:{agent_role}"),
                excerpt=raw.statement,
                confidence=raw.confidence,
            ),)
            emitted.append(self.emit(Claim.new(
                task_id=self._task_id,
                agent_role=agent_role,
                statement=raw.statement,
                status=raw.status,
                register=register,
                evidence=evidence,
                verifying_action=verifier,
                load_bearing=load_bearing,
                tags=tag_tuple,
                span_id=span_id,
            )))
        return emitted

    # ── Mutation: supersession ───────────────────────────────────────

    def supersede(self, *, claim_id: str, replacement: Claim) -> Claim:
        """Mark ``claim_id`` as contradicted by ``replacement``.

        The original claim stays in the ledger (with status flipped to
        CONTRADICTED and ``superseded_by`` set). The replacement is
        emitted normally. History is not erased.

        Returns the new (contradicted) form of the original claim.
        """
        original = self._claims.get(claim_id)
        if original is None:
            raise KeyError(f"claim {claim_id!r} not in ledger")
        if original.status is VerificationStatus.CONTRADICTED:
            raise LedgerError(
                f"claim {claim_id!r} already contradicted by "
                f"{original.superseded_by!r}; cannot re-supersede"
            )

        contradicted = replace(
            original,
            status=VerificationStatus.CONTRADICTED,
            superseded_by=replacement.claim_id,
        )
        self._claims[claim_id] = contradicted
        _persist(contradicted)
        self.emit(replacement)
        return contradicted

    # ── Queries ──────────────────────────────────────────────────────

    def by_id(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    def all(self) -> list[Claim]:
        """Every claim in the ledger, ordered by emission time."""
        return sorted(self._claims.values(), key=lambda c: c.created_at)

    def load_bearing(self) -> list[Claim]:
        return [c for c in self._claims.values() if c.load_bearing]

    def unverified_load_bearing(self) -> list[Claim]:
        return [
            c
            for c in self._claims.values()
            if c.load_bearing
            and c.status in (VerificationStatus.INFERRED, VerificationStatus.ASSUMED)
        ]

    def __len__(self) -> int:
        return len(self._claims)

    def __contains__(self, claim_id: object) -> bool:
        return isinstance(claim_id, str) and claim_id in self._claims


# ── Internal helpers ─────────────────────────────────────────────────

def _persist(claim: Claim) -> None:
    """Best-effort persistence. Imported lazily to avoid a circular
    import: span_writer reads from this module.
    """
    try:
        from app.epistemic.span_writer import persist_claim
        persist_claim(claim)
    except Exception as exc:  # pragma: no cover - belt-and-suspenders
        logger.debug("epistemic ledger: persist_claim failed: %s", exc)


def _dispatch_hooks(claim: Claim, ledger: Ledger) -> None:
    """Run every registered claim hook. A hook that raises is logged at
    WARNING and otherwise ignored — one bad hook must not poison the
    rest, and never breaks emission for the caller."""
    for hook in claim_hooks():
        try:
            hook(claim, ledger)
        except Exception as exc:
            logger.warning(
                "epistemic ledger: claim hook %r raised on claim %s: %s",
                getattr(hook, "__qualname__", hook),
                claim.claim_id,
                exc,
            )


def _format_invocation(tool_name: str, tool_args: Mapping[str, Any]) -> str:
    """Render a tool call as a single line of evidence excerpt.

    Format: ``$ tool key1=value1 key2=value2``. Long arg values are
    truncated to keep the excerpt cap honest.
    """
    parts = [f"$ {tool_name}"]
    for k, v in tool_args.items():
        rendered = repr(v) if not isinstance(v, str) else v
        if len(rendered) > 64:
            rendered = rendered[:64] + "…"
        parts.append(f"{k}={rendered}")
    return " ".join(parts)


def _parse_iso(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    # ``fromisoformat`` accepts the format produced by ``isoformat()``.
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

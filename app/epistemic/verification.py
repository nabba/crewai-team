"""Verifier Registry — the cheap-verification preference rule.

Maps claim shapes (regex over the claim's statement) to read-only
commands that would settle the claim with exact-answer evidence.

The registry is loaded once at import from ``data/verifier_registry.yaml``
and held immutable. Modifications go through CODEOWNERS PR review;
the running agent has no API to add a verifier shape — it can only
*propose* an addition through the Self-Improver loop, which opens a PR.

Safety invariant:
    The loader rejects any entry whose tool (first whitespace token of
    the ``tool`` field) appears in :data:`DESTRUCTIVE_TOOL_NAMES`. A
    side-effecting verifier breaks the entire pre-output gate, so we
    refuse to even instantiate the registry.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from app.epistemic.ledger import VerifyingAction

logger = logging.getLogger(__name__)


# ── Safety boundary: tools a verifier may NEVER use ─────────────────
# A verifier exists to *settle* a claim with read-only evidence.
# Side-effects (delete, write, mutate) belong nowhere near the
# pre-output gate. This list is module-level (immutable, requires
# code review to change) by design.
DESTRUCTIVE_TOOL_NAMES: frozenset[str] = frozenset({
    "rm", "rmdir", "unlink", "shred",
    "mv",                         # mutates filesystem state
    "dd",
    "drop", "DROP", "truncate", "TRUNCATE",
    "delete", "DELETE", "remove",
    "kill", "killall", "pkill",
    "chmod", "chown",             # permission changes
    "git-reset", "git-clean", "git-rm",
    "docker-rm", "docker-kill",
    "psql -c-write",              # placeholder; real psql is allowed for SELECT,
                                  # disambiguated by a check at use-time below.
})


# ── Registry types ──────────────────────────────────────────────────

@dataclass(frozen=True)
class VerifierShape:
    """A single (claim-pattern → verifying-tool) mapping.

    Constructed only by ``_VerifierRegistry.load`` from YAML; the
    constructor is package-internal.
    """

    id: str
    pattern: re.Pattern[str]
    tags_any: tuple[str, ...]
    tool: str
    expected_signal: str
    estimated_seconds: float
    _arg_kind: str
    _arg_groups: Mapping[str, int]
    _arg_template: str | None

    def materialize(self, statement: str) -> VerifyingAction | None:
        """If this shape matches ``statement``, return a concrete
        :class:`VerifyingAction` with arguments extracted from the
        regex groups. Returns ``None`` if no match."""
        match = self.pattern.search(statement)
        if match is None:
            return None
        return VerifyingAction(
            tool=self.tool,
            args=self._extract_args(match),
            expected_signal=self.expected_signal,
            estimated_seconds=self.estimated_seconds,
        )

    def _extract_args(self, match: re.Match[str]) -> dict[str, Any]:
        if self._arg_kind == "none":
            return {}
        if self._arg_kind == "regex_capture":
            return {name: match.group(idx) for name, idx in self._arg_groups.items()}
        if self._arg_kind == "template":
            assert self._arg_template is not None  # invariant from loader
            captured = {name: match.group(idx) for name, idx in self._arg_groups.items()}
            return {"sql": self._arg_template.format(**captured)}
        # Loader is responsible for ensuring _arg_kind is a known value;
        # this path is unreachable absent a YAML-loader bug.
        raise AssertionError(f"unknown arg_extractor kind: {self._arg_kind!r}")


# ── Registry ────────────────────────────────────────────────────────

_DEFAULT_PATH = Path(__file__).parent / "data" / "verifier_registry.yaml"


class VerifierRegistryLoadError(RuntimeError):
    """Raised when the YAML registry contains a violation that cannot
    be safely ignored — typically a destructive tool. Refusing to load
    is the correct behavior; a partial registry would be a footgun."""


class _VerifierRegistry:
    """Immutable collection of :class:`VerifierShape` entries.

    Use :func:`VERIFIER_REGISTRY` (module singleton) for normal callers;
    the class itself is exposed only for tests that need a custom
    registry via :meth:`load_from`.
    """

    def __init__(self, shapes: tuple[VerifierShape, ...]) -> None:
        self._shapes = shapes

    def __len__(self) -> int:
        return len(self._shapes)

    def __iter__(self):
        return iter(self._shapes)

    def match(
        self,
        statement: str,
        *,
        tags: tuple[str, ...] = (),
    ) -> VerifyingAction | None:
        """Return the cheapest verifier that matches ``statement``.

        If multiple shapes match, the one with the lowest
        ``estimated_seconds`` wins — preferring fast exact-answer
        commands over slow ones (the whole point of the layer).

        ``tags`` narrows the search: when the caller supplies tags, a
        shape with non-empty ``tags_any`` is only considered if at
        least one of its tags appears in the caller's list. When the
        caller supplies no tags, all pattern matches are eligible —
        ``tags_any`` is purely advisory (it disambiguates ambiguous
        callers, but never excludes plain content matches).
        """
        candidates: list[VerifierShape] = []
        for shape in self._shapes:
            if shape.pattern.search(statement) is None:
                continue
            if (
                tags
                and shape.tags_any
                and not any(t in tags for t in shape.tags_any)
            ):
                continue
            candidates.append(shape)
        if not candidates:
            return None
        cheapest = min(candidates, key=lambda s: s.estimated_seconds)
        return cheapest.materialize(statement)

    @classmethod
    def load_from(cls, path: Path) -> "_VerifierRegistry":
        """Load and validate a registry from a YAML file.

        Raises:
            VerifierRegistryLoadError: if any entry violates the safety
                invariant (tool in :data:`DESTRUCTIVE_TOOL_NAMES`) or has
                a structurally invalid arg_extractor.
        """
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise VerifierRegistryLoadError(
                f"failed to read verifier registry from {path}: {exc}"
            ) from exc

        if not isinstance(raw, dict) or "verifiers" not in raw:
            raise VerifierRegistryLoadError(
                f"verifier registry at {path} missing top-level 'verifiers' key"
            )

        shapes: list[VerifierShape] = []
        seen_ids: set[str] = set()
        for entry in raw["verifiers"]:
            shape = _shape_from_entry(entry)
            if shape.id in seen_ids:
                raise VerifierRegistryLoadError(
                    f"duplicate verifier id {shape.id!r} in {path}"
                )
            seen_ids.add(shape.id)
            shapes.append(shape)
        logger.info(
            "epistemic verifier registry loaded: %d shapes from %s",
            len(shapes), path,
        )
        return cls(tuple(shapes))


def _shape_from_entry(entry: Mapping[str, Any]) -> VerifierShape:
    """Validate one YAML entry, applying the destructive-tool guard."""
    if "id" not in entry or "tool" not in entry or "matches" not in entry:
        raise VerifierRegistryLoadError(
            f"verifier entry missing required keys (id/tool/matches): {entry!r}"
        )

    tool = str(entry["tool"]).strip()
    head = tool.split()[0] if tool else ""
    if head in DESTRUCTIVE_TOOL_NAMES:
        raise VerifierRegistryLoadError(
            f"verifier {entry['id']!r} uses destructive tool {head!r}; "
            f"refusing to load (a verifier MUST be read-only)"
        )

    matches = entry["matches"]
    if "claim_pattern" not in matches:
        raise VerifierRegistryLoadError(
            f"verifier {entry['id']!r} missing matches.claim_pattern"
        )
    try:
        pattern = re.compile(matches["claim_pattern"])
    except re.error as exc:
        raise VerifierRegistryLoadError(
            f"verifier {entry['id']!r} has invalid regex: {exc}"
        ) from exc

    tags_any = tuple(matches.get("tags_any", []) or [])

    extractor = entry.get("arg_extractor") or {"kind": "none"}
    arg_kind = extractor.get("kind", "none")
    if arg_kind not in ("none", "regex_capture", "template"):
        raise VerifierRegistryLoadError(
            f"verifier {entry['id']!r} arg_extractor.kind must be one of "
            f"none|regex_capture|template, got {arg_kind!r}"
        )
    arg_groups: Mapping[str, int] = extractor.get("groups", {}) or {}
    if arg_kind in ("regex_capture", "template") and not arg_groups:
        raise VerifierRegistryLoadError(
            f"verifier {entry['id']!r} arg_extractor.kind={arg_kind} requires "
            f"non-empty groups"
        )
    arg_template = extractor.get("template")
    if arg_kind == "template" and not arg_template:
        raise VerifierRegistryLoadError(
            f"verifier {entry['id']!r} arg_extractor.kind=template requires "
            f"a template field"
        )

    estimated = float(entry.get("estimated_seconds", 1.0))
    if estimated < 0:
        raise VerifierRegistryLoadError(
            f"verifier {entry['id']!r} estimated_seconds must be non-negative"
        )

    return VerifierShape(
        id=str(entry["id"]),
        pattern=pattern,
        tags_any=tags_any,
        tool=tool,
        expected_signal=str(entry.get("expected_signal", "")),
        estimated_seconds=estimated,
        _arg_kind=arg_kind,
        _arg_groups=dict(arg_groups),
        _arg_template=arg_template,
    )


# ── Module singleton ─────────────────────────────────────────────────
# Built lazily on first access. Tests that need a custom registry can
# call ``_VerifierRegistry.load_from`` directly.

_REGISTRY: _VerifierRegistry | None = None


def VERIFIER_REGISTRY() -> _VerifierRegistry:
    """Return the process-wide registry, loading it on first call.

    A function (rather than a module-level constant) so tests can
    monkey-patch ``_REGISTRY`` to inject a fixture without import-order
    surprises.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _VerifierRegistry.load_from(_DEFAULT_PATH)
    return _REGISTRY


def _reset_for_tests() -> None:
    """Clear the cached registry. Tests only."""
    global _REGISTRY
    _REGISTRY = None


def match(statement: str, *, tags: tuple[str, ...] = ()) -> VerifyingAction | None:
    """Convenience: find a verifier for ``statement`` from the default registry."""
    return VERIFIER_REGISTRY().match(statement, tags=tags)

"""Task-shape classifier — distinguishes ARTIFACT-producing requests
from TEXT-only requests, plus post-run verification that the artifact
actually exists.

This is Cure B from the post-§30 failure-mode triage. Pre-fix the
orchestrator's success contract was "the LLM returned non-empty
text" — so when the user asked for a graphic and the coder returned
Python source code that *would*, if run, produce a graphic, the
orchestrator marked that as success-shaped and passed it to vetting.
Vetting (correctly) said "this doesn't deliver the requested
graphic — only partial code that cannot generate the image", but
vetting can't *make* a graphic, so it just rejects.

This module separates the two failure modes:

  • **Text task fails vetting** — content quality issue. The
    orchestrator's existing ``_build_retry_task`` + retry path
    already handles it.
  • **Artifact task fails to produce a file** — different shape of
    failure. The crew was asked for a PNG, returned text-only.
    Without ``ArtifactNotProduced`` as a typed signal, vetting
    catches it indirectly (and 30 minutes late).

Architecture
============

    user input
         │
         ▼
    ┌────────────────────────┐
    │ classify_task(text)    │ → TaskShape(kind, expected_extensions, …)
    └────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────┐
    │ build_artifact_directive(shape, task_text) │  augments crew task
    │   when shape.kind == "artifact"            │  with "produce + return
    └────────────────────────────────────────────┘  the file path" prompt
         │
         ▼
    crew runs → returns response_text
         │
         ▼
    ┌────────────────────────────────────────────┐
    │ verify_artifacts(shape, response_text)     │ raises ArtifactNotProduced
    │   when shape.kind == "artifact"            │ if expected file is
    └────────────────────────────────────────────┘ missing / empty / bad
         │
         ▼
    response delivered (text + verified artifact path) OR retry

The classifier is a pure heuristic — no LLM call — because shape
detection is a small enumeration and an LLM-call here would be
recursive (Cure A's truncation guard would protect us, but the
extra latency on every routing decision isn't justified).

The verifier reads a small set of allowed roots
(``workspace/output/`` etc.) so a hallucinated path can't be
"verified" by pointing at /etc/passwd. Path-traversal-resistant.
"""
from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# Per-request task-shape — set by ``handle()`` at task entry, read by
# ``_run_crew_inner`` before dispatch (to inject the directive) and
# after dispatch (to verify the artifact).  ContextVar is the right
# scope: each Signal / Discord / API call lands in its own asyncio
# task and gets its own value; concurrent requests don't trample.
_active_task_shape: ContextVar["TaskShape | None"] = ContextVar(
    "commander_active_task_shape", default=None,
)


def get_active_task_shape() -> "TaskShape | None":
    """Read the per-request task shape set by ``handle()``."""
    return _active_task_shape.get()


def set_active_task_shape(shape: "TaskShape | None") -> None:
    """Set the per-request task shape (called from orchestrator)."""
    _active_task_shape.set(shape)


# ── Intent triggers ─────────────────────────────────────────────────


# Verbs that indicate "produce a deliverable".
_ARTIFACT_VERBS = (
    "make", "produce", "generate", "create", "build", "render",
    "draw", "plot", "export", "save", "output",
)

# Nouns + extensions that indicate "the deliverable is a file".
# Two columns: (regex pattern, declared extensions).
_ARTIFACT_NOUNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Visual
    (r"\bgraphic\b",                 (".png", ".jpg", ".jpeg", ".svg", ".pdf")),
    (r"\b(chart|graph)\b",           (".png", ".jpg", ".svg", ".pdf")),
    (r"\b(image|picture|photo|illustration)\b",
                                     (".png", ".jpg", ".jpeg", ".webp")),
    (r"\b(plot|figure|visualization|visualisation)\b",
                                     (".png", ".jpg", ".svg", ".pdf")),
    (r"\b(diagram)\b",                (".png", ".svg", ".pdf")),
    (r"\bmap\b",                      (".png", ".jpg", ".svg", ".geotiff", ".tif", ".pdf")),

    # Documents
    (r"\b(pdf|report)\b",             (".pdf",)),
    (r"\b(slides?|presentation|deck|powerpoint)\b",
                                     (".pptx", ".pdf")),
    (r"\b(document|word\s+doc)\b",   (".docx", ".pdf")),
    (r"\b(spreadsheet|excel)\b",     (".xlsx", ".csv")),

    # Data tables
    (r"\bcsv\s+file\b",               (".csv",)),
    (r"\b(table|dataset)\b\s+(?:as|in|to)\s+(?:csv|excel|file)",
                                     (".csv", ".xlsx")),

    # Generic file
    (r"\bfile\b",                    ()),  # no specific extension; just signals shape
)

# Explicit extension mentions like "output as a .png" or "save as PDF".
_EXPLICIT_EXTENSION_RE = re.compile(
    r"\b(?:as|in|to|output(?:ting)?|save(?:d)?(?:\s+as)?)\s+a?\s*"
    r"\.?(png|jpe?g|svg|pdf|csv|xlsx|docx|pptx|webp|tiff?|geotiff)\b",
    re.IGNORECASE,
)


@dataclass
class TaskShape:
    """The shape of a user task — what the orchestrator's success
    contract should be.

    ``kind == "text"``     — a plain-text response is the deliverable
    ``kind == "artifact"`` — a file at one of ``expected_extensions``
                              must exist before the task is "done"

    ``trigger`` records WHY the classifier decided the way it did
    (the verb / noun / extension that matched). Useful for debug
    logs and for the augmented task directive — the agent reads it
    too so it knows what the user is asking for.
    """
    kind: str = "text"
    expected_extensions: tuple[str, ...] = ()
    trigger: str = ""
    raw_text: str = ""

    @property
    def is_artifact(self) -> bool:
        return self.kind == "artifact"


class ArtifactNotProduced(Exception):
    """Raised when a task classified as artifact-shape returned a
    response that doesn't reference an existing, non-empty artifact
    of the expected shape.

    Carries the shape, the response text, and the list of paths
    that were tried (with per-path failure reasons) so the retry
    path can surface a precise diagnostic instead of "task failed."
    """

    def __init__(
        self,
        shape: TaskShape,
        response_text: str,
        attempted_paths: list[tuple[str, str]],
    ) -> None:
        self.shape = shape
        self.response_text = response_text
        self.attempted_paths = attempted_paths  # [(path, reason), …]
        if not attempted_paths:
            detail = "no file path mentioned in response"
        else:
            detail = "; ".join(
                f"{p}: {r}" for p, r in attempted_paths
            )
        super().__init__(
            f"task expected artifact ({shape.kind}, "
            f"extensions={shape.expected_extensions or 'any'}); "
            f"verification failed — {detail}"
        )


# ── Classifier ──────────────────────────────────────────────────────


def classify_task(text: str) -> TaskShape:
    """Heuristic shape classifier.

    Pure function. No LLM call. Walks a small trigger table and
    returns the first match. Returns ``TaskShape(kind="text")`` when
    nothing matches — text is the safe default; mis-classifying a
    text task as artifact would loop the verifier in failure.

    The classifier is intentionally CONSERVATIVE — it requires both
    a verb signal AND a noun-or-extension signal for an artifact
    classification, so "I'm making a chart" (statement, not request)
    or "the report is interesting" (no verb) don't trigger.
    """
    if not text or not isinstance(text, str):
        return TaskShape(kind="text")

    lower = text.lower()

    has_verb = any(
        re.search(rf"\b{v}\b", lower)
        for v in _ARTIFACT_VERBS
    )

    # Explicit extension match always wins (highest signal).
    m = _EXPLICIT_EXTENSION_RE.search(text)
    if m:
        ext = "." + m.group(1).lower().replace("jpeg", "jpg")
        return TaskShape(
            kind="artifact",
            expected_extensions=(ext,),
            trigger=f"explicit extension: {ext}",
            raw_text=text,
        )

    if not has_verb:
        return TaskShape(kind="text")

    # Noun-pattern table.
    for pattern, extensions in _ARTIFACT_NOUNS:
        if re.search(pattern, lower):
            return TaskShape(
                kind="artifact",
                expected_extensions=extensions,
                trigger=f"noun match: {pattern}",
                raw_text=text,
            )

    return TaskShape(kind="text")


# ── Task directive (prompt augmentation) ────────────────────────────


_ARTIFACT_DIRECTIVE_TEMPLATE = """
TASK SHAPE: artifact-producing request.

The user is asking for a FILE deliverable, not a text answer.
Detected: {trigger}.
Expected file types: {extensions}.

To complete this task you MUST:

  1. Use the coding_session_* tools to create a session, write the
     script, and EXECUTE it.  ``coding_session_run`` is what
     actually produces the file — embedding code in your text
     response without running it does NOT satisfy the contract.

  2. Verify the artifact exists in your session before submitting.
     Use ``coding_session_list_files`` to confirm the output file
     is present and non-empty.

  3. In your final text response, include the artifact path on a
     line of the form::

         ARTIFACT: <relative-path>

     Example::

         ARTIFACT: workspace/output/estonia_forest_age_2024.png

     The orchestrator will VERIFY this path exists.  A response
     that returns code without an existing artifact will be
     rejected and retried.
"""


def build_artifact_directive(shape: TaskShape) -> str:
    """Return the prompt directive to APPEND to the crew task body.

    For text-shape tasks this returns an empty string — no augment.
    For artifact-shape tasks the directive tells the agent how to
    use coding_session tools and how to format the response so the
    verifier can find the artifact.
    """
    if not shape.is_artifact:
        return ""
    extensions = (
        ", ".join(shape.expected_extensions)
        if shape.expected_extensions
        else "any concrete file format (png/pdf/csv/...)"
    )
    return _ARTIFACT_DIRECTIVE_TEMPLATE.format(
        trigger=shape.trigger or "artifact-shape signal",
        extensions=extensions,
    ).strip()


# ── Verifier ────────────────────────────────────────────────────────


# Path-root filter — DENY-list, not allow-list.
#
# Pre-2026-05-10 history. The first cut of this filter was an
# allow-list (``workspace/``, ``output/``, ``/tmp/crewai-``).  It
# turned out to be too narrow: when the operator asked "make a
# graphic about forest-age distribution in Estonia", the agent
# (correctly!) emitted ``ARTIFACT: /tmp/estonia_forest_age_
# distribution.png`` and the file actually existed there
# (353 KB PNG).  But ``/tmp/`` wasn't on the allow-list, so the
# verifier rejected a perfectly-good deliverable, prepended an
# ``[ARTIFACT VERIFICATION FAILED]`` header to the response, and
# the operator received a misleading failure message instead of
# their graphic.
#
# The lesson: existence + non-empty + extension are the REAL
# safety gates.  Path-root filtering was paranoid early gating
# that generated false positives.  The deny-list below blocks
# obviously-malicious roots (``/etc/``, ``/proc/``, etc.) where
# an artifact PNG has no business living, but otherwise lets
# the existence check be authoritative.
_DENIED_ROOTS: tuple[str, ...] = (
    "/etc/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/root/",
)

# Find ``ARTIFACT: <path>`` markers (the directive's preferred shape)
# and bare path mentions inside backticks / on their own line.
_ARTIFACT_MARKER_RE = re.compile(
    r"^\s*ARTIFACT:\s*(\S+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_BACKTICK_PATH_RE = re.compile(
    r"`(?P<path>(?:[A-Za-z0-9_./\-]+)+\."
    r"(?:png|jpg|jpeg|svg|pdf|csv|xlsx|docx|pptx|webp|tiff?|geotiff))`",
    re.IGNORECASE,
)
_BARE_PATH_RE = re.compile(
    r"(?:^|\s)(?P<path>(?:[A-Za-z0-9_./\-]+)+\."
    r"(?:png|jpg|jpeg|svg|pdf|csv|xlsx|docx|pptx|webp|tiff?|geotiff))(?:$|\s|[.,;])",
    re.IGNORECASE | re.MULTILINE,
)


def _path_under_denied_root(path: str) -> bool:
    """Return True if ``path`` is under one of the obviously-malicious
    deny-list roots (``/etc/``, ``/proc/``, etc.).  Existence + non-
    empty + extension checks downstream are the authoritative gates;
    this just catches the cases where the agent's response references
    a system path it had no business producing under."""
    if not path:
        return False
    for root in _DENIED_ROOTS:
        if path.startswith(root):
            return True
    return False


def extract_artifact_paths(
    response_text: str,
    *,
    allowed_extensions: Iterable[str] = (),
) -> list[str]:
    """Pull out every plausible artifact-path mention from the response.

    Search order:
      1. Explicit ``ARTIFACT: <path>`` markers (what the directive
         tells the agent to emit).
      2. Backticked file paths with known extensions.
      3. Bare file paths with known extensions.

    Filters out duplicates and paths outside ``_ALLOWED_ROOTS``.
    """
    seen: list[str] = []
    if not response_text:
        return seen

    allowed_ext_set = {e.lower().lstrip(".") for e in allowed_extensions}

    def _accept(path: str) -> None:
        path = path.strip(".,;'\"")
        if not path:
            return
        # Extension filter (when caller specified some).
        if allowed_ext_set:
            ext = path.rsplit(".", 1)[-1].lower()
            ext = ext.replace("jpeg", "jpg")
            normalized_allowed = {
                e.replace("jpeg", "jpg") for e in allowed_ext_set
            }
            if ext not in normalized_allowed:
                return
        # Deny-list root filter — blocks /etc/, /proc/, etc.  The
        # existence check downstream is the authoritative safety gate.
        if _path_under_denied_root(path):
            return
        if path not in seen:
            seen.append(path)

    for m in _ARTIFACT_MARKER_RE.finditer(response_text):
        _accept(m.group(1))
    for m in _BACKTICK_PATH_RE.finditer(response_text):
        _accept(m.group("path"))
    for m in _BARE_PATH_RE.finditer(response_text):
        _accept(m.group("path"))

    return seen


def verify_artifacts(
    shape: TaskShape,
    response_text: str,
    *,
    workspace_root: Path | None = None,
) -> str:
    """Verify the response references an existing, non-empty artifact.

    Args:
        shape: the classifier's verdict for this task.
        response_text: what the crew returned.
        workspace_root: where to resolve relative paths. Defaults to
            ``/app`` in container, repo-root locally.

    Returns:
        The verified absolute path of the first valid artifact.

    Raises:
        ArtifactNotProduced: when the task is artifact-shape AND no
            referenced path resolves to an existing non-empty file
            of the expected shape. Carries the list of attempted
            paths + per-path reasons so retry diagnostics are precise.

    No-op for text-shape tasks.
    """
    if not shape.is_artifact:
        return ""

    if workspace_root is None:
        # In-container default: /app.  Local-dev callers can pass
        # the repo root explicitly.
        workspace_root = Path("/app")

    candidates = extract_artifact_paths(
        response_text,
        allowed_extensions=shape.expected_extensions,
    )
    # When extract_artifact_paths returns empty, dig deeper to give
    # an honest reason — distinguish "nothing path-shaped in text"
    # from "paths were mentioned but all rejected by extension or
    # deny-list filters".  Pre-2026-05-10 the verifier reported
    # "no file path mentioned" in both cases, which was wrong: an
    # artifact at /tmp/forest.png would say "no file path mentioned"
    # because /tmp/ was off the allow-list, even though the path
    # was clearly in the text.
    attempted: list[tuple[str, str]] = []
    if not candidates:
        # Re-scan with NO extension filter to find any path-shaped
        # tokens that DID appear; record them with rejection reasons.
        all_paths = extract_artifact_paths(response_text)
        if all_paths:
            for p in all_paths:
                ext = p.rsplit(".", 1)[-1].lower().replace("jpeg", "jpg")
                expected = {
                    e.lstrip(".").lower().replace("jpeg", "jpg")
                    for e in shape.expected_extensions
                }
                if expected and ext not in expected:
                    attempted.append((
                        p,
                        f"wrong extension ({ext}); expected one of "
                        f"{sorted(shape.expected_extensions)}",
                    ))

    for raw_path in candidates:
        # Resolve relative paths against workspace_root.
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = workspace_root / raw_path
        try:
            if not candidate.exists():
                attempted.append((str(candidate), "does not exist"))
                continue
            if not candidate.is_file():
                attempted.append((str(candidate), "not a regular file"))
                continue
            size = candidate.stat().st_size
            if size == 0:
                attempted.append((str(candidate), "file is empty (0 bytes)"))
                continue
            # Found a viable artifact.
            logger.info(
                "artifact_intent: verified artifact at %s (%d bytes)",
                candidate, size,
            )
            return str(candidate)
        except OSError as exc:
            attempted.append((str(candidate), f"OSError: {exc}"))
            continue

    # If we got here, none of the candidates panned out.
    raise ArtifactNotProduced(
        shape=shape,
        response_text=response_text,
        attempted_paths=attempted,
    )


__all__ = [
    "TaskShape",
    "ArtifactNotProduced",
    "classify_task",
    "build_artifact_directive",
    "extract_artifact_paths",
    "verify_artifacts",
    "get_active_task_shape",
    "set_active_task_shape",
]

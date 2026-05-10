"""Library radar proposer — see :mod:`app.library_radar` package docstring."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_DAEMON_THREAD_NAME = "library-radar-proposer"
_WARMUP_S = 60
_POLL_INTERVAL_S = 24 * 3600  # daily

_DEFAULT_REQUIREMENTS_PATH = Path("/app/requirements.txt")
_INTERESTING_CATEGORIES = frozenset({"frameworks", "tools"})

# Tech-radar text format from app/crews/tech_radar_crew.py:
#   "[<category>] <title>: <summary>. Action: <action>"
_RADAR_LINE_RE = re.compile(
    r"^\[(?P<category>\w+)\]\s*(?P<title>.+?):\s*(?P<summary>.+?)"
    r"(?:\.\s*Action:\s*(?P<action>.+))?$"
)

# Crude package-name heuristic. Looks for tokens that look like Python
# package names in the title or summary: lowercase + hyphens/underscores,
# 2+ chars, not a sentence word. The operator vets the proposed name in
# the markdown anyway — this is just a starting suggestion.
_PKG_TOKEN_RE = re.compile(r"\b([a-z][a-z0-9_-]{2,30})\b")
_PKG_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "you", "are",
    "your", "have", "has", "use", "uses", "using", "new", "now", "can",
    "via", "across", "into", "more", "less", "than", "but", "not",
    "one", "two", "all", "any", "out", "off", "set", "get", "see",
    "model", "models", "agent", "agents", "framework", "frameworks",
    "tool", "tools", "library", "libraries", "package", "packages",
    "python", "code", "data", "test", "tests", "support", "release",
    "released", "open", "source", "based",
})


_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


def _enabled() -> bool:
    return os.getenv("LIBRARY_RADAR_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


@dataclass(frozen=True)
class Discovery:
    """A parsed tech-radar entry ready for adoption review."""

    category: str
    title: str
    summary: str
    action: str
    candidate_packages: list[str]


def _parse_radar_line(line: str) -> Discovery | None:
    m = _RADAR_LINE_RE.match(line.strip())
    if not m:
        return None
    category = (m.group("category") or "").lower()
    title = (m.group("title") or "").strip()
    summary = (m.group("summary") or "").strip()
    action = (m.group("action") or "").strip()
    candidates = _extract_package_names(f"{title} {summary}")
    return Discovery(
        category=category,
        title=title,
        summary=summary,
        action=action,
        candidate_packages=candidates,
    )


def _extract_package_names(text: str) -> list[str]:
    """Heuristic: pull plausible package name tokens. Operator vets."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _PKG_TOKEN_RE.findall(text.lower()):
        if tok in _PKG_STOPWORDS:
            continue
        if "-" not in tok and "_" not in tok and tok.isalpha() and len(tok) < 5:
            # Pure short alphabetic tokens are almost never package names.
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= 6:
            break
    return out


def _read_requirements(path: Path) -> set[str]:
    """Return lowercased set of package names already pinned."""
    if not path.exists():
        return set()
    out: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            # Strip extras + version specifiers.
            name = re.split(r"[\s\[<>=!~;]", line, maxsplit=1)[0]
            if name:
                out.add(name.lower().strip())
    except OSError:
        logger.debug("library_radar: cannot read %s", path, exc_info=True)
    return out


def _signature_for(discovery: Discovery) -> str:
    """Stable filesystem-safe signature for dedup."""
    raw = f"{discovery.category}|{discovery.title}".lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _slug(text: str, fallback: str = "library") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50]
    return s or fallback


def _build_coding_session_spec(d: Discovery, slug: str) -> dict:
    """Q2 §39: scaffold for trialing a library proposal.

    Library adoption is a structured workflow: confirm PyPI name +
    license, add a smoke-import test inside a worktree, run pytest
    + ruff + mypy, file a CR for the requirements.txt change. The
    spec captures that workflow as a coding-session.
    """
    candidates = " | ".join(d.candidate_packages[:3]) if d.candidate_packages else "?"
    return {
        "intent": (
            f"Trial adoption of {d.title[:60]} ({d.category})"
        ),
        "files": [
            {
                "path": "requirements.txt",
                "action": "edit",
                "purpose": (
                    f"add the chosen package (candidates: {candidates}) "
                    f"with a version pin"
                ),
                "size_estimate": "+1 line",
            },
            {
                "path": f"tests/library_trials/test_{slug[:30]}_smoke.py",
                "action": "create",
                "purpose": (
                    "smoke-import test that imports the package + "
                    "verifies a top-level public symbol works"
                ),
                "size_estimate": "~25 LOC",
            },
        ],
        "acceptance": [
            f"pip install -r requirements.txt",
            f"pytest tests/library_trials/test_{slug[:30]}_smoke.py -v",
            "verify license is compatible with the project license",
        ],
        "expected_duration_min": 30,
    }


def _render_proposal(d: Discovery, *, signature: str) -> str:
    pkg_list = ", ".join(f"`{p}`" for p in d.candidate_packages) or "(none extracted)"
    action_block = f"\n## Tech-radar suggested action\n\n{d.action}\n" if d.action else ""
    return (
        f"# Library adoption proposal — {d.title}\n"
        f"\n"
        f"> Auto-generated by `app.library_radar`. Filtered from "
        f"`scope_tech_radar` ChromaDB on category=`{d.category}`.\n"
        f"> Signature: `{signature}`\n"
        f"\n"
        f"## Summary\n"
        f"\n"
        f"{d.summary}\n"
        f"{action_block}"
        f"\n"
        f"## Candidate package name(s)\n"
        f"\n"
        f"{pkg_list}\n"
        f"\n"
        f"The operator vets the actual PyPI name; the radar's title/summary\n"
        f"is heuristically tokenised here as a starting point.\n"
        f"\n"
        f"## Adoption checklist\n"
        f"\n"
        f"- [ ] Confirm the PyPI name (or the GitHub repo if the package isn't on PyPI)\n"
        f"- [ ] Verify the license is compatible with the project license\n"
        f"- [ ] Check the package's release cadence + maintainer signal\n"
        f"- [ ] Trial in a coding-session worktree against a small benchmark\n"
        f"- [ ] If trial passes, file a change-request to add the pin to\n"
        f"      `requirements.txt` (standard /cp/changes gate)\n"
        f"\n"
        f"## Operator action\n"
        f"\n"
        f"- **Adopt**: file a change-request for the `requirements.txt` edit.\n"
        f"- **Defer**: leave this file in place; the dedup-by-signature\n"
        f"  guard prevents re-emitting the same discovery.\n"
        f"- **Reject**: delete this file. (The next radar pass may re-emit\n"
        f"  if the discovery resurfaces — that's a feature, not a bug, since\n"
        f"  it surfaces persistent operator-relevant signal.)\n"
    )


def run_one_pass(
    *,
    discoveries: list[str] | None = None,
    requirements_path: Path | str | None = None,
) -> dict:
    """Single proposer pass. Returns a structured result dict.

    Test/operator hooks:
      ``discoveries``        overrides the radar source (raw text lines)
                             so tests don't need ChromaDB.
      ``requirements_path``  overrides the dependency manifest path.
    Proposals are staged via ``app.proposal_bridge`` for promotion
    to the change-request gate; tests should monkeypatch
    ``PROPOSAL_BRIDGE_DIR`` and inspect via
    ``proposal_bridge.list_proposals(source='library_radar')``.
    """
    if not _enabled():
        return {"status": "disabled", "drafts_written": 0}

    req_path = Path(requirements_path) if requirements_path else _DEFAULT_REQUIREMENTS_PATH

    if discoveries is None:
        discoveries = _load_radar_lines()

    if not discoveries:
        return {"status": "no_evidence", "drafts_written": 0}

    parsed = [d for d in (_parse_radar_line(ln) for ln in discoveries) if d is not None]
    parsed = [d for d in parsed if d.category in _INTERESTING_CATEGORIES]
    if not parsed:
        return {
            "status": "no_relevant",
            "drafts_written": 0,
            "n_evidence": len(discoveries),
        }

    pinned = _read_requirements(req_path)
    if pinned:
        parsed = [
            d for d in parsed
            if not any(c in pinned for c in d.candidate_packages)
        ]
    if not parsed:
        return {
            "status": "all_already_pinned",
            "drafts_written": 0,
            "n_evidence": len(discoveries),
        }

    try:
        from app.proposal_bridge import stage
    except Exception:
        logger.warning("library_radar: proposal_bridge unavailable", exc_info=True)
        return {
            "status": "bridge_unavailable",
            "drafts_written": 0,
            "n_evidence": len(discoveries),
            "n_relevant": len(parsed),
        }

    written = 0
    skipped = 0
    for discovery in parsed:
        sig = _signature_for(discovery)
        slug = _slug(discovery.title)
        target_path = f"docs/proposed_libraries/{sig}-{slug}.md"
        try:
            state, was_new = stage(
                source="library_radar",
                signature=sig,
                title=discovery.title[:80] or "library proposal",
                body_markdown=_render_proposal(discovery, signature=sig),
                target_path=target_path,
                coding_session_spec=_build_coding_session_spec(discovery, slug),
            )
        except Exception:
            logger.warning(
                "library_radar: stage failed for %s",
                sig, exc_info=True,
            )
            continue
        if was_new:
            written += 1
            logger.info(
                "library_radar: staged %s (category=%s, status=%s)",
                sig, discovery.category, state.status.value,
            )
        else:
            skipped += 1

    return {
        "status": "ok",
        "n_evidence": len(discoveries),
        "n_relevant": len(parsed),
        "drafts_written": written,
        "drafts_skipped_dedup": skipped,
    }


def _load_radar_lines() -> list[str]:
    """Pull recent radar discoveries from ChromaDB. Failure → empty."""
    try:
        from app.memory.scoped_memory import retrieve_operational
    except Exception as exc:  # noqa: BLE001
        logger.debug("library_radar: scoped_memory import failed: %s", exc)
        return []
    try:
        results = retrieve_operational(
            scope="scope_tech_radar",
            query="library framework tool",
            n=50,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("library_radar: retrieve_operational failed: %s", exc)
        return []
    # retrieve_operational typically returns list[str] or list[dict].
    out: list[str] = []
    for r in results or []:
        if isinstance(r, str):
            out.append(r)
        elif isinstance(r, dict):
            text = r.get("document") or r.get("text") or r.get("content") or ""
            if text:
                out.append(text)
    return out


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            result = run_one_pass()
            if result.get("drafts_written", 0) > 0:
                logger.info(
                    "library_radar: %d new adoption proposal(s)",
                    result["drafts_written"],
                )
        except Exception:
            logger.debug("library_radar: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    global _driver_started
    if not _enabled():
        logger.info("library_radar: disabled via LIBRARY_RADAR_ENABLED")
        return
    with _driver_lock:
        if _is_running():
            return
        if _driver_started:
            logger.warning(
                "library_radar: previous thread is dead, re-spawning",
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info(
            "library_radar: daemon started (warm-up=%ds, poll=%dh)",
            _WARMUP_S, _POLL_INTERVAL_S // 3600,
        )


def stop() -> None:
    _stop_event.set()


# Eager start at import — same pattern as healing/monitors and the
# inquiry scheduler.
start()

"""
llm_registry_scanner.py — Discover models on the Ollama public registry
that aren't yet pulled to the local daemon.

Closes the gap exposed by the 2026-04-25 qwen3.5 incident:

  * ``llm_discovery.scan_ollama()`` only hits the LOCAL daemon's
    ``/api/tags`` — it sees what's already on disk.
  * Result: a strictly-better release like ``qwen3.5:35b-a3b-q4_K_M``
    (3 weeks live, fixes mem0 function-calling, vision + thinking modes)
    stayed invisible to discovery for 21 days because nobody had pulled
    it yet.

This module fetches ``ollama.com/library/<family>/tags`` for a small
allowlist of model families we care about, parses the listed variants,
filters by size + quantization heuristics, diffs against locally-pulled
tags, and produces governance proposals of shape
``request_type="local_model_pull"``.

The system NEVER auto-pulls — pulls are 5-50 GB on disk and require
explicit user approval via Signal or the dashboard. The scanner
surfaces *candidates*; the user decides.

Wired into ``llm_discovery.run_discovery_cycle()``; runs alongside the
local-daemon scan during the existing idle cycle. Off-host network calls
respect the existing ``LLM_REGISTRY_SCAN_ENABLED`` flag (default on so
new installs benefit; flip it off via env if you don't want public
registry calls from this host).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────
#
# Family allowlist — keep tight. Each family entry costs one HTTP GET to
# ollama.com per scan cycle. Default list covers the families this team
# has used productively (Qwen, Gemma, Llama, DeepSeek, Codestral). Users
# tune via LLM_REGISTRY_FAMILIES (comma-separated) without code changes.

_DEFAULT_FAMILIES: tuple[str, ...] = (
    "qwen3.5",       # MoE successor to qwen3 — vision + tools + thinking
    "qwen3",         # incumbent local default
    "gemma3",        # Google open-source — strong writing/reasoning
    "llama3.1",      # Meta — general-purpose fallback
    "llama4",        # Meta next-gen (when published)
    "deepseek-r1",   # reasoning-specialist
    "codestral",     # coding-specialist (Mistral)
)

# Size guard — anything bigger than this gets skipped (no proposal).
#
# 2026-04-25: replaced hardcoded constant with auto-detection that
# reads the actual host capability. Motivation — the deepseek-r1:32b
# SIGKILL spiral happened because TWO ~38 GB models loaded into 48 GB
# unified memory simultaneously. A static 24 GB cap was the right number
# for THIS Mac but would be wrong for a 16 GB Mac (catastrophically too
# big) or a 96 GB workstation (leaves capability on the table). Auto-detect
# computes the budget from observable signals so the scanner becomes
# portable.
#
# Manual override still respected via LLM_REGISTRY_MAX_SIZE_GB.
_DEFAULT_MAX_SIZE_GB_FALLBACK = 16.0   # conservative fallback if probe fails

# Resource-budget heuristics (tunable via env if needed)
_OS_BASELINE_GB_MACOS = 10.0    # macOS WindowServer + system services
_OS_BASELINE_GB_LINUX = 4.0     # bare-metal Linux baseline
_KV_CACHE_OVERHEAD = 1.20       # +20% for model's KV cache at load time
_SAFETY_FACTOR = 0.95           # leave 5% slack so we don't graze the ceiling

# Tag-suffix allowlist — only candidates matching one of these patterns
# get proposed. Keeps memory-bloated variants (fp16, bf16) and
# experimental quants (mxfp8, nvfp4) out of the proposal stream.
_PREFERRED_QUANT_SUFFIXES: tuple[str, ...] = (
    "-q4_K_M", "-q4_k_m",
    "-q5_K_M", "-q5_k_m",
    "-q8_0",        # only proposed if size ≤ max
    "-instruct-2507-q4_",
    "-instruct-2507-q4_K_M",
)

# Naming conventions worth a strong upvote — these directly map to the
# things we care about: MoE for speed, instruct for tool use, coding/
# reasoning specialization for crew matching.
_FEATURE_HINTS: dict[str, str] = {
    "-a3b": "moe-3b-active",
    "-a10b": "moe-10b-active",
    "-a22b": "moe-22b-active",
    "-instruct": "instruct-tuned",
    "-coding": "coding-specialist",
    "-thinking": "thinking-mode",
    "-fp16": "full-precision",
    "-bf16": "bfloat16",
}


# ── Data shape ───────────────────────────────────────────────────────────

@dataclass
class RegistryCandidate:
    """A model variant available on ollama.com but not yet pulled locally."""
    family: str           # e.g. "qwen3.5"
    tag: str              # e.g. "35b-a3b-q4_K_M"
    full_name: str        # "qwen3.5:35b-a3b-q4_K_M" — direct ollama-pull arg
    digest: str           # short manifest digest, e.g. "3460ffeede54"
    size_gb: float        # disk footprint
    context_k: int        # native context window in K tokens
    modality: str         # "Text" or "Text, Image"
    features: list[str] = field(default_factory=list)  # tags from _FEATURE_HINTS

    def to_proposal_detail(self, capacity: "HostCapacity | None" = None) -> dict:
        """Serializable dict for the governance request body.

        When ``capacity`` is provided, includes a fit verdict so the user
        can see at a glance whether this model is comfortably within the
        host budget or right at the edge.
        """
        detail = {
            "model": self.full_name,
            "family": self.family,
            "tag": self.tag,
            "digest": self.digest,
            "size_gb": self.size_gb,
            "context_k": self.context_k,
            "modality": self.modality,
            "features": self.features,
            "pull_command": f"ollama pull {self.full_name}",
        }
        if capacity:
            cap = capacity.max_model_size_gb
            if self.size_gb <= cap * 0.75:
                fit = "comfortable"
            elif self.size_gb <= cap:
                fit = "marginal"
            else:
                fit = "over_budget"
            detail["fit"] = fit
            detail["host_capacity"] = {
                "total_ram_gb": capacity.total_ram_gb,
                "ollama_budget_gb": capacity.ollama_budget_gb,
                "max_model_size_gb": cap,
                "max_loaded_models": capacity.max_loaded_models,
                "source": capacity.source,
            }
        return detail


# ── HTML parsing ──────────────────────────────────────────────────────────
#
# The /library/<family>/tags HTML renders each variant as roughly:
#
#   <tag_name> <digest12> • <size>GB • <context>K context window
#       • <modality> input • <age>
#
# Regex-only parse (no BeautifulSoup) — keeps the dependency surface small
# and the parser fast. The HTML format has been stable for 18+ months;
# if Ollama redesigns the page we'll see the parse return [] and emit no
# proposals (degrades gracefully — no false alarms, just no auto-suggest).

_TAG_ROW_PATTERN = re.compile(
    r"([a-z0-9.]+:[0-9a-zA-Z._-]+)"        # 1: full tag name
    r"\s+([a-f0-9]{12})"                    # 2: digest12
    r"\s*[•·]\s*([0-9.]+)\s*GB"             # 3: size in GB
    r"\s*[•·]\s*([0-9]+)\s*K\s+context"     # 4: context in K
    r"[^•·]*?[•·]\s*([^•·]+?)\s+input",     # 5: modality (Text, Image)
    re.I,
)


def _strip_html(raw: str) -> str:
    """Reduce HTML to a single-line text-content string for regex matching."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_tags_page(html: str, family: str) -> list[RegistryCandidate]:
    """Parse one ``/library/<family>/tags`` HTML page into candidates.

    Returns variants matching ``family:`` only. Multi-family pages
    aren't a thing on ollama.com today, but the family check keeps the
    parser strict against future changes.
    """
    text = _strip_html(html)
    out: list[RegistryCandidate] = []
    seen: set[str] = set()

    for m in _TAG_ROW_PATTERN.finditer(text):
        full_name = m.group(1)
        # The same tag is rendered multiple times on the page (once
        # detailed, once compact). Dedupe by name.
        if full_name in seen:
            continue
        if not full_name.startswith(f"{family}:"):
            continue
        seen.add(full_name)

        tag = full_name.split(":", 1)[1]
        try:
            size_gb = float(m.group(3))
        except ValueError:
            continue
        try:
            context_k = int(m.group(4))
        except ValueError:
            context_k = 0

        # Detect feature hints from the tag name
        features = [
            label for suffix, label in _FEATURE_HINTS.items()
            if suffix in tag.lower()
        ]

        out.append(RegistryCandidate(
            family=family,
            tag=tag,
            full_name=full_name,
            digest=m.group(2),
            size_gb=size_gb,
            context_k=context_k,
            modality=m.group(5).strip(),
            features=features,
        ))
    return out


# ── Filtering ─────────────────────────────────────────────────────────────

def _quant_score(tag: str) -> int:
    """Higher = better-fit quantization. Negative = skip."""
    t = tag.lower()
    if any(s.lower() in t for s in _PREFERRED_QUANT_SUFFIXES):
        if "q4" in t:
            return 3              # Q4_K_M is the sweet spot
        if "q5" in t:
            return 2
        if "q8" in t:
            return 1              # accept only if under size cap
    # Reject memory-bloated or experimental quants outright
    if any(x in t for x in ("-fp16", "-bf16", "-mxfp8", "-nvfp4")):
        return -1
    # Bare param-only tag like "35b" — Ollama defaults to a sensible quant
    # (usually q4_K_M). Accept as a fallback.
    if re.fullmatch(r"[0-9.]+b(?:-a[0-9]+b)?", t):
        return 1
    return 0


def filter_candidates(
    candidates: list[RegistryCandidate],
    *,
    max_size_gb: float = _DEFAULT_MAX_SIZE_GB_FALLBACK,
    min_size_gb: float = 4.0,
) -> list[RegistryCandidate]:
    """Drop variants that don't fit the host or use exotic quants.

    * Size in [min_size_gb, max_size_gb] (default uses safe fallback;
      production callers pass the auto-detected cap from
      ``probe_host_capacity()`` via ``scan_ollama_registry()``)
    * Preferred quantization or bare tag (rejects fp16/bf16/exotic)
    * MoE variants (-a3b/-a10b/-a22b) get a slight preference via sort
    """
    result = []
    for c in candidates:
        if not (min_size_gb <= c.size_gb <= max_size_gb):
            continue
        if _quant_score(c.tag) < 0:
            continue
        result.append(c)

    # Sort by: features richness (desc), quant score (desc), size (asc).
    # MoE + instruct variants surface first; among equals, smaller size
    # wins so the user's first proposal is the cheapest-to-pull.
    result.sort(
        key=lambda c: (
            -len(c.features),
            -_quant_score(c.tag),
            c.size_gb,
        )
    )
    return result


# ── Network fetch ─────────────────────────────────────────────────────────

def _fetch_family_html(family: str, *, timeout: float = 8.0) -> str:
    """GET ``ollama.com/library/<family>/tags``. Returns "" on any error."""
    import httpx
    url = f"https://ollama.com/library/{family}/tags"
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
        logger.debug(
            f"registry_scan: {family} returned {resp.status_code}"
        )
        return ""
    except Exception as exc:
        logger.debug(f"registry_scan: {family} fetch failed: {exc}")
        return ""


def _families_from_env() -> tuple[str, ...]:
    raw = os.getenv("LLM_REGISTRY_FAMILIES", "").strip()
    if not raw:
        return _DEFAULT_FAMILIES
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(parts) or _DEFAULT_FAMILIES


def _max_size_from_env() -> float:
    """Resolve the size cap.

    Priority:
      1. Manual override via env ``LLM_REGISTRY_MAX_SIZE_GB`` (escape hatch
         for ops who know better than the heuristic).
      2. Auto-detection — see ``probe_host_capacity()``.
      3. Conservative fallback if both fail.
    """
    raw = os.getenv("LLM_REGISTRY_MAX_SIZE_GB", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    detected = probe_host_capacity()
    if detected and detected.max_model_size_gb > 0:
        return detected.max_model_size_gb
    return _DEFAULT_MAX_SIZE_GB_FALLBACK


# ── Host-capacity probe ──────────────────────────────────────────────────
#
# The scanner needs to know "what's the biggest local model that fits on
# this machine without triggering the SIGKILL spiral we hit on 2026-04-25?"
# Static constants don't generalize across hardware tiers; we compute it
# from observable signals.
#
# Formula:
#
#     ollama_budget = total_ram - os_baseline - docker_overhead
#     max_model    = ollama_budget / max_loaded_models / kv_factor * safety
#
# Each input has a fallback chain so the probe always returns *something*
# usable rather than crashing the scan. Failures degrade to the
# fallback constant (16 GB) — strictly conservative.

@dataclass
class HostCapacity:
    """Auto-detected host resource budget for local model selection."""
    total_ram_gb: float          # detected total physical RAM
    os_baseline_gb: float        # OS/system reservation
    docker_overhead_gb: float    # sum of container memory limits
    ollama_budget_gb: float      # what's left for Ollama
    max_loaded_models: int       # OLLAMA_MAX_LOADED_MODELS (default 1)
    kv_factor: float             # KV-cache overhead multiplier
    max_model_size_gb: float     # final cap — what scanner uses
    source: str                  # which probe path produced these numbers


def probe_host_capacity() -> HostCapacity | None:
    """Detect the safe-to-pull model size from the running environment.

    Returns ``None`` if every probe path fails (caller should use the
    conservative fallback constant in that case).

    Docker-on-Mac caveat: ``/info`` returns the Docker VM's RAM, not the
    macOS host's. On a 48 GB Mac, Docker reports 23.4 GB — and Ollama
    runs OUTSIDE that VM, on the host, competing with macOS itself. The
    probe distinguishes "host-authoritative" sources (env override,
    sysctl) from "VM-view" sources (docker_info, proc_meminfo, psutil)
    and adjusts the ``docker_overhead_gb`` calculation accordingly:

      * host-authoritative source → docker_overhead is the SUM of
        container memory limits (each container competes with Ollama
        for host RAM).
      * VM-view source → docker_overhead is ALREADY the Docker VM cap;
        adding container limits would double-count.

    Set ``HOST_TOTAL_RAM_GB`` in your env to make this deterministic.
    Without it the probe degrades gracefully — VM-view sources cap
    aggressively (small Ollama budget = small models proposed = safe).
    """
    try:
        total_ram_gb = _detect_total_ram_gb()
        if total_ram_gb <= 0:
            return None

        source = _LAST_PROBE_SOURCE.get("source", "mixed")
        os_baseline_gb = _detect_os_baseline_gb()
        max_loaded = _detect_max_loaded_models()
        kv_factor = _KV_CACHE_OVERHEAD

        # See docstring — overhead semantics depend on whether total_ram
        # is the host's actual RAM or the Docker VM's slice of it.
        if source in ("env", "sysctl"):
            # total_ram is the macOS/Linux host's true RAM. Container
            # limits are the genuine overhead.
            docker_overhead_gb = _detect_docker_overhead_gb()
        else:
            # total_ram already reflects only what Docker can see.
            # Subtracting per-container limits would double-count.
            docker_overhead_gb = 0.0

        ollama_budget = max(0.0, total_ram_gb - os_baseline_gb - docker_overhead_gb)
        # Per-model budget = total budget / how many can be loaded at once
        per_model_budget = ollama_budget / max(1, max_loaded)
        # Apply KV-cache overhead and a small safety factor
        max_model_size_gb = (per_model_budget / kv_factor) * _SAFETY_FACTOR

        return HostCapacity(
            total_ram_gb=round(total_ram_gb, 1),
            os_baseline_gb=round(os_baseline_gb, 1),
            docker_overhead_gb=round(docker_overhead_gb, 1),
            ollama_budget_gb=round(ollama_budget, 1),
            max_loaded_models=max_loaded,
            kv_factor=kv_factor,
            max_model_size_gb=round(max_model_size_gb, 1),
            source=source,
        )
    except Exception as exc:
        logger.debug(f"probe_host_capacity: failed: {exc}", exc_info=True)
        return None


# Tracks which fallback path delivered each measurement (for telemetry).
_LAST_PROBE_SOURCE: dict[str, str] = {}


def _detect_total_ram_gb() -> float:
    """Total physical RAM in GB. Tries multiple sources in order.

    Inside Docker, the container's /proc/meminfo and psutil both report
    the cgroup memory limit, NOT the host's true RAM. Since Ollama runs
    on the HOST and competes with Docker containers for host memory,
    we must report the HOST's total — otherwise the budget calculation
    is nonsense (we'd subtract Docker overhead from a number that's
    already the container's limit).

    Resolution order prefers the most authoritative source:
      1. ``HOST_TOTAL_RAM_GB`` env (manual; trumps everything).
      2. Docker daemon ``/info`` endpoint (returns host stats — this is
         the right answer when running inside Docker).
      3. macOS sysctl (works when ``run_host.py`` bypasses Docker).
      4. ``/proc/meminfo`` and ``psutil`` (last resort; container view).
    """
    # 1. Explicit env override
    raw = os.getenv("HOST_TOTAL_RAM_GB", "").strip()
    if raw:
        try:
            _LAST_PROBE_SOURCE["source"] = "env"
            return float(raw)
        except ValueError:
            pass

    # 2. Docker /info — definitive host RAM when Docker is reachable
    try:
        import httpx
        docker_host = os.getenv("DOCKER_HOST", "")
        if docker_host.startswith("tcp://"):
            base = "http://" + docker_host[len("tcp://"):]
            resp = httpx.get(f"{base}/info", timeout=3)
            if resp.status_code == 200:
                mem_bytes = resp.json().get("MemTotal", 0)
                if mem_bytes > 0:
                    _LAST_PROBE_SOURCE["source"] = "docker_info"
                    return mem_bytes / (1024 ** 3)
    except Exception:
        pass

    # 3. macOS sysctl — works on host (run_host.py path)
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=2,
        )
        bytes_total = int(out.strip())
        _LAST_PROBE_SOURCE["source"] = "sysctl"
        return bytes_total / (1024 ** 3)
    except Exception:
        pass

    # 4. /proc/meminfo — container-view fallback
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    _LAST_PROBE_SOURCE["source"] = "proc_meminfo"
                    return kb / 1024 / 1024
    except Exception:
        pass

    # 5. psutil — final fallback
    try:
        import psutil  # type: ignore[import-not-found]
        bytes_total = psutil.virtual_memory().total
        _LAST_PROBE_SOURCE["source"] = "psutil"
        return bytes_total / (1024 ** 3)
    except Exception:
        pass

    return 0.0


def _detect_os_baseline_gb() -> float:
    """Estimate OS + system service overhead.

    macOS reserves ~10 GB for WindowServer + system frameworks; Linux
    (bare-metal or container) is closer to 4 GB.
    """
    raw = os.getenv("HOST_OS_BASELINE_GB", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    import platform
    return _OS_BASELINE_GB_MACOS if platform.system() == "Darwin" else _OS_BASELINE_GB_LINUX


def _detect_docker_overhead_gb() -> float:
    """Sum of memory limits across running Docker containers.

    Tries the local Docker socket via DOCKER_HOST (the gateway already
    talks to the docker-proxy). Returns 0 if Docker isn't reachable —
    the caller will then assume "no Docker overhead" which is correct
    for the host-only run_host.py path.
    """
    try:
        import httpx
        docker_host = os.getenv("DOCKER_HOST", "")
        if docker_host.startswith("tcp://"):
            base = "http://" + docker_host[len("tcp://"):]
        else:
            # Unix socket path or unset — fall through to no-overhead.
            return 0.0
        # /containers/json with size=0 returns memory limits per running
        # container. We sum HostConfig.Memory (bytes; 0 means unlimited
        # which we ignore — there's no way to bound an unlimited container).
        resp = httpx.get(f"{base}/containers/json", timeout=4)
        if resp.status_code != 200:
            return 0.0
        rows = resp.json()
        total = 0
        for row in rows:
            cid = row.get("Id", "")
            if not cid:
                continue
            try:
                detail = httpx.get(f"{base}/containers/{cid}/json", timeout=3).json()
                mem = (detail.get("HostConfig") or {}).get("Memory") or 0
                if mem and mem > 0:
                    total += mem
            except Exception:
                continue
        return total / (1024 ** 3) if total else 0.0
    except Exception:
        return 0.0


def _detect_max_loaded_models() -> int:
    """How many models Ollama keeps simultaneously hot.

    OLLAMA_MAX_LOADED_MODELS controls this. Default Ollama behavior is
    "as many as fit" which is unsafe on memory-constrained hosts —
    that's exactly what triggered the deepseek-r1:32b SIGKILL spiral.
    We assume 1 if unset (the scanner errs toward letting bigger
    individual models in, since serial loading is the safer pattern).
    """
    for var in ("OLLAMA_MAX_LOADED_MODELS",):
        raw = os.getenv(var, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    # Probe ollama daemon's reported config if reachable
    try:
        import httpx
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        resp = httpx.get(f"{ollama_url}/api/ps", timeout=2)
        if resp.status_code == 200:
            # /api/ps doesn't expose config but if multiple models are
            # already loaded we know MAX_LOADED ≥ count. Floor at 1.
            n_loaded = len(resp.json().get("models", []))
            return max(1, n_loaded)
    except Exception:
        pass
    return 1


# ── Public API ────────────────────────────────────────────────────────────

def scan_ollama_registry(
    *,
    families: tuple[str, ...] | None = None,
    max_size_gb: float | None = None,
    fetch: callable = _fetch_family_html,
) -> list[RegistryCandidate]:
    """Scan ollama.com for variants of the configured family allowlist.

    ``fetch`` is injectable for tests. Default uses httpx to hit
    ollama.com directly.
    """
    if not _is_enabled():
        return []
    fams = families or _families_from_env()
    cap = max_size_gb if max_size_gb is not None else _max_size_from_env()
    out: list[RegistryCandidate] = []
    for family in fams:
        html = fetch(family)
        if not html:
            continue
        parsed = parse_tags_page(html, family)
        kept = filter_candidates(parsed, max_size_gb=cap)
        out.extend(kept)
        logger.debug(
            f"registry_scan: family={family} parsed={len(parsed)} kept={len(kept)}"
        )
    return out


def diff_against_local(
    candidates: list[RegistryCandidate],
    local_tags: list[str],
) -> list[RegistryCandidate]:
    """Drop candidates whose full_name already appears in ``local_tags``.

    ``local_tags`` is the ``name`` list from Ollama's ``/api/tags`` —
    same source ``llm_discovery.scan_ollama()`` already uses. Match is
    exact; aliases (e.g. ``qwen3.5:35b`` vs ``qwen3.5:35b-a3b-q4_K_M``)
    are intentionally treated as distinct so the user can see both.
    """
    have = set(local_tags)
    return [c for c in candidates if c.full_name not in have]


# ── Tag-shape parser ──────────────────────────────────────────────────────
#
# Three filters added 2026-04-30 (response to user rejecting 9 of 9 pulls
# the scanner proposed: qwen3:4b-thinking-q8_0, qwen3:4b-instruct-q8_0,
# qwen3.5:4b-q8_0 × 3 idle-cycle repeats each). The user's rejections
# were the right call — they already had qwen3.5:35b-a3b-q4_K_M which
# strictly dominates the 4B siblings on capability. The scanner just
# didn't know.
#
# All three filters parse a model name like
#   qwen3.5:35b-a3b-q4_K_M
# into structural parts:
#   family = "qwen3.5"      (everything before ":")
#   size_b = 35             (the "<digits>b" token, B-params; None when absent)
#   variants = ("a3b",)     (-a3b / -a10b / -instruct / -thinking / -coding etc.)
#   quant = "q4_K_M"        (None when no quant suffix)
#
# Two same-family models are "comparable" when they share the same family
# AND (when both have explicit sizes) one is larger than the other.

_SIZE_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)b\b", re.IGNORECASE)
_QUANT_RE = re.compile(
    r"-(?P<q>q\d_\w_\w|q\d_\d|q\d_K_M|q\d_K_S|q\d|fp16|bf16|mxfp\d+|nvfp\d+)$",
    re.IGNORECASE,
)


@dataclass
class _ParsedTag:
    full_name: str
    family: str
    size_b: float | None     # numeric param count in B (qwen3.5:35b-a3b → 35.0)
    quant: str | None        # "q4_K_M", "q8_0", "fp16" — None when absent
    variants: tuple[str, ...]  # other suffix tokens (a3b, instruct, thinking, ...)


def _parse_model_id(name: str) -> _ParsedTag:
    """Parse an Ollama tag (``qwen3.5:35b-a3b-q4_K_M`` or just ``qwen3.5:35b``)
    into structural parts. Defensive — unrecognized shapes parse to a tag
    with everything as ``variants`` so downstream filters degrade gracefully
    (they only block when they can match POSITIVELY, never when they're
    confused).
    """
    if not name or ":" not in name:
        return _ParsedTag(full_name=name, family=name or "", size_b=None,
                          quant=None, variants=())
    family, rest = name.split(":", 1)
    family = family.strip().lower()
    rest_lower = rest.lower()
    quant_match = _QUANT_RE.search(rest_lower)
    quant = quant_match.group("q").lower() if quant_match else None
    if quant_match:
        rest_lower = rest_lower[: quant_match.start()]
    # Size token: first "Nb" / "N.Mb" found in the remaining tag
    size = None
    size_match = _SIZE_RE.search(rest_lower)
    if size_match:
        try:
            size = float(size_match.group(1))
        except ValueError:
            size = None
        # Strip the size from the remaining variants-bag
        rest_lower = (
            rest_lower[: size_match.start()] + rest_lower[size_match.end():]
        )
    # Whatever's left, split on '-' and discard empties
    variants = tuple(
        v for v in re.split(r"[-_]+", rest_lower)
        if v and v != "b"
    )
    return _ParsedTag(
        full_name=name, family=family, size_b=size,
        quant=quant, variants=variants,
    )


# Quant ranking — higher = larger / more precise. Used to decide:
# "is the proposed quant 'bigger' than the installed one?". When yes
# AND the rest of the tag (family + size + variants) matches, we skip
# the proposal (user already has a leaner equivalent).
_QUANT_RANK = {
    None:    0,
    "q2_k":  1, "q2": 1,
    "q3_k":  2, "q3": 2, "q3_k_m": 2, "q3_k_s": 2,
    "q4_0":  3,
    "q4_k_s": 3, "q4_k_m": 4, "q4_k": 4,
    "q5_k_s": 5, "q5_k_m": 5, "q5_k": 5, "q5_0": 5,
    "q6_k":  6, "q6": 6,
    "q8_0":  7, "q8":  7,
    "fp16":  9, "bf16": 9,
    "mxfp8": 7, "nvfp4": 4,
}


def _quant_rank(label: str | None) -> int:
    if not label:
        return 0
    return _QUANT_RANK.get(label.lower(), 0)


# Family lineage — split a family name like "qwen3.5" into a base
# ("qwen") and a version (3.5). This is what lets the dominance
# filter recognise that "qwen3" is dominated by "qwen3.5" (same
# lineage, lower major.minor) without conflating unrelated families
# that happen to share a prefix.
#
# Examples:
#   qwen3.5     → ("qwen", 3.5)
#   qwen3       → ("qwen", 3.0)
#   llama3.1    → ("llama", 3.1)
#   llama4      → ("llama", 4.0)
#   gemma4      → ("gemma", 4.0)
#   deepseek-r1 → ("deepseek-r", 1.0)
#   codestral   → ("codestral", None)  — no version digits
#   glm-ocr     → ("glm-ocr", None)
_FAMILY_VERSION_RE = re.compile(r"^(.+?)([0-9]+(?:\.[0-9]+)?)$")


def _family_base_and_version(family: str) -> tuple[str, float | None]:
    """Split a family name into (base, version). Returns (family, None)
    when no trailing version digits are present."""
    if not family:
        return ("", None)
    m = _FAMILY_VERSION_RE.match(family)
    if not m:
        return (family, None)
    try:
        ver = float(m.group(2))
    except ValueError:
        return (family, None)
    base = m.group(1).rstrip("-_.")
    if not base:
        return (family, None)
    return (base, ver)


# ── Filter 1: same-family dominance ───────────────────────────────────────

def filter_dominated_by_installed(
    candidates: list[RegistryCandidate],
    local_tags: list[str],
) -> list[RegistryCandidate]:
    """Skip candidates dominated by something already installed locally.

    Three layered dominance rules. Rules 2 + 3 added 2026-04-30 after
    the user surfaced a second wave of governance rejections:
    qwen3.5:latest, qwen3:14b-q4_K_M, qwen3:8b-q4_K_M — all proposed
    while qwen3.5:35b-a3b-q4_K_M was already installed.

      Rule 1 — same family + smaller size:
        installed: qwen3.5:35b-a3b-q4_K_M
        proposed:  qwen3.5:4b-q8_0       → SKIP (strict downgrade)

      Rule 2 — same family + sizeless tag (`:latest`, `:instruct`):
        installed: qwen3.5:35b-a3b-q4_K_M
        proposed:  qwen3.5:latest        → SKIP (user pinned a specific
                                           variant; the generic alias is
                                           almost always the smallest
                                           default and not what they want)

      Rule 3 — cross-version-within-base:
        installed: qwen3.5:35b-a3b-q4_K_M  (base=qwen, ver=3.5)
        proposed:  qwen3:14b-q4_K_M        (base=qwen, ver=3.0) → SKIP
        proposed:  qwen3:8b-q4_K_M         (base=qwen, ver=3.0) → SKIP
        Newer model lineage always wins — if you've installed any
        qwen3.5 variant, you don't want qwen3 proposals.

    Kept regardless:
        installed: qwen3.5:35b-a3b-q4_K_M
        proposed:  llama3.1:70b           → different base
        proposed:  qwen3.5:122b-a10b      → larger same-family
        proposed:  qwen4:14b              → newer-version-than-installed
    """
    if not candidates:
        return candidates

    # Map: family → max installed size_b (Rule 1)
    installed_max_size_by_family: dict[str, float] = {}
    # Set of families that have ANY installed member (Rule 2)
    installed_families: set[str] = set()
    # Map: lineage base → max installed version (Rule 3)
    installed_max_version_by_base: dict[str, float] = {}

    for tag in local_tags:
        parsed = _parse_model_id(tag)
        if not parsed.family:
            continue
        installed_families.add(parsed.family)
        if parsed.size_b is not None:
            prev = installed_max_size_by_family.get(parsed.family, 0.0)
            if parsed.size_b > prev:
                installed_max_size_by_family[parsed.family] = parsed.size_b
        base, ver = _family_base_and_version(parsed.family)
        if ver is not None:
            prev_ver = installed_max_version_by_base.get(base, -1.0)
            if ver > prev_ver:
                installed_max_version_by_base[base] = ver

    out: list[RegistryCandidate] = []
    for c in candidates:
        parsed = _parse_model_id(c.full_name)

        # Rule 1: same family + strictly smaller explicit size
        installed_max = installed_max_size_by_family.get(parsed.family)
        if (installed_max is not None
                and parsed.size_b is not None
                and parsed.size_b < installed_max):
            logger.debug(
                "registry_scan: skip %s — %s:%sB already installed (rule 1: smaller sibling)",
                c.full_name, parsed.family, installed_max,
            )
            continue

        # Rule 2: candidate is sizeless AND family already has a member
        if parsed.size_b is None and parsed.family in installed_families:
            logger.debug(
                "registry_scan: skip %s — already-installed family with sizeless candidate (rule 2: :latest of pinned family)",
                c.full_name,
            )
            continue

        # Rule 3: cross-version within the same lineage base
        c_base, c_ver = _family_base_and_version(parsed.family)
        if c_ver is not None:
            installed_ver = installed_max_version_by_base.get(c_base)
            if installed_ver is not None and c_ver < installed_ver:
                logger.debug(
                    "registry_scan: skip %s — newer lineage installed (%s%s, rule 3: cross-version dominance)",
                    c.full_name, c_base, installed_ver,
                )
                continue

        out.append(c)
    return out


# ── Filter 2: quantization preference ─────────────────────────────────────

def filter_quant_dominated(
    candidates: list[RegistryCandidate],
    local_tags: list[str],
) -> list[RegistryCandidate]:
    """Skip a candidate when an INSTALLED model has the same family +
    size + variants but a leaner (lower-rank) quantization.

    Example dropped:
      installed: qwen3.5:35b-a3b-q4_K_M
      proposed:  qwen3.5:35b-a3b-q8_0    → SKIP — q8_0 doubles disk for marginal quality
      proposed:  qwen3.5:35b-a3b-fp16    → SKIP — even larger, same reason

    Example kept:
      installed: qwen3.5:35b-a3b-q8_0
      proposed:  qwen3.5:35b-a3b-q4_K_M  → KEEP — leaner alternative
      proposed:  qwen3.5:35b-a3b-q5_K_M  → KEEP — between, plausible
    """
    if not candidates:
        return candidates
    # Build a lookup of (family, size, variants_set) → max installed quant rank
    installed_max_rank: dict[tuple, int] = {}
    for tag in local_tags:
        p = _parse_model_id(tag)
        key = (p.family, p.size_b, frozenset(p.variants))
        rank = _quant_rank(p.quant)
        installed_max_rank[key] = max(installed_max_rank.get(key, -1), rank)

    out: list[RegistryCandidate] = []
    for c in candidates:
        p = _parse_model_id(c.full_name)
        key = (p.family, p.size_b, frozenset(p.variants))
        installed = installed_max_rank.get(key)
        if installed is None:
            out.append(c)
            continue
        cand_rank = _quant_rank(p.quant)
        # Skip when candidate quant is HIGHER (bigger/more precise) than
        # the leanest installed equivalent. Equal-rank means we already
        # have it (caught by diff_against_local); lower-rank is a
        # cheaper alternative the user might want.
        if cand_rank > installed:
            logger.debug(
                "registry_scan: skipping %s — same base already installed at lower-cost quant (rank %d ≤ %d)",
                c.full_name, installed, cand_rank,
            )
            continue
        out.append(c)
    return out


# ── Filter 3: rejection learning ──────────────────────────────────────────

# How long after a rejection should we suppress the same model_id from
# proposals? 30 days = "user clearly doesn't want this; don't keep
# nagging across idle cycles."
_REJECTION_SUPPRESSION_DAYS = 30


def get_recently_rejected_models(window_days: int = _REJECTION_SUPPRESSION_DAYS) -> set[str]:
    """Return the set of model full_names rejected via governance in the
    last ``window_days``.

    Reads ``control_plane.governance_requests`` directly. Best-effort —
    DB unavailability returns an empty set so the scanner still
    proposes (loud over silent).
    """
    try:
        from app.control_plane.db import execute
        rows = execute(
            """SELECT detail_json
                 FROM control_plane.governance_requests
                WHERE request_type = 'local_model_pull'
                  AND status = 'rejected'
                  AND reviewed_at > NOW() - (%s || ' days')::interval""",
            (str(int(window_days)),),
            fetch=True,
        ) or []
        out: set[str] = set()
        for r in rows:
            detail = r.get("detail_json") or {}
            if isinstance(detail, str):
                try:
                    import json as _j
                    detail = _j.loads(detail)
                except Exception:
                    detail = {}
            model = detail.get("model") or detail.get("model_id") or ""
            if model:
                out.add(model)
        return out
    except Exception as exc:
        logger.debug("registry_scan: rejection lookup failed: %s", exc)
        return set()


def filter_recently_rejected(
    candidates: list[RegistryCandidate],
    *,
    window_days: int = _REJECTION_SUPPRESSION_DAYS,
    rejected_set: set[str] | None = None,
) -> list[RegistryCandidate]:
    """Skip candidates whose full_name was rejected in the recent window.

    ``rejected_set`` is injectable for tests; in production callers pass
    None and we fetch from the DB.
    """
    if rejected_set is None:
        rejected_set = get_recently_rejected_models(window_days)
    if not rejected_set:
        return candidates
    out: list[RegistryCandidate] = []
    for c in candidates:
        if c.full_name in rejected_set:
            logger.debug(
                "registry_scan: skipping %s — rejected via governance in last %d days",
                c.full_name, window_days,
            )
            continue
        out.append(c)
    return out


def _is_enabled() -> bool:
    val = os.getenv("LLM_REGISTRY_SCAN_ENABLED", "true").strip().lower()
    return val not in ("0", "false", "no", "off")

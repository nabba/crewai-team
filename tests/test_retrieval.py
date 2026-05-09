"""Tests for the shared retrieval orchestrator (Phase 0)."""

import os
import pytest
from datetime import datetime, timezone, timedelta

_LOW_MEM = os.environ.get("LOW_MEM_TESTS", "1") == "1"


# ── Temporal decay tests ────────────────────────────────────────────────────

def test_temporal_decay_basic():
    """Newer results should score higher than older ones."""
    from app.retrieval.temporal import apply_temporal_decay

    now = datetime.now(timezone.utc)
    results = [
        {"text": "old", "score": 0.8, "metadata": {"ingested_at": (now - timedelta(days=14)).isoformat()}},
        {"text": "new", "score": 0.8, "metadata": {"ingested_at": (now - timedelta(hours=1)).isoformat()}},
    ]
    out = apply_temporal_decay(results, now=now)
    assert len(out) == 2
    # New should have higher blended score.
    assert out[0]["text"] == "new"
    assert out[0]["blended_score"] > out[1]["blended_score"]


def test_temporal_decay_no_timestamps():
    """Results without timestamps get neutral temporal score."""
    from app.retrieval.temporal import apply_temporal_decay

    results = [
        {"text": "a", "score": 0.7, "metadata": {}},
        {"text": "b", "score": 0.9, "metadata": {}},
    ]
    out = apply_temporal_decay(results)
    # Without temporal signal, order follows semantic score.
    assert out[0]["text"] == "b"


def test_temporal_decay_zero_weight():
    """Zero weight should produce identical blended and semantic scores."""
    from app.retrieval.temporal import apply_temporal_decay

    now = datetime.now(timezone.utc)
    results = [
        {"text": "a", "score": 0.5, "metadata": {"ingested_at": now.isoformat()}},
    ]
    out = apply_temporal_decay(results, weight=0.0, now=now)
    assert out[0]["blended_score"] == 0.5


# ── Decomposer tests ───────────────────────────────────────────────────────

def test_decomposer_short_query():
    """Short queries bypass decomposition entirely."""
    from app.retrieval.decomposer import decompose_query

    result = decompose_query("hello", min_length=50)
    assert result == ["hello"]


def test_decomposer_disabled():
    """When disabled, always returns original query."""
    from app.retrieval import config as cfg
    original = cfg.DECOMPOSITION_ENABLED
    try:
        cfg.DECOMPOSITION_ENABLED = False
        from app.retrieval.decomposer import decompose_query
        result = decompose_query("a" * 200, min_length=10)
        assert result == ["a" * 200]
    finally:
        cfg.DECOMPOSITION_ENABLED = original


# ── RetrievalResult tests ──────────────────────────────────────────────────

def test_retrieval_result_dataclass():
    from app.retrieval.orchestrator import RetrievalResult
    r = RetrievalResult(text="hello", score=0.9, metadata={"source": "test"}, provenance={"collection": "c"})
    assert r.text == "hello"
    assert r.score == 0.9
    assert r.provenance["collection"] == "c"


# ── Config tests ───────────────────────────────────────────────────────────

def test_retrieval_config_defaults():
    from app.retrieval.config import RetrievalConfig
    cfg = RetrievalConfig()
    assert cfg.rerank_enabled is True
    assert cfg.rerank_top_k_input == 20
    assert cfg.temporal_enabled is False
    assert cfg.decomposition_enabled is True


# ── Reranker graceful degradation ──────────────────────────────────────────

def test_reranker_empty_input():
    from app.retrieval.reranker import rerank
    assert rerank("query", []) == []


def test_reranker_passthrough_on_missing_model():
    """If model fails, results returned unchanged."""
    from app.retrieval import reranker
    # Temporarily force model to fail.
    original_failed = reranker._model_failed
    original_model = reranker._model
    reranker._model_failed = True
    reranker._model = None
    try:
        docs = [{"text": "hello", "score": 0.5}, {"text": "world", "score": 0.8}]
        result = reranker.rerank("query", docs, top_k=1)
        assert len(result) == 1
        assert "rerank_score" in result[0]
    finally:
        reranker._model_failed = original_failed
        reranker._model = original_model


# ── OpenRouter rerank backend ──────────────────────────────────────────────

class _FakeResponse:
    """Stand-in for httpx.Response with the bits the reranker reads."""
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or ""

    def json(self):
        return self._payload


def _patch_openrouter_setup(monkeypatch, api_key: str = "sk-test"):
    """Stub out the OpenRouter API key plumbing so tests don't touch real config."""
    class _FakeSettings:
        class openrouter_api_key:
            @staticmethod
            def get_secret_value():
                return api_key

    monkeypatch.setattr("app.config.get_settings", lambda: _FakeSettings)


def test_openrouter_rerank_happy_path(monkeypatch):
    """Successful /rerank call: API indices map back to original docs, scores attached."""
    import httpx
    from app.retrieval import reranker

    _patch_openrouter_setup(monkeypatch)

    docs = [
        {"text": "alpha", "score": 0.5, "id": "a"},
        {"text": "bravo", "score": 0.4, "id": "b"},
        {"text": "charlie", "score": 0.3, "id": "c"},
    ]
    fake_payload = {
        "results": [
            {"index": 2, "relevance_score": 0.92},
            {"index": 0, "relevance_score": 0.81},
        ],
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(200, fake_payload))

    result = reranker._openrouter_rerank(
        "q", docs, top_k=2, text_key="text", model="cohere/rerank-multilingual-v3.0",
    )
    assert result is not None
    assert [d["id"] for d in result] == ["c", "a"]
    assert result[0]["rerank_score"] == 0.92
    assert result[1]["rerank_score"] == 0.81


def test_openrouter_rerank_http_error_returns_none(monkeypatch):
    """5xx / 4xx responses → None so caller falls back to local."""
    import httpx
    from app.retrieval import reranker

    _patch_openrouter_setup(monkeypatch)
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(503, {}, "upstream down"))

    result = reranker._openrouter_rerank(
        "q", [{"text": "a"}], top_k=1, text_key="text", model="cohere/rerank-multilingual-v3.0",
    )
    assert result is None


def test_openrouter_rerank_network_exception_returns_none(monkeypatch):
    """A raised exception from httpx.post must not propagate."""
    import httpx
    from app.retrieval import reranker

    _patch_openrouter_setup(monkeypatch)

    def _boom(*a, **kw):
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(httpx, "post", _boom)
    result = reranker._openrouter_rerank(
        "q", [{"text": "a"}], top_k=1, text_key="text", model="cohere/rerank-multilingual-v3.0",
    )
    assert result is None


def test_openrouter_rerank_missing_api_key(monkeypatch):
    """Empty API key → None without making any HTTP call."""
    from app.retrieval import reranker
    _patch_openrouter_setup(monkeypatch, api_key="")

    result = reranker._openrouter_rerank(
        "q", [{"text": "a"}], top_k=1, text_key="text", model="cohere/rerank-multilingual-v3.0",
    )
    assert result is None


def test_openrouter_rerank_malformed_response_graceful(monkeypatch):
    """Missing 'results' key → empty list, not crash."""
    import httpx
    from app.retrieval import reranker
    _patch_openrouter_setup(monkeypatch)

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(200, {"unexpected": "shape"}))

    result = reranker._openrouter_rerank(
        "q", [{"text": "a"}], top_k=1, text_key="text", model="cohere/rerank-multilingual-v3.0",
    )
    assert result == []


def test_openrouter_rerank_invalid_index_skipped(monkeypatch):
    """Out-of-range or non-int indices are silently skipped, others returned."""
    import httpx
    from app.retrieval import reranker
    _patch_openrouter_setup(monkeypatch)

    docs = [{"text": "a", "id": "a"}, {"text": "b", "id": "b"}]
    fake_payload = {
        "results": [
            {"index": 5, "relevance_score": 0.9},   # out of range — skip
            {"index": "x", "relevance_score": 0.8},  # not int — skip
            {"index": 1, "relevance_score": 0.7},   # valid
        ],
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(200, fake_payload))

    result = reranker._openrouter_rerank(
        "q", docs, top_k=5, text_key="text", model="cohere/rerank-multilingual-v3.0",
    )
    assert [d["id"] for d in result] == ["b"]


def test_rerank_dispatch_falls_back_to_local_on_openrouter_failure(monkeypatch):
    """End-to-end dispatch: backend=openrouter + HTTP error → local path runs."""
    import httpx
    from app.retrieval import reranker
    from app.retrieval import config as cfg

    _patch_openrouter_setup(monkeypatch)
    monkeypatch.setattr(cfg, "RERANKER_BACKEND", "openrouter")
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(500, {}, "boom"))

    # Force local model to also be unavailable so we hit the passthrough
    # branch deterministically — the assertion is that we don't crash.
    original_failed = reranker._model_failed
    original_model = reranker._model
    reranker._model_failed = True
    reranker._model = None
    try:
        docs = [{"text": "alpha", "score": 0.5}, {"text": "bravo", "score": 0.6}]
        result = reranker.rerank("q", docs, top_k=2)
        assert len(result) == 2
        assert all("rerank_score" in d for d in result)
    finally:
        reranker._model_failed = original_failed
        reranker._model = original_model

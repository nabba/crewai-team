"""Smoke tests for app.tool_runtime.measure (Phase 1c gate harness).

These tests don't validate exact token counts (those depend on tool
descriptions which evolve); they validate the *shape* of the
measurement and the modeling logic — so a future change to a tool
description doesn't accidentally make the gate report meaningless.

Critical invariants checked:
  * Cache-line independence is modeled correctly (system caches even
    when tools mutate).
  * Stock prefix is materially larger than LoadableAgent's initial
    prefix (the whole point of the architecture).
  * Verdict is GO iff ratio ≤ threshold.
"""
from __future__ import annotations


# ── Cache-line modeling ──────────────────────────────────────────────


class TestCacheModel:

    def test_iter_1_writes_both_cache_lines(self):
        from app.tool_runtime.measure import model_iterations
        out = model_iterations(
            system_tokens=1000,
            tools_tokens_per_iter=[500] * 5,
            new_turn_tokens_per_iter=200,
            num_iterations=5,
        )
        # iter 1: write system + tools, no reads
        assert out[0].cache_write_tokens == 1500
        assert out[0].cache_read_tokens == 0
        assert out[0].fresh_tokens == 0  # no prior turns yet

    def test_stable_tools_keep_cache_warm(self):
        from app.tool_runtime.measure import model_iterations
        out = model_iterations(
            system_tokens=1000,
            tools_tokens_per_iter=[500] * 5,
            new_turn_tokens_per_iter=200,
            num_iterations=5,
        )
        # iters 2-5: pure cache reads on the static prefix
        for i in (1, 2, 3, 4):
            assert out[i].cache_write_tokens == 0, (
                f"iter {i+1} should not rewrite the cache"
            )

    def test_tools_change_writes_only_tools_line(self):
        """Cache-line independence: tools mutation re-writes only the
        tools cache line, not system. This is the load-bearing
        property for LoadableAgent's win."""
        from app.tool_runtime.measure import model_iterations
        out = model_iterations(
            system_tokens=1000,
            tools_tokens_per_iter=[500, 700, 700, 700, 700],
            new_turn_tokens_per_iter=200,
            num_iterations=5,
        )
        # iter 2: tools changed (500 → 700). Should write 700 tokens
        # (just tools), and READ system from cache.
        iter2 = out[1]
        assert iter2.cache_write_tokens == 700, (
            "iter 2 should re-write only the tools cache line"
        )
        # The 1000 system tokens MUST appear in cache_read, NOT in
        # cache_write — that's the critical model property.
        assert iter2.cache_read_tokens >= 1000


# ── Real-agent prompt-shape extraction ───────────────────────────────


class TestRealAgentShape:

    def test_stock_coder_has_large_tools_block(self):
        """The stock coder really does ship a big tool prompt."""
        from app.agents.coder import create_coder
        from app.tool_runtime.measure import extract_prompt_shape
        agent = create_coder()
        shape = extract_prompt_shape(agent, name="stock_coder")
        # Phase 0 measurements showed ~9k tools_description tokens on
        # the stock coder. We give a wide band to allow for tool-list
        # evolution.
        assert shape.tools_description_tokens > 5000, (
            f"Stock coder has only {shape.tools_description_tokens} "
            "tokens of tools_description — has the toolset shrunk? "
            "(If yes, recalibrate the band.)"
        )

    def test_loadable_initial_is_an_order_of_magnitude_smaller(self):
        from app.agents.coder import create_coder
        from app.tool_runtime.measure import (
            _build_loadable_and_extract, extract_prompt_shape,
        )
        stock = extract_prompt_shape(create_coder(), name="stock")
        loadable = _build_loadable_and_extract(loaded_names=[])
        # The whole point: loadable initial prefix is much smaller.
        assert loadable.tools_description_tokens < stock.tools_description_tokens / 3, (
            "LoadableAgent initial prefix should be ≤1/3 of stock — "
            "if not, the architecture's premise is broken."
        )


# ── End-to-end gate verdict ──────────────────────────────────────────


class TestGateVerdict:

    def test_default_scenario_passes_gate(self):
        """The Phase 1c headline: 5 iterations, 2 mid-task loads,
        ratio ≤ 50%."""
        from app.tool_runtime.measure import compare_stock_vs_loadable
        report = compare_stock_vs_loadable()
        assert report["verdict"] == "GO", (
            f"Phase 1c gate FAILED: {report['ratio']:.1%} of stock "
            f"(threshold {report['threshold']:.0%}). "
            "Either the toolset got dramatically smaller, the cache "
            "model is wrong, or the architecture's premise is broken."
        )
        assert report["ratio"] <= report["threshold"]
        assert report["ratio"] > 0  # sanity: not zero

    def test_no_loads_scenario_is_dramatic_win(self):
        """Without mid-task loads, LoadableAgent should be a much
        bigger win — no cache resets at all."""
        from app.tool_runtime.measure import compare_stock_vs_loadable
        report = compare_stock_vs_loadable(num_mid_task_loads=0)
        assert report["ratio"] < 0.30, (
            f"No-load ratio {report['ratio']:.1%} should be << 30% "
            "since LoadableAgent's small prefix dominates."
        )

    def test_more_iterations_improves_ratio(self):
        """LoadableAgent's iter-2/3 cache-reset costs amortize over
        more iterations — longer tasks favor LoadableAgent more."""
        from app.tool_runtime.measure import compare_stock_vs_loadable
        short = compare_stock_vs_loadable(num_iterations=5)
        long = compare_stock_vs_loadable(num_iterations=10)
        assert long["ratio"] < short["ratio"], (
            "Longer task should have LOWER (better) ratio for "
            "LoadableAgent; if not, modeling has a bug."
        )

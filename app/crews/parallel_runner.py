"""
parallel_runner.py — Shared thread pool for running crews and sub-agents in parallel.

Provides run_parallel() which takes a list of callables, executes them
concurrently, and returns results with error isolation (one failure
doesn't kill the others).

Ollama concurrency control:
  Each crew/sub-agent makes multiple LLM calls (tool loops), so running
  N crews in parallel can exceed Ollama's OLLAMA_NUM_PARALLEL capacity.
  A semaphore limits how many crews hit Ollama simultaneously, queuing
  the rest at the application level rather than timing out in Ollama.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Single shared pool for the entire process — caps total concurrency
_pool = ThreadPoolExecutor(
    max_workers=settings.thread_pool_size,
    thread_name_prefix="crew-parallel",
)

# Ollama concurrency gate — limit how many crews run LLM calls at once.
# With OLLAMA_NUM_PARALLEL=4 and each crew making 3-8 LLM calls,
# allowing 2 concurrent crews keeps total in-flight requests manageable.
_ollama_concurrency = getattr(settings, "ollama_max_concurrent_crews", 2)
_ollama_semaphore = threading.Semaphore(_ollama_concurrency)


@dataclass
class ParallelResult:
    """Result from a single parallel task."""
    label: str
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None


def run_parallel(
    tasks: list[tuple[str, Callable[[], str]]],
    timeout_seconds: int = 600,
) -> list[ParallelResult]:
    """
    Run multiple callables in parallel and collect results.

    Args:
        tasks: List of (label, callable) tuples.  Each callable should
               return a string result.
        timeout_seconds: Max time to wait for all tasks (default 10 min).

    Returns:
        List of ParallelResult in the same order as input tasks.
    """
    if not tasks:
        return []

    def _throttled(fn):
        """Wrap callable with semaphore so only N crews hit Ollama at once."""
        with _ollama_semaphore:
            return fn()

    futures = {}
    for label, fn in tasks:
        future = _pool.submit(_throttled, fn)
        futures[future] = label

    results_map: dict[str, ParallelResult] = {}
    try:
        for future in as_completed(futures, timeout=timeout_seconds):
            label = futures[future]
            try:
                result = future.result()
                results_map[label] = ParallelResult(
                    label=label, success=True, result=str(result),
                )
                logger.info(f"Parallel task '{label}' completed successfully")
            except Exception as exc:
                logger.error(f"Parallel task '{label}' failed: {exc}")
                results_map[label] = ParallelResult(
                    label=label, success=False, error=str(exc)[:300],
                )
    except TimeoutError:
        logger.error("run_parallel: timed out waiting for tasks")

    # Return in original order; mark missing (timed-out) tasks
    ordered = []
    for label, _ in tasks:
        if label in results_map:
            ordered.append(results_map[label])
        else:
            ordered.append(ParallelResult(
                label=label, success=False, error="Timed out",
            ))
    return ordered

# Resilient API Orchestration Pattern

## Problem
API failures, rate limits (429), and timeouts cause agent crashes or incomplete data retrieval during lead enrichment tasks.

## Solution: The Resilient Wrapper Pattern
1. **Exponential Backoff**: Implement a decorator for all API calls that retries on 429/503 errors with increasing delays (1s, 2s, 4s, 8s).
2. **State Persistence**: Save progress to a local JSON file after every 10 successful requests to allow resume-from-failure.
3. **Concurrency Control**: Use `asyncio.Semaphore` to limit concurrent requests to a specific domain to avoid IP bans.

## Implementation Steps
- Define a `BaseAPIClient` class with built-in retry logic.
- Use a 'Job Queue' pattern for large datasets instead of a single loop.
- Log all quota-related errors to a `quota_log.txt` for the System Improvement Analyst to review.
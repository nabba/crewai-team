# Asynchronous API Orchestration

## Problem
Sequential API calls in the coding crew lead to inefficient execution times and increased risk of timeouts during large-scale data retrieval (e.g., regional PSP enrichment).

## Solution
Implement a standardized `AsyncClient` pattern using `httpx` and `asyncio`.

### Implementation Guidelines
1. **Concurrency Control**: Use `asyncio.Semaphore(limit)` to prevent overwhelming target APIs.
2. **Resilience**: Implement a decorator-based exponential backoff for 429 (Too Many Requests) and 5xx errors.
3. **Batching**: Group requests into chunks to optimize network overhead.

### Code Template
```python
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_with_retry(client, url, params=None):
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()

async def orchestrated_fetch(urls):
    semaphore = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        async def wrapped_fetch(url):
            async with semaphore:
                return await fetch_with_retry(client, url)
        return await asyncio.gather(*(wrapped_fetch(u) for u in urls))
```
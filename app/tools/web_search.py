from crewai.tools import tool
import requests
from app.config import get_brave_api_key

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Reusable HTTP session for Brave API calls
_session = requests.Session()
_session.headers.update({
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
})


def search_brave(query: str, count: int = 5) -> list[dict]:
    """Raw Brave Search — returns list of {title, url, description} dicts.

    Used by ATLAS modules (api_scout, learning_planner) for programmatic
    search. The @tool version below wraps this for CrewAI agent use.
    """
    params = {"q": query, "count": count}
    try:
        response = _session.get(
            BRAVE_SEARCH_URL,
            headers={"X-Subscription-Token": get_brave_api_key()},
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            }
            for item in data.get("web", {}).get("results", [])[:count]
        ]
    except Exception:
        return []


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the web using Brave Search API.
    Returns top 5 results as title + URL + snippet.
    """
    params = {"q": query, "count": 5}

    try:
        response = _session.get(
            BRAVE_SEARCH_URL,
            headers={"X-Subscription-Token": get_brave_api_key()},
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:5]:
            title = item.get("title", "No title")
            url = item.get("url", "")
            snippet = item.get("description", "No description")
            results.append(f"**{title}**\n{url}\n{snippet}\n")

        return "\n".join(results) if results else "No results found."
    except Exception:
        return "Search error: unable to reach search API."

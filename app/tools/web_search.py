from crewai.tools import tool
import requests
from app.config import get_brave_api_key

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the web using Brave Search API.
    Returns top 5 results as title + URL + snippet.
    """
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": get_brave_api_key(),
    }
    params = {"q": query, "count": 5}

    try:
        response = requests.get(
            BRAVE_SEARCH_URL, headers=headers, params=params, timeout=10
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

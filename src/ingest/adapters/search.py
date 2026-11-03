"""Web search adapter implementing SearchPort for project research."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ingest.models import IngestError, SearchResult

logger = logging.getLogger(__name__)


class SearchAdapter:
    """Search adapter supporting Tavily and Brave Search with strict grounding."""

    def __init__(
        self,
        api_key: str | None = None,
        brave_api_key: str | None = None,
        provider: str = "tavily",
    ) -> None:
        self.api_key = api_key or os.getenv("SEARCH_API_KEY") or os.getenv("TAVILY_API_KEY", "")
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY", "")
        self.provider = provider

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Perform a web search returning top-k SearchResult items.

        Raises IngestError('search', ...) on missing API key or remote failure.
        Returns an empty list [] if no relevant search results are found.
        """
        tavily_key = (self.api_key or "").strip()
        is_tavily_valid = bool(
            tavily_key
            and not tavily_key.startswith("tvly-...")
            and tavily_key not in ("mock", "dummy")
        )

        brave_key = (self.brave_api_key or "").strip()
        is_brave_valid = bool(
            brave_key
            and not brave_key.startswith("BSA...")
            and brave_key not in ("mock", "dummy")
        )

        if not is_tavily_valid and not is_brave_valid:
            raise IngestError("search", "Web search API key (Tavily or Brave) is not configured")

        try:
            if is_tavily_valid:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    res = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": tavily_key,
                            "query": query,
                            "max_results": k,
                            "search_depth": "basic",
                        },
                    )
                    res.raise_for_status()
                    data = res.json()
                    results: list[SearchResult] = []
                    for item in data.get("results", []):
                        results.append(
                            SearchResult(
                                title=item.get("title", "Untitled"),
                                url=item.get("url", ""),
                                snippet=item.get("content", ""),
                            )
                        )
                    return results[:k]

            if is_brave_valid:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    res = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers={
                            "X-Subscription-Token": brave_key,
                            "Accept": "application/json",
                        },
                        params={"q": query, "count": k},
                    )
                    res.raise_for_status()
                    data = res.json()
                    results = []
                    web_results = data.get("web", {}).get("results", [])
                    for item in web_results:
                        results.append(
                            SearchResult(
                                title=item.get("title", "Untitled"),
                                url=item.get("url", ""),
                                snippet=item.get("description", ""),
                            )
                        )
                    return results[:k]

            return []
        except IngestError:
            raise
        except Exception as e:
            logger.error("Web search API call failed for query '%s': %s", query, e, exc_info=True)
            raise IngestError("search", e) from e


# Alias for backward compatibility
TavilySearch = SearchAdapter


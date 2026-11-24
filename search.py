"""
Live web search, used by the web agent -- either because the router sent
a live/time-sensitive question straight there, or as the fallback after
the knowledge base couldn't answer confidently.

Uses the free DuckDuckGo search via the `ddgs` package. For a real deployment, swap
this for Tavily or SerpAPI (both have LangChain tool wrappers) for more
reliable results.
"""


def web_search(query: str, max_results: int = 5) -> str:
    """Returns the search result text, or "" if search is unavailable --
    callers decide how to tell the user, rather than us passing an error
    message off as if it were a search result."""
    try:
        # `ddgs` is the maintained successor of `duckduckgo_search`, which
        # LangChain's DuckDuckGoSearchRun still imports.
        from ddgs import DDGS
    except Exception as exc:
        print(f"[web search unavailable: {exc}]")
        return ""

    # Free scraped search engines rate-limit shared cloud IPs often, so
    # fall through a few backends before giving up.
    for backend in ("auto", "bing", "google"):
        try:
            results = DDGS().text(query, max_results=max_results, backend=backend) or []
        except Exception as exc:
            print(f"[web search via {backend} failed: {exc}]")
            continue
        text = "\n\n".join(
            f"{r.get('title', '')}: {r.get('body', '')}".strip(": ") for r in results
        )
        if text.strip():
            return text
    return ""

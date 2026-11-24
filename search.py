"""
Live web search, used by the web agent -- either because the router sent
a live/time-sensitive question straight there, or as the fallback after
the knowledge base couldn't answer confidently.

Uses LangChain's free DuckDuckGo search tool. For a real deployment, swap
this for Tavily or SerpAPI (both have LangChain tool wrappers) for more
reliable results.
"""


def web_search(query: str) -> str:
    """Returns the search result text, or "" if search is unavailable --
    callers decide how to tell the user, rather than us passing an error
    message off as if it were a search result."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)
    except Exception as exc:
        print(f"[web search unavailable: {exc}]")
        return ""

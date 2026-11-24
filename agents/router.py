"""Router agent: decides which specialist should handle the question.

- knowledge_base: answerable from the ingested documents (the default)
- web: needs live / time-sensitive public information
- direct: greetings and small talk, no lookup needed

Asks the LLM first; if the LLM gives no parseable answer (always the
case for the mock backend), falls back to simple rules.
"""
import re
from agents.state import AgentState, step

ROUTES = ("knowledge_base", "web", "direct")

ROUTER_PROMPT = """You route questions for an assistant backed by an internal knowledge base containing: {topics}.

Choose exactly one route:
- knowledge_base: the question could be answered from those internal documents.
- web: the question needs live or general public information (news, current events, prices, weather) clearly outside the internal documents.
- direct: a greeting, thanks or small talk that needs no information lookup.
When unsure, choose knowledge_base.

Reply in exactly this format:
route: <knowledge_base|web|direct>
reason: <one short sentence>

Question: {query}"""

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|bye)\b[\w\s!.,?]{0,20}$",
    re.I,
)
_LIVE_INFO = re.compile(
    r"\b(today|tonight|tomorrow|latest|current(ly)?|news|weather|price|stock|"
    r"right now|this (week|month|year))\b",
    re.I,
)


def _rule_based_route(query: str) -> tuple[str, str]:
    if _GREETING.match(query):
        return "direct", "greeting or small talk, no lookup needed"
    if _LIVE_INFO.search(query):
        return "web", "asks for live or time-sensitive information"
    return "knowledge_base", "try the internal knowledge base first"


def _parse(raw: str) -> tuple[str | None, str]:
    route, reason = None, ""
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(" *-").lower(), value.strip(" *`")
        if key == "route" and value.lower() in ROUTES:
            route = value.lower()
        elif key == "reason":
            reason = value
    return route, reason


def make_router_node(llm, sources: list[str]):
    topics = ", ".join(sources) or "internal documents"

    def router_node(state: AgentState) -> AgentState:
        query = state["query"]
        route, reason = _parse(llm.complete(ROUTER_PROMPT.format(topics=topics, query=query)))
        how = "LLM"
        if route is None:
            route, reason = _rule_based_route(query)
            how = "rules"
        return {
            "route": route,
            "route_reason": reason,
            "search_query": query,
            "attempts": 0,
            "trace": step("router", f"-> {route} ({how}): {reason}"),
        }

    return router_node


def route_from_router(state: AgentState) -> str:
    return state["route"]

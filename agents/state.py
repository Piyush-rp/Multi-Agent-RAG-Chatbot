"""Shared LangGraph state passed between all agents."""
import operator
from typing import Annotated, List, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    route: str              # knowledge_base | direct
    route_reason: str
    search_query: str       # query sent to retrieval (rewritten on retries)
    attempts: int           # knowledge-base retrieval attempts so far
    retrieved: List[dict]
    context: str
    answers: List[str]
    final_answer: str
    confidence: float
    verdict: str            # confident | uncertain
    verdict_reason: str
    gap_id: int
    evidence: List[dict]    # supporting excerpts picked by the context agent
    # Each agent appends what it did; the reducer concatenates steps.
    trace: Annotated[List[dict], operator.add]


def step(agent: str, detail: str) -> list[dict]:
    return [{"agent": agent, "detail": detail}]

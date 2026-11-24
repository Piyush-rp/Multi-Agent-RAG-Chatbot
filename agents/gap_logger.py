"""Gap-logger agent: records a question the knowledge base couldn't
answer confidently, so a human can see what the documents are missing,
and tells the user plainly instead of passing off a guess as an answer."""
from agents.state import AgentState, step
from gap_tracker import log_gap


def log_gap_node(state: AgentState) -> AgentState:
    gap_id = log_gap(
        query=state["query"],
        confidence=state["confidence"],
        best_answer=state["final_answer"],
    )
    answer = (
        "I couldn't find a reliable answer to that in the documents in the "
        "knowledge base, so I won't guess. "
        f"(Logged as knowledge gap #{gap_id}: {state.get('verdict_reason', 'low confidence')}.)"
    )
    return {
        "gap_id": gap_id,
        "final_answer": answer,
        "trace": step("gap_logger", f"logged knowledge gap #{gap_id}"),
    }

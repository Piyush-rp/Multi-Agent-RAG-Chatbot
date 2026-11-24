"""Gap-logger agent: records a question the knowledge base couldn't
answer confidently, so a human can see what the documents are missing."""
from agents.state import AgentState, step
from gap_tracker import log_gap


def log_gap_node(state: AgentState) -> AgentState:
    gap_id = log_gap(
        query=state["query"],
        confidence=state["confidence"],
        best_answer=state["final_answer"],
    )
    return {"gap_id": gap_id, "trace": step("gap_logger", f"logged knowledge gap #{gap_id}")}

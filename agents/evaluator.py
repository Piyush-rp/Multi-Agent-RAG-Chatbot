"""Evaluator agent: judges whether the knowledge-base answer can be
trusted, and decides what happens next (answer / retry / give up)."""
from agents.state import AgentState, step
from confidence import score_confidence


def evaluate_node(state: AgentState) -> AgentState:
    confidence, verdict, reason = score_confidence(state["answers"], state["context"])
    return {
        "confidence": confidence,
        "verdict": verdict,
        "verdict_reason": reason,
        "trace": step("evaluator", f"{verdict} (score {confidence:.2f}): {reason}"),
    }


def make_route_after_evaluation(max_retries: int):
    def route_after_evaluation(state: AgentState) -> str:
        if state["verdict"] == "confident":
            return "confident"
        if state.get("attempts", 0) <= max_retries:
            return "retry"
        return "give_up"

    return route_after_evaluation

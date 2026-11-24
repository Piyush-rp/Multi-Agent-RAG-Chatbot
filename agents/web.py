"""Web agent: answers from a live web search. Reached two ways:
- directly from the router, for live / time-sensitive questions
- as the fallback after the knowledge base gave up (a logged gap), in
  which case the answer is clearly flagged as unverified."""
import gap_tracker
from agents.state import AgentState, step
from search import web_search


def make_web_node(llm):
    def web_node(state: AgentState) -> AgentState:
        query = state["query"]
        is_fallback = state.get("gap_id") is not None
        results = web_search(query)

        if not results:
            if is_fallback:
                answer = (
                    f"{state['final_answer']}\n\n"
                    f"[Unverified: the knowledge base wasn't confident "
                    f"({state['verdict_reason']}) and live web search is "
                    f"unavailable. Logged as knowledge gap #{state['gap_id']}.]"
                )
            else:
                answer = "Live web search is unavailable right now, so I can't answer this reliably."
            return {
                "final_answer": answer,
                "supplemented": False,
                "trace": step("web", "search unavailable"),
            }

        web_answer = llm.generate(query, results)
        if is_fallback:
            gap_tracker.set_supplemented_answer(state["gap_id"], web_answer)
            answer = (
                f"{web_answer}\n\n"
                f"[Unverified: the knowledge base wasn't confident about this "
                f"({state['verdict_reason']}), so this answer comes from a live "
                f"web search. Logged as knowledge gap #{state['gap_id']}.]"
            )
        else:
            answer = f"{web_answer}\n\n[Source: live web search]"
        return {
            "final_answer": answer,
            "web_results": results,
            "supplemented": True,
            "trace": step("web", "answered from live web search"),
        }

    return web_node

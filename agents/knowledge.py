"""Knowledge-base agent: hybrid retrieval, then N grounded answer samples
for the evaluator to compare."""
from agents.state import AgentState, step
from retrieval import HybridRetriever


def _label(chunk: dict) -> str:
    """'[file.pdf, p. 12] ' prefix so the generator can cite its sources."""
    page = chunk.get("page")
    return f"[{chunk['source']}{f', p. {page}' if page else ''}] "


def make_retrieve_node(retriever: HybridRetriever):
    def retrieve_node(state: AgentState) -> AgentState:
        query = state.get("search_query") or state["query"]
        results = retriever.search(query)
        attempts = state.get("attempts", 0) + 1
        found_by = sorted({name for r in results for name in r["scores"]})
        detail = f"attempt {attempts}: '{query}' -> {len(results)} chunk(s)"
        if results:
            detail += f" (matched by: {', '.join(found_by)})"
        return {
            "retrieved": results,
            "context": "\n\n".join(_label(r) + r["text"] for r in results),
            "attempts": attempts,
            "trace": step("retriever", detail),
        }

    return retrieve_node


def make_generate_node(llm, n_samples: int):
    def generate_node(state: AgentState) -> AgentState:
        # Answer the user's original question, even if retrieval used a
        # rewritten search query.
        answers = [llm.generate(state["query"], state["context"]) for _ in range(n_samples)]
        return {
            "answers": answers,
            "final_answer": answers[0],
            "trace": step("generator", f"sampled {len(answers)} answer(s)"),
        }

    return generate_node

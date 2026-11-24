"""Query-rewriter agent: when the evaluator isn't confident, reformulate
the search query (fix spelling, expand abbreviations, add synonyms) and
send it back to the retriever for another attempt."""
from agents.state import AgentState, step
from lexical import tokenize

REWRITE_PROMPT = """The search below found nothing useful in an internal document collection.
Rewrite it as a better search query: fix spelling, expand abbreviations, and add likely synonyms or related terms.
Return ONLY the rewritten query on one line.

Original question: {query}
Previous search query: {search_query}"""


def make_rewrite_node(llm):
    def rewrite_node(state: AgentState) -> AgentState:
        previous = state.get("search_query") or state["query"]
        rewritten = llm.complete(
            REWRITE_PROMPT.format(query=state["query"], search_query=previous)
        ).strip().splitlines()
        rewritten = rewritten[0].strip(' "') if rewritten else ""
        how = "LLM"
        if not rewritten:
            # Rule-based fallback: keyword-only query, which shifts weight
            # toward the BM25 and trigram retrievers.
            rewritten = " ".join(tokenize(state["query"])) or state["query"]
            how = "rules"
        return {
            "search_query": rewritten,
            "trace": step("rewriter", f"({how}) '{previous}' -> '{rewritten}'"),
        }

    return rewrite_node

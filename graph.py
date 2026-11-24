"""
The core of the project: a LangGraph StateGraph that wires the specialist
agents (see agents/) into one multi-agent workflow.

Graph shape:

    router --knowledge_base--> retrieve -> generate -> evaluate --confident--> context -> END
       |                          ^                       |
       |                          +------ rewrite <--retry+
       |                                                  |
       |                                               give_up
       |                                                  v
       |                                               log_gap -> END
       +--direct--> direct -> END

- router:   picks knowledge_base / direct
- retrieve: hybrid search (dense + BM25 + trigram, fused with RRF)
- generate: N grounded answer samples
- evaluate: answer-agreement + abstention check -> confident / uncertain
- rewrite:  reformulates the search query, up to MAX_RETRIES times
- context:  picks the exact lines that support a confident answer
- log_gap:  records what the knowledge base is missing and says so plainly
"""
from langgraph.graph import StateGraph, END

from agents.context import context_node
from agents.direct import make_direct_node
from agents.evaluator import evaluate_node, make_route_after_evaluation
from agents.gap_logger import log_gap_node
from agents.knowledge import make_generate_node, make_retrieve_node
from agents.rewriter import make_rewrite_node
from agents.router import make_router_node, route_from_router
from agents.state import AgentState
from config import Config
from llm import get_llm
from retrieval import HybridRetriever
from vectorstore import VectorStore


def build_graph(vector_store: VectorStore = None, retriever: HybridRetriever = None):
    retriever = retriever or HybridRetriever(vector_store or VectorStore())
    llm = get_llm()

    graph = StateGraph(AgentState)
    graph.add_node("router", make_router_node(llm, retriever.sources))
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("generate", make_generate_node(llm, Config.N_SAMPLES))
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("context", context_node)
    graph.add_node("rewrite", make_rewrite_node(llm))
    graph.add_node("log_gap", log_gap_node)
    graph.add_node("direct", make_direct_node(llm, retriever.sources))

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {"knowledge_base": "retrieve", "direct": "direct"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        make_route_after_evaluation(Config.MAX_RETRIES),
        {"confident": "context", "retry": "rewrite", "give_up": "log_gap"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("context", END)
    graph.add_edge("log_gap", END)
    graph.add_edge("direct", END)

    return graph.compile()


def run_query(query: str, vector_store: VectorStore = None) -> AgentState:
    app = build_graph(vector_store)
    return app.invoke({"query": query})

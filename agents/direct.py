"""Direct-reply agent: greetings and small talk, no lookup."""
from agents.state import AgentState, step

DIRECT_PROMPT = """You are a friendly assistant for a knowledge base containing: {topics}.
Reply briefly (one or two sentences) to the message below, and invite a question about those topics.

Message: {query}"""


def make_direct_node(llm, sources: list[str]):
    topics = ", ".join(sources) or "internal documents"

    def direct_node(state: AgentState) -> AgentState:
        reply = llm.complete(DIRECT_PROMPT.format(topics=topics, query=state["query"])).strip()
        if not reply:
            reply = f"Hello! Ask me anything about the knowledge base ({topics})."
        return {"final_answer": reply, "trace": step("direct", "replied without lookup")}

    return direct_node

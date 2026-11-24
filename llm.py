"""
Pluggable LLM backend.

Every backend exposes two calls:
- generate(query, context): grounded answer from retrieved context
- complete(prompt): free-form completion, used by the router, rewriter
  and direct-reply agents

- "mock": deterministic-ish answers built directly from retrieved context.
  No API key, no network call. Use this to verify the LangGraph control
  flow for free before wiring up a real model. complete() returns "",
  so every agent falls back to its rule-based path.

- "anthropic": Claude via langchain-anthropic. Requires ANTHROPIC_API_KEY.

- "openai": GPT via langchain-openai. Requires OPENAI_API_KEY.

- "groq": Llama (or any Groq-hosted model) via Groq's OpenAI-compatible
  API. Requires GROQ_API_KEY. Free tier available at console.groq.com.
"""
import random
import re
from config import Config


GROUNDED_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say so plainly "
    "instead of guessing. Each context passage starts with its source in "
    "square brackets, e.g. [handbook.pdf, p. 12]; cite the sources you used "
    "at the end of your answer in that same format.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)


class MockLLM:
    """Simulates an LLM answering from context, with a little randomness
    injected so the confidence evaluator has something real to disagree
    about across samples -- otherwise every branch of the graph would
    trivially always look 'confident'."""

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            # No relevant context retrieved -- an honest model should
            # hedge here, and hedges *should* register as low confidence
            phrasing = random.choice([
                "I don't have enough information in the knowledge base to answer this confidently.",
                "This doesn't appear to be covered by the available documents.",
                "I'm not certain -- the retrieved context doesn't address this directly.",
            ])
            return phrasing

        # Drop the "[source, p. N]" label the retriever prefixes to chunks.
        snippet = re.sub(r"^\[[^\]]*\]\s*", "", context.strip()).split("\n")[0][:200]
        phrasing = random.choice([
            f"Based on the available documents: {snippet}",
            f"According to the knowledge base: {snippet}",
            f"The documents indicate: {snippet}",
        ])
        return phrasing

    def complete(self, prompt: str) -> str:
        return ""


class _LangChainLLM:
    def generate(self, query: str, context: str) -> str:
        return self.complete(GROUNDED_PROMPT.format(context=context, query=query))

    def complete(self, prompt: str) -> str:
        return self._model.invoke(prompt).content


class AnthropicLLM(_LangChainLLM):
    def __init__(self):
        from langchain_anthropic import ChatAnthropic
        self._model = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=0.7,
            max_tokens=400,
        )


class OpenAILLM(_LangChainLLM):
    def __init__(self):
        from langchain_openai import ChatOpenAI
        self._model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=Config.OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=400,
        )


class GroqLLM(_LangChainLLM):
    def __init__(self):
        from langchain_openai import ChatOpenAI
        # Groq's API is OpenAI-compatible: point ChatOpenAI at its base_url.
        self._model = ChatOpenAI(
            model=Config.GROQ_MODEL,
            api_key=Config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.7,
            max_tokens=400,
        )


_llm_singleton = None


def get_llm():
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton

    provider = Config.LLM_PROVIDER
    if provider == "mock":
        _llm_singleton = MockLLM()
    elif provider == "anthropic":
        _llm_singleton = AnthropicLLM()
    elif provider == "openai":
        _llm_singleton = OpenAILLM()
    elif provider == "groq":
        _llm_singleton = GroqLLM()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    return _llm_singleton

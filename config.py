"""Central place for all environment-driven configuration."""
import os
from dotenv import load_dotenv

load_dotenv()


def _streamlit_secret(name: str):
    """Streamlit Community Cloud exposes secrets via st.secrets; they only
    reach os.environ once st.secrets has been touched, which may be after
    this module is imported. Read them directly as a fallback."""
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        value = _streamlit_secret(name)
    return default if value is None else str(value)


class Config:
    LLM_PROVIDER = _env("LLM_PROVIDER", "mock").lower()
    EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
    GROQ_API_KEY = _env("GROQ_API_KEY", "")
    GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")

    N_SAMPLES = int(_env("N_SAMPLES", "3"))
    CONFIDENCE_THRESHOLD = float(_env("CONFIDENCE_THRESHOLD", "0.72"))
    # How many times the rewriter agent may reformulate the search query
    # and retry the knowledge base before the question is logged as a gap.
    MAX_RETRIES = int(_env("MAX_RETRIES", "1"))

    # --- Hybrid retrieval (dense + BM25 + trigram, fused with RRF) ---
    TOP_K = int(_env("TOP_K", "4"))
    # Candidates each retriever contributes before fusion.
    RETRIEVAL_CANDIDATES = int(_env("RETRIEVAL_CANDIDATES", "10"))
    # Dense chunks below this cosine similarity are treated as irrelevant
    # and dropped, rather than fed to the LLM as if they were useful
    # context. Calibrated for sentence-transformers (all-MiniLM-L6-v2).
    RETRIEVAL_MIN_SIMILARITY = float(_env("RETRIEVAL_MIN_SIMILARITY", "0.3"))
    # Trigram matches below this fuzzy-match score are dropped (0.0 - 1.0).
    TRIGRAM_MIN_SCORE = float(_env("TRIGRAM_MIN_SCORE", "0.2"))
    # Reciprocal Rank Fusion constant and per-retriever weights.
    RRF_K = int(_env("RRF_K", "60"))
    WEIGHT_DENSE = float(_env("WEIGHT_DENSE", "1.0"))
    WEIGHT_BM25 = float(_env("WEIGHT_BM25", "1.0"))
    WEIGHT_TRIGRAM = float(_env("WEIGHT_TRIGRAM", "0.5"))

    # Chunking: ~800 characters fits all-MiniLM-L6-v2's 256-token window.
    CHUNK_SIZE = int(_env("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "120"))

    CHROMA_DIR = _env("CHROMA_DIR", "./data/chroma")
    GAP_DB_PATH = _env("GAP_DB_PATH", "./data/gaps.db")

    SAMPLE_DOCS_DIR = _env("SAMPLE_DOCS_DIR", "./sample_docs")

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set in .env"
            )
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set in .env"
            )
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY to be set in .env"
            )

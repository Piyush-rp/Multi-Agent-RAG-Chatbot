"""Central place for all environment-driven configuration."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    N_SAMPLES = int(os.getenv("N_SAMPLES", "3"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.72"))
    # How many times the rewriter agent may reformulate the search query
    # and retry the knowledge base before the question is logged as a gap.
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

    # --- Hybrid retrieval (dense + BM25 + trigram, fused with RRF) ---
    TOP_K = int(os.getenv("TOP_K", "4"))
    # Candidates each retriever contributes before fusion.
    RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "10"))
    # Dense chunks below this cosine similarity are treated as irrelevant
    # and dropped, rather than fed to the LLM as if they were useful
    # context. Calibrated for sentence-transformers (all-MiniLM-L6-v2).
    RETRIEVAL_MIN_SIMILARITY = float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.3"))
    # Trigram matches below this fuzzy-match score are dropped (0.0 - 1.0).
    TRIGRAM_MIN_SCORE = float(os.getenv("TRIGRAM_MIN_SCORE", "0.2"))
    # Reciprocal Rank Fusion constant and per-retriever weights.
    RRF_K = int(os.getenv("RRF_K", "60"))
    WEIGHT_DENSE = float(os.getenv("WEIGHT_DENSE", "1.0"))
    WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "1.0"))
    WEIGHT_TRIGRAM = float(os.getenv("WEIGHT_TRIGRAM", "0.5"))

    CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
    GAP_DB_PATH = os.getenv("GAP_DB_PATH", "./data/gaps.db")

    SAMPLE_DOCS_DIR = os.getenv("SAMPLE_DOCS_DIR", "./sample_docs")

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

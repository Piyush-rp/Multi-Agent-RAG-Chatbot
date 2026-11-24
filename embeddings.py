"""
Semantic embeddings via a local sentence-transformers model
(all-MiniLM-L6-v2 by default, configurable with EMBEDDING_MODEL).
Downloads ~90MB from Hugging Face on first use.

Lexical matching (exact keywords, typos) is handled separately by the
BM25 and trigram retrievers in lexical.py, so this module only has to
do one job: meaning-based similarity.
"""
import numpy as np
from config import Config


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name or Config.EMBEDDING_MODEL)

    def embed_documents(self, texts):
        return self._model.encode(list(texts), convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


_embedder_singleton = None


def get_embedder():
    """Returns a process-wide singleton so we don't re-load the
    transformer model on every call."""
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = SentenceTransformerEmbedder()
    return _embedder_singleton


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

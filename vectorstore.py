"""Thin wrapper around a persistent Chroma collection holding the dense
(sentence-transformers) side of the hybrid retriever -- see retrieval.py."""
import os
import chromadb
from config import Config
from embeddings import get_embedder


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Simple fixed-size sliding-window chunker. Good enough for a demo
    corpus; swap for a token-aware splitter (e.g. LangChain's
    RecursiveCharacterTextSplitter) for real documents."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


class VectorStore:
    def __init__(self):
        os.makedirs(Config.CHROMA_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
        self._collection = self._client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedder()

    def reset(self):
        try:
            self._client.delete_collection("knowledge_base")
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_documents(self, doc_texts: dict):
        """doc_texts: {filename: full_text}"""
        all_chunks, all_ids, all_metas = [], [], []
        for filename, text in doc_texts.items():
            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{filename}::{i}")
                all_metas.append({"source": filename, "chunk_index": i})

        if not all_chunks:
            return 0

        embeddings = self._embedder.embed_documents(all_chunks)
        self._collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metas,
        )
        return len(all_chunks)

    def search(self, query: str, k: int = 3, min_similarity: float = None):
        """Returns retrieved chunks with a similarity score, filtering out
        anything below min_similarity. This matters: without a relevance
        floor, the vector store always returns its k-nearest chunks even
        when none of them are actually relevant to the query -- which lets
        the agent generate consistent, 'confident'-looking answers from
        context that has nothing to do with the question. Filtering here
        is what lets a truly out-of-scope query surface as low retrieval
        confidence instead of a smooth wrong answer."""
        count = self._collection.count()
        if count == 0:
            return []

        threshold = (Config.RETRIEVAL_MIN_SIMILARITY
                     if min_similarity is None else min_similarity)

        query_embedding = self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        out = []
        for cid, d, m, dist in zip(ids, docs, metas, distances):
            similarity = 1.0 - dist  # cosine space: distance = 1 - cosine_similarity
            if similarity >= threshold:
                out.append({
                    "id": cid,
                    "text": d,
                    "source": m.get("source", "unknown"),
                    "similarity": round(similarity, 4),
                })
        return out

    def all_chunks(self):
        """Every stored chunk -- used to build the in-memory BM25 and
        trigram indexes."""
        results = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": cid, "text": d, "source": (m or {}).get("source", "unknown")}
            for cid, d, m in zip(results["ids"], results["documents"], results["metadatas"])
        ]

    def count(self):
        return self._collection.count()

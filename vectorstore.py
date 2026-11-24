"""Thin wrapper around a persistent Chroma collection holding the dense
(sentence-transformers) side of the hybrid retriever -- see retrieval.py.

Each VectorStore points at one named collection. The Streamlit app gives
every browser session its own collection, so documents one visitor
uploads never leak into another visitor's answers.
"""
import os
import time
from typing import Callable, Iterable, Optional

import chromadb
from config import Config
from embeddings import get_embedder

# Chroma rejects upserts above ~5k records; embedding in moderate batches
# also lets the UI report progress on very large documents.
BATCH_SIZE = 256
SESSION_PREFIX = "session_"
# Source name for answers users taught to fill knowledge gaps.
LEARNED_SOURCE = "Learned answers"


def learned_text(entry: dict) -> str:
    """The stored text of a learned answer. The answer comes first (it is
    what the generator should use); the question is included so the same
    question retrieves it again, and any supporting passage the curator
    agent found in the documents is kept alongside it."""
    text = f"{entry['answer']}\n\nQuestion this answers: {entry['question']}"
    if entry.get("support"):
        text += f"\n\nSupporting passage: {entry['support']}"
    return text


def _chunk_text(text: str, chunk_size: int = None, overlap: int = None):
    """Fixed-size sliding-window chunker that prefers to break at sentence
    or word boundaries, so chunks don't start or end mid-word."""
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = Config.CHUNK_OVERLAP if overlap is None else overlap
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut < chunk_size // 2:
                cut = window.rfind(" ")
            if cut > chunk_size // 2:
                end = start + cut + 1
        chunks.append(text[start:end])
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # Begin the overlap at a word boundary rather than mid-word.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return [c.strip() for c in chunks if c.strip()]


class VectorStore:
    def __init__(self, collection_name: str = "knowledge_base"):
        os.makedirs(Config.CHROMA_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
        self.collection_name = collection_name
        self._collection = self._open()
        self._embedder = get_embedder()

    def _open(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "created_at": time.time()},
        )

    def reset(self):
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._open()

    # ------------------------------------------------------------------ ingest
    def _upsert_batched(self, ids, docs, metas, progress: Optional[Callable] = None):
        total = len(docs)
        for start in range(0, total, BATCH_SIZE):
            end = start + BATCH_SIZE
            embeddings = self._embedder.embed_documents(docs[start:end])
            self._collection.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                embeddings=embeddings,
                metadatas=metas[start:end],
            )
            if progress:
                progress(min(end, total), total)
        return total

    def ingest_pages(self, source: str, pages: Iterable[tuple[int, str]],
                     progress: Optional[Callable] = None) -> int:
        """Chunk and embed a paged document. pages: iterable of
        (page_number, text). Chunks never cross page boundaries, so every
        chunk can cite the page it came from. Only PDFs have real pages;
        for other formats the pseudo-page is stored as a "section" and not
        shown as a page citation. progress(done, total) is called after
        each embedded batch."""
        page_key = "page" if source.lower().endswith(".pdf") else "section"
        ids, docs, metas = [], [], []
        for page, text in pages:
            for i, chunk in enumerate(_chunk_text(text)):
                ids.append(f"{source}::p{page}::{i}")
                docs.append(chunk)
                metas.append({"source": source, page_key: int(page), "chunk_index": i})
        if not docs:
            return 0
        return self._upsert_batched(ids, docs, metas, progress)

    def ingest_documents(self, doc_texts: dict, progress: Optional[Callable] = None):
        """doc_texts: {filename: full_text} for page-less plain text."""
        ids, docs, metas = [], [], []
        for filename, text in doc_texts.items():
            for i, chunk in enumerate(_chunk_text(text)):
                ids.append(f"{filename}::{i}")
                docs.append(chunk)
                metas.append({"source": filename, "chunk_index": i})
        if not docs:
            return 0
        return self._upsert_batched(ids, docs, metas, progress)

    def copy_from(self, other: "VectorStore") -> int:
        """Copy every chunk (with its precomputed embedding) from another
        collection. Lets each new session start with the sample documents
        without re-embedding them."""
        data = other._collection.get(include=["documents", "metadatas", "embeddings"])
        ids = data["ids"]
        for start in range(0, len(ids), 5000):
            end = start + 5000
            self._collection.upsert(
                ids=ids[start:end],
                documents=data["documents"][start:end],
                metadatas=data["metadatas"][start:end],
                embeddings=data["embeddings"][start:end],
            )
        return len(ids)

    def upsert_learned(self, entries: list[dict]) -> int:
        """Add (or update) learned answers, one chunk each, keyed by gap id."""
        if not entries:
            return 0
        ids = [f"learned::{e['gap_id']}" for e in entries]
        docs = [learned_text(e) for e in entries]
        metas = [{"source": LEARNED_SOURCE, "gap_id": int(e["gap_id"]), "chunk_index": 0}
                 for e in entries]
        return self._upsert_batched(ids, docs, metas)

    def delete_learned(self, gap_id: int):
        self._collection.delete(ids=[f"learned::{gap_id}"])

    def delete_source(self, source: str):
        self._collection.delete(where={"source": source})

    def sources(self) -> dict[str, dict]:
        """{source: {"chunks": n, "pages": n}} for every stored document."""
        metas = self._collection.get(include=["metadatas"])["metadatas"]
        out: dict[str, dict] = {}
        for m in metas:
            m = m or {}
            entry = out.setdefault(m.get("source", "unknown"), {"chunks": 0, "pages": set()})
            entry["chunks"] += 1
            if "page" in m:
                entry["pages"].add(m["page"])
        return {s: {"chunks": e["chunks"], "pages": len(e["pages"])}
                for s, e in sorted(out.items())}

    # ------------------------------------------------------------------ search
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
                m = m or {}
                out.append({
                    "id": cid,
                    "text": d,
                    "source": m.get("source", "unknown"),
                    "page": m.get("page"),
                    "similarity": round(similarity, 4),
                })
        return out

    def all_chunks(self):
        """Every stored chunk -- used to build the in-memory BM25 and
        trigram indexes."""
        results = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": cid, "text": d, "source": (m or {}).get("source", "unknown"),
             "page": (m or {}).get("page")}
            for cid, d, m in zip(results["ids"], results["documents"], results["metadatas"])
        ]

    def count(self):
        return self._collection.count()

    # ------------------------------------------------------------ housekeeping
    @staticmethod
    def prune_sessions(max_age_hours: float = 12) -> int:
        """Delete per-session collections older than max_age_hours, so
        abandoned browser sessions don't fill the disk."""
        client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for col in client.list_collections():
            name = col if isinstance(col, str) else col.name
            if not name.startswith(SESSION_PREFIX):
                continue
            meta = client.get_collection(name).metadata or {}
            if meta.get("created_at", 0) < cutoff:
                client.delete_collection(name)
                removed += 1
        return removed

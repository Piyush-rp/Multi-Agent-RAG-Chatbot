"""
Hybrid retriever: runs dense (Chroma + sentence-transformers), BM25 and
trigram search in parallel and merges the three ranked lists with
Reciprocal Rank Fusion.

RRF only looks at *ranks*, not raw scores, which is what makes it safe to
combine retrievers whose scores live on completely different scales
(cosine similarity vs. unbounded BM25 vs. 0-1 trigram overlap). A chunk
found by several retrievers accumulates score from each of them.

Each retriever applies its own relevance floor before fusion, so a truly
out-of-scope query still comes back empty instead of returning the
k-nearest-but-irrelevant chunks.
"""
from config import Config
from lexical import BM25Index, TrigramIndex
from vectorstore import VectorStore


def reciprocal_rank_fusion(ranked_lists: dict, weights: dict, k: int = 60) -> list[tuple]:
    """ranked_lists: {retriever_name: [(chunk_id, score), ...] best-first}
    Returns [(chunk_id, fused_score, {retriever_name: score})] best-first."""
    fused = {}
    for name, hits in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for rank, (chunk_id, score) in enumerate(hits):
            entry = fused.setdefault(chunk_id, [0.0, {}])
            entry[0] += weight / (k + rank + 1)
            entry[1][name] = round(score, 4)
    ordered = sorted(fused.items(), key=lambda x: x[1][0], reverse=True)
    return [(cid, rrf, scores) for cid, (rrf, scores) in ordered]


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self._store = store
        self.refresh()

    def refresh(self):
        """(Re)build the in-memory lexical indexes from Chroma. Call this
        after ingesting new documents into an already-built retriever."""
        self._chunks = self._store.all_chunks()
        self._by_id = {c["id"]: c for c in self._chunks}
        texts = [c["text"] for c in self._chunks]
        self._bm25 = BM25Index(texts)
        self._trigram = TrigramIndex(texts)

    @property
    def sources(self) -> list[str]:
        return sorted({c["source"] for c in self._chunks})

    def search(self, query: str, k: int = None) -> list[dict]:
        k = k or Config.TOP_K
        n = Config.RETRIEVAL_CANDIDATES
        ranked = {
            "dense": [(r["id"], r["similarity"]) for r in self._store.search(query, k=n)],
            "bm25": [(self._chunks[i]["id"], s) for i, s in self._bm25.search(query, n)],
            "trigram": [(self._chunks[i]["id"], s) for i, s in
                        self._trigram.search(query, n, min_score=Config.TRIGRAM_MIN_SCORE)],
        }
        weights = {
            "dense": Config.WEIGHT_DENSE,
            "bm25": Config.WEIGHT_BM25,
            "trigram": Config.WEIGHT_TRIGRAM,
        }
        out = []
        for chunk_id, rrf, scores in reciprocal_rank_fusion(ranked, weights, Config.RRF_K)[:k]:
            chunk = self._by_id[chunk_id]
            out.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk.get("page"),
                "rrf_score": round(rrf, 5),
                "scores": scores,
            })
        return out

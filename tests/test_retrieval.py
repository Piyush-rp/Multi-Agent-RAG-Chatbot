"""
Unit tests for the lexical retrievers and rank fusion -- pure Python, no
model download, no network. Run with:

    python tests/test_retrieval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexical import BM25Index, TrigramIndex, tokenize, trigram_similarity
from retrieval import reciprocal_rank_fusion

DOCS = [
    "The library is open from 8 AM to 9 PM on weekdays.",
    "Hostel residents must return by 9:30 PM. Rule 6: no cooking appliances in rooms.",
    "Students must maintain 75 percent attendance to sit for exams.",
]


def test_tokenize_drops_stopwords_keeps_numbers():
    assert tokenize("What is rule no 6?") == ["rule", "6"]


def test_bm25_ranks_exact_keyword_match_first():
    hits = BM25Index(DOCS).search("rule 6 cooking", k=3)
    assert hits[0][0] == 1
    assert all(score > 0 for _, score in hits)


def test_bm25_returns_nothing_for_unrelated_query():
    assert BM25Index(DOCS).search("airspeed swallow", k=3) == []


def test_trigram_similarity():
    assert trigram_similarity("library", "library") == 1.0
    assert trigram_similarity("libary", "library") > 0.4
    assert trigram_similarity("swallow", "library") < 0.2


def test_trigram_matches_typos_that_bm25_misses():
    query = "libary weekdys"
    assert BM25Index(DOCS).search(query, k=3) == []
    hits = TrigramIndex(DOCS).search(query, k=3, min_score=0.3)
    assert hits and hits[0][0] == 0


def test_rrf_rewards_agreement_between_retrievers():
    fused = reciprocal_rank_fusion(
        {"dense": [("a", 0.9), ("b", 0.8)], "bm25": [("b", 5.0), ("c", 2.0)]},
        weights={"dense": 1.0, "bm25": 1.0},
    )
    assert fused[0][0] == "b"  # found by both retrievers
    assert fused[0][2] == {"dense": 0.8, "bm25": 5.0}


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} retrieval tests passed.")

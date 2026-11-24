"""
Smoke test: verifies the multi-agent LangGraph workflow end-to-end --
routing, hybrid retrieval, generation, confidence branching, query
rewrite retries, gap logging -- using the mock LLM so it needs no API
key. Needs the sentence-transformers model (downloaded on first run).
Run with:

    python tests/test_graph_smoke.py
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["LLM_PROVIDER"] = "mock"
os.environ["CHROMA_DIR"] = "./data/test_chroma"
os.environ["GAP_DB_PATH"] = "./data/test_gaps.db"

# Clean slate for a repeatable test
for path in ["./data/test_chroma", "./data/test_gaps.db"]:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)

from vectorstore import VectorStore
from graph import build_graph
import gap_tracker

store = VectorStore()
store.ingest_documents({
    "library.txt": (
        "The library is open from 8 AM to 9 PM on weekdays. "
        "Students may borrow up to 5 books for 14 days."
    ),
    "hostel.txt": "Rule 6: cooking appliances are not allowed inside hostel rooms.",
})
app = build_graph(store)


def show(label, result):
    print(f"[{label}] route={result['route']} verdict={result.get('verdict', '-')}")
    for s in result["trace"]:
        print(f"    {s['agent']}: {s['detail']}")
    print(f"  answer: {result['final_answer'][:160]}")


def test_confident_answer_on_covered_topic():
    result = app.invoke({"query": "What are the library hours?"})
    show("covered topic", result)
    assert result["route"] == "knowledge_base"
    assert result["verdict"] == "confident"
    assert result["retrieved"][0]["source"] == "library.txt"


def test_typo_is_recovered_by_trigram():
    result = app.invoke({"query": "libary borowing limit"})
    show("typo", result)
    assert any("trigram" in r["scores"] for r in result["retrieved"])
    assert result["retrieved"][0]["source"] == "library.txt"


def test_exact_keyword_found_by_bm25():
    result = app.invoke({"query": "rule 6"})
    show("keyword", result)
    assert result["retrieved"][0]["source"] == "hostel.txt"
    assert "bm25" in result["retrieved"][0]["scores"]


def test_uncovered_topic_retries_then_logs_gap():
    result = app.invoke({"query": "What is the airspeed velocity of an unladen swallow?"})
    show("uncovered topic", result)
    assert result["verdict"] == "uncertain"
    assert result["attempts"] == 2  # first try + one rewrite retry
    assert any(s["agent"] == "rewriter" for s in result["trace"])
    gaps = gap_tracker.list_gaps(include_resolved=True)
    assert gaps and gaps[0]["query"].startswith("What is the airspeed")


def test_router_sends_greeting_to_direct_agent():
    result = app.invoke({"query": "hello"})
    show("greeting", result)
    assert result["route"] == "direct"
    assert "retrieved" not in result


def test_out_of_scope_question_is_not_answered_with_a_guess():
    result = app.invoke({"query": "latest news about the Mars mission"})
    show("out of scope", result)
    assert result["route"] == "knowledge_base"
    assert result["verdict"] == "uncertain"
    assert "couldn't find a reliable answer" in result["final_answer"]
    assert "gap_id" in result


if __name__ == "__main__":
    print("Running smoke test (mock LLM + sentence-transformers, no API key needed)...\n")
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print()
    print(f"All {len(tests)} smoke tests passed.")

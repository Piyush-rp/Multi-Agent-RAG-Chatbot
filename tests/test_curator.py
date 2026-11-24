"""
Tests the teach-the-answer loop: a question hits a knowledge gap, the user
supplies the answer, the curator agent checks it, it is saved, and asking
again returns it. Uses the mock LLM (plus a fake LLM for the parsing
path), so no API key is needed. Run with:

    python tests/test_curator.py
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["LLM_PROVIDER"] = "mock"
os.environ["CHROMA_DIR"] = "./data/test_curator_chroma"
os.environ["GAP_DB_PATH"] = "./data/test_curator_gaps.db"

for path in ["./data/test_curator_chroma", "./data/test_curator_gaps.db"]:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)

import gap_tracker
from agents.curator import curate, support_text
from graph import build_graph
from llm import get_llm
from retrieval import HybridRetriever
from vectorstore import LEARNED_SOURCE, VectorStore

store = VectorStore()
store.ingest_documents({
    "library.txt": (
        "The library is open from 8 AM to 9 PM on weekdays. "
        "Students may borrow up to 5 books for 14 days."
    ),
})


def teach(gap_id, question, user_input, llm=None):
    """What the app's Save button does."""
    draft = curate(question, user_input, HybridRetriever(store), llm or get_llm())
    support = support_text(draft["evidence"])
    gap_tracker.save_learned_answer(gap_id, question, user_input, draft["answer"],
                                    draft["verdict"], draft["note"], support)
    store.upsert_learned([{"gap_id": gap_id, "question": question,
                           "answer": draft["answer"], "support": support}])
    return draft


def test_taught_answer_is_returned_next_time():
    question = "How often must the wifi password be changed?"
    first = build_graph(store).invoke({"query": question})
    assert first["verdict"] == "uncertain" and "gap_id" in first

    draft = teach(first["gap_id"], question, "every 90 days, by the IT desk")
    print(f"[teach] verdict={draft['verdict']} answer={draft['answer']!r}")
    assert draft["verdict"] == "not_in_documents"
    assert draft["answer"] == "Every 90 days, by the IT desk."
    assert gap_tracker.gap_stats()["resolved"] == 1

    again = build_graph(store).invoke({"query": question})
    print(f"[ask again] verdict={again['verdict']} answer={again['final_answer']!r}")
    assert again["verdict"] == "confident"
    assert again["retrieved"][0]["source"] == LEARNED_SOURCE
    assert "90 days" in again["final_answer"]


def test_rule_based_check_finds_supporting_lines():
    draft = curate("How long can I keep a borrowed book?", "14 days",
                   HybridRetriever(store), get_llm())
    print(f"[support] verdict={draft['verdict']} support={support_text(draft['evidence'])}")
    assert draft["verdict"] == "supported"
    assert draft["evidence"][0]["source"] == "library.txt"


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, prompt):
        return self.reply


def test_llm_verdict_and_rewrite_are_used():
    reply = ("answer: The library closes at 9 PM on weekdays.\n"
             "support: 1\nverdict: contradicted\n"
             "note: The user said 10 PM but the document says 9 PM.")
    draft = curate("When does the library close?", "10pm",
                   HybridRetriever(store), FakeLLM(reply))
    assert draft["how"] == "LLM"
    assert draft["verdict"] == "contradicted"
    assert draft["answer"] == "The library closes at 9 PM on weekdays."
    assert draft["evidence"] and draft["evidence"][0]["source"] == "library.txt"


def test_unparseable_llm_reply_falls_back_to_rules():
    draft = curate("When does the library close?", "9 pm", HybridRetriever(store),
                   FakeLLM("Sure! I think it's fine."))
    assert draft["how"] == "rules"


def test_forget_reopens_the_gap():
    gap_id = gap_tracker.list_learned_answers()[0]["gap_id"]
    gap_tracker.forget_learned_answer(gap_id)
    store.delete_learned(gap_id)
    assert gap_tracker.list_learned_answers() == []
    assert LEARNED_SOURCE not in store.sources()
    assert gap_tracker.gap_stats()["unresolved"] == 1


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}\n")
    print(f"All {len(tests)} curator tests passed.")

"""Curator agent: turns a user's answer to a knowledge gap into a clean
knowledge-base entry, checking it against the documents first.

Not a graph node: the UI calls curate() when a user teaches an answer,
shows the result for review, and only then saves it (see app.py).

How it works:
- searches the knowledge base with the question plus the user's answer,
  which often finds passages the original question missed;
- asks the LLM to rewrite the user's input as a clear, standalone answer
  and to judge whether the passages support it, contradict it, or don't
  mention it;
- picks the exact supporting lines with the context agent's selector.

Without a usable LLM reply (always the case for the mock backend), the
user's text is kept as the answer and support is judged by similarity
alone, which cannot detect contradictions.
"""
import re

from agents.context import select_evidence
from retrieval import HybridRetriever
from vectorstore import LEARNED_SOURCE

VERDICTS = ("supported", "contradicted", "not_in_documents")
# Rule-based fallback: an excerpt scoring at least this (similarity to the
# question + answer, plus word overlap) counts as supporting the answer.
SUPPORT_THRESHOLD = 0.65

CURATE_PROMPT = """A question could not be answered from the documents in a knowledge base. A user has now said what the answer should be.

1. Rewrite the user's input as a clear, complete, standalone answer to the question, in one to three sentences. Keep every fact the user gave, and do not add facts that appear in neither the user's input nor the passages.
2. Check the numbered passages from the documents and list the ones that support the answer.
3. Decide whether the documents support the answer, contradict it, or don't mention it.

Reply in exactly this format:
answer: <the rewritten answer>
support: <comma-separated passage numbers, or none>
verdict: <supported|contradicted|not_in_documents>
note: <one short sentence explaining the verdict>

Question: {question}
User's answer: {user_input}

Passages:
{passages}"""


def _parse(raw: str) -> dict | None:
    out = {}
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(" *-").lower(), value.strip(" *`")
        if key in ("answer", "support", "verdict", "note") and key not in out:
            out[key] = value
    if not out.get("answer") or out.get("verdict", "").lower() not in VERDICTS:
        return None
    out["verdict"] = out["verdict"].lower()
    out["support"] = [int(n) for n in re.findall(r"\d+", out.get("support", ""))]
    return out


def _tidy(text: str) -> str:
    text = " ".join(text.split())
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def support_text(evidence: list[dict]) -> str:
    """The highlighted lines of the best excerpt, with where they came from."""
    if not evidence:
        return ""
    e = evidence[0]
    lines = [t for t, highlighted in e["lines"] if highlighted] or [t for t, _ in e["lines"]]
    where = e["source"] + (f", p. {e['page']}" if e.get("page") else "")
    return f"\"{' '.join(lines)}\" ({where})"


def curate(question: str, user_input: str, retriever: HybridRetriever, llm) -> dict:
    """Returns {"answer", "verdict", "note", "evidence", "how"} for review."""
    candidates = [c for c in retriever.search(f"{question} {user_input}")
                  if c["source"] != LEARNED_SOURCE]

    passages = "\n\n".join(
        f"[{i}] ({c['source']}{f', p. {c['page']}' if c.get('page') else ''}) {c['text']}"
        for i, c in enumerate(candidates, 1)
    ) or "(no relevant passages found)"
    parsed = _parse(llm.complete(CURATE_PROMPT.format(
        question=question, user_input=user_input, passages=passages)))

    if parsed:
        answer = _tidy(parsed["answer"])
        supporting = [candidates[n - 1] for n in parsed["support"] if 0 < n <= len(candidates)]
        evidence = select_evidence(question, answer, supporting) if supporting else []
        verdict = parsed["verdict"]
        if verdict == "supported" and not evidence:
            verdict = "not_in_documents"
        return {"answer": answer, "verdict": verdict, "note": parsed.get("note", ""),
                "evidence": evidence, "how": "LLM"}

    answer = _tidy(user_input)
    evidence = [e for e in select_evidence(question, answer, candidates)
                if e["score"] >= SUPPORT_THRESHOLD]
    if evidence:
        verdict, note = "supported", "Found closely matching lines in the documents."
    else:
        verdict, note = "not_in_documents", "No matching lines found in the documents."
    return {"answer": answer, "verdict": verdict,
            "note": note + " (Rule-based check: contradictions can't be detected without an LLM.)",
            "evidence": evidence, "how": "rules"}

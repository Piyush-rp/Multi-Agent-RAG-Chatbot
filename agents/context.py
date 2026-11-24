"""Context agent: after an answer is judged confident, finds the exact
lines in the retrieved passages that support it, so the user sees a short,
highlighted excerpt (with file and page) instead of whole chunks.

How it picks the lines:
- every retrieved chunk is split into sentences;
- each sentence is scored by embedding similarity to the question plus
  the answer (the answer is the strongest signal of what the generator
  actually used), plus a boost for sharing the answer's own words;
- the best chunks (within a small margin of the top score) each yield an
  excerpt: their best sentence plus one sentence either side;
- inside an excerpt, sentences scoring close to the best one, or sharing a
  specific number with the answer ("24 days", "USD 60"), are highlighted.
"""
import re

import numpy as np

from agents.state import AgentState, step
from embeddings import get_embedder
from lexical import tokenize

LEXICAL_WEIGHT = 0.3

# Sentence boundaries: end punctuation followed by whitespace and a capital
# letter or digit ("... per year. A medical certificate ...").
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_NUMBER = re.compile(r"\d[\d,.]*")


def reflow(text: str) -> str:
    """Join the hard line breaks PDFs put at the end of every visual line,
    keeping paragraph breaks and numbered section headings."""
    text = re.sub(r"\n(?=\d+\.\s+[A-Z])", "\n\n", text)
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in re.split(r"\n\s*\n", reflow(text)):
        parts.extend(s.strip() for s in _SENTENCE_END.split(para) if s.strip())
    return parts


def _numbers(text: str) -> set[str]:
    return {n.rstrip(".,").replace(",", "") for n in _NUMBER.findall(text)}


def select_evidence(query: str, answer: str, retrieved: list[dict],
                    max_passages: int = 2, margin: float = 0.08,
                    highlight_margin: float = 0.06) -> list[dict]:
    """Return up to max_passages excerpts, best first:
        [{"source", "page", "score", "lines": [(sentence, highlighted), ...]}]"""
    chunks = [split_sentences(c["text"]) for c in retrieved]
    flat = [(ci, si) for ci, sents in enumerate(chunks) for si in range(len(sents))]
    if not flat:
        return []

    embedder = get_embedder()
    target = np.array(embedder.embed_query(f"{query} {answer}"))
    vecs = np.array(embedder.embed_documents([chunks[ci][si] for ci, si in flat]))
    target /= np.linalg.norm(target) or 1.0
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    # Semantic similarity, plus a boost for sharing the answer's words: the
    # embedding alone can't tell "Tuesday, Wednesday and Thursday" (the
    # answer) from "Monday and Friday are remote" (a neighbour).
    answer_words = set(tokenize(answer)) - set(tokenize(query))
    def overlap(sentence: str) -> float:
        return len(answer_words & set(tokenize(sentence))) / len(answer_words) if answer_words else 0.0
    semantic = vecs @ target
    score_of = {key: float(sem) + LEXICAL_WEIGHT * overlap(chunks[key[0]][key[1]])
                for key, sem in zip(flat, semantic)}

    best = {}  # chunk index -> (score, sentence index)
    for (ci, si), s in score_of.items():
        if ci not in best or s > best[ci][0]:
            best[ci] = (s, si)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    top = ranked[0][1][0]
    answer_numbers = _numbers(answer)

    out, seen = [], set()
    for ci, (s, si) in ranked:
        if len(out) >= max_passages or s < top - margin:
            break
        sents = chunks[ci]
        if sents[si] in seen:  # chunks overlap, so the same line can repeat
            continue
        seen.add(sents[si])
        lo, hi = max(si - 1, 0), min(si + 2, len(sents))
        lines = []
        for j in range(lo, hi):
            close = score_of[(ci, j)] >= s - highlight_margin
            shares_number = bool(answer_numbers & _numbers(sents[j]))
            lines.append((sents[j], close or shares_number))
        out.append({
            "source": retrieved[ci]["source"],
            "page": retrieved[ci].get("page"),
            "score": round(s, 3),
            "lines": lines,
        })
    return out


def context_node(state: AgentState) -> AgentState:
    evidence = select_evidence(state["query"], state["final_answer"],
                               state.get("retrieved", []))
    where = ", ".join(f"{e['source']}" + (f" p. {e['page']}" if e.get("page") else "")
                      for e in evidence) or "no supporting lines found"
    return {
        "evidence": evidence,
        "trace": step("context", f"supporting lines from {where}"),
    }

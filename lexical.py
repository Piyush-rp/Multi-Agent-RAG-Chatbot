"""
Lexical retrievers that complement dense semantic search.

- BM25Index: Okapi BM25 over word tokens. Catches exact keyword, ID and
  number matches ("rule 6", "48 hours") that embeddings tend to blur.
- TrigramIndex: character-trigram fuzzy matching (the same idea as
  Postgres pg_trgm). Catches typos and spelling variants ("libary",
  "hostle") that both BM25 and embeddings miss.

Both are pure Python and built in memory from the chunks already stored
in Chroma, so there is nothing extra to persist or keep in sync.
"""
import math
import re
from collections import Counter

STOPWORDS = frozenset("""
a about an and any are as at be been but by can could did do does for from
had has have how i if in into is it its me my no not of on or our should so
than that the their them then there these they this to us was we were what
when where which who why will with would you your
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self._k1, self._b = k1, b
        self._tfs = [Counter(tokenize(d)) for d in docs]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avgdl = (sum(self._lens) / len(docs)) if docs else 0.0
        n = len(docs)
        df = Counter(term for tf in self._tfs for term in tf)
        # BM25+ style idf: always positive, even for terms in most docs
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, dl in zip(self._tfs, self._lens):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                    score += self._idf[term] * f * (self._k1 + 1) / (f + norm)
            out.append(score)
        return out

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """(doc_index, score) for the top-k docs sharing any query term."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]


def trigrams(word: str) -> set[str]:
    padded = f"  {word} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def trigram_similarity(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    return len(ta & tb) / len(ta | tb)


class TrigramIndex:
    """Scores a chunk by how well each query word fuzzy-matches *some*
    word in the chunk: per query word, take the best trigram similarity
    against the chunk's vocabulary (ignoring matches below
    min_word_similarity, 0.3 like pg_trgm's default), then average over
    the query words."""

    def __init__(self, docs: list[str], min_word_similarity: float = 0.3):
        self._min_word_sim = min_word_similarity
        self._doc_vocab = [set(tokenize(d)) for d in docs]
        self._vocab = set().union(*self._doc_vocab) if docs else set()

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        if not terms:
            return [0.0] * len(self._doc_vocab)
        # Compare each query word against the corpus vocabulary once,
        # then reuse those similarities for every chunk.
        term_sims = {
            term: {w: trigram_similarity(term, w) for w in self._vocab}
            for term in terms
        }
        out = []
        for vocab in self._doc_vocab:
            total = 0.0
            for term in terms:
                best = max((term_sims[term][w] for w in vocab), default=0.0)
                if best >= self._min_word_sim:
                    total += best
            out.append(total / len(terms))
        return out

    def search(self, query: str, k: int, min_score: float = 0.0) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0 and s >= min_score]

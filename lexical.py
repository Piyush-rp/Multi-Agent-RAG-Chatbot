"""
Lexical retrievers that complement dense semantic search.

- BM25Index: Okapi BM25 over word tokens. Catches exact keyword, ID and
  number matches ("rule 6", "48 hours") that embeddings tend to blur.
- TrigramIndex: character-trigram fuzzy matching (the same idea as
  Postgres pg_trgm). Catches typos and spelling variants ("libary",
  "hostle") that both BM25 and embeddings miss.

Both are pure Python and built in memory from the chunks already stored
in Chroma, so there is nothing extra to persist or keep in sync. Both use
inverted indexes (term -> chunks), so a query only touches chunks that
share a term with it -- this keeps search fast on corpora of tens of
thousands of chunks (documents running to thousands of pages).
"""
import math
import re
from collections import Counter, defaultdict

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
        self._n = len(docs)
        self._lens = []
        # term -> [(doc_index, term_frequency), ...]
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            tf = Counter(tokenize(d))
            self._lens.append(sum(tf.values()))
            for term, f in tf.items():
                self._postings[term].append((i, f))
        self._avgdl = (sum(self._lens) / self._n) if self._n else 0.0
        n = self._n
        # BM25+ style idf: always positive, even for terms in most docs
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                     for t, p in self._postings.items()}

    def _sparse_scores(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for i, f in postings:
                norm = self._k1 * (1 - self._b + self._b * self._lens[i] / self._avgdl)
                scores[i] += idf * f * (self._k1 + 1) / (f + norm)
        return scores

    def scores(self, query: str) -> list[float]:
        sparse = self._sparse_scores(query)
        return [sparse.get(i, 0.0) for i in range(self._n)]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """(doc_index, score) for the top-k docs sharing any query term."""
        ranked = sorted(self._sparse_scores(query).items(), key=lambda x: x[1], reverse=True)
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
        self._n = len(docs)
        # word -> chunks containing it, and trigram -> words containing it,
        # so a query word is only compared against plausible candidates.
        self._word_docs: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(docs):
            for w in set(tokenize(d)):
                self._word_docs[w].append(i)
        self._gram_words: dict[str, set[str]] = defaultdict(set)
        for w in self._word_docs:
            for g in trigrams(w):
                self._gram_words[g].add(w)

    def _matches(self, term: str) -> dict[str, float]:
        """Vocabulary words whose trigram similarity to term clears the floor."""
        candidates = set()
        for g in trigrams(term):
            candidates |= self._gram_words.get(g, set())
        out = {}
        for w in candidates:
            sim = trigram_similarity(term, w)
            if sim >= self._min_word_sim:
                out[w] = sim
        return out

    def _sparse_scores(self, query: str) -> dict[int, float]:
        terms = set(tokenize(query))
        if not terms:
            return {}
        totals: dict[int, float] = defaultdict(float)
        for term in terms:
            best: dict[int, float] = {}
            for w, sim in self._matches(term).items():
                for i in self._word_docs[w]:
                    if sim > best.get(i, 0.0):
                        best[i] = sim
            for i, sim in best.items():
                totals[i] += sim
        return {i: t / len(terms) for i, t in totals.items()}

    def scores(self, query: str) -> list[float]:
        sparse = self._sparse_scores(query)
        return [sparse.get(i, 0.0) for i in range(self._n)]

    def search(self, query: str, k: int, min_score: float = 0.0) -> list[tuple[int, float]]:
        ranked = sorted(self._sparse_scores(query).items(), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0 and s >= min_score]

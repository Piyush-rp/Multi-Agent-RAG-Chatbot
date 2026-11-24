"""
Confidence evaluator.

Generates N candidate answers for the same query, embeds each one, and
measures how much they agree with each other (mean pairwise cosine
similarity). High agreement -> the model is saying roughly the same thing
every time -> confident. Low agreement -> the model is inconsistent ->
treat as uncertain rather than presenting whichever answer came out first.

Two checks run before agreement, because agreement alone is fooled by
them:
- no context retrieved: nothing to ground an answer in.
- the model abstains ("the context doesn't say"): N answers can agree
  perfectly on "I don't know", which is a consistent non-answer, not a
  confident one.

This is a simplified, embedding-based proxy for the "semantic entropy"
technique (Farquhar et al., 2024) used for LLM hallucination detection.
Full semantic entropy clusters generations using NLI-based equivalence
rather than raw cosine similarity -- see the README's "Honest scope note"
for what to say if an interviewer asks about this tradeoff.
"""
from itertools import combinations
from config import Config
from embeddings import get_embedder, cosine_similarity

ABSTENTION_MARKERS = (
    "don't have enough information", "do not have enough information",
    "not covered", "doesn't address", "does not address",
    "doesn't contain", "does not contain", "doesn't mention",
    "does not mention", "not mentioned", "no information",
    "not in the context", "not provided in the context",
    "cannot answer", "can't answer", "unable to answer", "i'm not certain",
)


def is_abstention(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return any(marker in text for marker in ABSTENTION_MARKERS)


def score_confidence(answers: list[str], context: str) -> tuple[float, str, str]:
    """Returns (confidence_score, verdict, reason) where verdict is
    'confident' or 'uncertain'."""
    if not context.strip():
        return 0.0, "uncertain", "no relevant chunks were retrieved"

    abstained = sum(is_abstention(a) for a in answers)
    if abstained * 2 > len(answers):
        return 0.0, "uncertain", (
            f"{abstained}/{len(answers)} answers said the context doesn't cover it")

    if len(answers) < 2:
        return 1.0, "confident", "single sample, nothing to compare against"

    vectors = get_embedder().embed_documents(answers)
    pairs = list(combinations(range(len(vectors)), 2))
    similarities = [cosine_similarity(vectors[i], vectors[j]) for i, j in pairs]
    mean_similarity = sum(similarities) / len(similarities)

    if mean_similarity >= Config.CONFIDENCE_THRESHOLD:
        return mean_similarity, "confident", "answers agree"
    return mean_similarity, "uncertain", "answers disagree with each other"

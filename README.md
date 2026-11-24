# Self-Improving Knowledge-Gap-Aware RAG Agent

A Retrieval-Augmented Generation system that monitors its own confidence.
When it doesn't know something, it doesn't guess — it logs the gap, tries a
live search fallback, and tells you clearly what it's unsure about.

Built with **LangGraph** (multi-agent orchestration), **LangChain**
(LLM wrappers), **hybrid retrieval** (dense sentence-transformer embeddings
in Chroma + BM25 + character-trigram fuzzy matching, fused with Reciprocal
Rank Fusion), and a **Streamlit** dashboard.

## Why this is different from a standard RAG chatbot

Most RAG projects assume the document corpus is complete and always answer
confidently. This one:

1. Generates multiple candidate answers per query
2. Embeds them and measures how much they *agree* with each other
   (a lightweight version of the "semantic entropy" technique used for
   LLM hallucination detection)
3. If the answers disagree (low confidence), it does **not** present a
   confident-sounding guess. It logs the query as a "knowledge gap,"
   optionally tries a live web search, and clearly labels the response
   as unverified.
4. A dashboard shows you a running list of what the knowledge base is
   missing — turning a passive Q&A bot into an active knowledge
   maintenance tool.

## Architecture

A team of specialist agents, wired as a LangGraph `StateGraph` (see `graph.py`):

```
                 +--direct--> Direct-reply agent ------------------------------> answer
User query -> Router agent
                 +--web-----> Web agent ---------------------------------------> answer
                 |
                 +--knowledge_base--> Retriever -> Generator -> Evaluator --confident--> answer
                                         ^                        |
                                         +---- Rewriter <--retry--+  (up to MAX_RETRIES)
                                                                  |
                                                               give up
                                                                  v
                                                    Gap logger -> Web agent -> flagged answer
```

| Agent | Job |
|---|---|
| Router | Picks knowledge base / web / direct reply (LLM, with rule-based fallback) |
| Retriever | Hybrid search: dense + BM25 + trigram, fused with RRF |
| Generator | Samples N grounded answers |
| Evaluator | Confident only if context was found, the model didn't abstain, and the N answers agree |
| Rewriter | Reformulates the search query and retries the knowledge base |
| Gap logger | Records what the knowledge base is missing (SQLite) |
| Web agent | Answers from live web search, flagged as unverified on fallback |
| Direct | Greetings / small talk, no lookup |

### Hybrid retrieval

Each retriever catches what the others miss:

| Retriever | Good at | Example |
|---|---|---|
| Dense (all-MiniLM-L6-v2) | Meaning, paraphrases | "when must I be back in the hostel" -> curfew |
| BM25 | Exact keywords, numbers, IDs | "rule 6", "48 hours" |
| Trigram (pg_trgm-style) | Typos, spelling variants | "libary", "atendance" |

Results are merged with **Reciprocal Rank Fusion**, which uses ranks rather
than raw scores, so cosine similarity, unbounded BM25 scores and 0-1
trigram overlap can be combined safely. Each retriever applies its own
relevance floor first, so an out-of-scope question comes back with no
context instead of the nearest irrelevant chunks.

## Project layout

```
config.py          - environment/config loading
embeddings.py      - sentence-transformers embeddings
lexical.py         - BM25 and trigram indexes (pure Python)
retrieval.py       - hybrid retriever + Reciprocal Rank Fusion
vectorstore.py     - Chroma wrapper: ingest + dense search
llm.py             - pluggable LLM backend (mock / Anthropic Claude / OpenAI)
confidence.py      - abstention check + answer-agreement confidence scorer
gap_tracker.py     - SQLite-backed knowledge-gap log
search.py          - live web search (DuckDuckGo)
agents/            - one module per agent (router, knowledge, evaluator, rewriter, gap_logger, web, direct)
graph.py           - wires the agents into the LangGraph StateGraph
ingest.py          - CLI to load sample_docs/ into the vector store
app.py             - Streamlit dashboard (chat + agent trace + gap report)
tests/test_retrieval.py   - BM25 / trigram / RRF unit tests (no downloads)
tests/test_graph_smoke.py - end-to-end multi-agent test with the mock LLM
```

## Run modes

**Demo mode (no API key)**: a `MockLLM` answers from retrieved context
directly, and the agents use rule-based routing and rewriting. Good for
verifying the control flow before spending API credits.

```bash
export LLM_PROVIDER=mock
python ingest.py
streamlit run app.py
```

**Production mode**
```bash
export LLM_PROVIDER=anthropic        # or: openai
export ANTHROPIC_API_KEY=sk-ant-...  # or OPENAI_API_KEY
python ingest.py
streamlit run app.py
```
Both modes use the `all-MiniLM-L6-v2` embedding model, downloaded from
Hugging Face (~90MB) on first run.

Re-run `python ingest.py` whenever documents change (or after changing
`EMBEDDING_MODEL`). Retrieval weights and thresholds are tunable in `.env`
-- see `.env.example`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
python ingest.py       # builds the vector store from sample_docs/
streamlit run app.py   # opens the dashboard in your browser

python tests/test_retrieval.py    # retrieval unit tests
python tests/test_graph_smoke.py  # end-to-end multi-agent test (mock LLM)
```

## Run on Windows (one click)

Double-click `run.bat`. On first run it creates the venv, installs
dependencies, copies `.env.example` to `.env` and builds the knowledge
base; after that it just starts the dashboard at http://localhost:8501.

## Run with Docker

```bash
docker build -t rag-agent .
docker run --rm -p 8501:8501 --env-file .env -v rag-agent-data:/app/data rag-agent
```

The container re-ingests `sample_docs/` on every start and keeps the
knowledge-gap log in the `rag-agent-data` volume. The embedding model is
baked into the image, so it starts without downloading anything.

## What to actually build on this for your resume/portfolio

1. Replace `sample_docs/` with a real document set (your university's
   academic policies, your HAL internship domain, a company's product docs)
2. Run it for real usage over a week or two, and screenshot the gap report
   — that's your "before/after" evidence for an interview
3. Swap DuckDuckGo in `search.py` for a more reliable search API
   (Tavily, SerpAPI)
4. Add a real confidence threshold calibration step: log confidence scores
   against human-labeled "was this actually right" judgments, and tune the
   threshold instead of guessing one

## Honest scope note

The confidence evaluator here is a simplified, embedding-based proxy for
semantic entropy (full semantic entropy clusters generations by NLI-based
equivalence, not raw cosine similarity). This simplification is intentional
for a project buildable in a few weeks — it's still a meaningfully harder
and more differentiated project than a standard RAG chatbot, but be ready
to explain this tradeoff honestly if asked about it in an interview.

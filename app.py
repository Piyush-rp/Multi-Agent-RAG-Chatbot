import os
import time
import uuid

import streamlit as st

import gap_tracker
from config import Config
from documents import SUPPORTED_EXTENSIONS, count_pages, iter_pages
from graph import build_graph
from ingest import ingest_directory
from vectorstore import SESSION_PREFIX, VectorStore

APP_NAME = "Multi-Agent Chatbot"
SAMPLE_PDF = "lumora_employee_handbook.pdf"
SAMPLE_PDF_PATH = os.path.join(Config.SAMPLE_DOCS_DIR, SAMPLE_PDF)

SAMPLE_QUESTIONS = {
    "Employee handbook (sample PDF)": [
        "How many days of paid annual leave do employees get, and how many can be carried forward?",
        "What is the daily meal allowance for international travel?",
        "Which days of the week are office days?",
        "When is the performance bonus paid and how large can it be?",
        "How quickly must a lost laptop be reported?",
        "What is the notice period after probation?",
    ],
    "Campus documents": [
        "What are the library opening hours on weekends?",
        "What is the minimum attendance needed to sit the end-semester exam?",
        "What time is the hostel curfew on weekdays?",
    ],
    "Other agents": [
        "Hi! What can you help me with?",
        "Who won the 2022 FIFA World Cup?",
    ],
}

ROUTE_LABELS = {"knowledge_base": "Knowledge base", "web": "Web search", "direct": "Direct reply"}

st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------
# Knowledge base setup
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the sample knowledge base...")
def base_store() -> VectorStore:
    """Sample documents, embedded once per server process. Each browser
    session copies these precomputed embeddings into its own collection."""
    VectorStore.prune_sessions()
    store = VectorStore("base_samples")
    store.reset()
    ingest_directory(store, Config.SAMPLE_DOCS_DIR)
    return store


def reset_session_store():
    store = VectorStore(f"{SESSION_PREFIX}{uuid.uuid4().hex}")
    store.copy_from(base_store())
    st.session_state.vector_store = store
    rebuild_graph()


def rebuild_graph():
    st.session_state.compiled_graph = build_graph(st.session_state.vector_store)


if "vector_store" not in st.session_state:
    reset_session_store()
if "history" not in st.session_state:
    st.session_state.history = []

store: VectorStore = st.session_state.vector_store
sample_sources = set(base_store().sources())


def add_document(uploaded) -> str:
    """Parse, chunk and embed one uploaded file, with progress bars."""
    name, data = uploaded.name, uploaded.getvalue()
    started = time.time()
    total_pages = count_pages(data, name)

    bar = st.progress(0.0, text=f"Reading {name} ...")
    pages = []
    for page_no, text in iter_pages(data, name):
        pages.append((page_no, text))
        if len(pages) % 20 == 0 or len(pages) == total_pages:
            bar.progress(min(page_no / max(total_pages, 1), 1.0),
                         text=f"Reading {name}: page {page_no} of {total_pages}")
    if not pages:
        bar.empty()
        return f"⚠️ {name}: no extractable text (scanned PDFs need OCR first)."

    if name in store.sources():
        store.delete_source(name)  # re-uploading a file replaces it

    embed_started = time.time()

    def on_progress(done, total):
        elapsed = time.time() - embed_started
        remaining = elapsed / done * (total - done) if done else 0
        eta = f" · about {remaining / 60:.0f} min left" if remaining > 90 else (
            f" · about {remaining:.0f}s left" if remaining > 0 else "")
        bar.progress(done / total, text=f"Embedding {name}: {done:,} of {total:,} chunks{eta}")

    n_chunks = store.ingest_pages(name, pages, progress=on_progress)
    bar.empty()
    return (f"✅ {name}: {len(pages):,} pages → {n_chunks:,} chunks "
            f"in {time.time() - started:.0f}s")


# --------------------------------------------------------------------------
# Sidebar: documents
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader(
            "Add documents",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Add to knowledge base", type="primary",
                                          use_container_width=True)
    if submitted and files:
        messages = []
        for f in files:
            try:
                messages.append(add_document(f))
            except Exception as exc:
                messages.append(f"❌ {f.name}: {exc}")
        with st.spinner("Rebuilding search indexes..."):
            rebuild_graph()
        st.session_state.upload_messages = messages
        st.rerun()

    for msg in st.session_state.pop("upload_messages", []):
        st.write(msg)

    st.subheader("In the knowledge base")
    for source, info in store.sources().items():
        col_name, col_btn = st.columns([5, 1])
        tag = " · sample" if source in sample_sources else ""
        pages = f"{info['pages']:,} pages · " if info["pages"] else ""
        col_name.markdown(f"**{source}**  \n<small>{pages}{info['chunks']:,} chunks{tag}</small>",
                          unsafe_allow_html=True)
        if col_btn.button(":material/delete:", key=f"del_{source}", help=f"Remove {source}"):
            store.delete_source(source)
            rebuild_graph()
            st.rerun()

    if st.button("Reset to sample documents", use_container_width=True):
        reset_session_store()
        st.session_state.history = []
        st.rerun()

    if os.path.exists(SAMPLE_PDF_PATH):
        with open(SAMPLE_PDF_PATH, "rb") as f:
            st.download_button("⬇️ Download the sample PDF", f.read(), file_name=SAMPLE_PDF,
                               mime="application/pdf", use_container_width=True)


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title(f"🤖 {APP_NAME}")
st.caption(
    f"LLM: `{Config.LLM_PROVIDER}`  |  Embeddings: `{Config.EMBEDDING_MODEL}`  |  "
    f"Retrieval: dense + BM25 + trigram (RRF)  |  "
    f"Confidence threshold: `{Config.CONFIDENCE_THRESHOLD}`"
)


def ask(query: str):
    with st.spinner("Routing, retrieving, generating, and checking confidence..."):
        try:
            result = st.session_state.compiled_graph.invoke({"query": query})
        except Exception as exc:
            st.error(f"Something went wrong answering that: {exc}")
            return
    st.session_state.history.insert(0, result)


tab_chat, tab_gaps = st.tabs(["💬 Ask a question", "📊 Knowledge gap report"])

with tab_chat:
    doc_count = store.count()
    st.markdown(f"**{len(store.sources())} documents · {doc_count:,} chunks indexed.** "
                "Ask about the sample documents, or upload your own in the sidebar.")

    with st.expander("💡 Sample questions: click one to try it", expanded=not st.session_state.history):
        for group, questions in SAMPLE_QUESTIONS.items():
            st.markdown(f"**{group}**")
            cols = st.columns(2)
            for i, q in enumerate(questions):
                if cols[i % 2].button(q, key=f"sample_{group}_{i}", use_container_width=True):
                    st.session_state.pending_query = q

    with st.form("ask", clear_on_submit=True):
        col_q, col_btn = st.columns([6, 1])
        query = col_q.text_input("Your question", placeholder="Ask anything about the documents...",
                                 label_visibility="collapsed")
        ask_clicked = col_btn.form_submit_button("Ask", type="primary", use_container_width=True,
                                                 disabled=(doc_count == 0))
    if ask_clicked and query.strip():
        ask(query.strip())
    elif "pending_query" in st.session_state:
        ask(st.session_state.pop("pending_query"))

    for result in st.session_state.history:
        route = ROUTE_LABELS.get(result.get("route"), result.get("route", ""))

        with st.container(border=True):
            st.markdown(f"**Q: {result['query']}**")
            if "verdict" in result:
                badge = ("✅ Confident" if result["verdict"] == "confident"
                         else "⚠️ Uncertain (logged as a gap)")
                st.markdown(f"Route: `{route}`  ·  **{badge}**  ·  "
                            f"score: `{result['confidence']:.2f}`")
            else:
                st.markdown(f"Route: `{route}`")
            st.write(result["final_answer"])
            if result.get("retrieved"):
                cited = []
                for r in result["retrieved"]:
                    label = f"{r['source']}" + (f" p. {r['page']}" if r.get("page") else "")
                    if label not in cited:
                        cited.append(label)
                st.caption("Sources: " + " · ".join(cited))
            if "retrieved" in result:
                with st.expander("Retrieved context used"):
                    if result["retrieved"]:
                        for r in result["retrieved"]:
                            scores = "  ·  ".join(f"{k}: `{v}`" for k, v in r["scores"].items())
                            page = f" p. {r['page']}" if r.get("page") else ""
                            st.markdown(f"**Source:** `{r['source']}{page}`  ·  "
                                        f"RRF `{r['rrf_score']}`  ·  {scores}")
                            st.text(r["text"])
                    else:
                        st.text("No documents retrieved.")
            if result.get("trace"):
                with st.expander("Agent trace"):
                    for s in result["trace"]:
                        st.markdown(f"- **{s['agent']}**: {s['detail']}")

with tab_gaps:
    st.subheader("Knowledge gaps the agent has flagged")
    stats = gap_tracker.gap_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total gaps logged", stats["total"])
    c2.metric("Unresolved", stats["unresolved"])
    c3.metric("Resolved", stats["resolved"])

    show_resolved = st.checkbox("Show resolved gaps too", value=False)
    gaps = gap_tracker.list_gaps(include_resolved=show_resolved)

    if not gaps:
        st.info("No knowledge gaps logged yet -- ask some questions the "
                 "documents don't cover to see this fill up.")
    for gap in gaps:
        with st.container(border=True):
            st.markdown(f"**Query:** {gap['query']}")
            st.markdown(f"Confidence: `{gap['confidence']:.2f}`  ·  Logged: {gap['created_at']}")
            st.text(gap["best_answer"])
            if not gap["resolved"]:
                if st.button("Mark as resolved", key=f"resolve_{gap['id']}"):
                    gap_tracker.mark_resolved(gap["id"])
                    st.rerun()

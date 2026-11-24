import os
import re
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

SAMPLE_TITLE = "Lumora Labs employee handbook (6-page PDF, fictional company)"
SAMPLE_QUESTIONS = [
    "How many days of paid annual leave do employees get, and how many can be carried forward?",
    "What is the daily meal allowance for international travel?",
    "Which days of the week are office days?",
    "When is the performance bonus paid and how large can it be?",
    "How quickly must a lost laptop be reported?",
    "What is the notice period after probation?",
    "How many days of sick leave do employees get?",
    "How long is parental leave for the primary caregiver?",
    "What are the core working hours?",
    "Within how many days must expense claims be submitted?",
    "What is the annual learning budget?",
    "How long is the probation period?",
    "What is the minimum password length?",
    "How much health insurance cover does Lumora provide?",
    "What is the home-office allowance for new employees?",
]

# Inline citations like 【file.pdf】 or [file.pdf, p. 3]: sources are shown
# separately under each answer, so strip them from the answer text.
_INLINE_CITATION = re.compile(
    r"\s*(【[^】]*】|\[[^\]]*\.(?:pdf|docx|txt|md)(?:,\s*p\.\s*\d+)?\])", re.I
)


def clean_answer(text: str) -> str:
    return _INLINE_CITATION.sub("", text).strip()

st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------
# Knowledge base setup
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the sample document...")
def sample_store() -> VectorStore:
    """The sample PDF, embedded once per server process. Adding it to a
    session copies these precomputed embeddings instead of re-embedding."""
    VectorStore.prune_sessions()
    store = VectorStore("base_samples")
    store.reset()
    ingest_directory(store, Config.SAMPLE_DOCS_DIR)
    return store


def rebuild_graph():
    st.session_state.compiled_graph = build_graph(st.session_state.vector_store)


def add_sample_document():
    st.session_state.vector_store.copy_from(sample_store())
    rebuild_graph()


if "vector_store" not in st.session_state:
    # Every browser session starts with its own empty knowledge base.
    st.session_state.vector_store = VectorStore(f"{SESSION_PREFIX}{uuid.uuid4().hex}")
    rebuild_graph()
if "history" not in st.session_state:
    st.session_state.history = []

store: VectorStore = st.session_state.vector_store


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

    st.subheader("Sample document")
    st.caption(SAMPLE_TITLE)
    sample_loaded = SAMPLE_PDF in store.sources()
    col_add, col_dl = st.columns(2)
    if col_add.button("Added ✓" if sample_loaded else "Add to knowledge base",
                      disabled=sample_loaded, use_container_width=True, key="add_sample"):
        add_sample_document()
        st.rerun()
    if os.path.exists(SAMPLE_PDF_PATH):
        with open(SAMPLE_PDF_PATH, "rb") as f:
            col_dl.download_button("Download", f.read(), file_name=SAMPLE_PDF,
                                   mime="application/pdf", use_container_width=True)

    sources = store.sources()
    if sources:
        st.subheader("In the knowledge base")
        for source, info in sources.items():
            col_name, col_btn = st.columns([3, 2])
            pages = f"{info['pages']:,} pages · " if info["pages"] else ""
            col_name.markdown(f"**{source}**  \n<small>{pages}{info['chunks']:,} chunks</small>",
                              unsafe_allow_html=True)
            if col_btn.button("Remove", key=f"del_{source}", use_container_width=True):
                store.delete_source(source)
                rebuild_graph()
                st.rerun()


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
    st.markdown(f"**{len(store.sources())} documents · {doc_count:,} chunks indexed.**")

    if doc_count == 0:
        st.info("The knowledge base is empty. Upload your own documents in the sidebar, "
                "or add the sample document.")

    def _on_question():
        """Queue the typed or picked question, then clear the box."""
        q = (st.session_state.get("question_box") or "").strip()
        st.session_state.question_box = None
        if q:
            st.session_state.pending_query = q

    # A selectbox that accepts free text works as a type-ahead: typing filters
    # the cached sample-document questions, and Enter on anything else asks
    # that instead.
    st.selectbox(
        "Your question",
        options=SAMPLE_QUESTIONS,
        index=None,
        key="question_box",
        placeholder="Ask anything about the documents...",
        accept_new_options=True,
        on_change=_on_question,
        label_visibility="collapsed",
    )

    if "pending_query" in st.session_state:
        q = st.session_state.pending_query
        if q in SAMPLE_QUESTIONS and SAMPLE_PDF not in store.sources():
            # A suggested question needs the sample document: add it, then
            # rerun so the sidebar and counts reflect it before answering.
            with st.spinner("Adding the sample document to the knowledge base..."):
                add_sample_document()
            st.rerun()
        st.session_state.pop("pending_query")
        if store.count() == 0:
            st.warning("Add a document to the knowledge base first.")
        else:
            ask(q)

    for result in st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**Q: {result['query']}**")
            st.write(clean_answer(result["final_answer"]))
            details = []
            if result.get("verdict") == "confident" and result.get("retrieved"):
                pages_by_source: dict[str, set] = {}
                for r in result["retrieved"]:
                    pages = pages_by_source.setdefault(r["source"], set())
                    if r.get("page"):
                        pages.add(r["page"])
                cited = [src + (f" p. {', '.join(str(p) for p in sorted(pg))}" if pg else "")
                         for src, pg in pages_by_source.items()]
                details.append("Sources: " + " · ".join(cited))
            if "confidence" in result:
                details.append(f"Confidence: {result['confidence']:.2f}")
            if details:
                st.caption("  ·  ".join(details))

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

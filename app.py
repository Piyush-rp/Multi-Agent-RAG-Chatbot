import html
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
UPLOAD_TYPES = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS]

# Inline citations like 【file.pdf】 or [file.pdf, p. 3]: sources are shown
# separately under each answer, so strip them from the answer text.
_INLINE_CITATION = re.compile(
    r"\s*(【[^】]*】|\[[^\]]*\.(?:pdf|docx|txt|md)(?:,\s*p\.\s*\d+)?\])", re.I
)


def clean_answer(text: str) -> str:
    return _INLINE_CITATION.sub("", text).strip()


st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")

st.markdown("""
<style>
/* Keep the context panel in view while scrolling through answers. */
[data-testid="stColumn"]:has(.st-key-context_panel) {
    position: sticky;
    top: 3.75rem;
    align-self: flex-start;
    max-height: calc(100vh - 5rem);
    overflow-y: auto;
}
.passage {
    white-space: pre-wrap;
    font-size: 0.92rem;
    line-height: 1.5;
}
.welcome-title { text-align: center; margin-bottom: 0.25rem; }
.welcome-sub { text-align: center; opacity: 0.8; font-size: 1.1rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Knowledge base
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


def new_session():
    """Each browser session gets its own empty knowledge base."""
    st.session_state.vector_store = VectorStore(f"{SESSION_PREFIX}{uuid.uuid4().hex}")
    st.session_state.history = []
    st.session_state.context_for = None
    st.session_state.page = "welcome"
    rebuild_graph()


def rebuild_graph():
    st.session_state.compiled_graph = build_graph(st.session_state.vector_store)


def add_sample_document():
    st.session_state.vector_store.copy_from(sample_store())
    rebuild_graph()


if "vector_store" not in st.session_state:
    new_session()

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


def process_uploads(files):
    """Ingest uploaded files; move to the chat page once anything is indexed."""
    messages = []
    for f in files:
        try:
            messages.append(add_document(f))
        except Exception as exc:
            messages.append(f"❌ {f.name}: {exc}")
    with st.spinner("Rebuilding search indexes..."):
        rebuild_graph()
    st.session_state.upload_messages = messages
    if store.count():
        st.session_state.page = "chat"
    st.rerun()


def sample_download_button(label: str, key: str):
    if os.path.exists(SAMPLE_PDF_PATH):
        with open(SAMPLE_PDF_PATH, "rb") as f:
            st.download_button(label, f.read(), file_name=SAMPLE_PDF, mime="application/pdf",
                               use_container_width=True, key=key)


# --------------------------------------------------------------------------
# Start page
# --------------------------------------------------------------------------
def welcome_page():
    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.markdown(f"<h1 class='welcome-title'>👋 Welcome to {APP_NAME}</h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='welcome-sub'>Ask questions about your documents. A team of AI agents "
            "finds the answer, checks how sure it is, and shows you exactly where it came "
            "from.</p>",
            unsafe_allow_html=True,
        )

        f1, f2, f3 = st.columns(3)
        with f1.container(border=True):
            st.markdown("**📚 Documents of any size**")
            st.caption("PDF, Word, text or Markdown files, even ones with thousands of pages.")
        with f2.container(border=True):
            st.markdown("**🔎 Answers with proof**")
            st.caption("Every answer lists its sources, and you can open the exact passages.")
        with f3.container(border=True):
            st.markdown("**🛡️ Honest when unsure**")
            st.caption("If your documents don't cover a question, it says so instead of guessing.")

        st.markdown("### Get started")
        left, right = st.columns(2)
        with left.container(border=True):
            st.markdown("**Upload your documents**")
            with st.form("welcome_upload", clear_on_submit=True, border=False):
                files = st.file_uploader("Choose files", type=UPLOAD_TYPES,
                                         accept_multiple_files=True,
                                         label_visibility="collapsed")
                submitted = st.form_submit_button("Upload and start", type="primary",
                                                  use_container_width=True)
            if submitted:
                if files:
                    process_uploads(files)
                else:
                    st.warning("Choose at least one file first.")
        with right.container(border=True):
            st.markdown("**Or try the sample document**")
            st.caption(SAMPLE_TITLE)
            if st.button("Use the sample document", type="primary", use_container_width=True,
                         key="welcome_sample"):
                with st.spinner("Adding the sample document..."):
                    add_sample_document()
                st.session_state.page = "chat"
                st.rerun()
            sample_download_button("Download the sample PDF", key="welcome_download")

        for msg in st.session_state.pop("upload_messages", []):
            st.write(msg)


# --------------------------------------------------------------------------
# Chat page
# --------------------------------------------------------------------------
def documents_sidebar():
    with st.sidebar:
        st.header("📄 Documents")

        with st.form("upload", clear_on_submit=True):
            files = st.file_uploader("Add documents", type=UPLOAD_TYPES,
                                     accept_multiple_files=True)
            submitted = st.form_submit_button("Add to knowledge base", type="primary",
                                              use_container_width=True)
        if submitted and files:
            process_uploads(files)

        for msg in st.session_state.pop("upload_messages", []):
            st.write(msg)

        st.subheader("Sample document")
        st.caption(SAMPLE_TITLE)
        sample_loaded = SAMPLE_PDF in store.sources()
        col_add, col_dl = st.columns(2)
        if col_add.button("Added ✓" if sample_loaded else "Add", disabled=sample_loaded,
                          use_container_width=True, key="add_sample"):
            add_sample_document()
            st.rerun()
        with col_dl:
            sample_download_button("Download", key="sidebar_download")

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

        st.divider()
        if st.button("Start over", use_container_width=True):
            new_session()
            st.rerun()


def ask(query: str):
    with st.spinner("Routing, retrieving, generating, and checking confidence..."):
        try:
            result = st.session_state.compiled_graph.invoke({"query": query})
        except Exception as exc:
            st.error(f"Something went wrong answering that: {exc}")
            return
    result["id"] = uuid.uuid4().hex
    st.session_state.history.insert(0, result)


def source_summary(result: dict) -> str:
    pages_by_source: dict[str, set] = {}
    for r in result["retrieved"]:
        pages = pages_by_source.setdefault(r["source"], set())
        if r.get("page"):
            pages.add(r["page"])
    return " · ".join(src + (f" p. {', '.join(str(p) for p in sorted(pg))}" if pg else "")
                      for src, pg in pages_by_source.items())


def answer_card(result: dict):
    with st.container(border=True):
        st.markdown(f"**Q: {result['query']}**")
        st.write(clean_answer(result["final_answer"]))
        details = []
        confident = result.get("verdict") == "confident"
        if confident and result.get("retrieved"):
            details.append("Sources: " + source_summary(result))
        if "confidence" in result:
            details.append(f"Confidence: {result['confidence']:.2f}")
        if details:
            st.caption("  ·  ".join(details))
        if result.get("retrieved"):
            is_open = st.session_state.context_for == result["id"]
            label = "Hide context" if is_open else "📄 View context"
            if st.button(label, key=f"ctx_{result['id']}"):
                st.session_state.context_for = None if is_open else result["id"]
                st.rerun()


def reflow(text: str) -> str:
    """Join the hard line breaks PDFs put at the end of every visual line,
    keeping paragraph breaks and numbered section headings."""
    text = re.sub(r"\n(?=\d+\.\s+[A-Z])", "\n\n", text)
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def context_panel(result: dict):
    with st.container(border=True, key="context_panel"):
        head, close = st.columns([5, 1], vertical_alignment="center")
        head.markdown("##### 📄 Source passages")
        if close.button("✕", key="close_context", help="Close"):
            st.session_state.context_for = None
            st.rerun()
        st.caption(f"Q: {result['query']}")
        if result.get("verdict") != "confident":
            st.info("The agent wasn't confident these passages answer the question; "
                    "they are the closest matches it found.")
        for i, r in enumerate(result["retrieved"], start=1):
            page = f" · page {r['page']}" if r.get("page") else ""
            with st.container(border=True):
                st.markdown(f"**{i}. {r['source']}{page}**")
                passage = html.escape(reflow(r["text"]))
                # Stop Markdown turning "3. Leave Policy" into a list item.
                passage = re.sub(r"(^|\n)(\d+)\. ", r"\1\2&#46; ", passage)
                st.markdown(f"<div class='passage'>{passage}</div>",
                            unsafe_allow_html=True)


def chat_page():
    documents_sidebar()

    st.title(f"🤖 {APP_NAME}")
    st.caption(
        f"LLM: `{Config.LLM_PROVIDER}`  |  Embeddings: `{Config.EMBEDDING_MODEL}`  |  "
        f"Retrieval: dense + BM25 + trigram (RRF)  |  "
        f"Confidence threshold: `{Config.CONFIDENCE_THRESHOLD}`"
    )

    tab_chat, tab_gaps = st.tabs(["💬 Ask a question", "📊 Knowledge gap report"])

    with tab_chat:
        selected = next((r for r in st.session_state.history
                         if r.get("id") == st.session_state.context_for), None)
        if selected:
            chat_col, panel_col = st.columns([3, 2], gap="medium")
        else:
            chat_col, panel_col = st.container(), None

        with chat_col:
            doc_count = store.count()
            st.markdown(f"**{len(store.sources())} documents · {doc_count:,} chunks indexed.**")
            if doc_count == 0:
                st.info("The knowledge base is empty. Upload documents or add the sample "
                        "document in the sidebar.")

            def _on_question():
                """Queue the typed or picked question, then clear the box."""
                q = (st.session_state.get("question_box") or "").strip()
                st.session_state.question_box = None
                if q:
                    st.session_state.pending_query = q

            # A selectbox that accepts free text works as a type-ahead: typing
            # filters the cached sample-document questions, and Enter on
            # anything else asks that instead.
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
                    with st.spinner("Adding the sample document to the knowledge base..."):
                        add_sample_document()
                    st.rerun()
                st.session_state.pop("pending_query")
                if store.count() == 0:
                    st.warning("Add a document to the knowledge base first.")
                else:
                    ask(q)

            for result in st.session_state.history:
                answer_card(result)

        if panel_col is not None:
            with panel_col:
                context_panel(selected)

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


if st.session_state.page == "welcome":
    welcome_page()
else:
    chat_page()

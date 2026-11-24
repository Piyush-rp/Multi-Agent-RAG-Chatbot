import streamlit as st
from config import Config
from vectorstore import VectorStore
from graph import build_graph
import gap_tracker

st.set_page_config(page_title="Self-Improving RAG Agent", layout="wide")

st.title("Self-Improving Knowledge-Gap-Aware RAG Agent")
st.caption(
    f"LLM: `{Config.LLM_PROVIDER}`  |  Embeddings: `{Config.EMBEDDING_MODEL}`  |  "
    f"Retrieval: dense + BM25 + trigram (RRF)  |  "
    f"Confidence threshold: `{Config.CONFIDENCE_THRESHOLD}`"
)

ROUTE_LABELS = {"knowledge_base": "Knowledge base", "web": "Web search", "direct": "Direct reply"}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "compiled_graph" not in st.session_state:
    st.session_state.compiled_graph = build_graph(st.session_state.vector_store)
if "history" not in st.session_state:
    st.session_state.history = []

doc_count = st.session_state.vector_store.count()
if doc_count == 0:
    # Hosted deployments (e.g. Streamlit Community Cloud) have no separate
    # build step to run ingest.py, so index sample_docs/ on first load.
    from ingest import load_documents
    docs = load_documents(Config.SAMPLE_DOCS_DIR)
    if docs:
        with st.spinner("Indexing sample documents into the knowledge base..."):
            st.session_state.vector_store.ingest_documents(docs)
        doc_count = st.session_state.vector_store.count()
if doc_count == 0:
    st.warning(
        "The vector store is empty. Add .txt files to sample_docs/ and run "
        "`python ingest.py` to load them into the knowledge base."
    )

tab_chat, tab_gaps = st.tabs(["Ask a question", "Knowledge gap report"])

with tab_chat:
    st.subheader(f"Knowledge base: {doc_count} chunks indexed")

    query = st.text_input("Ask something about the ingested documents:")
    ask_clicked = st.button("Ask", type="primary", disabled=(doc_count == 0))

    if ask_clicked and query.strip():
        with st.spinner("Routing, retrieving, generating, and checking confidence..."):
            result = st.session_state.compiled_graph.invoke({"query": query})
        st.session_state.history.insert(0, result)

    for result in st.session_state.history:
        route = ROUTE_LABELS.get(result.get("route"), result.get("route", ""))

        with st.container(border=True):
            st.markdown(f"**Q: {result['query']}**")
            if "verdict" in result:
                badge = ("Confident" if result["verdict"] == "confident"
                         else "Uncertain (logged as a gap)")
                st.markdown(f"Route: `{route}`  ·  **{badge}**  ·  "
                            f"score: `{result['confidence']:.2f}`")
            else:
                st.markdown(f"Route: `{route}`")
            st.write(result["final_answer"])
            if "retrieved" in result:
                with st.expander("Retrieved context used"):
                    if result["retrieved"]:
                        for r in result["retrieved"]:
                            scores = "  ·  ".join(f"{k}: `{v}`" for k, v in r["scores"].items())
                            st.markdown(f"**Source:** `{r['source']}`  ·  RRF `{r['rrf_score']}`  ·  {scores}")
                            st.text(r["text"])
                    else:
                        st.text("No documents retrieved.")

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

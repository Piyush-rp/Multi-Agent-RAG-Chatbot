"""Run this once (or whenever your documents change) to (re)build the
vector store from everything in sample_docs/ (PDF, DOCX, TXT, MD).

The Streamlit app does this automatically on first load, so running it
by hand is only needed for the command-line / Docker workflow."""
import sys
from config import Config
from documents import load_directory
from vectorstore import VectorStore


def load_documents(directory: str) -> dict:
    """{filename: [(page, text), ...]} for every supported file."""
    return load_directory(directory)


def ingest_directory(store: VectorStore, directory: str, progress=None) -> int:
    total = 0
    for filename, pages in load_documents(directory).items():
        total += store.ingest_pages(filename, pages, progress=progress)
    return total


def main():
    print(f"Loading documents from {Config.SAMPLE_DOCS_DIR} ...")
    docs = load_documents(Config.SAMPLE_DOCS_DIR)
    if not docs:
        print("No supported documents found. Add some to sample_docs/ first.")
        sys.exit(1)

    print(f"Found {len(docs)} document(s): {', '.join(docs.keys())}")
    store = VectorStore()
    store.reset()
    n_chunks = ingest_directory(store, Config.SAMPLE_DOCS_DIR)
    print(f"Ingested {n_chunks} chunks into the vector store "
          f"(embedding model: {Config.EMBEDDING_MODEL}).")
    print(f"Vector store persisted at: {Config.CHROMA_DIR}")


if __name__ == "__main__":
    main()

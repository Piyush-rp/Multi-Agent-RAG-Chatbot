"""Run this once (or whenever your documents change) to (re)build the
vector store from everything in sample_docs/."""
import os
import sys
from config import Config
from vectorstore import VectorStore


def load_documents(directory: str) -> dict:
    docs = {}
    if not os.path.isdir(directory):
        print(f"No such directory: {directory}")
        return docs
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs[filename] = f.read()
    return docs


def main():
    print(f"Loading documents from {Config.SAMPLE_DOCS_DIR} ...")
    docs = load_documents(Config.SAMPLE_DOCS_DIR)
    if not docs:
        print("No .txt documents found. Add some to sample_docs/ first.")
        sys.exit(1)

    print(f"Found {len(docs)} document(s): {', '.join(docs.keys())}")
    store = VectorStore()
    store.reset()
    n_chunks = store.ingest_documents(docs)
    print(f"Ingested {n_chunks} chunks into the vector store "
          f"(embedding model: {Config.EMBEDDING_MODEL}).")
    print(f"Vector store persisted at: {Config.CHROMA_DIR}")


if __name__ == "__main__":
    main()

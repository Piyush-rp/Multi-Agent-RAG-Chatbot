FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# Hugging Face Spaces (and other hardened hosts) run the container as a
# non-root user with UID 1000, so the app directory must be owned by it.
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA (~2GB+), which
# sentence-transformers doesn't need here.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers start offline.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY . .
RUN mkdir -p /app/data && chown -R user:user /app
USER user

EXPOSE 8501

# Re-ingest sample_docs/ on every start so document changes are picked up;
# mount a volume on /app/data to keep the knowledge-gap log across runs.
CMD ["sh", "-c", "python ingest.py && streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]

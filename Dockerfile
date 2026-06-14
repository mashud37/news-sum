FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
ENV HF_HOME=/app/.cache/huggingface
# CPU-only torch first, from the dedicated index, so sentence-transformers does
# not pull the ~2.5 GB CUDA build from PyPI.
RUN pip install --no-cache-dir "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY . .
ENV JOB=ingest_rss
CMD ["sh", "-c", "python ${JOB}.py"]

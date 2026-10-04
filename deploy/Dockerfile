# DocuMind on Hugging Face Spaces (Docker SDK). Built by deploy/build_space.py, which puts this
# file at the root of the Space next to the code and the prebuilt search index.
FROM python:3.12-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
RUN pip install --no-cache-dir uv==0.12.5

USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# Dependencies first, so code changes don't reinstall them. CPU-only PyTorch (see pyproject).
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Bake the models into the image, so a cold start doesn't download them.
RUN uv run --no-sync python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user . .

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "streamlit", "run", "ui/app.py", \
     "--server.port", "7860", "--server.address", "0.0.0.0"]

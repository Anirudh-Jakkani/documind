"""Build and upload the Hugging Face Space.

    uv run python -m deploy.build_space                 # build deploy/space/ only
    uv run python -m deploy.build_space --upload        # build and upload (needs HF_TOKEN)
    uv run python -m deploy.build_space --index-only    # refresh deploy/index (Streamlit Cloud)

The Space gets: every file tracked by git, the Dockerfile, a README with the Space's settings
block, and the prebuilt search index for the shipped configuration (section chunks, BM25 and
the bge-small vectors). The index can't be rebuilt on the Space because RBI's server blocks
scripted PDF downloads.

The answer model's key is NOT uploaded: add GEMINI_API_KEY as a secret in the Space settings.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from documind.config import ROOT_DIR, Settings
from documind.retrieval.search import collection_name, index_paths

SPACE_DIR = ROOT_DIR / "deploy" / "space"
SPACE_HEADER = """---
title: DocuMind
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Questions about RBI regulations, answered with citations
---

"""


def copy_tracked_files(target: Path) -> int:
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT_DIR, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for name in files:
        if name.startswith(("deploy/", ".github/", ".claude/")):
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT_DIR / name, destination)
    return len(files)


def copy_index(index_target: Path, settings: Settings) -> int:
    """Copy the shipped index: chunks and BM25 files, and only the vectors of the shipped
    collection (the experiment collections stay local)."""
    strategy = settings.chunk_strategy
    source = index_paths(settings, strategy)
    destination = index_target / strategy
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source["chunks"], destination / "chunks.jsonl")
    shutil.copy2(source["bm25"], destination / "bm25.pkl.gz")

    name = collection_name(strategy, settings.embedding_model)
    src = QdrantClient(path=str(source["qdrant"]))
    dst = QdrantClient(path=str(index_target / "qdrant"))
    size = src.get_collection(name).config.params.vectors.size
    dst.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
    offset, copied = None, 0
    while True:
        points, offset = src.scroll(name, limit=500, offset=offset, with_vectors=True)
        dst.upsert(
            name, points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points]
        )
        copied += len(points)
        if offset is None:
            break
    count = dst.count(name).count
    src.close()
    dst.close()
    if count != copied:
        raise RuntimeError(f"copied {copied} vectors but the new index has {count}")
    return count


def build() -> Path:
    settings = Settings()
    if SPACE_DIR.exists():
        shutil.rmtree(SPACE_DIR)
    SPACE_DIR.mkdir(parents=True)
    n_files = copy_tracked_files(SPACE_DIR)
    shutil.copy2(ROOT_DIR / "deploy" / "Dockerfile", SPACE_DIR / "Dockerfile")
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    (SPACE_DIR / "README.md").write_text(SPACE_HEADER + readme, encoding="utf-8")
    vectors = copy_index(SPACE_DIR / "data" / "index", settings)
    total = sum(f.stat().st_size for f in SPACE_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"Built {SPACE_DIR}: {n_files} tracked files, {vectors:,} vectors, {total:.1f} MB")
    return SPACE_DIR


def upload(folder: Path, repo_id: str, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(folder),
        commit_message="Deploy DocuMind",
        delete_patterns=["*"],  # the Space mirrors the build exactly
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and upload the Hugging Face Space")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="write the shipped index to deploy/index (committed, for Streamlit Cloud)",
    )
    parser.add_argument("--space", default="Anijack/documind", help="user/space-name")
    args = parser.parse_args()

    if args.index_only:
        settings = Settings()
        target = ROOT_DIR / "deploy" / "index"
        if settings.index_dir.resolve() == target.resolve():
            print("No local index in data/index to copy from. Build it first.")
            return 1
        if target.exists():
            shutil.rmtree(target)
        print(f"Wrote {copy_index(target, settings):,} vectors to {target}")
        return 0

    folder = build()
    if args.upload:
        import os

        from dotenv import dotenv_values

        token = os.environ.get("HF_TOKEN") or dotenv_values(ROOT_DIR / ".env").get("HF_TOKEN")
        if not token:
            print("Set HF_TOKEN in .env to upload.")
            return 1
        url = upload(folder, args.space, token)
        print(f"Uploaded: {url}\nAdd GEMINI_API_KEY under Settings → Variables and secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

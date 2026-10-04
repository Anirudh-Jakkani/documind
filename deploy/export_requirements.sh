#!/usr/bin/env sh
# Regenerate requirements.txt (used by Streamlit Community Cloud) from uv.lock.
# PyTorch's CPU index is added so pip can find the "+cpu" build pinned in the lock file.
set -e
{
  echo "--extra-index-url https://download.pytorch.org/whl/cpu"
  uv export --no-dev --no-hashes --no-emit-project --no-header --format requirements-txt
} > requirements.txt

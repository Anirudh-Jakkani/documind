"""Logging: readable lines on the console, and one JSON line per question in a file.

The request log (logs/requests.jsonl by default) records what was asked, whether it was
answered, which sources were cited, timings, tokens and whether the cache answered it. It is
rotated at REQUEST_LOG_MAX_BYTES so a long-running demo can't fill the disk.
"""

import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

REQUEST_LOG_MAX_BYTES = 5_000_000
_write_lock = threading.Lock()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_request(path: Path | None, record: dict) -> None:
    """Append one JSON line; rotate the file to *.1 when it gets too big."""
    if path is None:
        return
    record = {"time": datetime.now(UTC).isoformat(timespec="seconds"), **record}
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > REQUEST_LOG_MAX_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

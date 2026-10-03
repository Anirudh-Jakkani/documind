"""Collect the PDFs listed in data/documents.csv into data/raw/.

RBI's document server shows a bot check to scripts, so the PDFs are downloaded in a normal
browser (open docs/download_links.html and click each link), then imported from that folder:

    uv run python -m documind.ingest.download --from-folder "C:/Users/<you>/Downloads"

Direct download is still tried first when no folder is given (it works for sources without
a bot check):

    uv run python -m documind.ingest.download

Each file is checked to be a real PDF and saved as data/raw/<doc_id>.pdf. A manifest with
sizes and SHA-256 hashes is written to data/raw/manifest.csv.
"""

import argparse
import csv
import hashlib
import html
import shutil
import sys
import time
from pathlib import Path

import httpx

from documind.config import ROOT_DIR, get_settings

USER_AGENT = "DocuMind/0.1 (educational RAG project; polite single-threaded downloader)"
DELAY_SECONDS = 1.0


class BotCheckError(RuntimeError):
    """The server returned a bot-check page instead of the file."""


def load_document_list(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def source_filename(doc: dict) -> str:
    """The file name the browser saves the PDF as, e.g. 169MD.PDF."""
    return doc["pdf_url"].rsplit("/", 1)[-1]


def is_pdf(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(4) == b"%PDF"


def find_in_folder(folder: Path, name: str) -> Path | None:
    """Find a downloaded file by name, ignoring case and browser suffixes like ' (1)'."""
    stem, suffix = name.rsplit(".", 1)
    candidates = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == f".{suffix.lower()}"
        and (p.stem.lower() == stem.lower() or p.stem.lower().startswith(stem.lower() + " ("))
    ]
    pdfs = [p for p in candidates if is_pdf(p)]
    return max(pdfs, key=lambda p: p.stat().st_mtime) if pdfs else None


def fetch(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        if b"captcha" in response.content.lower() or b"TSPD" in response.content:
            raise BotCheckError("the server returned a bot check instead of the PDF")
        raise RuntimeError("response is not a PDF")
    return response.content


def write_link_page(documents: list[dict], target: Path) -> None:
    rows = "\n".join(
        f'<li><a href="{html.escape(d["pdf_url"])}" target="_blank">'
        f'{html.escape(d["title"])}</a> <small>{html.escape(source_filename(d))}, '
        f'{d["size_kb"]} KB</small></li>'
        for d in documents
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "<!doctype html><meta charset='utf-8'><title>DocuMind: documents to download</title>"
        "<style>body{font:15px/1.6 system-ui;max-width:900px;margin:40px auto;padding:0 16px}"
        "small{color:#666}li{margin:6px 0}a:visited{color:#888}</style>"
        f"<h1>Download {len(documents)} RBI documents</h1>"
        "<p>Click each link. If a PDF opens in the browser instead of downloading, press "
        "Ctrl+S and save it with its suggested name. Visited links turn grey. Then run "
        "<code>uv run python -m documind.ingest.download --from-folder &lt;your Downloads "
        "folder&gt;</code>.</p>"
        f"<ol>{rows}</ol>",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-folder", type=Path, help="import PDFs already downloaded here")
    parser.add_argument("--force", action="store_true", help="replace files already in data/raw")
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    documents = load_document_list(settings.data_dir / "documents.csv")
    link_page = ROOT_DIR / "docs" / "download_links.html"

    missing: list[str] = []
    client = None if args.from_folder else httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
    )
    try:
        for i, doc in enumerate(documents, 1):
            target = raw_dir / f"{doc['doc_id']}.pdf"
            label = f"[{i:>2}/{len(documents)}] {doc['doc_id']}"
            if target.exists() and is_pdf(target) and not args.force:
                print(f"{label} already in data/raw")
                continue

            if args.from_folder:
                found = find_in_folder(args.from_folder, source_filename(doc))
                if found is None:
                    print(f"{label} not found ({source_filename(doc)})")
                    missing.append(doc["doc_id"])
                    continue
                shutil.copyfile(found, target)
                print(f"{label} imported from {found.name}")
                continue

            try:
                target.write_bytes(fetch(client, doc["pdf_url"]))
                print(f"{label} downloaded")
                time.sleep(DELAY_SECONDS)
            except BotCheckError:
                write_link_page(documents, link_page)
                print(
                    f"\nThe document server shows a bot check to scripts, so stopping.\n"
                    f"Open {link_page} in your browser, download the PDFs, then run:\n"
                    f'  uv run python -m documind.ingest.download --from-folder "<Downloads>"'
                )
                return 2
            except (httpx.HTTPError, RuntimeError) as error:
                print(f"{label} FAILED: {error}")
                missing.append(doc["doc_id"])
    finally:
        if client:
            client.close()

    manifest = []
    for doc in documents:
        path = raw_dir / f"{doc['doc_id']}.pdf"
        if path.exists():
            content = path.read_bytes()
            manifest.append(
                {
                    "doc_id": doc["doc_id"],
                    "file": path.name,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    with (raw_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "file", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(manifest)

    total_mb = sum(m["bytes"] for m in manifest) / 1024 / 1024
    print(f"\n{len(manifest)} of {len(documents)} documents ready ({total_mb:.1f} MB)")
    if missing:
        print("Missing:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

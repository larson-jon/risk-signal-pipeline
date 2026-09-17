"""Download the all-MiniLM-L6-v2 sentence-transformers model to a local folder.

Why this exists: on networks where huggingface.co (and its cdn-lfs CDN) is
blocked or returning 503s, the topic-clustering stage cannot download its
embedding model on demand. This script pulls the complete sentence-transformers
file set from a community GitHub mirror into `all-MiniLM-L6-v2-local`, which
topic_clustering.py discovers automatically and loads fully offline.

NOTE: This pulls from a third-party community mirror, not the official
Hugging Face repo. The weights are the standard public MiniLM model. If your
environment requires provenance guarantees, obtain the model from an approved
internal source and drop it in `all-MiniLM-L6-v2-local/` instead.

Usage:
    python download_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

# Community GitHub mirror containing the complete sentence-transformers layout.
GH_OWNER = "henrytanner52"
GH_REPO = "all-MiniLM-L6-v2"
GH_BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}"
TREE_API = (
    f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/git/trees/{GH_BRANCH}?recursive=1"
)

MODEL_DIR = Path(__file__).parent / "all-MiniLM-L6-v2-local"
REQUEST_TIMEOUT = 30
USER_AGENT = "risk-signal-pipeline/1.0"

# Files we skip (docs, not needed to load the model).
SKIP = {"README.md"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _list_files(session: requests.Session) -> list[dict]:
    r = session.get(TREE_API, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return [
        item for item in r.json().get("tree", [])
        if item["type"] == "blob" and item["path"] not in SKIP
    ]


def _download_file(session: requests.Session, rel_path: str, size: int) -> None:
    dest = MODEL_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  [cached] {rel_path}")
        return
    url = f"{RAW_BASE}/{rel_path}"
    label = f"({size/1_000_000:.1f} MB)" if size > 1_000_000 else ""
    print(f"  [get] {rel_path} {label}".rstrip())
    with session.get(url, timeout=REQUEST_TIMEOUT * 3, stream=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def main() -> int:
    session = _session()
    print(f"[model] listing files from {GH_OWNER}/{GH_REPO}@{GH_BRANCH}...")
    try:
        files = _list_files(session)
    except requests.RequestException as exc:
        print(f"[model] ERROR listing files: {exc}")
        return 1

    print(f"[model] downloading {len(files)} files -> {MODEL_DIR}")
    for item in files:
        try:
            _download_file(session, item["path"], item.get("size", 0))
        except requests.RequestException as exc:
            print(f"[model] ERROR downloading {item['path']}: {exc}")
            return 1

    print(f"\n[model] done -> {MODEL_DIR}")
    print("topic_clustering.py will now discover and use this model offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

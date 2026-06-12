# Provides small shared helpers for hashing, directories, and text normalization.
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def md5_text(text: str) -> str:
    """Return the MD5 hex digest for UTF-8 encoded text."""

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def md5_file(path: Path) -> str:
    """Return the MD5 hex digest for a file by reading it in chunks."""

    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> None:
    """Create a directory and any missing parents if they do not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """Remove lightweight markup and normalize whitespace for chunk text."""

    text = re.sub(r"<[^>]+>", "", text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Loads and saves ChunkRecord objects from final chunk JSON files.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from offline_index.schema import ChunkRecord
from offline_index.utils import ensure_dir


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Load chunk records from a chunk JSON file."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    raw_chunks = _extract_raw_chunks(payload)
    chunks: list[ChunkRecord] = []
    for index, item in enumerate(raw_chunks):
        _validate_raw_chunk(item, index, path)
        chunks.append(ChunkRecord.model_validate(item))
    return chunks


def save_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    """Save chunk records to a UTF-8 JSON chunk file."""

    ensure_dir(path.parent)
    payload = [chunk.model_dump() for chunk in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_raw_chunks(payload: Any) -> list[Any]:
    """Extract a raw chunk list from list payloads or dictionaries with a chunks field."""

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    raise ValueError("chunk JSON must be a list or a dict containing a 'chunks' list")


def _validate_raw_chunk(item: Any, index: int, path: Path) -> None:
    """Validate that one raw chunk contains the fields required by ChunkRecord."""

    if not isinstance(item, dict):
        raise ValueError(f"chunk #{index} in {path} must be an object")
    missing = [field for field in ("id", "document", "metadata") if field not in item]
    if missing:
        raise ValueError(f"chunk #{index} in {path} is missing required fields: {', '.join(missing)}")

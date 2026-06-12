# Wraps ChromaDB client, collection, add, delete, and query operations.
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from offline_index.schema import ChunkRecord
from offline_index.utils import ensure_dir


def _prepare_windows_sqlite_dll_path() -> None:
    """Add conda SQLite DLL directories before importing ChromaDB on Windows."""

    if os.name != "nt":
        return
    candidates = [
        Path(sys.prefix) / "Library" / "bin",
    ]
    if len(Path(sys.prefix).parents) >= 2:
        candidates.append(Path(sys.prefix).parents[1] / "Library" / "bin")
    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except (FileNotFoundError, OSError):
                pass


_prepare_windows_sqlite_dll_path()
import chromadb  # noqa: E402


def get_chroma_client(persist_dir: Path):
    """Create a persistent ChromaDB client rooted at persist_dir."""

    ensure_dir(Path(persist_dir))
    return chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))


def get_or_create_collection(client, collection_name: str):
    """Return an existing Chroma collection or create it."""

    return client.get_or_create_collection(name=collection_name)


def reset_collection(client, collection_name: str):
    """Delete and recreate a Chroma collection."""

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(name=collection_name)


def add_chunks(collection, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
    """Add chunk records and their embeddings to a Chroma collection."""

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    ids = [chunk.id for chunk in chunks]
    documents = [chunk.document for chunk in chunks]
    metadatas = [_sanitize_metadata(chunk.metadata) for chunk in chunks]
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        raise ValueError("ids, documents, metadatas, and embeddings length mismatch")
    if not chunks:
        return
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def delete_by_doc_id(collection, doc_id: str) -> None:
    """Delete all chunks in a Chroma collection for one doc_id."""

    collection.delete(where={"doc_id": doc_id})


def query(collection, query_embedding: list[float], top_k: int = 5, where: dict | None = None):
    """Query a Chroma collection by one embedding and optional metadata filter."""

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert metadata to Chroma-compatible flat scalar values."""

    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = ""
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False)
    return sanitized


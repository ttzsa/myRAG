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


def sync_chunks_by_location(collection, chunks: list[ChunkRecord], embedder) -> dict[str, int]:
    """Synchronize chunks by meta_location and content_md5 before embedding new content."""

    added = 0
    updated = 0
    skipped = 0
    deleted = 0
    chunks_to_add: list[ChunkRecord] = []
    if not chunks:
        return {"added": 0, "updated": 0, "skipped": 0, "deleted": 0, "embedded": 0}

    doc_id = _doc_id_for_chunks(chunks)
    existing_by_location = _existing_chunks_by_location(collection, doc_id)
    current_locations: set[str] = set()
    ids_to_delete: list[str] = []

    for chunk in chunks:
        meta_location = _require_metadata(chunk, "meta_location")
        content_md5 = _require_metadata(chunk, "content_md5")
        if meta_location in current_locations:
            raise ValueError(f"duplicate meta_location in current chunks: {meta_location}")
        current_locations.add(meta_location)
        existing = existing_by_location.get(meta_location)
        if existing is None:
            chunks_to_add.append(chunk)
            added += 1
            continue
        if existing.get("content_md5") == content_md5:
            skipped += 1
            continue
        ids_to_delete.append(str(existing["id"]))
        chunks_to_add.append(chunk)
        added += 1
        updated += 1

    for meta_location, existing in existing_by_location.items():
        if meta_location not in current_locations:
            ids_to_delete.append(str(existing["id"]))
            deleted += 1

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    embeddings_to_add = embedder.embed_documents([chunk.document for chunk in chunks_to_add])
    add_chunks(collection, chunks_to_add, embeddings_to_add)
    return {"added": added, "updated": updated, "skipped": skipped, "deleted": deleted, "embedded": len(chunks_to_add)}


def _doc_id_for_chunks(chunks: list[ChunkRecord]) -> str:
    """Return the single doc_id shared by a group of chunks."""

    doc_ids = {_require_metadata(chunk, "doc_id") for chunk in chunks}
    if len(doc_ids) != 1:
        raise ValueError("sync_chunks_by_location requires chunks from exactly one doc_id")
    return next(iter(doc_ids))


def _existing_chunks_by_location(collection, doc_id: str) -> dict[str, dict[str, str]]:
    """Fetch existing chunk metadata for one doc_id and index it by meta_location."""

    existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
    existing_metadatas = existing.get("metadatas", []) if isinstance(existing, dict) else []
    by_location: dict[str, dict[str, str]] = {}
    for item_id, metadata in zip(existing_ids, existing_metadatas):
        if not isinstance(metadata, dict):
            continue
        meta_location = str(metadata.get("meta_location", ""))
        if not meta_location:
            continue
        by_location[meta_location] = {
            "id": str(item_id),
            "content_md5": str(metadata.get("content_md5", "")),
        }
    return by_location


def _require_metadata(chunk: ChunkRecord, key: str) -> str:
    """Read a required string metadata value from one chunk."""

    value = str(chunk.metadata.get(key, ""))
    if not value:
        raise ValueError(f"chunk {chunk.id} missing metadata.{key}")
    return value


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

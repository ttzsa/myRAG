# Loads, saves, queries, and upserts the JSON processed_pdfs registry.
from __future__ import annotations

import json
from pathlib import Path

from offline_index.schema import DocumentManifest, DocumentRecord
from offline_index.utils import ensure_dir


def load_manifest(path: Path) -> DocumentManifest:
    """Load a manifest from UTF-8 JSON or return an empty manifest if missing."""

    if not path.exists():
        return DocumentManifest()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return DocumentManifest.model_validate(payload)


def save_manifest(manifest: DocumentManifest, path: Path) -> None:
    """Save a manifest to UTF-8 JSON using a stable human-readable format."""

    ensure_dir(path.parent)
    payload = manifest.model_dump()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_by_md5(manifest: DocumentManifest, pdf_md5: str) -> DocumentRecord | None:
    """Find a document in the manifest by PDF content MD5."""

    return manifest.find_by_md5(pdf_md5)


def find_by_source_path(manifest: DocumentManifest, source_path: str) -> DocumentRecord | None:
    """Find a document in the manifest by source PDF path."""

    return manifest.find_by_source_path(source_path)


def upsert_document(manifest: DocumentManifest, record: DocumentRecord) -> DocumentRecord:
    """Insert or replace a document record matched by source path, then doc_id, then MD5."""

    for index, existing in enumerate(manifest.documents):
        if (
            existing.source_path == record.source_path
            or existing.doc_id == record.doc_id
            or existing.pdf_md5 == record.pdf_md5
        ):
            manifest.documents[index] = record
            return record
    manifest.documents.append(record)
    return record

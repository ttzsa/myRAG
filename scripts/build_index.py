# Command-line entry point for embedding preview chunks and writing them to ChromaDB.
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from offline_index.chunk_loader import load_chunks
from offline_index.chunk_loader import save_chunks
from offline_index.config_loader import load_config, resolve_value
from offline_index.document_manifest import load_manifest, save_manifest, upsert_document
from offline_index.embedder import create_embedder
from offline_index.offline_pipeline import build_chunks_from_mineru_content, create_visual_summarizer
from offline_index.schema import ChunkRecord


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building the ChromaDB index."""

    parser = argparse.ArgumentParser(description="Embed chunks_preview JSON and write chunks to ChromaDB.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--chunks-path", type=Path)
    group.add_argument("--chunks-dir", type=Path)
    group.add_argument("--content-list-v2", type=Path)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--file-name", default="")
    parser.add_argument("--doc-id", default="")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--output-preview", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--persist-dir", type=Path, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--embedder", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--rebuild-doc", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Load chunks, embed documents, write Chroma records, update manifest, and print counts."""

    args = parse_args()
    config = load_config(args.env_file)
    args.chunks_dir = args.chunks_dir or config.paths.debug_dir
    args.manifest_path = resolve_value(args.manifest_path, config.paths.manifest_path)
    args.persist_dir = resolve_value(args.persist_dir, config.chroma.persist_dir)
    args.collection = resolve_value(args.collection, config.chroma.collection)
    args.embedder = resolve_value(args.embedder, config.embedding.provider)
    args.chunk_size = resolve_value(args.chunk_size, config.chunking.chunk_size)
    args.chunk_overlap = resolve_value(args.chunk_overlap, config.chunking.chunk_overlap)
    chunk_paths: list[Path] = []
    if args.content_list_v2:
        chunks = _build_chunks_from_content_list(args, config)
        if args.output_preview:
            save_chunks(args.output_preview, chunks)
            chunk_paths = [args.output_preview.resolve()]
    else:
        chunk_paths = _collect_chunk_paths(args)
        chunks = _load_all_chunks(chunk_paths)
    chunks_by_doc = _group_chunks_by_doc_id(chunks)
    from offline_index.chroma_store import add_chunks, delete_by_doc_id, get_chroma_client, get_or_create_collection, reset_collection

    embedder = create_embedder(args.embedder, config.embedding)
    client = get_chroma_client(args.persist_dir)
    collection = reset_collection(client, args.collection) if args.reset else get_or_create_collection(client, args.collection)

    if args.rebuild_doc:
        for doc_id in chunks_by_doc:
            delete_by_doc_id(collection, doc_id)

    total_counts: Counter = Counter()
    embedding_dimension: int | None = None
    manifest = load_manifest(args.manifest_path)
    for doc_id, doc_chunks in chunks_by_doc.items():
        embeddings = embedder.embed_documents([chunk.document for chunk in doc_chunks])
        if embeddings:
            current_dimension = len(embeddings[0])
            if embedding_dimension is None:
                embedding_dimension = current_dimension
            elif embedding_dimension != current_dimension:
                raise ValueError(
                    f"embedding dimension mismatch while indexing: {embedding_dimension} != {current_dimension}"
                )
        add_chunks(collection, doc_chunks, embeddings)
        counts = Counter(chunk.metadata.get("chunk_type", "") for chunk in doc_chunks)
        total_counts.update(counts)
        _update_manifest_doc(manifest, doc_id, doc_chunks, counts)

    save_manifest(manifest, args.manifest_path)
    print(f"chunk files: {len(chunk_paths)}")
    print(f"documents indexed: {len(chunks_by_doc)}")
    print(f"chunks written: {len(chunks)}")
    print(f"text chunks: {total_counts.get('text', 0)}")
    print(f"image chunks: {total_counts.get('image', 0)}")
    print(f"table chunks: {total_counts.get('table', 0)}")
    print(f"embedding dimension: {embedding_dimension if embedding_dimension is not None else 0}")
    print(f"persist dir: {args.persist_dir.resolve()}")
    print(f"collection: {args.collection}")
    return 0


def _build_chunks_from_content_list(args: argparse.Namespace, config) -> list[ChunkRecord]:
    """Build chunks inline from one MinerU content_list_v2 input before embedding."""

    if not args.images_dir:
        raise ValueError("--images-dir is required when using --content-list-v2")
    if not args.file_name:
        raise ValueError("--file-name is required when using --content-list-v2")
    if not args.doc_id:
        raise ValueError("--doc-id is required when using --content-list-v2")

    summarizer = None
    cache = None
    if config.vlm.enabled:
        summarizer, cache = create_visual_summarizer(config.vlm)
    chunks = build_chunks_from_mineru_content(
        content_list_v2_path=args.content_list_v2,
        images_dir=args.images_dir,
        doc_id=args.doc_id,
        file_name=args.file_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        summarizer=summarizer,
    )
    if cache is not None:
        cache.save()
    return chunks


def _collect_chunk_paths(args: argparse.Namespace) -> list[Path]:
    """Collect one chunks file or all *_chunks_preview.json files in a directory."""

    if args.chunks_path:
        return [args.chunks_path.resolve()]
    return sorted(path.resolve() for path in args.chunks_dir.glob("*_chunks_preview.json") if path.is_file())


def _load_all_chunks(paths: list[Path]) -> list[ChunkRecord]:
    """Load chunks from all selected preview files."""

    chunks: list[ChunkRecord] = []
    for path in paths:
        chunks.extend(load_chunks(path))
    return chunks


def _group_chunks_by_doc_id(chunks: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    """Group chunks by metadata.doc_id."""

    grouped: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id", ""))
        if not doc_id:
            raise ValueError(f"chunk {chunk.id} missing metadata.doc_id")
        grouped[doc_id].append(chunk)
    return dict(grouped)


def _update_manifest_doc(manifest, doc_id: str, chunks: list[ChunkRecord], counts: Counter) -> None:
    """Update a manifest record after successful Chroma indexing."""

    record = next((document for document in manifest.documents if document.doc_id == doc_id), None)
    if record is None:
        first = chunks[0]
        record = _make_minimal_record_from_chunk(first)
    record.index_status = "indexed"
    record.chunk_count = len(chunks)
    record.text_chunk_count = counts.get("text", 0)
    record.image_chunk_count = counts.get("image", 0)
    record.table_chunk_count = counts.get("table", 0)
    record.indexed_at = datetime.now(timezone.utc).isoformat()
    record.error_message = ""
    upsert_document(manifest, record)


def _make_minimal_record_from_chunk(chunk: ChunkRecord):
    """Create a minimal manifest record when chunks exist without a manifest entry."""

    from offline_index.schema import DocumentRecord

    return DocumentRecord(
        doc_id=str(chunk.metadata["doc_id"]),
        file_name=str(chunk.metadata.get("file_name", "")),
        source_path="",
        pdf_md5="",
        mineru_output_dir="",
    )


if __name__ == "__main__":
    raise SystemExit(main())

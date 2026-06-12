# Command-line entry point for scanning PDFs and maintaining rag_documents.json.
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from offline_index.config_loader import load_config, resolve_value
from offline_index.document_manifest import load_manifest, save_manifest, upsert_document
from offline_index.index_report import IndexReport, format_report
from offline_index.mineru_output_locator import locate_mineru_output
from offline_index.offline_pipeline import build_chunks_from_mineru_content, create_visual_summarizer
from offline_index.source_file_finder import scan_pdfs
from offline_index.schema import ChunkRecord, DocumentRecord, MinerUOutputLocation, PdfCandidate
from offline_index.chunk_loader import save_chunks
from offline_index.utils import ensure_dir, md5_text


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manifest construction."""

    parser = argparse.ArgumentParser(description="Build or update rag_documents.json from source PDFs.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--pdf-root", type=Path, default=None)
    parser.add_argument("--mineru-output-root", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-vlm", action="store_true")
    parser.add_argument("--build-chunks-preview", action="store_true")
    parser.add_argument("--preview-output-dir", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Scan PDFs, update manifest records, optionally write per-document chunk previews, and print a report."""

    args = parse_args()
    config = load_config(args.env_file)
    args.pdf_root = resolve_value(args.pdf_root, config.paths.pdf_root)
    args.mineru_output_root = resolve_value(args.mineru_output_root, config.paths.mineru_output_root)
    args.manifest_path = resolve_value(args.manifest_path, config.paths.manifest_path)
    args.recursive = resolve_value(args.recursive, config.paths.pdf_recursive)
    args.force = args.force or config.paths.force_rebuild
    args.preview_output_dir = resolve_value(args.preview_output_dir, config.paths.debug_dir)
    args.chunk_size = resolve_value(args.chunk_size, config.chunking.chunk_size)
    args.chunk_overlap = resolve_value(args.chunk_overlap, config.chunking.chunk_overlap)
    args.config = config
    report = IndexReport()
    manifest = load_manifest(args.manifest_path)
    candidates = _filter_output_root_candidates(
        scan_pdfs(args.pdf_root, recursive=args.recursive),
        args.mineru_output_root,
    )
    report.scanned_pdfs = len(candidates)

    total_start = perf_counter()
    total_candidates = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        document_start = perf_counter()
        print(f"processing document: [{index}/{total_candidates}] {candidate.file_name}")
        _process_candidate(candidate, args, manifest, report)
        print(f"document elapsed: [{index}/{total_candidates}] {candidate.file_name}: {_format_elapsed(perf_counter() - document_start)}")

    save_manifest(manifest, args.manifest_path)
    print(format_report(report))
    print(f"total elapsed: {_format_elapsed(perf_counter() - total_start)}")
    print(f"manifest: {args.manifest_path.resolve()}")
    return 1 if report.failed else 0


def _process_candidate(
    candidate: PdfCandidate,
    args: argparse.Namespace,
    manifest,
    report: IndexReport,
) -> None:
    """Apply manifest update and optional chunk preview generation for one PDF."""

    try:
        existing_by_md5 = manifest.find_by_md5(candidate.pdf_md5)
        existing_by_source = manifest.find_by_source_path(candidate.source_path)
        if existing_by_md5 and not args.force:
            if _has_content_list_v2(Path(existing_by_md5.mineru_output_dir)):
                if not args.build_chunks_preview or _has_existing_chunks(existing_by_md5):
                    existing_by_md5.index_status = "skipped"
                    existing_by_md5.error_message = ""
                    report.existing_skipped += 1
                    return
            else:
                existing_by_md5.index_status = "pending"
                existing_by_md5.error_message = "content_list_v2.json missing; run parse_pdfs.py"
                report.mineru_output_missing += 1
                return

        is_new = existing_by_source is None and existing_by_md5 is None
        if is_new:
            report.new_documents += 1
        elif existing_by_source and existing_by_source.pdf_md5 != candidate.pdf_md5:
            report.md5_changed += 1

        location = locate_mineru_output(Path(candidate.source_path), args.mineru_output_root)
        record = _make_document_record(candidate, location, existing_by_source or existing_by_md5)

        if location.success:
            report.mineru_output_found += 1
            record.index_status = "parsed"
            if args.build_chunks_preview:
                counts, preview_path, summarizer = _build_chunks_preview(candidate, record.doc_id, location, args)
                _apply_chunk_counts(record, counts)
                record.index_status = "chunked"
                record.indexed_at = _now_iso()
                print(f"chunk preview: {preview_path}")
                if summarizer is not None:
                    report.vlm_cache_hits += getattr(summarizer, "cache_hits", 0)
                    report.vlm_generated += getattr(summarizer, "generated", 0)
                    report.vlm_failed += getattr(summarizer, "failed", 0)
                    for message in getattr(summarizer, "failure_messages", []):
                        print(f"vlm warning: {message}")
                report.chunk_preview_generated += 1
        else:
            report.mineru_output_missing += 1
            record.index_status = "pending"
            record.error_message = location.error_message

        upsert_document(manifest, record)
    except Exception as exc:
        report.failed += 1
        failed_record = _make_document_record(
            candidate,
            MinerUOutputLocation(mineru_output_dir="", success=False, error_message=str(exc)),
            manifest.find_by_source_path(candidate.source_path),
        )
        failed_record.index_status = "failed"
        failed_record.error_message = str(exc)
        upsert_document(manifest, failed_record)
        print(f"failed: {candidate.source_path}: {exc}")


def _make_document_record(
    candidate: PdfCandidate,
    location: MinerUOutputLocation,
    existing: DocumentRecord | None,
) -> DocumentRecord:
    """Create a manifest record from a PDF candidate and located MinerU output."""

    doc_id = existing.doc_id if existing and existing.pdf_md5 == candidate.pdf_md5 else _make_doc_id(candidate)
    return DocumentRecord(
        doc_id=doc_id,
        file_name=candidate.file_name,
        source_path=candidate.source_path,
        pdf_md5=candidate.pdf_md5,
        file_size=candidate.file_size,
        mineru_output_dir=location.mineru_output_dir,
        index_status="pending",
        error_message=location.error_message,
    )


def _build_chunks_preview(
    candidate: PdfCandidate,
    doc_id: str,
    location: MinerUOutputLocation,
    args: argparse.Namespace,
) -> tuple[Counter, Path, object | None]:
    """Build chunks from located MinerU content_list_v2 and write a per-document preview JSON."""

    images_dir = Path(location.images_dir) if location.images_dir else Path(location.mineru_output_dir) / "images"
    summarizer = None
    cache = None
    if args.config.vlm.enabled:
        summarizer, cache = create_visual_summarizer(args.config.vlm, force_vlm=args.force_vlm)
    chunks = build_chunks_from_mineru_content(
        content_list_v2_path=Path(location.content_list_v2_path),
        images_dir=images_dir,
        doc_id=doc_id,
        file_name=candidate.file_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        summarizer=summarizer,
    )
    if cache is not None:
        cache.save()
    preview_dir = args.preview_output_dir.resolve()
    ensure_dir(preview_dir)
    preview_path = preview_dir / _make_preview_file_name(candidate)
    _print_chunk_progress(candidate.file_name, chunks)
    save_chunks(preview_path, chunks)
    return Counter(chunk.metadata["chunk_type"] for chunk in chunks), preview_path, summarizer


def _make_preview_file_name(candidate: PdfCandidate) -> str:
    """Create a human-readable chunk preview file name from PDF stem and full content MD5."""

    return f"{Path(candidate.file_name).stem}_{candidate.pdf_md5}_chunks_preview.json"


def _apply_chunk_counts(record: DocumentRecord, counts: Counter) -> None:
    """Copy per-type chunk counts into a manifest document record."""

    record.text_chunk_count = counts.get("text", 0)
    record.image_chunk_count = counts.get("image", 0)
    record.table_chunk_count = counts.get("table", 0)
    record.chunk_count = record.text_chunk_count + record.image_chunk_count + record.table_chunk_count
    record.error_message = ""


def _print_chunk_progress(file_name: str, chunks: list[ChunkRecord]) -> None:
    """Print per-chunk progress for one document."""

    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        chunk_type = metadata.get("chunk_type", "unknown")
        page_start = int(metadata.get("page_start", 0) or 0)
        page_end = int(metadata.get("page_end", page_start) or page_start)
        print(f"processing chunk: {file_name} [{index}/{total_chunks}] {chunk_type} {chunk.id} pages {_format_page_range(page_start, page_end)}")


def _format_page_range(page_start: int, page_end: int) -> str:
    """Format a single page or page range for progress output."""

    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for CLI progress output."""

    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"{minutes}m {remaining_seconds:.2f}s"


def _make_doc_id(candidate: PdfCandidate) -> str:
    """Create a stable document id from absolute source path and content MD5."""

    return "doc_" + md5_text(f"{candidate.source_path}{candidate.pdf_md5}")


def _has_content_list_v2(output_dir: Path) -> bool:
    """Return True when a MinerU output directory contains content_list_v2 JSON."""

    return output_dir.is_dir() and any(output_dir.glob("*_content_list_v2.json"))


def _has_existing_chunks(record: DocumentRecord) -> bool:
    """Return True when a manifest record already has built chunk counts."""

    return record.chunk_count > 0


def _filter_output_root_candidates(candidates: list[PdfCandidate], output_root: Path) -> list[PdfCandidate]:
    """Remove PDFs nested under the MinerU output root from source candidates."""

    output_root = output_root.resolve()
    return [candidate for candidate in candidates if not _is_relative_to(Path(candidate.source_path), output_root)]


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or nested under parent."""

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

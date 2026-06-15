# Tracks and formats scan, manifest, MinerU output, and chunk file statistics.
from __future__ import annotations

from pydantic import BaseModel


class IndexReport(BaseModel):
    """Collects counters for one build_manifest run."""

    scanned_pdfs: int = 0
    new_documents: int = 0
    existing_skipped: int = 0
    md5_changed: int = 0
    mineru_output_found: int = 0
    mineru_output_missing: int = 0
    chunk_files_generated: int = 0
    vlm_cache_hits: int = 0
    vlm_generated: int = 0
    vlm_failed: int = 0
    failed: int = 0


def format_report(report: IndexReport) -> str:
    """Format index report counters as readable log lines."""

    return "\n".join(
        [
            f"scanned PDFs: {report.scanned_pdfs}",
            f"new documents: {report.new_documents}",
            f"existing skipped: {report.existing_skipped}",
            f"MD5 changed: {report.md5_changed}",
            f"MinerU output found: {report.mineru_output_found}",
            f"MinerU output missing: {report.mineru_output_missing}",
            f"chunk files generated: {report.chunk_files_generated}",
            f"VLM cache hits: {report.vlm_cache_hits}",
            f"VLM generated: {report.vlm_generated}",
            f"VLM failed: {report.vlm_failed}",
            f"failed: {report.failed}",
        ]
    )

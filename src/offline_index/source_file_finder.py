# Finds source PDF files and produces file metadata candidates for indexing.
from __future__ import annotations

from pathlib import Path

from offline_index.schema import PdfCandidate
from offline_index.utils import md5_file


def scan_pdfs(root_path: Path, recursive: bool = True) -> list[PdfCandidate]:
    """Scan one PDF file or a directory and return metadata for source PDF candidates."""

    root_path = root_path.resolve()
    paths = _find_pdf_paths(root_path, recursive=recursive)
    candidates: list[PdfCandidate] = []
    for path in paths:
        stat = path.stat()
        candidates.append(
            PdfCandidate(
                source_path=str(path),
                file_name=path.name,
                file_size=stat.st_size,
                pdf_md5=compute_file_md5(path),
                modified_time=stat.st_mtime,
            )
        )
    return candidates


def compute_file_md5(path: Path) -> str:
    """Compute an MD5 hash from file contents."""

    return md5_file(path)


def _find_pdf_paths(root_path: Path, recursive: bool) -> list[Path]:
    """Find non-temporary PDF paths under a file or directory."""

    if root_path.is_file():
        return [root_path] if _is_source_pdf(root_path) else []
    if not root_path.exists() or not root_path.is_dir():
        return []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    paths = [path.resolve() for path in root_path.glob(pattern) if _is_source_pdf(path)]
    return sorted(paths, key=lambda path: (len(path.relative_to(root_path).parts), str(path).lower()))


def _is_source_pdf(path: Path) -> bool:
    """Return True for normal PDF files and False for temp or non-PDF paths."""

    if not path.is_file():
        return False
    if path.suffix.lower() != ".pdf":
        return False
    return not (path.name.startswith("~$") or path.name.startswith("."))

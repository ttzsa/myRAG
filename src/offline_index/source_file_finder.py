# Finds source PDF files and produces file metadata candidates for indexing.
from __future__ import annotations

from pathlib import Path

from offline_index.schema import DocumentManifest, PdfCandidate
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


def find_pdf_paths(root_path: Path, recursive: bool = True) -> list[Path]:
    """Find source PDF paths with the same filtering used by PDF discovery."""

    return _find_pdf_paths(root_path.resolve(), recursive=recursive)


def select_pdf_candidates(
    candidates: list[PdfCandidate],
    manifest: DocumentManifest,
    scope: str = "new",
    mineru_output_root: Path | None = None,
) -> list[PdfCandidate]:
    """Select discovered PDFs for this run using content MD5 and a new/all scope."""

    normalized = scope.strip().lower()
    if normalized == "all":
        return candidates
    if normalized != "new":
        raise ValueError(f"unsupported pdf scope: {scope}")
    selected: list[PdfCandidate] = []
    for candidate in candidates:
        document = manifest.find_by_md5(candidate.pdf_md5)
        if document is None:
            selected.append(candidate)
            continue
        if not _has_content_list_v2(candidate, document, mineru_output_root):
            selected.append(candidate)
    return selected


def _has_content_list_v2(candidate: PdfCandidate, document, mineru_output_root: Path | None) -> bool:
    """Return True when a recorded PDF still has an existing content_list_v2 JSON file."""

    if document.content_list_v2_path and Path(document.content_list_v2_path).is_file():
        return True
    output_dir = Path(document.mineru_output_dir) if document.mineru_output_dir else None
    if output_dir and output_dir.is_dir() and any(output_dir.glob("*_content_list_v2.json")):
        return True
    if mineru_output_root is not None:
        expected_dir = (Path(mineru_output_root).resolve() / Path(candidate.file_name).stem / "auto").resolve()
        return expected_dir.is_dir() and any(expected_dir.glob("*_content_list_v2.json"))
    return False


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

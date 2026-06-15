# Runs MinerU CLI in batch mode for source files and reports per-file parse results.
from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from pathlib import Path

from pydantic import BaseModel

from offline_index.document_manifest import load_manifest, save_manifest, upsert_document
from offline_index.source_file_finder import compute_file_md5, find_pdf_paths
from offline_index.schema import DocumentRecord, PdfCandidate
from offline_index.utils import md5_text


class MinerUParseResult(BaseModel):
    """Represents the result of one MinerU CLI parse attempt."""

    pdf_path: str
    output_root: str
    expected_output_dir: str
    success: bool
    skipped: bool
    return_code: int
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


def find_pdfs(input_path: Path, recursive: bool = True) -> list[Path]:
    """Find PDF files from one PDF path or a directory, optionally recursively."""

    return find_pdf_paths(input_path, recursive=recursive)


def expected_mineru_output_dir(pdf_path: Path, output_root: Path) -> Path:
    """Return MinerU's expected auto output directory for a PDF."""

    return (output_root.resolve() / pdf_path.stem / "auto").resolve()


def is_already_parsed(pdf_path: Path, output_root: Path) -> bool:
    """Return True when the expected auto directory contains MinerU parse outputs."""

    output_dir = expected_mineru_output_dir(pdf_path, output_root)
    if not output_dir.is_dir():
        return False
    return any(output_dir.glob("*_content_list_v2.json"))


def run_mineru_cli(
    pdf_path: Path,
    output_root: Path,
    mineru_exe: Path,
    backend: str = "pipeline",
    method: str = "auto",
    parser_method: str = "mineru",
    force: bool = False,
    timeout: int | None = None,
    manifest_path: Path | None = None,
    model_source: str | None = None,
    tools_config_json: Path | None = None,
) -> MinerUParseResult:
    """Run MinerU CLI for one PDF and return captured command status."""

    pdf_path = pdf_path.resolve()
    output_root = output_root.resolve()
    mineru_exe = mineru_exe.resolve()
    output_dir = expected_mineru_output_dir(pdf_path, output_root)
    current_md5 = compute_file_md5(pdf_path)
    manifest = load_manifest(manifest_path) if manifest_path else None
    existing_by_md5 = manifest.find_by_md5(current_md5) if manifest else None

    if not force and existing_by_md5:
        existing_output_dir = Path(existing_by_md5.mineru_output_dir) if existing_by_md5.mineru_output_dir else output_dir
        if _has_content_list_v2(existing_output_dir):
            return MinerUParseResult(
                pdf_path=str(pdf_path),
                output_root=str(output_root),
                expected_output_dir=str(existing_output_dir.resolve()),
                success=True,
                skipped=True,
                return_code=0,
                error_message="same PDF MD5 already parsed with content_list_v2.json",
            )

    command = [str(mineru_exe), "-p", str(pdf_path), "-o", str(output_root), "-b", backend]
    env = os.environ.copy()

    if model_source:
        env["MINERU_MODEL_SOURCE"] = model_source
    if tools_config_json:
        env["MINERU_TOOLS_CONFIG_JSON"] = str(tools_config_json.resolve())

    if backend == "pipeline" and method:
        command.extend(["-m", method])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        result = MinerUParseResult(
            pdf_path=str(pdf_path),
            output_root=str(output_root),
            expected_output_dir=str(output_dir),
            success=False,
            skipped=False,
            return_code=-1,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error_message=f"MinerU CLI timed out after {timeout} seconds",
        )
        _save_parse_record(manifest, manifest_path, pdf_path, output_dir, current_md5, result, parser_method)
        return result
    except OSError as exc:
        result = MinerUParseResult(
            pdf_path=str(pdf_path),
            output_root=str(output_root),
            expected_output_dir=str(output_dir),
            success=False,
            skipped=False,
            return_code=-1,
            error_message=str(exc),
        )
        _save_parse_record(manifest, manifest_path, pdf_path, output_dir, current_md5, result, parser_method)
        return result

    success = completed.returncode == 0

    result = MinerUParseResult(
        pdf_path=str(pdf_path),
        output_root=str(output_root),
        expected_output_dir=str(output_dir),
        success=success,
        skipped=False,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error_message="" if success else _summarize_error(completed.stderr, completed.stdout, completed.returncode),
    )
    _save_parse_record(manifest, manifest_path, pdf_path, output_dir, current_md5, result, parser_method)
    return result


def parse_pdf_dir(
    input_path: Path,
    output_root: Path,
    mineru_exe: Path,
    recursive: bool = True,
    backend: str = "pipeline",
    method: str = "auto",
    force: bool = False,
    timeout: int | None = None,
    manifest_path: Path | None = None,
    model_source: str | None = None,
    tools_config_json: Path | None = None,
) -> list[MinerUParseResult]:
    """Parse every discovered source PDF and continue after individual failures."""

    output_root = output_root.resolve()
    pdfs = [path for path in find_pdfs(input_path, recursive=recursive) if not _is_relative_to(path, output_root)]
    results: list[MinerUParseResult] = []

    for pdf_path in pdfs:
        results.append(
            run_mineru_cli(
                pdf_path=pdf_path,
                output_root=output_root,
                mineru_exe=mineru_exe,
                backend=backend,
                method=method,
                parser_method="mineru",
                force=force,
                timeout=timeout,
                manifest_path=manifest_path,
                model_source=model_source,
                tools_config_json=tools_config_json,
            )
        )

    return results


def parse_pdf_candidates(
    candidates: list[PdfCandidate],
    output_root: Path,
    mineru_exe: Path,
    parser_method: str = "mineru",
    backend: str = "pipeline",
    method: str = "auto",
    timeout: int | None = None,
    manifest_path: Path | None = None,
    model_source: str | None = None,
    tools_config_json: Path | None = None,
) -> list[MinerUParseResult]:
    """Parse exactly the PDF candidates selected by the discover stage."""

    normalized = parser_method.strip().lower()
    if normalized != "mineru":
        raise ValueError(f"parser method is not implemented: {parser_method}")
    results: list[MinerUParseResult] = []
    for candidate in candidates:
        results.append(
            run_mineru_cli(
                pdf_path=Path(candidate.source_path),
                output_root=output_root,
                mineru_exe=mineru_exe,
                backend=backend,
                method=method,
                parser_method=normalized,
                force=True,
                timeout=timeout,
                manifest_path=manifest_path,
                model_source=model_source,
                tools_config_json=tools_config_json,
            )
        )
    return results


def _summarize_error(stderr: str, stdout: str, return_code: int) -> str:
    """Build a full error message from stderr/stdout and return code."""

    text = (stderr or stdout or "").strip()
    if not text:
        return f"MinerU CLI failed with return code {return_code}"
    return text or f"MinerU CLI failed with return code {return_code}"
    # return text


def _has_content_list_v2(output_dir: Path) -> bool:
    """Return True when a MinerU output directory contains content_list_v2 JSON."""

    return output_dir.is_dir() and any(output_dir.glob("*_content_list_v2.json"))

def _save_parse_record(
    manifest,
    manifest_path: Path | None,
    pdf_path: Path,
    output_dir: Path,
    pdf_md5: str,
    result: MinerUParseResult,
    parser_method: str = "mineru",
) -> None:
    """Persist a manifest record for a parse attempt when manifest_path is configured."""

    if manifest is None or manifest_path is None:
        return
    record = DocumentRecord(
        doc_id=_make_doc_id(pdf_path, pdf_md5),
        file_name=pdf_path.name,
        source_path=str(pdf_path),
        pdf_md5=pdf_md5,
        file_size=pdf_path.stat().st_size,
        mineru_output_dir=str(output_dir),
        parser_method=parser_method,
        parse_status="parsed" if result.success else "failed",
        parse_output_dir=str(output_dir),
        parse_error=result.error_message,
        index_status="parsed" if result.success else "failed",
        error_message=result.error_message,
    )
    upsert_document(manifest, record)
    save_manifest(manifest, manifest_path)


def _make_doc_id(pdf_path: Path, pdf_md5: str) -> str:
    """Create a stable document id from source path and PDF content MD5."""

    return "doc_" + md5_text(f"{pdf_path}{pdf_md5}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or nested under parent."""

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

# Command-line entry point for batch parsing PDFs with MinerU CLI.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from offline_index.config_loader import load_config, resolve_value
from offline_index.document_manifest import load_manifest
from offline_index.mineru_runner import MinerUParseResult, parse_pdf_candidates
from offline_index.source_file_finder import scan_pdfs, select_pdf_candidates


def parse_bool(value: str) -> bool:
    """Parse a command-line boolean value for --recursive."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch MinerU PDF parsing."""

    parser = argparse.ArgumentParser(description="Batch parse PDFs with MinerU CLI.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to .env configuration file.")
    parser.add_argument("--input", type=Path, default=None, help="PDF file or directory containing PDF files.")
    parser.add_argument("--output-root", type=Path, default=None, help="MinerU output root directory.")
    parser.add_argument("--mineru-exe", type=Path, default=None, help="Path to mineru.exe.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Path to processed_pdfs.json.")
    parser.add_argument("--recursive", type=parse_bool, default=None, help="Whether to scan directories recursively.")
    parser.add_argument("--backend", default=None, help="MinerU backend, default: pipeline.")
    parser.add_argument("--method", default=None, help="MinerU method for pipeline backend, default: auto.")
    parser.add_argument("--pdf-scope", choices=["new", "all"], default="new", help="PDF selection scope from processed_pdfs.json.")
    parser.add_argument("--parser-method", choices=["mineru", "pymupdf"], default="mineru", help="PDF parser method.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds for each PDF.")
    return parser.parse_args()


def main() -> int:
    """Run batch parsing and print a summary for scanned, skipped, success, and failed PDFs."""

    args = parse_args()
    config = load_config(args.env_file)
    input_path = resolve_value(args.input, config.paths.pdf_root)
    output_root = resolve_value(args.output_root, config.paths.mineru_output_root)
    mineru_exe = resolve_value(args.mineru_exe, config.mineru.exe)
    manifest_path = resolve_value(args.manifest_path, config.paths.manifest_path)
    recursive = resolve_value(args.recursive, config.paths.pdf_recursive)
    backend = resolve_value(args.backend, config.mineru.backend)
    method = resolve_value(args.method, config.mineru.method)
    manifest = load_manifest(manifest_path)
    candidates = _filter_output_root_candidates(scan_pdfs(input_path, recursive=recursive), output_root)
    selected = select_pdf_candidates(candidates, manifest, scope=args.pdf_scope, mineru_output_root=output_root)
    results = parse_pdf_candidates(
        candidates=selected,
        output_root=output_root,
        mineru_exe=mineru_exe,
        parser_method=args.parser_method,
        backend=backend,
        method=method,
        timeout=args.timeout,
        manifest_path=manifest_path,
        model_source=config.mineru.model_source,
        tools_config_json=config.mineru.tools_config_json,
    )
    skipped = [result for result in results if result.skipped]
    failed = [result for result in results if not result.success]
    parsed_success = [result for result in results if result.success and not result.skipped]

    print(f"scanned PDFs: {len(candidates)}")
    print(f"selected PDFs: {len(selected)}")
    print(f"parsed success: {len(parsed_success)}")
    print(f"skipped: {len(skipped)}")
    print(f"failed: {len(failed)}")

    for result in results:
        status = _status_label(result)
        print(f"[{status}] {result.pdf_path}")
        print(f"  output: {result.expected_output_dir}")

    if failed:
        print("failed files:")
        for result in failed:
            print(f"- {result.pdf_path}: {result.error_message}")

    return 1 if failed else 0


def _status_label(result: MinerUParseResult) -> str:
    """Return a short display label for one parse result."""

    if result.skipped:
        return "skipped"
    if result.success:
        return "parsed"
    return "failed"


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or nested under parent."""

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _filter_output_root_candidates(candidates, output_root: Path):
    """Remove PDFs nested under the MinerU output root from source candidates."""

    output_root = output_root.resolve()
    return [candidate for candidate in candidates if not _is_relative_to(Path(candidate.source_path), output_root)]


if __name__ == "__main__":
    raise SystemExit(main())

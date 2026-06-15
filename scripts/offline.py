# Unified command-line entry point for the four-stage offline RAG workflow.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for the unified offline workflow."""

    parser = argparse.ArgumentParser(description="Run offline PDF discovery, parsing, chunking, and indexing.")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--pdf-scope", choices=["new", "all"], default="new")
    parser.add_argument("--parser-method", choices=["mineru", "pymupdf"], default="mineru")
    parser.add_argument("--vlm-mode", choices=["auto", "refresh", "off"], default="auto")
    parser.add_argument("--reset", action="store_true")

    chunk = subparsers.add_parser("chunk", help="Build chunk files from content_list_v2 JSON files.")
    chunk.add_argument("--env-file", type=Path, default=Path(".env"))
    chunk.add_argument("--content-list-v2", type=Path, action="append", default=[])
    chunk.add_argument("--vlm-mode", choices=["auto", "refresh", "off"], default="auto")

    index = subparsers.add_parser("index", help="Write chunk files into the vector store.")
    index.add_argument("--env-file", type=Path, default=Path(".env"))
    index.add_argument("--reset", action="store_true")

    return parser.parse_args(argv)


def build_stage_commands(args: argparse.Namespace) -> list[list[str]]:
    """Build subprocess commands for the requested offline stage or full workflow."""

    if args.command == "chunk":
        command = _base_command("build_manifest.py", args.env_file)
        for path in args.content_list_v2:
            command.extend(["--content-list-v2", str(path)])
        command.extend(["--vlm-mode", args.vlm_mode])
        return [command]
    if args.command == "index":
        command = _base_command("build_index.py", args.env_file)
        if args.reset:
            command.append("--reset")
        return [command]

    parse_command = _base_command("parse_pdfs.py", args.env_file)
    parse_command.extend(["--pdf-scope", args.pdf_scope, "--parser-method", args.parser_method])
    chunk_command = _base_command("build_manifest.py", args.env_file)
    chunk_command.extend(["--vlm-mode", args.vlm_mode])
    index_command = _base_command("build_index.py", args.env_file)
    if args.reset:
        index_command.append("--reset")
    return [parse_command, chunk_command, index_command]


def main(argv: list[str] | None = None) -> int:
    """Run each planned offline stage command in order."""

    args = parse_args(argv)
    for command in build_stage_commands(args):
        completed = subprocess.run(command)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def _base_command(script_name: str, env_file: Path) -> list[str]:
    """Return a Python command for one script, adding --env-file only when non-default."""

    command = [sys.executable, str(Path("scripts") / script_name)]
    if Path(env_file) != Path(".env"):
        command.extend(["--env-file", str(env_file)])
    return command


if __name__ == "__main__":
    raise SystemExit(main())

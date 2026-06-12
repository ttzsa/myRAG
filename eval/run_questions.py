from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for batch online QA evaluation."""

    parser = argparse.ArgumentParser(description="Run scripts/ask.py for each question in a text file.")
    parser.add_argument("--questions-file", type=Path, default=EVAL_ROOT / "questions.txt")
    parser.add_argument("--output-dir", type=Path, default=EVAL_ROOT)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args(argv)


def read_questions(path: Path) -> list[str]:
    """Read non-empty questions from a text file, accepting common Chinese encodings."""

    text = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_command(args: argparse.Namespace, question: str) -> list[str]:
    """Build the scripts/ask.py command for one question."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ask.py"),
        "--env-file",
        str(args.env_file),
        "--query",
        question,
    ]
    if args.top_k is not None:
        command.extend(["--top-k", str(args.top_k)])
    if not args.no_debug:
        command.append("--show-debug")
    return command


def simplify_chunk(chunk: dict) -> dict:
    """Keep the retrieval fields needed for manual evaluation."""

    metadata = chunk.get("metadata") or {}
    return {
        "rank": chunk.get("rank"),
        "score": chunk.get("score"),
        "distance": chunk.get("distance"),
        "channel": chunk.get("channel"),
        "chunk_id": chunk.get("chunk_id"),
        "document": chunk.get("document"),
        "file_name": metadata.get("file_name", ""),
        "page_start": metadata.get("page_start", ""),
        "page_end": metadata.get("page_end", ""),
        "chunk_type": metadata.get("chunk_type", ""),
        "source": metadata.get("source", ""),
        "metadata": metadata,
    }


def build_eval_record(
    *,
    index: int,
    question: str,
    duration_ms: float,
    exit_code: int,
    result: dict | None,
    stderr: str,
    parse_error: str | None,
) -> dict:
    """Build the JSON record saved for one evaluation question."""

    if result is None:
        return {
            "index": index,
            "question": question,
            "answer": "",
            "rag_results": [],
            "citations": [],
            "confidence": "",
            "ok": False,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "error": {
                "exit_code": exit_code,
                "stderr": stderr,
                "parse_error": parse_error,
            },
        }

    retrieval_debug_info = result.get("retrieval_debug_info") or {}
    retrieved_chunks = retrieval_debug_info.get("chunks") or result.get("used_chunks") or []
    return {
        "index": index,
        "question": question,
        "answer": result.get("answer", ""),
        "rag_results": [simplify_chunk(chunk) for chunk in retrieved_chunks],
        "used_chunks": [simplify_chunk(chunk) for chunk in result.get("used_chunks", [])],
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", ""),
        "ok": exit_code == 0 and parse_error is None,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "retrieval_debug_info": {
            "retrieval_mode": retrieval_debug_info.get("retrieval_mode", ""),
            "retrieved_count": retrieval_debug_info.get("retrieved_count", len(retrieved_chunks)),
            "context_chars": retrieval_debug_info.get("context_chars", 0),
        },
        "error": None,
    }


def run_question(args: argparse.Namespace, index: int, question: str) -> dict:
    """Run one question through scripts/ask.py and return a serializable record."""

    command = build_command(args, question)
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    result = None
    parse_error = None
    if completed.stdout.strip():
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)

    return build_eval_record(
        index=index,
        question=question,
        duration_ms=duration_ms,
        exit_code=completed.returncode,
        result=result,
        stderr=completed.stderr,
        parse_error=parse_error,
    )


def main() -> int:
    """Run all questions and write a timestamped JSON result file."""

    args = parse_args()
    questions_file = args.questions_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = read_questions(questions_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}.json"

    records = []
    exit_code = 0
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question}", flush=True)
        record = run_question(args, index, question)
        records.append(record)
        if not record["ok"]:
            exit_code = record["exit_code"] or 1
            if args.stop_on_error:
                break

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "questions_file": str(questions_file),
        "total_questions": len(questions),
        "completed_questions": len(records),
        "success_count": sum(1 for record in records if record["ok"]),
        "failure_count": sum(1 for record in records if not record["ok"]),
        "records": records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"Saved results to {output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

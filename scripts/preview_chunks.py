# Command-line entry point for generating chunk JSON from MinerU output.
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from offline_index.chunk_loader import save_chunks
from offline_index.config_loader import load_config, resolve_value
from offline_index.offline_pipeline import build_chunks_from_mineru_content, create_visual_summarizer
from offline_index.utils import ensure_dir


DEFAULT_AUTO_DIR = PROJECT_ROOT / "documents" / "output_pipeline" / "DarkIR Robust Low-Light Image Restoration" / "auto"
DEFAULT_CONTENT_LIST = DEFAULT_AUTO_DIR / "DarkIR Robust Low-Light Image Restoration_content_list_v2.json"
DEFAULT_IMAGES_DIR = DEFAULT_AUTO_DIR / "images"
DEFAULT_FILE_NAME = "DarkIR Robust Low-Light Image Restoration.pdf"
DEFAULT_DOC_ID = "doc_darkir_robust_low_light_image_restoration"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "chunks" / "chunks.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and provide defaults for the current DarkIR test output."""

    parser = argparse.ArgumentParser(description="Generate chunks from MinerU content_list_v2.json.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--content-list-v2", type=Path, default=DEFAULT_CONTENT_LIST)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--file-name", default=DEFAULT_FILE_NAME)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--vlm-mode", choices=["auto", "refresh", "off"], default="auto")
    return parser.parse_args()


def main() -> int:
    """Run the MinerU-to-chunks pipeline and print chunk statistics."""

    args = parse_args()
    config = load_config(args.env_file)
    args.chunk_size = resolve_value(args.chunk_size, config.chunking.chunk_size)
    args.chunk_overlap = resolve_value(args.chunk_overlap, config.chunking.chunk_overlap)

    summarizer = None
    cache = None
    if config.vlm.enabled and args.vlm_mode != "off":
        summarizer, cache = create_visual_summarizer(config.vlm, force_vlm=args.vlm_mode == "refresh")

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

    ensure_dir(args.output.parent)
    save_chunks(args.output, chunks)

    counts = Counter(chunk.metadata["chunk_type"] for chunk in chunks)
    print(f"text chunk count: {counts.get('text', 0)}")
    print(f"image chunk count: {counts.get('image', 0)}")
    print(f"table chunk count: {counts.get('table', 0)}")
    print(f"total chunk count: {len(chunks)}")
    if summarizer is not None:
        print(f"VLM cache hits: {summarizer.cache_hits}")
        print(f"VLM generated: {summarizer.generated}")
        print(f"VLM failed: {summarizer.failed}")
        for message in summarizer.failure_messages:
            print(f"vlm warning: {message}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

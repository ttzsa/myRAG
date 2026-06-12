# Shared offline pipeline helpers for MinerU blocks, optional VLM enrichment, and chunk generation.
from __future__ import annotations

from pathlib import Path

from offline_index.block_converter import convert_blocks
from offline_index.chunk_builder import build_chunks
from offline_index.config_loader import VLMConfig
from offline_index.mineru_output_reader import flatten_blocks, read_content_list_v2
from offline_index.schema import ChunkRecord, SemanticBlock
from offline_index.summary_cache import SummaryCache
from offline_index.visual_summarizer import VisualBlockSummarizer
from offline_index.vlm_client import create_vlm_client


def build_semantic_blocks_from_mineru_content(
    content_list_v2_path: Path,
    images_dir: Path,
    doc_id: str,
    file_name: str,
    summarizer: VisualBlockSummarizer | None = None,
) -> list[SemanticBlock]:
    """Read MinerU JSON, normalize blocks, and optionally enrich visual blocks."""

    pages = read_content_list_v2(Path(content_list_v2_path))
    flattened = flatten_blocks(pages)
    blocks = convert_blocks(flattened, doc_id=doc_id, images_dir=Path(images_dir))
    if summarizer is not None:
        blocks = summarizer.enrich_blocks(blocks, file_name=file_name)
    return blocks


def build_chunks_from_mineru_content(
    content_list_v2_path: Path,
    images_dir: Path,
    doc_id: str,
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    summarizer: VisualBlockSummarizer | None = None,
) -> list[ChunkRecord]:
    """Build final ChunkRecord objects from MinerU JSON and optional VLM enhancement."""

    blocks = build_semantic_blocks_from_mineru_content(
        content_list_v2_path=content_list_v2_path,
        images_dir=images_dir,
        doc_id=doc_id,
        file_name=file_name,
        summarizer=summarizer,
    )
    return build_chunks(
        blocks,
        file_name=file_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def create_visual_summarizer(vlm_config: VLMConfig, force_vlm: bool = False) -> tuple[VisualBlockSummarizer, SummaryCache]:
    """Create a configured visual summarizer and its cache."""

    cache = SummaryCache(vlm_config.cache_path)
    client = create_vlm_client(vlm_config)
    summarizer = VisualBlockSummarizer(
        client=client,
        cache=cache,
        model=vlm_config.model,
        max_images_per_doc=vlm_config.max_images_per_doc,
        force_vlm=force_vlm,
    )
    return summarizer, cache

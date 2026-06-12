# Builds final ChunkRecord objects from normalized SemanticBlock records.
from __future__ import annotations

from offline_index.schema import ChunkRecord, SemanticBlock
from offline_index.utils import md5_text, normalize_text


def build_chunks(
    blocks: list[SemanticBlock],
    file_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[ChunkRecord]:
    """Build text, image, and table chunks from ordered SemanticBlock records."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[ChunkRecord] = []
    ordered_blocks = sorted(blocks, key=lambda block: block.reading_order)
    text_blocks = [block for block in ordered_blocks if block.rag_type == "text" and block.text]

    chunks.extend(_build_text_chunks(text_blocks, file_name, chunk_size, chunk_overlap))
    for block in ordered_blocks:
        if block.rag_type == "image":
            chunks.append(_build_special_chunk(block, file_name, "image"))
        elif block.rag_type == "table":
            chunks.append(_build_special_chunk(block, file_name, "table"))

    return chunks


def _build_text_chunks(
    text_blocks: list[SemanticBlock],
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    """Merge text blocks into a stream, split it, and format text chunk records."""

    if not text_blocks:
        return []

    stream = "\n\n".join(block.text for block in text_blocks if block.text)
    page_markers = _page_markers(text_blocks)
    split_texts = _split_text(stream, chunk_size, chunk_overlap)
    chunks: list[ChunkRecord] = []
    cursor = 0

    for text in split_texts:
        start = stream.find(text[: min(len(text), 32)], cursor)
        if start < 0:
            start = cursor
        end = start + len(text)
        cursor = max(start + 1, end - chunk_overlap)
        page_start, page_end = _pages_for_span(page_markers, start, end)
        body = _format_text_document(file_name, page_start, page_end, text)
        chunks.append(_make_chunk(text_blocks[0].doc_id, file_name, "text", page_start, page_end, "", body))

    return chunks


def _page_markers(blocks: list[SemanticBlock]) -> list[tuple[int, int, int]]:
    """Record character spans for each text block so split chunks keep page ranges."""

    markers: list[tuple[int, int, int]] = []
    cursor = 0
    for index, block in enumerate(blocks):
        text = block.text
        start = cursor
        end = start + len(text)
        markers.append((start, end, block.page_start))
        cursor = end
        if index < len(blocks) - 1:
            cursor += 2
    return markers


def _pages_for_span(markers: list[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    """Return the page range touched by a character span in the text stream."""

    pages = [page for marker_start, marker_end, page in markers if marker_end >= start and marker_start <= end]
    if not pages:
        pages = [markers[0][2]]
    return min(pages), max(pages)


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split normalized text into overlapping chunks with simple boundary preference."""

    text = normalize_text(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = _best_break(text, start, hard_end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def _best_break(text: str, start: int, hard_end: int) -> int:
    """Choose a readable split point near hard_end when a delimiter is available."""

    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    for delimiter in ("\n\n", "\n", ". ", "。", "; ", "；", ", ", "，", " "):
        position = window.rfind(delimiter)
        if position >= int(len(window) * 0.6):
            return start + position + len(delimiter)
    return hard_end


def _format_text_document(file_name: str, page_start: int, page_end: int, text: str) -> str:
    """Return pure text content for a text chunk document."""

    return normalize_text(text)


def _build_special_chunk(block: SemanticBlock, file_name: str, chunk_type: str) -> ChunkRecord:
    """Build one standalone image or table chunk from a SemanticBlock."""

    if chunk_type == "image":
        summary = _mock_image_summary(block, file_name)
    elif chunk_type == "table":
        summary = _mock_table_summary(block, file_name)
    else:
        raise ValueError(f"unsupported special chunk type: {chunk_type}")
    return _make_chunk(
        block.doc_id,
        file_name,
        chunk_type,
        block.page_start,
        block.page_end,
        block.source,
        summary,
    )


def _mock_image_summary(block: SemanticBlock, file_name: str) -> str:
    """Create pure image summary text from MinerU caption and footnote text."""

    return normalize_text(block.text or "No caption or footnote was extracted.")


def _mock_table_summary(block: SemanticBlock, file_name: str) -> str:
    """Create pure table text from MinerU caption, footnote, and HTML text."""

    return normalize_text(block.text or "No structured table text was extracted.")


def _make_chunk(
    doc_id: str,
    file_name: str,
    chunk_type: str,
    page_start: int,
    page_end: int,
    source: str,
    document: str,
) -> ChunkRecord:
    """Create a ChunkRecord with flat metadata, content_hash, and stable chunk id."""

    document = normalize_text(document)
    content_hash = md5_text(document)
    chunk_id = "chunk_" + md5_text(f"{doc_id}{chunk_type}{page_start}{page_end}{content_hash}")
    return ChunkRecord(
        id=chunk_id,
        document=document,
        metadata={
            "doc_id": doc_id,
            "file_name": file_name,
            "chunk_type": chunk_type,
            "page_start": page_start,
            "page_end": page_end,
            "source": source,
            "content_hash": content_hash,
        },
    )


def _page_range(page_start: int, page_end: int) -> str:
    """Format a single page or page range for chunk document text."""

    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"

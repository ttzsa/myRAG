# Builds final ChunkRecord objects from normalized SemanticBlock records.
from __future__ import annotations

from offline_index.schema import ChunkRecord, SemanticBlock
from offline_index.utils import md5_text, normalize_text


PAGE_SENTENCE_ENDINGS = {"。", "！", "？", ".", "!", "?"}
OVERLAP_SEMANTIC_BOUNDARIES = {"。", "！", "？", ".", "!", "?", "；", ";", "，", ",", "\n"}


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
    page_block_indices = _page_block_indices(ordered_blocks)
    text_blocks = [block for block in ordered_blocks if block.rag_type == "text" and block.text]

    chunks.extend(_build_text_chunks(text_blocks, file_name, chunk_size, chunk_overlap, page_block_indices))
    for block in ordered_blocks:
        if block.rag_type == "image":
            chunks.append(_build_special_chunk(block, file_name, "image", page_block_indices))
        elif block.rag_type == "table":
            chunks.append(_build_special_chunk(block, file_name, "table", page_block_indices))

    return chunks


def _page_block_indices(blocks: list[SemanticBlock]) -> dict[tuple[int, str, int], int]:
    """Return each SemanticBlock's one-based ordinal within its page."""

    counters: dict[int, int] = {}
    indices: dict[tuple[int, str, int], int] = {}
    for block in blocks:
        page = block.page_start
        counters[page] = counters.get(page, 0) + 1
        indices[(page, block.block_id, block.reading_order)] = counters[page]
    return indices


def _build_text_chunks(
    text_blocks: list[SemanticBlock],
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    page_block_indices: dict[tuple[int, str, int], int],
) -> list[ChunkRecord]:
    """Build page-aware text chunks from MinerU text blocks."""

    if not text_blocks:
        return []

    pages = _group_text_blocks_by_page(text_blocks)
    _move_cross_page_lead_sentences(pages)
    chunks: list[ChunkRecord] = []

    for page in sorted(pages):
        current_blocks: list[SemanticBlock] = []
        for block in pages[page]:
            text = normalize_text(block.text)
            if not text:
                continue
            if len(text) > chunk_size:
                _flush_text_block_chunk(chunks, current_blocks, file_name, page_block_indices)
                current_blocks = []
                chunks.extend(_build_large_block_chunks(block, file_name, chunk_size, chunk_overlap, page_block_indices))
                continue
            if current_blocks and _merged_blocks_length(current_blocks, block) > chunk_size:
                _flush_text_block_chunk(chunks, current_blocks, file_name, page_block_indices)
                current_blocks = [block]
            else:
                current_blocks.append(block)
        _flush_text_block_chunk(chunks, current_blocks, file_name, page_block_indices)

    return chunks


def _group_text_blocks_by_page(text_blocks: list[SemanticBlock]) -> dict[int, list[SemanticBlock]]:
    """Group copied text blocks by page so cross-page preprocessing can edit text safely."""

    pages: dict[int, list[SemanticBlock]] = {}
    for block in sorted(text_blocks, key=lambda item: item.reading_order):
        text = normalize_text(block.text)
        if not text:
            continue
        copied = block.model_copy(update={"text": text})
        pages.setdefault(copied.page_start, []).append(copied)
    return pages


def _move_cross_page_lead_sentences(pages: dict[int, list[SemanticBlock]]) -> None:
    """Move the next page's lead sentence into an incomplete previous page ending."""

    sorted_pages = sorted(pages)
    for index, page in enumerate(sorted_pages[:-1]):
        current_blocks = pages.get(page, [])
        next_blocks = pages.get(sorted_pages[index + 1], [])
        if not current_blocks or not next_blocks:
            continue
        last_block = current_blocks[-1]
        if _ends_with_sentence(last_block.text):
            continue
        lead_sentence, remainder = _take_lead_sentence(next_blocks[0].text)
        if not lead_sentence:
            continue
        current_blocks[-1] = last_block.model_copy(update={"text": normalize_text(last_block.text.rstrip() + lead_sentence.lstrip())})
        if remainder:
            next_blocks[0] = next_blocks[0].model_copy(update={"text": remainder})
        else:
            del next_blocks[0]


def _ends_with_sentence(text: str) -> bool:
    """Return whether text ends with a configured sentence terminator."""

    text = normalize_text(text)
    return bool(text) and text[-1] in PAGE_SENTENCE_ENDINGS


def _take_lead_sentence(text: str) -> tuple[str, str]:
    """Take text from the beginning through the first sentence terminator."""

    text = normalize_text(text).lstrip()
    for index, char in enumerate(text):
        if char in PAGE_SENTENCE_ENDINGS:
            return text[: index + 1].strip(), text[index + 1 :].lstrip()
    return "", text


def _merged_blocks_length(blocks: list[SemanticBlock], next_block: SemanticBlock) -> int:
    """Return normalized character length after appending one block with block separators."""

    texts = [block.text for block in blocks] + [next_block.text]
    return len(normalize_text("\n\n".join(text for text in texts if text)))


def _flush_text_block_chunk(
    chunks: list[ChunkRecord],
    blocks: list[SemanticBlock],
    file_name: str,
    page_block_indices: dict[tuple[int, str, int], int],
) -> None:
    """Append one chunk for merged small text blocks."""

    if not blocks:
        return
    meta_location = _meta_location_for_blocks(blocks[0].doc_id, blocks[0].page_start, blocks, page_block_indices, part=0)
    document = _format_text_document(
        file_name,
        blocks[0].page_start,
        blocks[-1].page_end,
        "\n\n".join(block.text for block in blocks if block.text),
    )
    chunks.append(
        _make_chunk(
            blocks[0].doc_id,
            file_name,
            "text",
            blocks[0].page_start,
            blocks[-1].page_end,
            "",
            document,
            meta_location,
        )
    )


def _build_large_block_chunks(
    block: SemanticBlock,
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    page_block_indices: dict[tuple[int, str, int], int],
) -> list[ChunkRecord]:
    """Split one oversized text block into sentence-aware overlapping chunks."""

    text = normalize_text(block.text)
    chunks: list[ChunkRecord] = []
    start = 0
    part_index = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end, ended_by_sentence = _best_sentence_break(text, start, hard_end)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                _make_chunk(
                    block.doc_id,
                    file_name,
                    "text",
                    block.page_start,
                    block.page_end,
                    "",
                    _format_text_document(file_name, block.page_start, block.page_end, chunk_text),
                    _meta_location_for_blocks(block.doc_id, block.page_start, [block], page_block_indices, part=part_index),
                )
            )
            part_index += 1
        if end >= len(text):
            break
        if ended_by_sentence:
            start = _optimized_next_overlap_start(text, start, end, chunk_overlap)
        else:
            start = max(start + 1, end - chunk_overlap)
    return chunks


def _best_sentence_break(text: str, start: int, hard_end: int) -> tuple[int, bool]:
    """Choose a split point after a sentence terminator near hard_end when available."""

    if hard_end >= len(text):
        return len(text), True
    window = text[start:hard_end]
    min_pos = int(len(window) * 0.6)
    best = -1
    for index, char in enumerate(window):
        if index >= min_pos and char in PAGE_SENTENCE_ENDINGS:
            best = index
    if best >= 0:
        return start + best + 1, True
    return hard_end, False


def _optimized_next_overlap_start(text: str, current_start: int, end: int, chunk_overlap: int) -> int:
    """Move the next oversized-block window start to a nearby semantic boundary when possible."""

    rough_start = max(current_start + 1, end - chunk_overlap)
    search_start = max(current_start + 1, end - (chunk_overlap * 2))
    candidates = _overlap_starts_between(text, search_start, end)
    left = [candidate for candidate in candidates if candidate <= rough_start]
    if left:
        return max(left)
    right = [candidate for candidate in candidates if candidate > rough_start]
    if right:
        return min(right)
    return rough_start


def _overlap_starts_between(text: str, start: int, end: int) -> list[int]:
    """Return semantic-boundary start offsets between start and end."""

    starts: list[int] = []
    if start <= 0:
        starts.append(0)
    for index in range(max(0, start - 1), min(len(text), end)):
        if text[index] not in OVERLAP_SEMANTIC_BOUNDARIES:
            continue
        sentence_start = index + 1
        while sentence_start < len(text) and text[sentence_start].isspace():
            sentence_start += 1
        if start <= sentence_start < end:
            starts.append(sentence_start)
    return starts


def _format_text_document(file_name: str, page_start: int, page_end: int, text: str) -> str:
    """Return pure text content for a text chunk document."""

    return normalize_text(text)


def _build_special_chunk(
    block: SemanticBlock,
    file_name: str,
    chunk_type: str,
    page_block_indices: dict[tuple[int, str, int], int],
) -> ChunkRecord:
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
        _meta_location_for_blocks(block.doc_id, block.page_start, [block], page_block_indices, part=0),
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
    meta_location: str,
) -> ChunkRecord:
    """Create a ChunkRecord with flat metadata, content_md5, and stable chunk id."""

    document = normalize_text(document)
    content_md5 = md5_text(document)
    chunk_id = "chunk_" + md5_text(f"{meta_location}{content_md5}")
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
            "content_md5": content_md5,
            "meta_location": meta_location,
        },
    )


def _meta_location_for_blocks(
    doc_id: str,
    page: int,
    blocks: list[SemanticBlock],
    page_block_indices: dict[tuple[int, str, int], int],
    part: int,
) -> str:
    """Build a stable source-location id for one final chunk."""

    block_indices = [_semantic_block_page_index(block, page_block_indices) for block in blocks]
    return f"{doc_id}:p{page}:b{min(block_indices)}-{max(block_indices)}:part{part}"


def _semantic_block_page_index(
    block: SemanticBlock,
    page_block_indices: dict[tuple[int, str, int], int],
) -> int:
    """Return one SemanticBlock's one-based page-local index."""

    return page_block_indices[(block.page_start, block.block_id, block.reading_order)]


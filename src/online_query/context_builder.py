# Builds cited prompt context from retrieved chunks.
from __future__ import annotations

from online_query.schema import Citation, RetrievedChunk


def build_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> tuple[str, list[Citation], list[RetrievedChunk]]:
    """Build numbered evidence text, citations, and the included chunk list."""

    blocks: list[str] = []
    citations: list[Citation] = []
    used_chunks: list[RetrievedChunk] = []
    current_length = 0

    for chunk in chunks:
        citation_id = len(citations) + 1
        citation = Citation.from_chunk(chunk, citation_id=citation_id)
        header = _format_header(citation)
        block = f"{header}\n{chunk.document.strip()}"
        next_length = current_length + len(block) + (2 if blocks else 0)
        if blocks and next_length > max_chars:
            break
        blocks.append(block)
        citations.append(citation)
        used_chunks.append(chunk)
        current_length = next_length
        if current_length >= max_chars:
            break

    return "\n\n".join(blocks), citations, used_chunks


def _format_header(citation: Citation) -> str:
    """Format one citation header for prompt context."""

    pages = f"{citation.page_start}-{citation.page_end}"
    return f"[{citation.citation_id}] {citation.file_name}, pages {pages}, type={citation.chunk_type}"

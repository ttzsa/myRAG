# Defines shared schemas for online dense RAG answers, citations, and retrieved chunks.
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Normalized retrieval result independent of Chroma's raw response shape."""

    chunk_id: str
    document: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    channel: str
    rank: int
    score: float = 0.0
    distance: float | None = None


class Citation(BaseModel):
    """Citation metadata returned with a generated answer."""

    citation_id: int
    chunk_id: str
    file_name: str = ""
    page_start: int = 0
    page_end: int = 0
    chunk_type: str = ""
    source: str = ""

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, citation_id: int) -> "Citation":
        """Build a citation from one retrieved chunk's metadata."""

        metadata = chunk.metadata or {}
        return cls(
            citation_id=citation_id,
            chunk_id=chunk.chunk_id,
            file_name=str(metadata.get("file_name", "")),
            page_start=_as_int(metadata.get("page_start")),
            page_end=_as_int(metadata.get("page_end")),
            chunk_type=str(metadata.get("chunk_type", "")),
            source=str(metadata.get("source", "")),
        )


class AnswerResult(BaseModel):
    """Final online QA response returned by the pipeline and CLI."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence: str = "low"
    retrieval_debug_info: dict[str, Any] | None = None


def _as_int(value: Any) -> int:
    """Convert scalar metadata values to int, returning 0 when missing."""

    if value is None or value == "":
        return 0
    return int(value)

# Defines shared Pydantic schemas for MinerU blocks, chunks, PDF candidates, and manifests.
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SemanticBlock(BaseModel):
    """Represents one normalized MinerU block before chunk construction."""

    block_id: str
    doc_id: str
    page_start: int
    page_end: int
    raw_type: str
    rag_type: str
    text: str = ""
    caption: str = ""
    source: str = ""
    bbox: str = ""
    reading_order: int


class ChunkRecord(BaseModel):
    """Represents one final RAG chunk record."""

    id: str
    document: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """Represents document-level indexing metadata for future manifest storage."""

    doc_id: str
    file_name: str
    source_path: str = ""
    pdf_md5: str = ""
    file_size: int = 0
    mineru_output_dir: str = ""
    parser_method: str = ""
    parse_status: str = ""
    parse_output_dir: str = ""
    content_list_v2_path: str = ""
    content_list_v2_md5: str = ""
    parse_error: str = ""
    chunk_status: str = ""
    chunk_path: str = ""
    chunk_file_md5: str = ""
    chunk_config_hash: str = ""
    vlm_mode: str = ""
    chunk_error: str = ""
    chunk_count: int = 0
    text_chunk_count: int = 0
    image_chunk_count: int = 0
    table_chunk_count: int = 0
    indexed_chunk_file_md5: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    index_error: str = ""
    indexed_at: str = ""
    index_status: str = "pending"
    error_message: str = ""


class PdfCandidate(BaseModel):
    """Represents one source PDF discovered during filesystem scanning."""

    source_path: str
    file_name: str
    file_size: int
    pdf_md5: str
    modified_time: float


class MinerUOutputLocation(BaseModel):
    """Represents located MinerU output paths for one source PDF."""

    mineru_output_dir: str
    content_list_v2_path: str = ""
    content_list_path: str = ""
    markdown_path: str = ""
    images_dir: str = ""
    success: bool = False
    warning: str = ""
    error_message: str = ""


class DocumentManifest(BaseModel):
    """Represents the processed_pdfs.json payload."""

    documents: list[DocumentRecord] = Field(default_factory=list)

    def find_by_md5(self, pdf_md5: str) -> DocumentRecord | None:
        """Find the first document record with the given PDF content MD5."""

        for document in self.documents:
            if document.pdf_md5 == pdf_md5:
                return document
        return None

    def find_by_source_path(self, source_path: str) -> DocumentRecord | None:
        """Find the first document record with the given normalized source path."""

        for document in self.documents:
            if document.source_path == source_path:
                return document
        return None

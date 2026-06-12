# Converts MinerU raw blocks into normalized SemanticBlock records.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from offline_index.schema import SemanticBlock
from offline_index.utils import normalize_text


NOISE_TYPES = {"page_number", "page_aside_text", "page_footnote", "header", "footer"}
TEXT_TYPES = {"title", "paragraph", "text", "list", "equation_inline", "equation_interline"}
SPECIAL_TYPES = {"image": "image", "table": "table"}


def convert_blocks(blocks: list[dict[str, Any]], doc_id: str, images_dir: Path) -> list[SemanticBlock]:
    """Convert raw MinerU blocks to SemanticBlock records and discard noise blocks."""

    semantic_blocks: list[SemanticBlock] = []
    images_dir = images_dir.resolve()

    for index, block in enumerate(blocks):
        raw_type = str(block.get("type", "")).strip()
        if raw_type in NOISE_TYPES:
            continue

        rag_type = _rag_type(raw_type)
        if rag_type is None:
            continue

        page = int(block.get("page_idx", 0)) + 1
        reading_order = int(block.get("reading_order", index))
        source = ""
        caption = ""
        if rag_type in {"image", "table"}:
            source = _resolve_source(_source_path(block), images_dir)
            caption = _extract_caption(raw_type, block)

        semantic_blocks.append(
            SemanticBlock(
                block_id=f"block_{reading_order:06d}",
                doc_id=doc_id,
                page_start=page,
                page_end=page,
                raw_type=raw_type,
                rag_type=rag_type,
                text=_extract_block_text(raw_type, block),
                caption=caption,
                source=source,
                bbox=json.dumps(block.get("bbox", ""), ensure_ascii=False),
                reading_order=reading_order,
            )
        )

    return semantic_blocks


def _rag_type(raw_type: str) -> str | None:
    """Map a MinerU block type to a RAG block type."""

    if raw_type in TEXT_TYPES:
        return "text"
    if raw_type in SPECIAL_TYPES:
        return SPECIAL_TYPES[raw_type]
    return None


def _extract_block_text(raw_type: str, block: dict[str, Any]) -> str:
    """Extract searchable text from a MinerU block according to its raw type."""

    content = block.get("content", {})
    if not isinstance(content, dict):
        return normalize_text(str(content))

    if raw_type == "title":
        return extract_all_content(content.get("title_content"))
    if raw_type in {"paragraph", "text"}:
        return extract_all_content(content.get("paragraph_content", content.get("text", content)))
    if raw_type == "list":
        return _extract_list_text(content.get("list_items", []))
    if raw_type == "image":
        return _join_parts(
            [
                extract_all_content(content.get("image_caption")),
                extract_all_content(content.get("image_footnote")),
            ]
        )
    if raw_type == "table":
        return _join_parts(
            [
                extract_all_content(content.get("table_caption")),
                extract_all_content(content.get("table_footnote")),
                _extract_table_body(content),
            ]
        )
    return extract_all_content(content)


def _extract_list_text(items: Any) -> str:
    """Extract newline-separated text from MinerU list items."""

    lines: list[str] = []
    if not isinstance(items, list):
        return extract_all_content(items)
    for item in items:
        if isinstance(item, dict):
            text = extract_all_content(item.get("item_content", item))
        else:
            text = extract_all_content(item)
        if text:
            lines.append(text)
    return normalize_text("\n".join(lines))


def extract_all_content(value: Any) -> str:
    """Recursively extract text content from common MinerU nested structures."""

    parts: list[str] = []

    def visit(node: Any) -> None:
        """Append text fragments from one nested node into the outer parts list."""

        if node is None:
            return
        if isinstance(node, str):
            parts.append(node)
            return
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if isinstance(node, dict):
            handled = False
            for key in (
                "content",
                "title_content",
                "paragraph_content",
                "item_content",
                "math_content",
                "text",
                "html",
                "table_body",
                "table_text",
            ):
                if key in node:
                    handled = True
                    visit(node[key])
            if handled:
                return
            for nested in node.values():
                if isinstance(nested, (list, dict)):
                    visit(nested)

    visit(value)
    return normalize_text(" ".join(parts))


def _extract_caption(raw_type: str, block: dict[str, Any]) -> str:
    """Extract the complete caption text for image and table blocks."""

    content = block.get("content", {})
    if not isinstance(content, dict):
        return ""
    if raw_type == "image":
        return extract_all_content(content.get("image_caption"))
    if raw_type == "table":
        return extract_all_content(content.get("table_caption"))
    return ""


def _extract_table_body(content: dict[str, Any]) -> str:
    """Extract fallback table body text from common MinerU table fields."""

    for key in ("table_body", "table_text", "html"):
        text = extract_all_content(content.get(key))
        if text:
            return text
    return ""


def _join_parts(parts: list[str]) -> str:
    """Join non-empty text parts with newlines and normalize the result."""

    return normalize_text("\n".join(part for part in parts if part))


def _source_path(block: dict[str, Any]) -> str:
    """Read the relative or absolute image source path from a MinerU block."""

    content = block.get("content", {})
    if not isinstance(content, dict):
        return ""
    image_source = content.get("image_source", {})
    if isinstance(image_source, dict):
        return str(image_source.get("path", "") or "")
    return ""


def _resolve_source(raw_path: str, images_dir: Path) -> str:
    """Resolve a MinerU asset path to an absolute local path under images_dir."""

    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())
    if path.parts and path.parts[0].lower() == images_dir.name.lower():
        return str((images_dir.parent / path).resolve())
    return str((images_dir / path).resolve())

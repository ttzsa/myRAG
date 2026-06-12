# Reads MinerU output JSON and flattens page-organized blocks.
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def read_content_list_v2(path: Path) -> list[Any]:
    """Load MinerU content_list_v2 JSON from disk."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"content_list_v2 must be a list: {path}")
    return data


def flatten_blocks(pages: list[Any]) -> list[dict[str, Any]]:
    """Flatten page-organized MinerU blocks and add page/order metadata."""

    flattened: list[dict[str, Any]] = []
    reading_order = 0

    for page_idx, page in enumerate(pages):
        page_blocks = page if isinstance(page, list) else [page]
        for block in page_blocks:
            if not isinstance(block, dict):
                continue
            item = copy.deepcopy(block)
            item["page_idx"] = page_idx
            item["reading_order"] = reading_order
            flattened.append(item)
            reading_order += 1

    return flattened

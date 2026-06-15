# Stores cached VLM summaries on disk to avoid repeated image/table requests.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline_index.schema import SemanticBlock
from offline_index.utils import ensure_dir, md5_file


class SummaryCache:
    """Simple JSON-backed cache for VLM block summaries."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    def get(self, key: str) -> str | None:
        """Return a cached summary when present."""

        item = self._items.get(key)
        if not item:
            return None
        summary = item.get("summary")
        return str(summary) if summary else None

    def set(self, key: str, summary: str, metadata: dict[str, Any]) -> None:
        """Store one summary and any lightweight metadata."""

        payload = dict(metadata)
        payload["summary"] = summary
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self._items[key] = payload

    def save(self) -> None:
        """Persist the cache file as UTF-8 JSON."""

        ensure_dir(self.path.parent)
        payload = {"items": self._items}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def make_key(self, block: SemanticBlock, model: str, prompt_version: str) -> str:
        """Build a cache key from only the source image bytes."""

        source = Path(block.source) if block.source else None
        return md5_file(source) if source and source.exists() else ""

    def _load(self) -> None:
        """Load any existing cache file."""

        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("items"), dict):
            self._items = payload["items"]

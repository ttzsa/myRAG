# Provides embedding interfaces and a deterministic mock embedder for offline indexing tests.
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from offline_index.config_loader import EmbeddingConfig


class BaseEmbedder(ABC):
    """Defines the common interface for document and query embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query text using the document embedding implementation."""

        return self.embed_documents([text])[0]


class MockEmbedder(BaseEmbedder):
    """Generates deterministic pseudo embeddings from MD5 hashes."""

    def __init__(self, dimension: int = 384) -> None:
        """Store the output vector dimension for mock embeddings."""

        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each text into a deterministic vector with the configured dimension."""

        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Map one text to a stable pseudo-random vector in the range [-1, 1]."""

        vector: list[float] = []
        for index in range(self.dimension):
            digest = hashlib.md5(f"{text}|{index}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:4], byteorder="big", signed=False)
            vector.append((value / 0xFFFFFFFF) * 2.0 - 1.0)
        return vector


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Generates embeddings through an OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        api_key_env: str = "EMBEDDING_API_KEY",
        batch_size: int = 32,
        timeout: float = 60,
        dimension: int | None = None,
        max_retries: int = 3,
    ) -> None:
        """Store OpenAI-compatible API settings and read the API key from the environment."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if dimension is not None and dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.batch_size = batch_size
        self.timeout = timeout
        self.dimension = dimension
        self.max_retries = max_retries

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts by sending batched OpenAI-compatible API requests."""

        if not texts:
            return []
        if not self.base_url:
            raise ValueError("OpenAI-compatible embedder requires base_url")
        if not self.model:
            raise ValueError("OpenAI-compatible embedder requires model")
        if not self.api_key:
            raise ValueError(f"environment variable {self.api_key_env} is not set")

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one embedding request batch and parse the response vectors."""

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimension is not None:
            payload["dimensions"] = self.dimension

        request = urllib.request.Request(
            url=self._embeddings_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._send_request_with_retries(request)
        return self._parse_embeddings_response(body, expected_count=len(texts))

    def _send_request_with_retries(self, request: urllib.request.Request) -> str:
        """Send one HTTP request and retry transient network or server failures."""

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if not self._is_retryable_http_status(exc.code) or attempt >= self.max_retries:
                    raise RuntimeError(f"embedding request failed with HTTP {exc.code}: {error_body[:500]}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"embedding request failed: {exc.reason}") from exc
                last_error = exc
            self._sleep_before_retry(attempt)
        raise RuntimeError(f"embedding request failed after retries: {last_error}")

    def _is_retryable_http_status(self, status_code: int) -> bool:
        """Return whether an HTTP status is worth retrying."""

        return status_code == 429 or 500 <= status_code < 600

    def _sleep_before_retry(self, attempt: int) -> None:
        """Sleep briefly with exponential backoff before the next retry."""

        time.sleep(min(2**attempt, 8))

    def _embeddings_url(self) -> str:
        """Build the OpenAI-compatible embeddings endpoint URL."""

        return f"{self.base_url.rstrip('/')}/embeddings"

    def _parse_embeddings_response(self, body: str, expected_count: int) -> list[list[float]]:
        """Validate and extract embeddings from an OpenAI-compatible response body."""

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("embedding response is not valid JSON") from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError("embedding response missing data list")
        if all(isinstance(item, dict) and "index" in item for item in data):
            data = sorted(data, key=lambda item: int(item["index"]))

        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ValueError("embedding response item missing embedding list")
            embedding = [float(value) for value in item["embedding"]]
            if self.dimension is not None and len(embedding) != self.dimension:
                raise ValueError(
                    f"embedding dimension mismatch: configured {self.dimension}, response {len(embedding)}"
                )
            embeddings.append(embedding)
        if len(embeddings) != expected_count:
            raise ValueError(f"embedding response count mismatch: expected {expected_count}, got {len(embeddings)}")

        return embeddings


def create_embedder(name: str, config: "EmbeddingConfig | None" = None) -> BaseEmbedder:
    """Create an embedder by name for CLI scripts."""

    normalized = name.strip().lower()
    if normalized == "mock":
        dimension = config.mock_dimension if config else 384
        return MockEmbedder(dimension=dimension)
    if normalized in {"openai", "openai-compatible", "openai_compatible"}:
        if config is None:
            raise ValueError("OpenAI-compatible embedder requires EmbeddingConfig")
        return OpenAICompatibleEmbedder(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            batch_size=config.batch_size,
            timeout=config.timeout,
            dimension=config.dimension,
        )
    raise ValueError(f"unsupported embedder: {name}")

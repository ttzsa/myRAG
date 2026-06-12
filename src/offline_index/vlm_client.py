# Calls an OpenAI-compatible VLM endpoint with local image files encoded as data URLs.
from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from offline_index.config_loader import VLMConfig


class OpenAICompatibleVLMClient:
    """Minimal OpenAI-compatible image summarization client implemented with urllib."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60,
        max_retries: int = 3,
    ) -> None:
        if not base_url.strip():
            raise ValueError("VLM base_url is required")
        if not api_key.strip():
            raise ValueError("VLM api_key is required")
        if not model.strip():
            raise ValueError("VLM model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)

    def summarize_image(self, image_path: Path, prompt: str) -> str:
        """Send one local image plus prompt text and return the model summary text."""

        image_url = _to_data_url(Path(image_path))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for _ in range(self.max_retries):
            req = request.Request(url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return _extract_text(data)
            except (error.URLError, error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError(f"VLM request failed after {self.max_retries} attempts: {last_error}") from last_error


def create_vlm_client(config: VLMConfig, api_key: str | None = None) -> OpenAICompatibleVLMClient:
    """Create the configured VLM client or fail with a clear configuration error."""

    provider = config.provider.strip().lower()
    if provider not in {"openai-compatible", "openai_compatible"}:
        raise ValueError(f"unsupported VLM provider: {config.provider}")
    if not config.base_url.strip():
        raise ValueError("VLM base_url is required")
    if not config.model.strip():
        raise ValueError("VLM model is required")
    resolved_key = api_key if api_key is not None else config.api_key or os.getenv("VLM_API_KEY", "")
    if not resolved_key.strip():
        raise ValueError("VLM API key is required in VLM_API_KEY")
    return OpenAICompatibleVLMClient(
        base_url=config.base_url,
        api_key=resolved_key,
        model=config.model,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _to_data_url(image_path: Path) -> str:
    """Read one local image and encode it as a data URL."""

    image_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_text(payload: dict[str, Any]) -> str:
    """Extract the first assistant message text from an OpenAI-compatible response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("VLM response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    raise ValueError("VLM response content has unsupported shape")

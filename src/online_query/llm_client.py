# OpenAI-compatible chat client for online QA.
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Common interface for answer generation clients."""

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Generate an answer for one question and evidence context."""


class OpenAICompatibleChatClient(BaseLLMClient):
    """Calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60,
        max_retries: int = 3,
    ) -> None:
        """Store API settings and read the API key from the environment."""

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("CHAT_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, question: str, context: str) -> str:
        """Generate a grounded answer from supplied evidence."""

        if not self.base_url:
            raise ValueError("CHAT_BASE_URL is required for OpenAI-compatible chat")
        if not self.model:
            raise ValueError("CHAT_MODEL is required for OpenAI-compatible chat")
        if not self.api_key:
            raise ValueError("CHAT_API_KEY is required")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个严格基于证据回答的 RAG 助手。"
                        "只能使用用户提供的证据回答；证据不足时要明确说明。"
                        "回答中必须使用 [1]、[2] 这样的证据编号标注依据。"
                    ),
                },
                {"role": "user", "content": _build_user_prompt(question, context)},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._send_request_with_retries(request)
        return self._parse_answer(body)

    def _send_request_with_retries(self, request: urllib.request.Request) -> str:
        """Send one HTTP request and retry transient failures."""

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if not _is_retryable_http_status(exc.code) or attempt >= self.max_retries:
                    raise RuntimeError(f"chat request failed with HTTP {exc.code}: {error_body[:500]}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"chat request failed: {exc.reason}") from exc
                last_error = exc
            time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"chat request failed after retries: {last_error}")

    def _parse_answer(self, body: str) -> str:
        """Parse an OpenAI-compatible chat completion response."""

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("chat response is not valid JSON") from exc
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("chat response missing message content")
        return content.strip()


def _build_user_prompt(question: str, context: str) -> str:
    """Build the user prompt containing question and evidence."""

    return (
        "请基于以下证据回答问题。\n\n"
        f"问题：{question}\n\n"
        "证据：\n"
        f"{context if context.strip() else '未检索到证据。'}\n\n"
        "回答要求：\n"
        "1. 只基于证据回答。\n"
        "2. 每个关键结论后标注证据编号，例如 [1]。\n"
        "3. 如果证据不足，直接说明无法从当前知识库确认。"
    )


def _is_retryable_http_status(status_code: int) -> bool:
    """Return whether a chat HTTP status is worth retrying."""

    return status_code == 429 or 500 <= status_code < 600

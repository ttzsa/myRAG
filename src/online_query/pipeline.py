# Orchestrates the V0 online dense RAG question-answering loop.
from __future__ import annotations

from typing import Any

from online_query.context_builder import build_context
from online_query.llm_client import BaseLLMClient
from online_query.schema import AnswerResult, RetrievedChunk


class OnlineQAPipeline:
    """Runs query -> dense retrieval -> context construction -> LLM answer."""

    def __init__(
        self,
        dense_retriever: Any,
        llm_client: BaseLLMClient,
        top_k: int = 5,
        max_context_chars: int = 6000,
    ) -> None:
        """Store pipeline dependencies and runtime settings."""

        self.dense_retriever = dense_retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def answer(self, query: str, include_debug: bool = False) -> AnswerResult:
        """Answer one query using dense retrieval and cited context."""

        retrieved_chunks = self.dense_retriever.retrieve(query, top_k=self.top_k)
        context, citations, used_chunks = build_context(retrieved_chunks, max_chars=self.max_context_chars)
        answer = self.llm_client.generate(query, context)
        return AnswerResult(
            answer=answer,
            citations=citations,
            used_chunks=used_chunks,
            confidence=_estimate_confidence(used_chunks),
            retrieval_debug_info=_build_debug_info(query, retrieved_chunks, context) if include_debug else None,
        )


def _estimate_confidence(chunks: list[RetrievedChunk]) -> str:
    """Estimate a coarse confidence label from retrieved evidence count."""

    if len(chunks) >= 3:
        return "high"
    if len(chunks) >= 1:
        return "medium"
    return "low"


def _build_debug_info(query: str, chunks: list[RetrievedChunk], context: str) -> dict[str, Any]:
    """Build compact retrieval debug information for CLI inspection."""

    return {
        "query": query,
        "retrieval_mode": "dense",
        "retrieved_count": len(chunks),
        "context_chars": len(context),
        "chunks": [chunk.model_dump() for chunk in chunks],
    }

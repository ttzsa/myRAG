# Dense retrieval wrapper around the existing Chroma collection and embedder.
from __future__ import annotations

from typing import Any

from online_query.schema import RetrievedChunk


class DenseRetriever:
    """Embeds a user query and retrieves nearest chunks from Chroma."""

    def __init__(self, collection: Any, embedder: Any) -> None:
        """Store Chroma collection and embedder dependencies."""

        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return normalized dense retrieval results."""

        query_embedding = self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            document = documents[index] if index < len(documents) and documents[index] else ""
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    document=str(document),
                    metadata=dict(metadata),
                    channel="dense",
                    rank=index + 1,
                    distance=float(distance) if distance is not None else None,
                    score=_distance_to_score(distance),
                )
            )
        return chunks


def _distance_to_score(distance: Any) -> float:
    """Convert a Chroma distance into a simple higher-is-better score."""

    if distance is None:
        return 0.0
    value = float(distance)
    return 1.0 / (1.0 + max(value, 0.0))

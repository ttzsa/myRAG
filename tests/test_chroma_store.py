# Regression tests for ChromaDB collection add, query, and delete wrappers.
import tempfile
import unittest

from offline_index.chroma_store import (
    add_chunks,
    delete_by_doc_id,
    get_chroma_client,
    get_or_create_collection,
    query,
    reset_collection,
    sync_chunks_by_location,
)
from offline_index.embedder import MockEmbedder
from offline_index.schema import ChunkRecord


class ChromaStoreTests(unittest.TestCase):
    """Covers ChromaDB persistence wrapper behavior with mock embeddings."""

    def test_add_query_and_delete_chunks_by_doc_id(self):
        """Verify chunks can be written, queried, and deleted by doc_id."""

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            client = get_chroma_client(tmp)
            collection = reset_collection(client, "rag_chunks_test")
            chunks = [
                ChunkRecord(
                    id="chunk_1",
                    document="DarkIR low light restoration",
                    metadata={
                        "doc_id": "doc_1",
                        "file_name": "paper.pdf",
                        "chunk_type": "text",
                        "page_start": 1,
                        "page_end": 1,
                        "source": "",
                        "content_md5": "a",
                        "meta_location": "doc_1:p1:b1-1:part0",
                    },
                ),
                ChunkRecord(
                    id="chunk_2",
                    document="table metrics",
                    metadata={
                        "doc_id": "doc_1",
                        "file_name": "paper.pdf",
                        "chunk_type": "table",
                        "page_start": 2,
                        "page_end": 2,
                        "source": ["complex", "metadata"],
                        "content_md5": "b",
                        "meta_location": "doc_1:p2:b1-1:part0",
                    },
                ),
            ]
            embedder = MockEmbedder(dimension=16)
            embeddings = embedder.embed_documents([chunk.document for chunk in chunks])

            add_chunks(collection, chunks, embeddings)
            self.assertEqual(collection.count(), 2)

            result = query(collection, embedder.embed_query("DarkIR"), top_k=1)
            self.assertEqual(len(result["ids"][0]), 1)
            self.assertIn(result["ids"][0][0], {"chunk_1", "chunk_2"})

            delete_by_doc_id(collection, "doc_1")
            self.assertEqual(collection.count(), 0)

    def test_get_or_create_collection_reuses_existing_collection(self):
        """Verify collection lookup is stable across calls."""

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            client = get_chroma_client(tmp)
            first = get_or_create_collection(client, "rag_chunks_test")
            second = get_or_create_collection(client, "rag_chunks_test")

            self.assertEqual(first.name, second.name)

    def test_sync_chunks_by_location_embeds_only_new_or_changed_chunks(self):
        """Verify meta_location drives incremental updates while content_md5 detects changes."""

        class FakeCollection:
            def __init__(self):
                self.items = {
                    "old_id": {
                        "metadata": {"doc_id": "doc_1", "meta_location": "doc_1:p1:b1-1:part0", "content_md5": "old"},
                        "document": "old",
                        "embedding": [0.0],
                    },
                    "same_id": {
                        "metadata": {"doc_id": "doc_1", "meta_location": "doc_1:p1:b2-2:part0", "content_md5": "same-hash"},
                        "document": "same",
                        "embedding": [1.0],
                    },
                    "stale_id": {
                        "metadata": {"doc_id": "doc_1", "meta_location": "doc_1:p1:b3-3:part0", "content_md5": "stale-hash"},
                        "document": "stale",
                        "embedding": [4.0],
                    },
                }
                self.deleted = []
                self.added = []

            def get(self, where=None, include=None):
                doc_id = where["doc_id"]
                matches = [
                    (item_id, item)
                    for item_id, item in self.items.items()
                    if item["metadata"]["doc_id"] == doc_id
                ]
                return {
                    "ids": [item_id for item_id, _ in matches],
                    "metadatas": [item["metadata"] for _, item in matches],
                }

            def delete(self, ids=None, where=None):
                self.deleted.extend(ids or [])
                for item_id in ids or []:
                    del self.items[item_id]

            def add(self, ids, documents, embeddings, metadatas):
                for item_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
                    self.added.append(item_id)
                    self.items[item_id] = {
                        "metadata": metadata,
                        "document": document,
                        "embedding": embedding,
                    }

        class CountingEmbedder:
            def __init__(self):
                self.texts = []

            def embed_documents(self, texts):
                self.texts.extend(texts)
                return [[float(index)] for index, _ in enumerate(texts)]

        collection = FakeCollection()
        embedder = CountingEmbedder()
        chunks = [
            ChunkRecord(
                id="new_id",
                document="new",
                metadata={"doc_id": "doc_1", "meta_location": "doc_1:p1:b1-1:part0", "content_md5": "new"},
            ),
            ChunkRecord(
                id="same_new_id",
                document="same",
                metadata={"doc_id": "doc_1", "meta_location": "doc_1:p1:b2-2:part0", "content_md5": "same-hash"},
            ),
            ChunkRecord(
                id="brand_new_id",
                document="brand new",
                metadata={"doc_id": "doc_1", "meta_location": "doc_1:p1:b4-4:part0", "content_md5": "brand-new-hash"},
            ),
        ]

        result = sync_chunks_by_location(collection, chunks, embedder)

        self.assertEqual(result, {"added": 2, "updated": 1, "skipped": 1, "deleted": 1, "embedded": 2})
        self.assertEqual(collection.deleted, ["old_id", "stale_id"])
        self.assertEqual(collection.added, ["new_id", "brand_new_id"])
        self.assertEqual(embedder.texts, ["new", "brand new"])


if __name__ == "__main__":
    unittest.main()

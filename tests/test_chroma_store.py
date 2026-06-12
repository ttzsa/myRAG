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
                        "content_hash": "a",
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
                        "content_hash": "b",
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


if __name__ == "__main__":
    unittest.main()

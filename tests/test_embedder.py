# Regression tests for deterministic mock embeddings and chunk preview loading.
import json
import os
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_index.chunk_loader import load_chunks
from offline_index.embedder import MockEmbedder, OpenAICompatibleEmbedder


class FakeHttpResponse:
    """Provides a minimal context-manager response for urllib tests."""

    def __init__(self, payload):
        """Store a JSON-serializable response payload."""

        self.payload = payload

    def __enter__(self):
        """Return this fake response from a with block."""

        return self

    def __exit__(self, exc_type, exc, tb):
        """Do not suppress exceptions from the with block."""

        return False

    def read(self):
        """Return the payload as UTF-8 encoded JSON bytes."""

        return json.dumps(self.payload).encode("utf-8")


class EmbedderTests(unittest.TestCase):
    """Covers the mock embedder and chunk preview loader."""

    def test_mock_embedder_is_deterministic_and_dimensioned(self):
        """Verify the same text always maps to the same fixed-size vector."""

        embedder = MockEmbedder(dimension=8)

        first = embedder.embed_documents(["DarkIR", "other"])
        second = embedder.embed_documents(["DarkIR"])

        self.assertEqual(len(first), 2)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[0], first[1])
        self.assertEqual(embedder.embed_query("DarkIR"), first[0])
        self.assertTrue(all(isinstance(value, float) for value in first[0]))

    def test_load_chunks_accepts_list_and_dict_payloads(self):
        """Verify chunks_preview JSON can be loaded from list or {'chunks': [...]} shapes."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = {
                "id": "chunk_1",
                "document": "hello",
                "metadata": {"doc_id": "doc_1", "chunk_type": "text"},
            }
            list_path = root / "list.json"
            dict_path = root / "dict.json"
            list_path.write_text(json.dumps([chunk]), encoding="utf-8")
            dict_path.write_text(json.dumps({"chunks": [chunk]}), encoding="utf-8")

            self.assertEqual(load_chunks(list_path)[0].id, "chunk_1")
            self.assertEqual(load_chunks(dict_path)[0].document, "hello")

    def test_load_chunks_rejects_missing_required_fields(self):
        """Verify malformed chunk records fail with a clear ValueError."""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps([{"id": "chunk_1", "metadata": {}}]), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_chunks(path)

    def test_openai_compatible_embedder_posts_embedding_request_with_dimension(self):
        """Verify OpenAI-compatible embeddings use the configured endpoint and dimensions."""

        response = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        with patch.dict(os.environ, {"TEST_EMBEDDING_KEY": "secret"}):
            with patch("offline_index.embedder.urllib.request.urlopen", return_value=FakeHttpResponse(response)) as urlopen:
                embedder = OpenAICompatibleEmbedder(
                    base_url="https://example.test/compatible-mode/v1",
                    model="text-embedding-v4",
                    api_key_env="TEST_EMBEDDING_KEY",
                    batch_size=8,
                    timeout=12.5,
                    dimension=3,
                )

                embeddings = embedder.embed_documents(["alpha", "beta"])

        self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://example.test/compatible-mode/v1/embeddings")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(payload["model"], "text-embedding-v4")
        self.assertEqual(payload["input"], ["alpha", "beta"])
        self.assertEqual(payload["dimensions"], 3)
        self.assertEqual(payload["encoding_format"], "float")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 12.5)

    def test_openai_compatible_embedder_requires_api_key(self):
        """Verify missing API keys fail before sending an HTTP request."""

        with patch.dict(os.environ, {}, clear=True):
            embedder = OpenAICompatibleEmbedder(
                base_url="https://example.test/v1",
                model="text-embedding-v4",
                api_key_env="MISSING_KEY",
            )

            with self.assertRaises(ValueError):
                embedder.embed_documents(["alpha"])

    def test_openai_compatible_embedder_retries_transient_url_errors(self):
        """Verify transient URL errors are retried before returning embeddings."""

        response = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
        transient_error = urllib.error.URLError("temporary ssl eof")
        with patch.dict(os.environ, {"TEST_EMBEDDING_KEY": "secret"}):
            with patch(
                "offline_index.embedder.urllib.request.urlopen",
                side_effect=[transient_error, FakeHttpResponse(response)],
            ) as urlopen:
                with patch("offline_index.embedder.time.sleep") as sleep:
                    embedder = OpenAICompatibleEmbedder(
                        base_url="https://example.test/v1",
                        model="text-embedding-v4",
                        api_key_env="TEST_EMBEDDING_KEY",
                        max_retries=1,
                    )

                    embeddings = embedder.embed_documents(["alpha"])

        self.assertEqual(embeddings, [[0.1, 0.2]])
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    def test_openai_compatible_embedder_rejects_unexpected_dimension(self):
        """Verify responses must match the configured embedding dimension."""

        response = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
        with patch.dict(os.environ, {"TEST_EMBEDDING_KEY": "secret"}):
            with patch("offline_index.embedder.urllib.request.urlopen", return_value=FakeHttpResponse(response)):
                embedder = OpenAICompatibleEmbedder(
                    base_url="https://example.test/v1",
                    model="text-embedding-v4",
                    api_key_env="TEST_EMBEDDING_KEY",
                    dimension=3,
                )

                with self.assertRaises(ValueError):
                    embedder.embed_documents(["alpha"])


if __name__ == "__main__":
    unittest.main()

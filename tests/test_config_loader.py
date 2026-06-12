# Regression tests for .env based application configuration loading.
import tempfile
import unittest
from pathlib import Path

from offline_index.config_loader import load_config, optional_value, resolve_value


class ConfigLoaderTests(unittest.TestCase):
    """Covers AppConfig loading, type conversion, and CLI override helpers."""

    def test_load_config_converts_env_values_to_nested_config(self):
        """Verify .env fields become typed nested AppConfig values."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "PDF_ROOT=pdfs",
                        "PDF_RECURSIVE=false",
                        "FORCE_REBUILD=true",
                        "MINERU_EXE=C:/mineru/mineru.exe",
                        "MINERU_BACKEND=pipeline",
                        "MINERU_METHOD=auto",
                        "MINERU_OUTPUT_ROOT=out",
                        "RAG_DOCUMENTS_PATH=data/index/rag_documents.json",
                        "DEBUG_DIR=data/debug",
                        "CHROMA_PERSIST_DIRECTORY=data/chroma",
                        "CHROMA_COLLECTION_NAME=rag_chunks",
                        "INGEST_CHUNK_SIZE=900",
                        "INGEST_CHUNK_OVERLAP=100",
                        "EMBEDDING_PROVIDER=mock",
                        "EMBEDDING_API_KEY=embedding-secret",
                        "EMBEDDING_BASE_URL=https://example.test/v1",
                        "EMBEDDING_MODEL=text-embedding",
                        "EMBEDDING_BATCH_SIZE=16",
                        "EMBEDDING_TIMEOUT_SECONDS=12.5",
                        "EMBEDDING_DIMENSION=384",
                        "MOCK_EMBEDDING_DIMENSION=128",
                        "VLM_ENABLED=true",
                        "VLM_PROVIDER=openai-compatible",
                        "VLM_API_KEY=vlm-secret",
                        "VLM_BASE_URL=https://example.test/v1",
                        "VLM_MODEL=qwen-vl",
                        "VLM_TIMEOUT_SECONDS=45",
                        "VLM_MAX_RETRIES=4",
                        "VLM_CACHE_PATH=data/cache/vlm_summaries.json",
                        "VLM_MAX_IMAGES_PER_DOC=12",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(env_path)

            self.assertEqual(config.paths.pdf_root, Path("pdfs"))
            self.assertFalse(config.paths.pdf_recursive)
            self.assertTrue(config.paths.force_rebuild)
            self.assertEqual(config.mineru.exe, Path("C:/mineru/mineru.exe"))
            self.assertEqual(config.chunking.chunk_size, 900)
            self.assertEqual(config.chunking.chunk_overlap, 100)
            self.assertEqual(config.embedding.api_key, "embedding-secret")
            self.assertEqual(config.embedding.batch_size, 16)
            self.assertEqual(config.embedding.timeout, 12.5)
            self.assertEqual(config.embedding.dimension, 384)
            self.assertEqual(config.embedding.mock_dimension, 128)
            self.assertTrue(config.vlm.enabled)
            self.assertEqual(config.vlm.provider, "openai-compatible")
            self.assertEqual(config.vlm.api_key, "vlm-secret")
            self.assertEqual(config.vlm.base_url, "https://example.test/v1")
            self.assertEqual(config.vlm.model, "qwen-vl")
            self.assertEqual(config.vlm.timeout, 45)
            self.assertEqual(config.vlm.max_retries, 4)
            self.assertEqual(config.vlm.cache_path, Path("data/cache/vlm_summaries.json"))
            self.assertEqual(config.vlm.max_images_per_doc, 12)

    def test_resolve_value_prefers_cli_value_over_config_value(self):
        """Verify explicit CLI values override .env derived config values."""

        self.assertEqual(resolve_value("cli", "env"), "cli")
        self.assertEqual(resolve_value(None, "env"), "env")
        self.assertIsNone(optional_value(None))
        self.assertEqual(optional_value("x"), "x")


if __name__ == "__main__":
    unittest.main()

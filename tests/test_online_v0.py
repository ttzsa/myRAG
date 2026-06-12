# Integration tests for the V0 online dense RAG question-answering loop.
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from offline_index.chroma_store import get_chroma_client, get_or_create_collection
from offline_index.config_loader import load_config, resolve_value
from offline_index.embedder import create_embedder
from online_query.config import load_online_config
from online_query.dense_retriever import DenseRetriever
from online_query.llm_client import OpenAICompatibleChatClient
from online_query.pipeline import OnlineQAPipeline


class OnlineV0IntegrationTests(unittest.TestCase):
    """Covers the real V0 query -> dense retrieval -> context -> LLM path."""

    def test_online_config_uses_question_answering_model_values(self):
        """Verify online config loads the question-answering LLM settings."""

        config = load_online_config(Path(".env"))

        self.assertTrue(config.chat.base_url)
        self.assertTrue(config.chat.api_key)
        self.assertTrue(config.chat.model)
        self.assertGreater(config.retrieval.top_k, 0)
        self.assertGreater(config.retrieval.max_context_chars, 0)

    def test_ask_parse_args_accepts_query_and_debug_flag(self):
        """Verify CLI argument parsing without mock-only options."""

        from scripts.ask import parse_args

        args = parse_args(["--query", "DarkIR?", "--show-debug", "--top-k", "2"])

        self.assertEqual(args.query, "DarkIR?")
        self.assertTrue(args.show_debug)
        self.assertEqual(args.top_k, 2)
        self.assertFalse(hasattr(args, "mock_llm"))
        self.assertFalse(hasattr(args, "mock_dimension"))

    def test_pipeline_returns_real_llm_answer_with_citations(self):
        """Verify real embedding, Chroma retrieval, prompt construction, and LLM generation."""

        offline_config = load_config(Path(".env"))
        online_config = load_online_config(Path(".env"))
        embedder = create_embedder(offline_config.embedding.provider, offline_config.embedding)
        client = get_chroma_client(offline_config.chroma.persist_dir)
        collection = get_or_create_collection(client, offline_config.chroma.collection)
        retriever = DenseRetriever(collection=collection, embedder=embedder)
        llm_client = OpenAICompatibleChatClient(
            base_url=online_config.chat.base_url,
            model=online_config.chat.model,
            api_key=online_config.chat.api_key,
            timeout=online_config.chat.timeout,
            max_retries=online_config.chat.max_retries,
        )
        top_k = resolve_value(2, online_config.retrieval.top_k)
        pipeline = OnlineQAPipeline(
            dense_retriever=retriever,
            llm_client=llm_client,
            top_k=top_k,
            max_context_chars=online_config.retrieval.max_context_chars,
        )

        result = pipeline.answer("DarkIR 在 LOLBlur 数据集上的表现如何？", include_debug=True)

        self.assertTrue(result.answer.strip())
        self.assertGreaterEqual(len(result.citations), 1)
        self.assertGreaterEqual(len(result.used_chunks), 1)
        self.assertIsNotNone(result.retrieval_debug_info)
        self.assertEqual(result.retrieval_debug_info["retrieval_mode"], "dense")
        self.assertIn("[1]", result.answer)


if __name__ == "__main__":
    unittest.main()

# Regression tests for the offline indexing reader, converter, and chunk builder.
import json
import tempfile
import unittest
from pathlib import Path

from offline_index.block_converter import convert_blocks, extract_all_content
from offline_index.chunk_builder import build_chunks
from offline_index.config_loader import VLMConfig
from offline_index.mineru_output_reader import flatten_blocks, read_content_list_v2
from offline_index.offline_pipeline import build_chunks_from_mineru_content, create_visual_summarizer
from offline_index.schema import SemanticBlock
from offline_index.summary_cache import SummaryCache
from offline_index.vlm_client import create_vlm_client
from offline_index.visual_summarizer import VisualBlockSummarizer


class OfflineIndexingTests(unittest.TestCase):
    """Covers the first-stage offline indexing behavior."""

    def test_flatten_blocks_adds_page_and_reading_order(self):
        """Verify flattened MinerU blocks include zero-based page index and order."""

        pages = [
            [{"type": "title", "content": {"title_content": [{"content": "A"}]}}],
            [{"type": "paragraph", "content": {"paragraph_content": [{"content": "B"}]}}],
        ]

        flattened = flatten_blocks(pages)

        self.assertEqual(flattened[0]["page_idx"], 0)
        self.assertEqual(flattened[0]["reading_order"], 0)
        self.assertEqual(flattened[1]["page_idx"], 1)
        self.assertEqual(flattened[1]["reading_order"], 1)

    def test_convert_blocks_discards_noise_and_resolves_image_source(self):
        """Verify noise blocks are dropped and image sources resolve to real paths."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            image_path = images_dir / "fig.jpg"
            image_path.write_bytes(b"fake image")
            blocks = [
                {"type": "page_number", "content": {"text": "1"}, "page_idx": 0, "reading_order": 0},
                {
                    "type": "title",
                    "content": {"title_content": [{"type": "text", "content": "Paper Title"}]},
                    "page_idx": 0,
                    "reading_order": 1,
                    "bbox": [1, 2, 3, 4],
                },
                {
                    "type": "image",
                    "content": {
                        "image_source": {"path": "images/fig.jpg"},
                        "image_caption": [{"type": "text", "content": "Figure 1"}],
                        "image_footnote": [{"type": "text", "content": "A note"}],
                    },
                    "page_idx": 0,
                    "reading_order": 2,
                },
            ]

            converted = convert_blocks(blocks, doc_id="doc_test", images_dir=images_dir)

            self.assertEqual([block.raw_type for block in converted], ["title", "image"])
            self.assertEqual(converted[0].rag_type, "text")
            self.assertEqual(converted[0].text, "Paper Title")
            self.assertEqual(converted[0].page_start, 1)
            self.assertEqual(converted[1].rag_type, "image")
            self.assertEqual(converted[1].source, str(image_path.resolve()))
            self.assertIn("Figure 1", converted[1].text)
            self.assertIn("A note", converted[1].text)

    def test_convert_blocks_treats_equations_as_text_blocks(self):
        """Verify equation blocks are merged into normal text handling with LaTeX preserved."""

        blocks = [
            {
                "type": "equation_inline",
                "content": {"math_content": r"E = mc^2"},
                "page_idx": 0,
                "reading_order": 0,
            },
            {
                "type": "equation_interline",
                "content": {"math_content": r"score(q, d) = cosine(E(q), E(d))"},
                "page_idx": 0,
                "reading_order": 1,
            },
        ]

        converted = convert_blocks(blocks, doc_id="doc_test", images_dir=Path.cwd())

        self.assertEqual([block.rag_type for block in converted], ["text", "text"])
        self.assertEqual(converted[0].text, r"E = mc^2")
        self.assertEqual(converted[1].text, r"score(q, d) = cosine(E(q), E(d))")

    def test_extract_all_content_merges_nested_caption(self):
        """Verify nested MinerU caption fragments are merged instead of truncating."""

        value = [
            {"content": [{"content": "Figure"}, {"content": "1"}]},
            {"extra": {"content": [{"content": "DarkIR"}, {"content": "results"}]}},
            {"content": {"content": "comparison"}},
        ]

        self.assertEqual(extract_all_content(value), "Figure 1 DarkIR results comparison")

    def test_convert_blocks_keeps_complete_caption(self):
        """Verify image/table caption fields stay complete and fallback text keeps all parts."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "fig.jpg").write_bytes(b"fig")
            (images_dir / "table.jpg").write_bytes(b"table")

            blocks = [
                {
                    "type": "image",
                    "content": {
                        "image_source": {"path": "images/fig.jpg"},
                        "image_caption": [{"content": "Figure"}, {"content": "1"}],
                        "image_footnote": [{"content": "footnote"}],
                    },
                    "page_idx": 0,
                    "reading_order": 0,
                },
                {
                    "type": "table",
                    "content": {
                        "image_source": {"path": "images/table.jpg"},
                        "table_caption": [{"content": "Table"}, {"content": "1"}],
                        "table_footnote": [{"content": "footnote"}],
                        "table_body": [{"content": "cell value"}],
                    },
                    "page_idx": 0,
                    "reading_order": 1,
                },
            ]

            converted = convert_blocks(blocks, doc_id="doc_test", images_dir=images_dir)

            self.assertEqual(converted[0].caption, "Figure 1")
            self.assertEqual(converted[0].text, "Figure 1\nfootnote")
            self.assertEqual(converted[1].caption, "Table 1")
            self.assertIn("Table 1", converted[1].text)
            self.assertIn("footnote", converted[1].text)
            self.assertIn("cell value", converted[1].text)

    def test_build_chunks_outputs_flat_metadata_and_stable_ids(self):
        """Verify chunk output contains stable ids, flat metadata, and no noise text."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content_path = root / "content_list_v2.json"
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "fig.jpg").write_bytes(b"fake image")
            pages = [
                [
                    {
                        "type": "title",
                        "content": {"title_content": [{"type": "text", "content": "Intro"}]},
                        "bbox": [1, 2, 3, 4],
                    },
                    {
                        "type": "paragraph",
                        "content": {"paragraph_content": [{"type": "text", "content": "A" * 900}]},
                    },
                    {
                        "type": "image",
                        "content": {
                            "image_source": {"path": "images/fig.jpg"},
                            "image_caption": [{"type": "text", "content": "Figure 1"}],
                            "image_footnote": [],
                        },
                    },
                    {
                        "type": "table",
                        "content": {
                            "table_caption": [{"type": "text", "content": "Table 1"}],
                            "table_footnote": [],
                            "html": "<table><tr><td>x</td></tr></table>",
                        },
                    },
                    {"type": "page_footnote", "content": {"text": "ignored"}},
                ]
            ]
            content_path.write_text(json.dumps(pages), encoding="utf-8")

            flattened = flatten_blocks(read_content_list_v2(content_path))
            blocks = convert_blocks(flattened, doc_id="doc_test", images_dir=images_dir)
            chunks = build_chunks(blocks, file_name="paper.pdf", chunk_size=800, chunk_overlap=120)

            chunk_types = [chunk.metadata["chunk_type"] for chunk in chunks]
            self.assertGreaterEqual(chunk_types.count("text"), 2)
            self.assertEqual(chunk_types.count("image"), 1)
            self.assertEqual(chunk_types.count("table"), 1)
            for chunk in chunks:
                self.assertTrue(chunk.id.startswith("chunk_"))
                self.assertIn("document", chunk.model_dump())
                self.assertEqual(
                    set(chunk.metadata.keys()),
                    {"doc_id", "file_name", "chunk_type", "page_start", "page_end", "source", "content_hash"},
                )
                self.assertEqual(chunk.metadata["doc_id"], "doc_test")
                self.assertEqual(chunk.metadata["file_name"], "paper.pdf")
                self.assertNotIn("ignored", chunk.document)

    def test_build_chunks_documents_do_not_repeat_metadata_headers(self):
        """Verify documents store pure chunk content instead of Type/File/Pages template headers."""

        blocks = [
            convert_blocks(
                [
                    {
                        "type": "paragraph",
                        "content": {"paragraph_content": [{"type": "text", "content": "Plain body text"}]},
                        "page_idx": 0,
                        "reading_order": 0,
                    },
                    {
                        "type": "image",
                        "content": {
                            "image_source": {"path": "images/fig.jpg"},
                            "image_caption": [{"type": "text", "content": "Figure caption"}],
                            "image_footnote": [{"type": "text", "content": "Figure footnote"}],
                        },
                        "page_idx": 0,
                        "reading_order": 1,
                    },
                    {
                        "type": "table",
                        "content": {
                            "table_caption": [{"type": "text", "content": "Table caption"}],
                            "table_footnote": [{"type": "text", "content": "Table footnote"}],
                            "html": "<table><tr><td>x</td></tr></table>",
                        },
                        "page_idx": 0,
                        "reading_order": 2,
                    },
                ],
                doc_id="doc_test",
                images_dir=Path.cwd(),
            )
        ][0]

        chunks = build_chunks(blocks, file_name="paper.pdf", chunk_size=800, chunk_overlap=120)

        for chunk in chunks:
            self.assertNotIn("Type:", chunk.document)
            self.assertNotIn("File:", chunk.document)
            self.assertNotIn("Pages:", chunk.document)
            self.assertNotIn("Content:", chunk.document)

    def test_visual_summarizer_cache_hit(self):
        """Verify cached summaries replace block text without calling the client again."""

        class StubClient:
            def __init__(self) -> None:
                self.calls = 0

            def summarize_image(self, image_path: Path, prompt: str) -> str:
                self.calls += 1
                return "fresh summary"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "figure.jpg"
            image_path.write_bytes(b"fake-image")
            block = SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="image",
                rag_type="image",
                text="fallback text",
                caption="Figure 1",
                source=str(image_path),
                reading_order=1,
            )
            cache = SummaryCache(root / "cache.json")
            key = cache.make_key(block=block, model="vlm-model", prompt_version="v1")
            cache.set(key, "cached summary", {"model": "vlm-model"})

            client = StubClient()
            summarizer = VisualBlockSummarizer(client=client, cache=cache, model="vlm-model")
            result = summarizer.enrich_blocks([block], file_name="paper.pdf")

            self.assertEqual(result[0].text, "cached summary")
            self.assertEqual(client.calls, 0)

    def test_visual_summarizer_force_vlm_bypasses_and_updates_cache(self):
        """Verify force VLM refreshes cached summaries and uses the fresh text."""

        class StubClient:
            def __init__(self) -> None:
                self.calls = 0

            def summarize_image(self, image_path: Path, prompt: str) -> str:
                self.calls += 1
                return "fresh summary"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "figure.jpg"
            image_path.write_bytes(b"fake-image")
            block = SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="image",
                rag_type="image",
                text="fallback text",
                caption="Figure 1",
                source=str(image_path),
                reading_order=1,
            )
            cache = SummaryCache(root / "cache.json")
            key = cache.make_key(block=block, model="vlm-model", prompt_version="v1")
            cache.set(key, "cached summary", {"model": "vlm-model"})

            client = StubClient()
            summarizer = VisualBlockSummarizer(client=client, cache=cache, model="vlm-model", force_vlm=True)
            result = summarizer.enrich_blocks([block], file_name="paper.pdf")

            self.assertEqual(result[0].text, "fresh summary")
            self.assertEqual(cache.get(key), "fresh summary")
            self.assertEqual(client.calls, 1)
            self.assertEqual(summarizer.cache_hits, 0)
            self.assertEqual(summarizer.generated, 1)

    def test_visual_summarizer_fallback_on_error(self):
        """Verify VLM exceptions do not interrupt the pipeline and fallback text survives."""

        class RaisingClient:
            def summarize_image(self, image_path: Path, prompt: str) -> str:
                raise RuntimeError("vlm failed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "figure.jpg"
            image_path.write_bytes(b"fake-image")
            block = SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="image",
                rag_type="image",
                text="fallback text",
                caption="Figure 1",
                source=str(image_path),
                reading_order=1,
            )

            summarizer = VisualBlockSummarizer(
                client=RaisingClient(),
                cache=SummaryCache(root / "cache.json"),
                model="vlm-model",
            )
            result = summarizer.enrich_blocks([block], file_name="paper.pdf")

            self.assertEqual(result[0].text, "fallback text")
            self.assertEqual(summarizer.failed, 1)
            self.assertIn("vlm failed", summarizer.failure_messages[0])

    def test_visual_summarizer_skips_missing_source(self):
        """Verify blocks without a readable source image keep their fallback text."""

        class StubClient:
            def __init__(self) -> None:
                self.calls = 0

            def summarize_image(self, image_path: Path, prompt: str) -> str:
                self.calls += 1
                return "summary"

        block = SemanticBlock(
            block_id="block_000001",
            doc_id="doc_test",
            page_start=1,
            page_end=1,
            raw_type="image",
            rag_type="image",
            text="fallback text",
            caption="Figure 1",
            source="",
            reading_order=1,
        )
        client = StubClient()
        summarizer = VisualBlockSummarizer(client=client, cache=SummaryCache(Path("unused.json")), model="vlm-model")

        result = summarizer.enrich_blocks([block], file_name="paper.pdf")

        self.assertEqual(result[0].text, "fallback text")
        self.assertEqual(client.calls, 0)

    def test_visual_summarizer_tracks_cache_and_generation_counts(self):
        """Verify VLM diagnostics distinguish cache hits from fresh generation."""

        class StubClient:
            def __init__(self) -> None:
                self.calls = 0

            def summarize_image(self, image_path: Path, prompt: str) -> str:
                self.calls += 1
                return "fresh summary"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_image = root / "cached.jpg"
            second_image = root / "fresh.jpg"
            first_image.write_bytes(b"cached-image")
            second_image.write_bytes(b"fresh-image")
            cached_block = SemanticBlock(
                block_id="block_cached",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="image",
                rag_type="image",
                text="fallback text",
                caption="Figure 1",
                source=str(first_image),
                reading_order=1,
            )
            fresh_block = SemanticBlock(
                block_id="block_fresh",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="image",
                rag_type="image",
                text="fallback text",
                caption="Figure 2",
                source=str(second_image),
                reading_order=2,
            )
            cache = SummaryCache(root / "cache.json")
            cache_key = cache.make_key(block=cached_block, model="vlm-model", prompt_version="v1")
            cache.set(cache_key, "cached summary", {"model": "vlm-model"})

            client = StubClient()
            summarizer = VisualBlockSummarizer(client=client, cache=cache, model="vlm-model")
            result = summarizer.enrich_blocks([cached_block, fresh_block], file_name="paper.pdf")

            self.assertEqual(result[0].text, "cached summary")
            self.assertEqual(result[1].text, "fresh summary")
            self.assertEqual(client.calls, 1)
            self.assertEqual(summarizer.cache_hits, 1)
            self.assertEqual(summarizer.generated, 1)
            self.assertEqual(summarizer.failed, 0)

    def test_chunk_builder_uses_enhanced_semanticblock_text(self):
        """Verify enhanced image/table SemanticBlock.text becomes ChunkRecord.document."""

        blocks = [
            SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="image",
                rag_type="image",
                text="VLM summary",
                caption="Figure 1",
                source="image.jpg",
                reading_order=1,
            ),
            SemanticBlock(
                block_id="block_000002",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="table",
                rag_type="table",
                text="Table summary",
                caption="Table 1",
                source="table.jpg",
                reading_order=2,
            ),
        ]

        chunks = build_chunks(blocks, file_name="paper.pdf")

        self.assertEqual(chunks[0].document, "VLM summary")
        self.assertEqual(chunks[1].document, "Table summary")

    def test_create_vlm_client_rejects_invalid_config(self):
        """Verify unsupported providers and missing required fields fail clearly."""

        unsupported = VLMConfig(
            enabled=True,
            provider="other",
            api_key="secret",
            base_url="https://example.test/v1",
            model="qwen-vl",
            timeout=60,
            max_retries=3,
            cache_path=Path("data/cache/vlm.json"),
            max_images_per_doc=10,
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_vlm_client(unsupported, api_key="secret")

        missing_base_url = VLMConfig(
            enabled=True,
            provider="openai-compatible",
            api_key="secret",
            base_url="",
            model="qwen-vl",
            timeout=60,
            max_retries=3,
            cache_path=Path("data/cache/vlm.json"),
            max_images_per_doc=10,
        )
        with self.assertRaisesRegex(ValueError, "base_url"):
            create_vlm_client(missing_base_url, api_key="secret")

    def test_create_visual_summarizer_passes_force_vlm(self):
        """Verify visual summarizer factory forwards the force VLM option."""

        config = VLMConfig(
            enabled=True,
            provider="openai-compatible",
            api_key="secret",
            base_url="https://example.test/v1",
            model="qwen-vl",
            timeout=60,
            max_retries=3,
            cache_path=Path("data/cache/vlm.json"),
            max_images_per_doc=10,
        )

        summarizer, _ = create_visual_summarizer(config, force_vlm=True)

        self.assertTrue(summarizer.force_vlm)

    def test_build_chunks_from_mineru_content_uses_visual_enrichment(self):
        """Verify the shared offline pipeline can replace image block fallback text before chunking."""

        class StubSummarizer:
            def enrich_blocks(self, blocks: list[SemanticBlock], file_name: str) -> list[SemanticBlock]:
                return [block.model_copy(update={"text": "enhanced summary"}) if block.rag_type == "image" else block for block in blocks]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content_path = root / "content_list_v2.json"
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "fig.jpg").write_bytes(b"fake image")
            pages = [
                [
                    {
                        "type": "image",
                        "content": {
                            "image_source": {"path": "images/fig.jpg"},
                            "image_caption": [{"type": "text", "content": "Figure 1"}],
                            "image_footnote": [],
                        },
                    }
                ]
            ]
            content_path.write_text(json.dumps(pages), encoding="utf-8")

            chunks = build_chunks_from_mineru_content(
                content_list_v2_path=content_path,
                images_dir=images_dir,
                doc_id="doc_test",
                file_name="paper.pdf",
                chunk_size=800,
                chunk_overlap=120,
                summarizer=StubSummarizer(),
            )

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].document, "enhanced summary")


if __name__ == "__main__":
    unittest.main()

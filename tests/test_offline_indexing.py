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
from offline_index.utils import md5_file
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
                    {"doc_id", "file_name", "chunk_type", "page_start", "page_end", "source", "content_md5", "meta_location"},
                )
                self.assertEqual(chunk.metadata["doc_id"], "doc_test")
                self.assertEqual(chunk.metadata["file_name"], "paper.pdf")
                self.assertTrue(chunk.metadata["meta_location"])
                self.assertTrue(chunk.metadata["content_md5"])
                self.assertNotIn("ignored", chunk.document)

    def test_special_meta_location_stays_stable_when_visual_summary_changes(self):
        """Verify image/table chunks can be updated in-place when only summary text changes."""

        base = SemanticBlock(
            block_id="block_000001",
            doc_id="doc_test",
            page_start=1,
            page_end=1,
            raw_type="image",
            rag_type="image",
            text="old summary",
            caption="Figure 1",
            source="image.jpg",
            reading_order=1,
        )
        updated = base.model_copy(update={"text": "new summary"})

        old_chunk = build_chunks([base], file_name="paper.pdf")[0]
        new_chunk = build_chunks([updated], file_name="paper.pdf")[0]

        self.assertEqual(old_chunk.metadata["meta_location"], new_chunk.metadata["meta_location"])
        self.assertNotEqual(old_chunk.metadata["content_md5"], new_chunk.metadata["content_md5"])
        self.assertNotEqual(old_chunk.id, new_chunk.id)

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

    def test_build_chunks_moves_next_page_lead_sentence_to_incomplete_page_end(self):
        """Verify text chunks repair a sentence split across adjacent pages before chunking."""

        blocks = [
            SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="paragraph",
                rag_type="text",
                text="第一页最后一句没有结束",
                reading_order=1,
            ),
            SemanticBlock(
                block_id="block_000002",
                doc_id="doc_test",
                page_start=2,
                page_end=2,
                raw_type="paragraph",
                rag_type="text",
                text="所以这里补完。第二页保留内容。",
                reading_order=2,
            ),
        ]

        chunks = build_chunks(blocks, file_name="paper.pdf", chunk_size=100, chunk_overlap=10)

        text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text"]
        self.assertEqual([chunk.metadata["page_start"] for chunk in text_chunks], [1, 2])
        self.assertEqual(text_chunks[0].document, "第一页最后一句没有结束所以这里补完。")
        self.assertEqual(text_chunks[1].document, "第二页保留内容。")
        self.assertEqual(text_chunks[0].metadata["meta_location"], "doc_test:p1:b1-1:part0")
        self.assertEqual(text_chunks[1].metadata["meta_location"], "doc_test:p2:b1-1:part0")

    def test_build_chunks_treats_comma_as_incomplete_for_cross_page_merge(self):
        """Verify cross-page repair still requires strict sentence endings, not commas."""

        blocks = [
            SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="paragraph",
                rag_type="text",
                text="第一页末尾仍然是半句，",
                reading_order=1,
            ),
            SemanticBlock(
                block_id="block_000002",
                doc_id="doc_test",
                page_start=2,
                page_end=2,
                raw_type="paragraph",
                rag_type="text",
                text="下一页把句子补完。第二页后续。",
                reading_order=2,
            ),
        ]

        chunks = build_chunks(blocks, file_name="paper.pdf", chunk_size=100, chunk_overlap=10)

        text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text"]
        self.assertEqual(text_chunks[0].document, "第一页末尾仍然是半句，下一页把句子补完。")
        self.assertEqual(text_chunks[1].document, "第二页后续。")

    def test_build_chunks_repairs_english_sentence_split_across_pages(self):
        """Verify cross-page repair can take an English lead sentence ending with a period."""

        blocks = [
            SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="paragraph",
                rag_type="text",
                text="Therefore, we extend FourLLIE to the exposure correction task and evaluate it in the",
                reading_order=1,
            ),
            SemanticBlock(
                block_id="block_000002",
                doc_id="doc_test",
                page_start=2,
                page_end=2,
                raw_type="paragraph",
                rag_type="text",
                text="SICE [3] dataset, which contains 512 pairs for training, and 60 image pairs for testing. Note that the frequency stage should estimate an additional amplitude transform map.",
                reading_order=2,
            ),
        ]

        chunks = build_chunks(blocks, file_name="paper.pdf", chunk_size=240, chunk_overlap=20)

        text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text"]
        self.assertEqual(
            text_chunks[0].document,
            "Therefore, we extend FourLLIE to the exposure correction task and evaluate it in theSICE [3] dataset, which contains 512 pairs for training, and 60 image pairs for testing.",
        )
        self.assertEqual(text_chunks[1].document, "Note that the frequency stage should estimate an additional amplitude transform map.")

    def test_build_chunks_keeps_pages_separate_while_merging_small_blocks(self):
        """Verify small text blocks merge inside one page but never merge across pages."""

        blocks = [
            SemanticBlock(
                block_id="block_000001",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="paragraph",
                rag_type="text",
                text="第一页第一段。",
                reading_order=1,
            ),
            SemanticBlock(
                block_id="block_000002",
                doc_id="doc_test",
                page_start=1,
                page_end=1,
                raw_type="paragraph",
                rag_type="text",
                text="第一页第二段。",
                reading_order=2,
            ),
            SemanticBlock(
                block_id="block_000003",
                doc_id="doc_test",
                page_start=2,
                page_end=2,
                raw_type="paragraph",
                rag_type="text",
                text="第二页第一段。",
                reading_order=3,
            ),
        ]

        chunks = build_chunks(blocks, file_name="paper.pdf", chunk_size=100, chunk_overlap=10)

        text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text"]
        self.assertEqual(len(text_chunks), 2)
        self.assertEqual(text_chunks[0].metadata["page_start"], 1)
        self.assertEqual(text_chunks[0].document, "第一页第一段。\n\n第一页第二段。")
        self.assertEqual(text_chunks[0].metadata["meta_location"], "doc_test:p1:b1-2:part0")
        self.assertEqual(text_chunks[1].metadata["page_start"], 2)
        self.assertEqual(text_chunks[1].document, "第二页第一段。")
        self.assertEqual(text_chunks[1].metadata["meta_location"], "doc_test:p2:b1-1:part0")

    def test_build_chunks_splits_large_block_with_sentence_overlap_window(self):
        """Verify only oversized single blocks use overlap and prefer complete sentence starts."""

        text = "第一句内容较长。第二句内容也很重要。第三句继续说明。"
        block = SemanticBlock(
            block_id="block_000001",
            doc_id="doc_test",
            page_start=1,
            page_end=1,
            raw_type="paragraph",
            rag_type="text",
            text=text,
            reading_order=1,
        )

        chunks = build_chunks([block], file_name="paper.pdf", chunk_size=18, chunk_overlap=8)

        text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text"]
        self.assertEqual([chunk.document for chunk in text_chunks], ["第一句内容较长。第二句内容也很重要。", "第二句内容也很重要。第三句继续说明。"])
        self.assertEqual(text_chunks[0].metadata["page_start"], 1)
        self.assertEqual(text_chunks[1].metadata["page_start"], 1)
        self.assertEqual(text_chunks[0].metadata["meta_location"], "doc_test:p1:b1-1:part0")
        self.assertEqual(text_chunks[1].metadata["meta_location"], "doc_test:p1:b1-1:part1")

    def test_build_chunks_overlap_can_start_after_comma_boundary(self):
        """Verify oversized-block overlap can use comma boundaries for a cleaner next window."""

        text = "第一句内容很长很长。第二句前半部分很长很长很长，后半部分继续说明。第三句结束。"
        block = SemanticBlock(
            block_id="block_000001",
            doc_id="doc_test",
            page_start=1,
            page_end=1,
            raw_type="paragraph",
            rag_type="text",
            text=text,
            reading_order=1,
        )

        chunks = build_chunks([block], file_name="paper.pdf", chunk_size=33, chunk_overlap=6)

        text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text"]
        self.assertEqual(text_chunks[0].document, "第一句内容很长很长。第二句前半部分很长很长很长，后半部分继续说明。")
        self.assertTrue(text_chunks[1].document.startswith("后半部分继续说明。"))

    def test_summary_cache_key_uses_only_source_image_md5(self):
        """Verify VLM cache identity depends only on the source image bytes."""

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "figure.jpg"
            image_path.write_bytes(b"same-image")
            cache = SummaryCache(Path(tmp) / "cache.json")
            first = SemanticBlock(
                block_id="block_000001",
                doc_id="doc_a",
                page_start=1,
                page_end=1,
                raw_type="image",
                rag_type="image",
                text="caption A",
                caption="Figure A",
                source=str(image_path),
                reading_order=1,
            )
            second = first.model_copy(
                update={
                    "block_id": "block_999999",
                    "doc_id": "doc_b",
                    "text": "caption B",
                    "caption": "Figure B",
                    "reading_order": 999,
                }
            )

            self.assertEqual(cache.make_key(first, model="model-a", prompt_version="v1"), md5_file(image_path))
            self.assertEqual(cache.make_key(first, model="model-a", prompt_version="v1"), cache.make_key(second, model="model-b", prompt_version="v2"))

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

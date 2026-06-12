# VLM Visual Block Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cached VLM summarization for MinerU image/table blocks so enhanced `SemanticBlock.text` flows into chunk preview generation and final Chroma indexing.

**Architecture:** Keep MinerU parsing, visual summarization, chunk building, and indexing as separate stages. `block_converter.py` keeps complete fallback extraction, `visual_summarizer.py` upgrades only image/table `SemanticBlock.text`, and both `scripts/preview_chunks.py` and `scripts/build_index.py` execute the same enhancement path when `VLM_ENABLED=true`.

**Tech Stack:** Python, Pydantic, dotenv, urllib, unittest, JSON cache files, ChromaDB pipeline scripts

---

### Task 1: Add failing parser and config tests

**Files:**
- Modify: `tests/test_offline_indexing.py`
- Modify: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_all_content_merges_nested_caption(self):
    value = [
        {"content": [{"content": "Figure"}, {"content": "1"}]},
        {"extra": {"content": [{"content": "DarkIR"}, {"content": "results"}]}},
    ]
    self.assertEqual(extract_all_content(value), "Figure 1 DarkIR results")


def test_convert_blocks_keeps_complete_caption(self):
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
                "table_body": [{"content": "cell"}],
            },
            "page_idx": 0,
            "reading_order": 1,
        },
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_offline_indexing.py tests/test_config_loader.py -q`
Expected: FAIL because `extract_all_content`, `SemanticBlock.caption`, and `config.vlm` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class SemanticBlock(BaseModel):
    ...
    caption: str = ""


@dataclass(frozen=True)
class VLMConfig:
    enabled: bool
    provider: str
    base_url: str
    api_key_env: str
    model: str
    timeout: float
    max_retries: int
    cache_path: Path
    max_images_per_doc: int
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_offline_indexing.py tests/test_config_loader.py -q`
Expected: PASS for the new parser/config coverage.

- [ ] **Step 5: Commit**

```bash
git add tests/test_offline_indexing.py tests/test_config_loader.py src/offline_index/schema.py src/offline_index/block_converter.py src/offline_index/config_loader.py .env.example
git commit -m "test: cover visual block parsing and vlm config"
```

### Task 2: Add failing visual summarizer tests

**Files:**
- Modify: `tests/test_offline_indexing.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_visual_summarizer_cache_hit(self):
    cache = SummaryCache(cache_path)
    cache.set(key, "cached summary", {})
    client = StubClient()
    summarizer = VisualBlockSummarizer(client, cache)
    result = summarizer.enrich_blocks([block], file_name="paper.pdf")
    self.assertEqual(result[0].text, "cached summary")
    self.assertEqual(client.calls, 0)


def test_visual_summarizer_fallback_on_error(self):
    client = RaisingClient()
    summarizer = VisualBlockSummarizer(client, SummaryCache(cache_path))
    result = summarizer.enrich_blocks([block], file_name="paper.pdf")
    self.assertEqual(result[0].text, "fallback text")


def test_visual_summarizer_skips_missing_source(self):
    block = SemanticBlock(..., rag_type="image", source="", text="fallback text", caption="caption")
    result = summarizer.enrich_blocks([block], file_name="paper.pdf")
    self.assertEqual(result[0].text, "fallback text")


def test_chunk_builder_uses_enhanced_semanticblock_text(self):
    block = SemanticBlock(..., rag_type="image", text="VLM summary", caption="caption")
    chunks = build_chunks([block], file_name="paper.pdf")
    self.assertEqual(chunks[0].document, "VLM summary")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_offline_indexing.py -q`
Expected: FAIL because `SummaryCache` and `VisualBlockSummarizer` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class SummaryCache:
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, summary: str, metadata: dict) -> None: ...
    def save(self) -> None: ...


class VisualBlockSummarizer:
    def enrich_blocks(self, blocks: list[SemanticBlock], file_name: str) -> list[SemanticBlock]: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_offline_indexing.py -q`
Expected: PASS for cache-hit, fallback, missing-source, and chunk propagation coverage.

- [ ] **Step 5: Commit**

```bash
git add tests/test_offline_indexing.py src/offline_index/summary_cache.py src/offline_index/visual_summarizer.py
git commit -m "feat: add cached visual block summarization"
```

### Task 3: Add failing VLM client tests and implement client

**Files:**
- Modify: `tests/test_offline_indexing.py`
- Create: `src/offline_index/vlm_client.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_openai_compatible_vlm_client_validates_required_config(self):
    with self.assertRaisesRegex(ValueError, "base_url"):
        OpenAICompatibleVLMClient(base_url="", api_key="x", model="m")


def test_openai_compatible_vlm_client_rejects_unsupported_provider(self):
    config = VLMConfig(provider="other", ...)
    with self.assertRaisesRegex(ValueError, "unsupported"):
        create_vlm_client(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_offline_indexing.py -q`
Expected: FAIL because the VLM client factory is missing.

- [ ] **Step 3: Write minimal implementation**

```python
class OpenAICompatibleVLMClient:
    def summarize_image(self, image_path: Path, prompt: str) -> str:
        ...


def create_vlm_client(config: VLMConfig) -> OpenAICompatibleVLMClient:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_offline_indexing.py -q`
Expected: PASS for validation and factory behavior.

- [ ] **Step 5: Commit**

```bash
git add tests/test_offline_indexing.py src/offline_index/vlm_client.py
git commit -m "feat: add openai-compatible vlm client"
```

### Task 4: Wire both entry scripts to the shared enhancement pipeline

**Files:**
- Modify: `scripts/preview_chunks.py`
- Modify: `scripts/build_index.py`
- Modify: `src/offline_index/chunk_loader.py`
- Modify: `src/offline_index/document_manifest.py`

- [ ] **Step 1: Write the failing integration tests**

```python
def test_preview_pipeline_uses_visual_enrichment_when_enabled(self):
    ...


def test_build_index_pipeline_preserves_vlm_enhanced_chunk_documents(self):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_offline_indexing.py -q`
Expected: FAIL because entry scripts do not build enhanced chunks yet.

- [ ] **Step 3: Write minimal implementation**

```python
blocks = convert_blocks(...)
if config.vlm.enabled:
    client = create_vlm_client(config.vlm)
    cache = SummaryCache(config.vlm.cache_path)
    summarizer = VisualBlockSummarizer(client, cache, config.vlm.max_images_per_doc)
    blocks = summarizer.enrich_blocks(blocks, file_name=file_name)
    cache.save()
chunks = build_chunks(blocks, ...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_offline_indexing.py tests/test_config_loader.py -q`
Expected: PASS with entrypoint coverage and no regressions.

- [ ] **Step 5: Commit**

```bash
git add scripts/preview_chunks.py scripts/build_index.py src/offline_index/chunk_loader.py src/offline_index/document_manifest.py tests/test_offline_indexing.py tests/test_config_loader.py
git commit -m "feat: wire vlm summaries into preview and index pipelines"
```

### Task 5: Final verification and manual preview run

**Files:**
- Verify only

- [ ] **Step 1: Run focused regression tests**

Run: `python -m pytest tests/test_offline_indexing.py tests/test_config_loader.py -q`
Expected: PASS

- [ ] **Step 2: Run the preview script with VLM disabled**

Run: `python scripts/preview_chunks.py --content-list-v2 "documents/output_pipeline/DarkIR Robust Low-Light Image Restoration/auto/DarkIR Robust Low-Light Image Restoration_content_list_v2.json" --images-dir "documents/output_pipeline/DarkIR Robust Low-Light Image Restoration/auto/images" --output "data/debug/manual_chunks_preview.json"`
Expected: exit 0 and a generated preview JSON.

- [ ] **Step 3: Inspect one image/table chunk document**

Run: `python -c "import json, pathlib; data=json.loads(pathlib.Path('data/debug/manual_chunks_preview.json').read_text(encoding='utf-8')); print(next(item['document'] for item in data if item['metadata']['chunk_type'] in {'image','table'}))"`
Expected: fallback text with VLM disabled, VLM summary text when enabled and configured.

- [ ] **Step 4: Document operator usage**

Include:
- required `.env` keys
- how cache invalidation works
- how to verify image/table chunks carry VLM summaries before indexing

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-06-09-vlm-visual-block-enhancement.md
git commit -m "docs: add vlm visual block enhancement plan"
```

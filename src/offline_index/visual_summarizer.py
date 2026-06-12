# Replaces image/table SemanticBlock.text with cached or freshly generated VLM summaries.
from __future__ import annotations

from pathlib import Path

from offline_index.schema import SemanticBlock
from offline_index.summary_cache import SummaryCache
from offline_index.utils import normalize_text


IMAGE_PROMPT_VERSION = "v1"
TABLE_PROMPT_VERSION = "v1"


class VisualBlockSummarizer:
    """Enrich visual SemanticBlocks without mutating fallback blocks in place."""

    def __init__(
        self,
        client,
        cache: SummaryCache,
        model: str,
        max_images_per_doc: int = 50,
        force_vlm: bool = False,
    ) -> None:
        self.client = client
        self.cache = cache
        self.model = model
        self.max_images_per_doc = max_images_per_doc
        self.force_vlm = force_vlm
        self.cache_hits = 0
        self.generated = 0
        self.failed = 0
        self.failure_messages: list[str] = []

    def enrich_blocks(self, blocks: list[SemanticBlock], file_name: str) -> list[SemanticBlock]:
        """Return blocks with image/table text replaced by VLM summaries when available."""

        enhanced: list[SemanticBlock] = []
        visual_count = 0
        for block in blocks:
            if block.rag_type not in {"image", "table"}:
                enhanced.append(block)
                continue
            if visual_count >= self.max_images_per_doc:
                enhanced.append(block)
                continue
            visual_count += 1
            enhanced.append(self._enrich_block(block, file_name))
        return enhanced

    def _enrich_block(self, block: SemanticBlock, file_name: str) -> SemanticBlock:
        """Return one updated visual block or the original block on any failure."""

        if not block.source:
            return block
        source_path = Path(block.source)
        if not source_path.exists() or not source_path.is_file():
            return block

        prompt_version = IMAGE_PROMPT_VERSION if block.rag_type == "image" else TABLE_PROMPT_VERSION
        cache_key = self.cache.make_key(block=block, model=self.model, prompt_version=prompt_version)
        if not self.force_vlm:
            cached = self.cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                return block.model_copy(update={"text": normalize_text(cached)})

        try:
            summary = self.client.summarize_image(source_path, self._build_prompt(block, file_name))
        except Exception as exc:
            self.failed += 1
            self._remember_failure(block, exc)
            return block

        summary = normalize_text(summary)
        if not summary:
            self.failed += 1
            self._remember_failure(block, RuntimeError("VLM returned empty summary"))
            return block
        self.cache.set(
            cache_key,
            summary,
            {
                "model": self.model,
                "source": block.source,
                "caption_hash": self.cache.make_key(block=block, model="", prompt_version="caption"),
                "prompt_version": prompt_version,
            },
        )
        self.generated += 1
        return block.model_copy(update={"text": summary})

    def _remember_failure(self, block: SemanticBlock, exc: Exception) -> None:
        """Keep a small sample of VLM failures for CLI diagnostics."""

        if len(self.failure_messages) >= 5:
            return
        source_name = Path(block.source).name if block.source else block.block_id
        self.failure_messages.append(f"{block.rag_type}:{source_name}: {exc}")

    def _build_prompt(self, block: SemanticBlock, file_name: str) -> str:
        """Build a compact Chinese prompt for one image or table block."""

        if block.rag_type == "table":
            return (
                "你是 RAG 系统中的表格理解模块。"
                "你的第一个任务是从图片中的表格里，完整、客观、逐项提取可见数据。"
                "第二个任务是进一步对表格进行客观分析。"
                f"\n文件名: {file_name}"
                f"\n页码: {block.page_start}"
                f"\n表格标题: {block.caption or '无'}"
                "要求："
                "\n1. 必须保留表格的行列对应关系，特别注意多级表头、合并单元格、行标题、列标题。"
                "\n2. 尽可能完整输出每一个单元格的内容，不能只挑重点。"
                "\n3. 如果是流程图、架构图、算法图，说明模块关系和数据流。"
                "\n4. 如果是曲线图、柱状图、对比图，说明横轴、纵轴、趋势、对比结论。"
                "\n5. 如果是论文实验图，说明实验对象、方法对比、主要发现。"
                "\n6. 数值、单位、百分号、正负号、范围、括号、上标/下标（若可辨认）都要保留。"
                "\n7. 如果图片内容无法判断，请明确说明不确定，不要编造。"
                "\n8. 输出应适合直接作为 RAG chunk 的 document 字段。"
                "\n9. 客观概述表格主题、字段和主要内容。不遗漏关键字段范围。"
                "\n10. 两个任务需要逐一完成，得到的结果进行拼接。"
                "\n直接返回完整的数据提取结果和客观分析结果。数据提取结果原语言返回，客观分析结果用尽可能用中文返回。"
                "不确定时明确说明，不要编造。"
            )
        return (
            "你是 RAG 系统中的图片理解模块。"
            f"\n文件名: {file_name}"
            f"\n页码: {block.page_start}"
            f"\n图片标题: {block.caption or '无'}"
            "\n请输出适合向量检索的中文语义摘要，解释核心信息、关系、趋势或结论；"
            "不确定时明确说明，不要编造。"
        )
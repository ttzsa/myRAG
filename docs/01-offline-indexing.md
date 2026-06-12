# 离线索引构建层设计

## 1. 目标

离线索引构建层负责把 `documents/source_documents` 下的 PDF 转换为可检索的本地知识库。

当前代码已经采用三段式流程：

```text
1. parse_pdfs.py
   调用 MinerU CLI 解析 PDF，生成 content_list_v2.json / images / markdown

2. build_manifest.py --build-chunks-preview
   扫描源 PDF，维护 rag_documents.json，并根据 MinerU 输出生成 chunks_preview.json

3. build_index.py
   读取 chunks_preview.json，调用 embedding，写入 ChromaDB
```

离线层最终产物：

```text
documents/source_documents/*.pdf
  -> documents/output_pipeline/<pdf_stem>/auto/*
  -> data/debug/*_chunks_preview.json
  -> data/chroma/
  -> data/index/rag_documents.json
```

## 2. 当前代码的总体流程

当前实现的真实数据流如下：

```text
1. 读取 .env 配置
2. 扫描 documents/source_documents 下的 PDF
3. 计算每个 PDF 的 MD5
4. 调用 MinerU CLI 解析 PDF
5. 在 documents/output_pipeline/<pdf_stem>/auto/ 下定位 content_list_v2.json 和 images/
6. 读取 content_list_v2.json
7. flatten 所有页面 block
8. 将原始 MinerU block 转换为 SemanticBlock
9. 丢弃 page_number / page_aside_text / page_footnote / header / footer
10. 将 image/table 的 caption 完整提取并写入 SemanticBlock.caption
11. 将 image/table 的 fallback 文本写入 SemanticBlock.text
12. 如果启用 VLM，则使用真实图片文件增强 image/table 的 SemanticBlock.text
13. 将 text block 合并后按 chunk_size / chunk_overlap 切分
14. 将 image/table 作为独立 chunk 输出
15. 生成 ChunkRecord，写入 data/debug/*_chunks_preview.json
16. 对 chunk.document 调用 embedding
17. 写入 ChromaDB rag_chunks collection
18. 更新 rag_documents.json 状态与计数
```

当前实现没有统一的 `index_runner.py`。实际执行方式是三条命令串联：

```powershell
python scripts\parse_pdfs.py
python scripts\build_manifest.py --build-chunks-preview --force
python scripts\build_index.py --reset
```

## 3. 配置设计

当前代码统一使用 `.env`，并通过 `src/offline_index/config_loader.py` 读取为 `AppConfig`。

当前配置结构：

```text
AppConfig
  paths
  mineru
  chroma
  chunking
  embedding
  vlm
```

`.env.example` 当前字段：

```dotenv
MINERU_EXE=
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_OUTPUT_ROOT=documents/output_pipeline

PDF_ROOT=documents/source_documents
PDF_RECURSIVE=true
FORCE_REBUILD=false

RAG_DOCUMENTS_PATH=data/index/rag_documents.json
DEBUG_DIR=data/debug

CHROMA_PERSIST_DIRECTORY=data/chroma
CHROMA_COLLECTION_NAME=rag_chunks

INGEST_CHUNK_SIZE=800
INGEST_CHUNK_OVERLAP=120

EMBEDDING_PROVIDER=
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=
EMBEDDING_BATCH_SIZE=8
EMBEDDING_TIMEOUT_SECONDS=120
EMBEDDING_DIMENSION=512

VLM_ENABLED=false
VLM_PROVIDER=openai-compatible
VLM_API_KEY=
VLM_BASE_URL=
VLM_MODEL=
VLM_TIMEOUT_SECONDS=60
VLM_MAX_RETRIES=3
VLM_CACHE_PATH=data/cache/vlm_summaries.json
VLM_MAX_IMAGES_PER_DOC=50

CHAT_PROVIDER=openai-compatible
CHAT_API_KEY=
CHAT_BASE_URL=
CHAT_MODEL=qwen-plus
CHAT_TIMEOUT_SECONDS=60
CHAT_MAX_RETRIES=3
CHAT_RETRIEVAL_TOP_K=5
CHAT_MAX_CONTEXT_CHARS=6000

HOST=127.0.0.1
PORT=8000
```

说明：

```text
PDF_ROOT:
  原始 PDF 输入目录。

MINERU_OUTPUT_ROOT:
  MinerU CLI 输出根目录。每篇 PDF 的真实输出目录为：
  documents/output_pipeline/<pdf_stem>/auto/

RAG_DOCUMENTS_PATH:
  文档级 manifest 文件。

DEBUG_DIR:
  chunks_preview.json 的默认输出目录。

CHROMA_PERSIST_DIRECTORY:
  ChromaDB 本地持久化目录。

INGEST_CHUNK_SIZE / INGEST_CHUNK_OVERLAP:
  文本 chunk 切分参数。

EMBEDDING_*:
  embedding 调用配置。

VLM_*:
  image/table 摘要增强配置。
```

## 4. MinerU 解析层

当前代码通过 `scripts/parse_pdfs.py` 调用 `src/offline_index/mineru_runner.py`。

真实命令行为：

```powershell
python scripts\parse_pdfs.py
```

脚本会：

```text
1. 从 PDF_ROOT 扫描 PDF
2. 自动排除位于 MINERU_OUTPUT_ROOT 下的 PDF
3. 对每个 PDF 调用 mineru.exe
4. 生成 documents/output_pipeline/<pdf_stem>/auto/
5. 更新 rag_documents.json 中的 parsed / failed 状态
```

MinerU 输出定位由 `mineru_output_locator.py` 完成。当前代码只把下列文件作为离线索引输入上下文：

```text
*_content_list_v2.json   主输入
*_content_list.json      仅记录路径，不参与主流程
*.md                     仅记录路径，不参与主流程
images/                  image/table 的真实图片资源
```

也就是说，当前代码构建 chunk 时只读取 `content_list_v2.json`，没有真的使用 markdown 兜底。

## 5. rag_documents 设计

`rag_documents.json` 是文档级 manifest，不进入 ChromaDB。

当前结构来自 `DocumentRecord`：

```json
{
  "documents": [
    {
      "doc_id": "doc_xxx",
      "file_name": "paper.pdf",
      "source_path": "D:\\...\\documents\\source_documents\\paper.pdf",
      "pdf_md5": "32fd75e7...",
      "file_size": 37690385,
      "mineru_output_dir": "D:\\...\\documents\\output_pipeline\\paper\\auto",
      "chunk_count": 106,
      "text_chunk_count": 76,
      "image_chunk_count": 19,
      "table_chunk_count": 11,
      "indexed_at": "2026-06-09T02:04:15.387382+00:00",
      "index_status": "indexed",
      "error_message": ""
    }
  ]
}
```

当前代码中会出现的状态包括：

```text
pending
parsed
chunked
indexed
failed
skipped
```

当前去重逻辑：

```text
1. 以 pdf_md5 为主判断是否已处理
2. 若 md5 已存在且未 force，则跳过或提示 MinerU 输出缺失
3. doc_id = md5(source_path + pdf_md5)
4. manifest 只记录文档状态，不负责自动删除旧 Chroma 记录
```

注意：

```text
文档发生变化时，manifest 会更新 doc_id 和状态；
真正删除旧 Chroma 记录需要 build_index.py 配合 --rebuild-doc 或 --reset。
```

## 6. MinerU block 到 RAG 类型的映射

当前代码中的真实映射定义在 `block_converter.py`：

```text
NOISE_TYPES:
  page_number
  page_aside_text
  page_footnote
  header
  footer

TEXT_TYPES:
  title
  paragraph
  text
  list
  equation_inline
  equation_interline

SPECIAL_TYPES:
  image -> image
  table -> table
```

因此当前 chunk 类型只有三种：

```text
text
image
table
```

公式不会单独建 chunk，而是并入 `text`。

## 7. SemanticBlock 设计

当前 `SemanticBlock` 真实字段如下：

```python
class SemanticBlock(BaseModel):
    block_id: str
    doc_id: str
    page_start: int
    page_end: int
    raw_type: str
    rag_type: str
    text: str = ""
    caption: str = ""
    source: str = ""
    bbox: str = ""
    reading_order: int
```

字段说明：

```text
text:
  当前 block 最终用于后续 chunk 构造的文本。
  对 image/table 来说，它先是 fallback 文本；
  如果启用 VLM，则会被替换成 VLM 摘要。

caption:
  image_caption / table_caption 的完整递归合并结果。

source:
  image/table 对应的真实图片路径。

bbox:
  从 MinerU 原始 block 直接序列化而来，当前不写入 ChunkRecord.metadata。
```

## 8. caption 与 fallback 文本提取

当前代码不是只取第一个 caption 片段，而是使用 `extract_all_content()` 递归提取所有 `content`。

这意味着：

```text
image_caption 被拆成多个 span 时，会先完整合并再写入 caption
table_caption 被拆成多个 span 时，也会完整合并
```

当前 fallback 文本构造规则：

```text
image:
  text = image_caption + image_footnote

table:
  text = table_caption + table_footnote + table_body/table_text/html
```

所以当前代码已经满足：

```text
1. caption 完整提取
2. VLM 失败时仍保留完整 fallback 文本
```

## 9. VLM 摘要增强

当前代码已经实现真实 VLM 摘要增强，位置在 `SemanticBlock` 阶段。

真实链路：

```text
content_list_v2.json
  -> convert_blocks()
  -> SemanticBlock
  -> VisualBlockSummarizer.enrich_blocks()
  -> 增强后的 SemanticBlock
  -> build_chunks()
```

实现模块：

```text
src/offline_index/vlm_client.py
src/offline_index/summary_cache.py
src/offline_index/visual_summarizer.py
```

当前行为：

```text
1. 仅对 rag_type=image/table 的 block 尝试增强
2. 必须存在真实 source 图片文件
3. 使用 caption + file_name + page_start 构造中文 prompt
4. 调用 OpenAI-compatible /chat/completions
5. 将返回结果写回 SemanticBlock.text
6. 若失败、无 source、无结果，则保留原 fallback text
7. 使用 data/cache/vlm_summaries.json 做缓存
8. 每个文档最多处理 VLM_MAX_IMAGES_PER_DOC 个 image/table block
```

当前提示词实现是“紧凑版中文 prompt”，不是文档设计中那种长模板，也没有结构化 JSON 输出要求。

## 10. ChunkRecord 与 rag_chunks 设计

当前 `ChunkRecord` 真实结构：

```python
class ChunkRecord(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any]
```

当前 `metadata` 字段：

```json
{
  "doc_id": "...",
  "file_name": "...",
  "chunk_type": "text | image | table",
  "page_start": 1,
  "page_end": 1,
  "source": "",
  "content_hash": "..."
}
```

当前 `chunk_id` 生成规则：

```text
chunk_id = "chunk_" + md5(doc_id + chunk_type + page_start + page_end + content_hash)
```

其中：

```text
content_hash = md5(document)
```

ChromaDB 写入时使用：

```text
ids        = chunk.id
documents  = chunk.document
embeddings = embedding(chunk.document)
metadatas  = chunk.metadata
```

## 11. 当前 chunk 切分实现

这里和早期设计文档差异最大。

当前代码没有使用 `RecursiveCharacterTextSplitter`，而是自定义 `_split_text()`：

```text
1. 先将所有 text block 按 reading_order 排序
2. 用 \n\n 拼成一个连续 text stream
3. 用 page_markers 记录字符区间到页码的映射
4. 按 chunk_size / chunk_overlap 做滑窗切分
5. 优先在以下分隔符附近断开：
   \n\n, \n, ". ", "。", "; ", "；", ", ", "，", " "
6. image/table 不参与文本切分，而是各自独立生成一个 chunk
```

因此当前实现更准确的描述应该是：

```text
结构感知来自 MinerU block 分类；
文本切分使用项目内自定义滑窗分割器，而不是 LangChain 的 RecursiveCharacterTextSplitter。
```

## 12. build_manifest.py 与 chunks_preview

`build_manifest.py --build-chunks-preview` 是当前离线阶段的核心中间产物生成器。

它会：

```text
1. 扫描 PDF
2. 更新 rag_documents.json
3. 定位 content_list_v2.json
4. 构建 SemanticBlock
5. 可选做 VLM 增强
6. 构建 ChunkRecord
7. 输出到 data/debug/<pdf_stem>_<pdf_md5>_chunks_preview.json
```

当前 preview 文件名规则：

```text
<pdf_stem>_<pdf_md5>_chunks_preview.json
```

这一步结束后：

```text
chunk.document 已经是最终准备向量化的文本
但还没有生成 embedding
```

## 13. build_index.py 与 Chroma 写入

当前 `build_index.py` 支持两种输入模式：

```text
1. 读取已有 chunks_preview.json
2. 直接用 --content-list-v2 + --images-dir 临时构建 chunk 再入库
```

当前默认模式是：

```text
读取 data/debug 下所有 *_chunks_preview.json
  -> 对每个 chunk.document 调用 embedding
  -> 写入 ChromaDB
  -> 更新 rag_documents.json 为 indexed
```

`--reset` 会重建整个 collection。  
`--rebuild-doc` 会按 `doc_id` 先删除后重建指定文档的 chunk。

## 14. 当前模块划分

当前代码中的真实模块应该写成：

```text
offline_index.config_loader
  读取 .env 配置

offline_index.source_file_finder
  扫描 PDF、计算 MD5

offline_index.mineru_runner
  调用 MinerU CLI

offline_index.mineru_output_locator
  定位 MinerU 输出目录和 content_list_v2.json

offline_index.mineru_output_reader
  读取 content_list_v2.json，并 flatten 页面 block

offline_index.block_converter
  将原始 MinerU block 转为 SemanticBlock

offline_index.visual_summarizer
  用 VLM 增强 image/table 的 SemanticBlock.text

offline_index.summary_cache
  缓存 VLM 摘要

offline_index.vlm_client
  调用 OpenAI-compatible 多模态接口

offline_index.chunk_builder
  生成 ChunkRecord

offline_index.chunk_loader
  读写 chunks_preview.json

offline_index.embedder
  调用 embedding 接口

offline_index.chroma_store
  写入、删除、查询 ChromaDB

offline_index.document_manifest
  读写 rag_documents.json

offline_index.offline_pipeline
  串联 block 转换、VLM 增强、chunk 构建
```

当前代码中并不存在这些名称：

```text
MinerUParser
image_summarizer
table_processor
chroma_writer
index_runner
```

文档里如果继续使用这些名字，会和实际代码脱节。

## 15. 当前实现边界

按现有代码，以下内容已经完成：

```text
1. .env 配置读取
2. PDF 文件夹扫描
3. PDF MD5 去重
4. rag_documents.json 文档清单维护
5. MinerU CLI 调用
6. content_list_v2.json 读取
7. header/footer/page_number 等噪声丢弃
8. text/image/table chunk 构建
9. image/table 的真实 VLM 摘要增强
10. embedding API 调用
11. ChromaDB rag_chunks 写入
12. 索引统计输出
```

当前尚未实现或尚未统一收口的部分：

```text
1. 统一的 index_runner.py
2. 基于 content_list.json / markdown 的自动兜底构建
3. 多版本索引管理
4. chunk 级增量更新
5. BM25 / 关键词索引
6. 表格结构化查询
7. 图片多轮理解
```

## 16. 推荐使用方式

当前项目推荐按下面顺序运行：

```powershell
python scripts\parse_pdfs.py
python scripts\build_manifest.py --build-chunks-preview --force
python scripts\build_index.py --reset
```

作用分别是：

```text
parse_pdfs.py:
  原始 PDF -> MinerU 输出

build_manifest.py --build-chunks-preview:
  MinerU 输出 -> rag_documents.json + chunks_preview.json

build_index.py --reset:
  chunks_preview.json -> embedding -> ChromaDB
```

这是当前代码逻辑下最准确的离线索引文档描述。

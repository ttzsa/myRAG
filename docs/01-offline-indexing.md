# 离线索引构建层设计

## 1. 目标

离线索引构建层负责把 `documents/source_documents` 下的 PDF 转换为可检索的本地知识库。

当前流程分为四个阶段：

```text
1. discover
   扫描源 PDF，计算 MD5，并维护 processed_pdfs.json。

2. parse
   根据 --parser-method 调用解析器，生成 content_list_v2.json / images / markdown。

3. chunk
   读取 content_list_v2.json，生成经过可选 VLM 增强的最终 chunk JSON。

4. index
   读取 chunk JSON，调用 embedding，并按 chunk_key + content_hash 增量写入 ChromaDB。
```

离线层最终产物：

```text
documents/source_documents/*.pdf
  -> documents/output_pipeline/<pdf_stem>/auto/*
  -> data/chunks/*_chunks.json
  -> data/chroma/
  -> data/index/processed_pdfs.json
```

## 2. 统一入口

默认完整流程：

```powershell
python scripts\offline.py
```

等价于：

```powershell
python scripts\offline.py --pdf-scope new --parser-method mineru --vlm-mode auto
```

重新处理全部 PDF：

```powershell
python scripts\offline.py --pdf-scope all --parser-method mineru
```

只重新生成指定 `content_list_v2.json` 的 chunk：

```powershell
python scripts\offline.py chunk --content-list-v2 path\to\xxx_content_list_v2.json --vlm-mode refresh
```

强制重建整个向量库：

```powershell
python scripts\offline.py index --reset
```

## 3. 配置

`.env.example` 当前 offline 相关字段：

```dotenv
MINERU_EXE=
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_OUTPUT_ROOT=documents/output_pipeline

PDF_ROOT=documents/source_documents
PDF_RECURSIVE=true
FORCE_REBUILD=false

PROCESSED_PDFS_PATH=data/index/processed_pdfs.json
CHUNKS_DIR=data/chunks

CHROMA_PERSIST_DIRECTORY=data/chroma
CHROMA_COLLECTION_NAME=rag_chunks

INGEST_CHUNK_SIZE=800
INGEST_CHUNK_OVERLAP=120

EMBEDDING_PROVIDER=openai-compatible
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
```

## 4. discover 阶段

discover 扫描 `PDF_ROOT` 下的 PDF，计算内容 MD5，并根据 `--pdf-scope` 选择本轮要处理的 PDF。

```text
--pdf-scope new:
  只选择 MD5 不在 processed_pdfs.json 中的 PDF。

--pdf-scope all:
  选择当前扫描到的全部 PDF。
```

PDF 扫描会排除临时文件和隐藏 PDF。

## 5. parse 阶段

parse 只处理 discover 选出的 PDF。

```text
--parser-method mineru:
  使用 MinerU CLI 解析 PDF。

--parser-method pymupdf:
  预留接口，目前会明确报未实现。
```

当前 MinerU 输出定位规则仍是：

```text
documents/output_pipeline/<pdf_stem>/auto/
```

后续主流程只消费 `*_content_list_v2.json`。

## 6. chunk 阶段

chunk 阶段把 `content_list_v2.json` 转换为最终 chunk 文件。

输出文件命名：

```text
data/chunks/<pdf_stem>_<pdf_md5>_chunks.json
```

参数：

```text
--content-list-v2:
  可重复传入多个 content_list_v2.json 路径。
  不传时处理当前扫描范围内可定位到的全部 content_list_v2.json。

--vlm-mode auto:
  有缓存则复用，没有缓存才调用 VLM。

--vlm-mode refresh:
  对选中的 content_list_v2.json 中的 image/table 重新调用 VLM 摘要。

--vlm-mode off:
  不调用 VLM，只使用 MinerU fallback 文本。
```

`--content-list-v2` 只决定输入范围，`--vlm-mode` 只决定摘要策略，两者互不影响。

## 7. index 阶段

index 阶段默认读取 `CHUNKS_DIR` 下的 `*_chunks.json`，并写入 ChromaDB。

默认增量行为：

```text
新 chunk:
  直接入库。

chunk_key 相同且 content_hash 未变:
  跳过。

chunk_key 相同但 content_hash 变化:
  删除旧 chunk 后写入新 chunk。
```

`--reset` 会清空整个 collection 后重新写入当前 chunk 文件。

## 8. chunk 身份

每个 chunk 同时携带：

```text
chunk_key:
  稳定身份，用于判断是否为同一个逻辑 chunk。

content_hash:
  当前 chunk.document 的内容 hash，用于判断内容是否变化。
```

这允许 image/table 的 VLM 摘要发生变化时，只更新对应逻辑 chunk。

## 9. processed_pdfs.json

`processed_pdfs.json` 是 PDF 处理记录表，不进入 ChromaDB。

它记录：

```text
doc_id
file_name
source_path
pdf_md5
file_size

parser_method
parse_status
parse_output_dir
content_list_v2_path
content_list_v2_md5
parse_error

chunk_status
chunk_path
chunk_file_md5
chunk_config_hash
vlm_mode
chunk_error

index_status
indexed_chunk_file_md5
embedding_model
embedding_dimension
indexed_at
index_error
```

## 10. 当前模块划分

```text
offline_index.config_loader
  读取 .env 配置。

offline_index.source_file_finder
  扫描 PDF、计算 MD5、根据 new/all scope 选择候选 PDF。

offline_index.mineru_runner
  调用 MinerU CLI。

offline_index.mineru_output_locator
  定位 MinerU 输出目录和 content_list_v2.json。

offline_index.mineru_output_reader
  读取 content_list_v2.json，并 flatten 页面 block。

offline_index.block_converter
  将原始 MinerU block 转为 SemanticBlock。

offline_index.visual_summarizer
  用 VLM 增强 image/table 的 SemanticBlock.text。

offline_index.summary_cache
  缓存 VLM 摘要。

offline_index.vlm_client
  调用 OpenAI-compatible 多模态接口。

offline_index.chunk_builder
  生成 ChunkRecord，并维护 chunk_key/content_hash。

offline_index.chunk_loader
  读写 chunk JSON。

offline_index.embedder
  调用 embedding 接口。

offline_index.chroma_store
  写入、删除、查询 ChromaDB，并支持按 chunk_key 增量同步。

offline_index.document_manifest
  读写 processed_pdfs.json。

offline_index.offline_pipeline
  串联 block 转换、VLM 增强、chunk 构建。
```

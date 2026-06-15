# myRAG

一个面向本地 PDF 资料的 RAG 原型项目。当前代码已经打通了：

- 离线索引：扫描 PDF、调用 MinerU 解析、生成 chunk、写入 ChromaDB
- 在线问答：基于 Chroma dense 检索取证据，再调用大模型生成带引用答案
- 批量评测：按问题列表批量调用问答脚本并落盘结果


## 1. 项目结构

```text
myRAG/
├─ documents/
│  ├─ source_documents/          # 你放 PDF 的地方
│  └─ output_pipeline/           # MinerU 解析输出目录
├─ data/
│  ├─ chunks/                    # 最终准备入库的 chunk JSON (便于debug)
│  ├─ chroma/                    # ChromaDB 持久化目录
│  ├─ vlm_cache/                 # VLM 摘要缓存
│  └─ index/processed_pdfs.json  # 文档处理状态
├─ scripts/
│  ├─ offline.py                 # 离线总入口
│  ├─ parse_pdfs.py              # 解析 PDF
│  ├─ build_manifest.py          # 生成 chunk + 更新 manifest
│  ├─ build_index.py             # 写入向量库
│  └─ ask.py                     # 在线问答
├─ src/
│  ├─ offline_index/             # 离线构建相关代码
│  └─ online_query/              # 在线查询相关代码
└─ eval/
   ├─ run_questions.py           # 批量测试问题  
   └─ questions.txt              # 问题列表
```

## 2. 前置条件

### 2.1 MinerU 怎么安装

官方参考：[MinerU GitHub 仓库](https://github.com/opendatalab/MinerU)

本项目不是自己解析 PDF，而是通过 `scripts/parse_pdfs.py` 调用本机安装好的 `mineru` CLI。

根据 MinerU 官方 README，目前推荐安装方式是：

```powershell
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
```

如果你不想用 `uv`，也可以先看官方仓库文档再选择其它安装方式。对本项目来说，关键不是安装方式，而是最后你本机能拿到 `mineru.exe` 或可执行的 `mineru` 命令。

### 2.2 MinerU 装完以后，文件放哪里

这里分两类文件，不要混：

- `mineru.exe` / `mineru` 可执行程序：放在你的 Python 环境 `Scripts` 目录里即可，通常无需手工复制
- ⭐待解析 PDF：放到本项目的 `documents/source_documents/` 目录下

本项目默认约定如下：

- PDF 输入目录：`documents/source_documents`
- MinerU 输出根目录：`documents/output_pipeline`
- 单个 PDF 的默认解析输出：`documents/output_pipeline/<pdf_stem>/auto/`

例如你放入：

```text
documents/source_documents/myPDF.pdf
```

解析后通常会得到类似目录：

```text
documents/output_pipeline/myPDF/auto/
```

这个目录下面包含：

- `myPDF_content_list_v2.json`
- `images/`
- markdown 等 MinerU 产物

### 2.3 MinerU 路径怎么填到项目里

修改 `.env` 里的关键项：

```dotenv
MINERU_EXE=D:\path\to\your\python_env\Scripts\mineru.exe
PDF_ROOT=documents/source_documents
MINERU_OUTPUT_ROOT=documents/output_pipeline
PROCESSED_PDFS_PATH=data/index/processed_pdfs.json
CHUNKS_DIR=data/chunks
CHROMA_PERSIST_DIRECTORY=data/chroma
CHROMA_COLLECTION_NAME=rag_chunks
```

如果你不知道 `mineru.exe` 在哪里，可以在激活环境后执行：

```powershell
where.exe mineru
```

把输出路径填进 `MINERU_EXE`。

### 2.4 大模型与向量化相关配置

当前项目会用到三类模型能力：

1. PDF 解析：由 MinerU CLI 提供
2. Embedding：离线建库时使用
3. Chat：在线问答时使用
4. VLM：图片表格摘要时使用

`.env` 里至少要检查这些项：

```dotenv
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=
EMBEDDING_DIMENSION=512

VLM_ENABLED=false
VLM_PROVIDER=openai-compatible
VLM_API_KEY=
VLM_BASE_URL=
VLM_MODEL=

CHAT_PROVIDER=openai-compatible
CHAT_API_KEY=
CHAT_BASE_URL=
CHAT_MODEL=qwen-plus
```

说明：

- 如果你只想先跑通离线流程，可以先用真实 embedding 配置，`VLM_ENABLED=false`
- 如果 `VLM_ENABLED=true`，离线 chunk 阶段会对图片/表格摘要调用 VLM
- 在线问答一定需要可用的 `CHAT_*` 配置

## 3. 安装项目依赖

本仓库已补充 `requirements.txt`。在虚拟环境中执行：

```powershell
pip install -r requirements.txt
```

当前 `requirements.txt` 只覆盖本项目 Python 依赖，不负责安装 MinerU 模型或 MinerU CLI 本身。MinerU 仍需按上一节单独安装。

## 4. Quick Start

### 4.1 第一次启动前建议按这个顺序做

1. 安装 Python 虚拟环境并激活
2. `pip install -r requirements.txt`
3. 按官方方式安装 MinerU
4. 复制 `.env.example` 为 `.env`
5. 配置 `MINERU_EXE`、Embedding、Chat 等参数
6. 把 PDF 放入 `documents/source_documents/`
7. 运行离线索引
8. 运行在线问答

### 4.2 推荐：统一运行 offline 四阶段

默认只处理新增 PDF，使用 MinerU 解析，VLM 使用缓存优先策略，并增量写入向量库：

```cmd
python scripts\offline.py
```

它等价于先后执行下面三条子命令：

```cmd
python scripts\parse_pdfs.py --pdf-scope new --parser-method mineru
python scripts\build_manifest.py --vlm-mode auto
python scripts\build_index.py
```

也就是说，`python scripts\offline.py` 会完成：

1. 扫描 `PDF_ROOT` 下的 PDF，计算 MD5，并根据 `processed_pdfs.json` 判断本轮要处理哪些 PDF。
2. 对选中的待处理 PDF 调用 MinerU，生成 `content_list_v2.json`、`images/`、markdown 等解析产物。
3. 把 `content_list_v2.json` 转成最终 chunk 文件，默认输出到 `CHUNKS_DIR`。
4. 把 chunk 做 embedding，并按 `chunk_key + content_hash` 增量写入 ChromaDB。

默认参数等价于：

```cmd
python scripts\offline.py --pdf-scope new --parser-method mineru --vlm-mode auto
```

参数含义：

- `--pdf-scope new`：只选择新增 PDF，以及已记录但 `content_list_v2.json` 缺失的 PDF。
- `--pdf-scope all`：选择当前扫描到的全部 PDF。
- `--parser-method mineru`：使用 MinerU 解析 PDF。
- `--parser-method pymupdf`：预留接口，目前尚未实现。
- `--vlm-mode auto`：生成 chunk 时，image/table 有 VLM 缓存就复用，没有缓存才调用 VLM。
- `--vlm-mode refresh`：对选中的 `content_list_v2.json` 中的 image/table 强制重新调用 VLM 摘要。
- `--vlm-mode off`：不调用 VLM，只使用 MinerU 提取出的 caption/table fallback 文本。
- `--reset`：完整流程的最后一步传给 `build_index.py`，清空并重建整个 Chroma collection。
- `--env-file .env.local`：使用非默认配置文件。

### 4.3 重新处理全部 PDF

```cmd
python scripts\offline.py --pdf-scope all --parser-method mineru
```

这等价于：

```cmd
python scripts\parse_pdfs.py --pdf-scope all --parser-method mineru
python scripts\build_manifest.py --vlm-mode auto
python scripts\build_index.py
```

### 4.4 只重新生成指定 `content_list_v2.json` 的 chunk

```cmd
python scripts\offline.py chunk --content-list-v2 path\to\xxx_content_list_v2.json --vlm-mode refresh
```

这只执行 chunk 阶段，等价于：

```cmd
python scripts\build_manifest.py --content-list-v2 path\to\xxx_content_list_v2.json --vlm-mode refresh
```

注意：这里必须直接传 `content_list_v2.json` 的路径，不传 PDF 文件名或 PDF 路径。

### 4.5 强制重建整个向量库

```cmd
python scripts\offline.py index --reset
```

这只执行 index 阶段，等价于：

```cmd
python scripts\build_index.py --reset
```

如果想重新解析全部 PDF、重新生成 chunk，并最终重建整个向量库，可以执行：

```cmd
python scripts\offline.py --pdf-scope all --parser-method mineru --vlm-mode refresh --reset
```

它等价于：

```cmd
python scripts\parse_pdfs.py --pdf-scope all --parser-method mineru
python scripts\build_manifest.py --vlm-mode refresh
python scripts\build_index.py --reset
```

## 5. 在线问答

### 5.1 最小用法

```cmd
python scripts\ask.py --query "DarkIR 在 LOLBlur 数据集上的表现如何？"
```

返回结果是 JSON，包含：

- `answer`：LLM 基于检索证据生成的答案
- `citations`：引用来源，包括文档名、页码、chunk 类型和图片/表格 source
- `used_chunks`：实际拼入 prompt 的证据块
- `confidence`：当前基于证据数量的粗略置信度
- `retrieval_debug_info`：仅在开启 debug 时返回

### 5.2 查看检索调试信息

```cmd
python scripts\ask.py --query "DarkIR 在 LOLBlur 数据集上的表现如何？" --show-debug
```

### 5.3 指定召回数量

```cmd
python scripts\ask.py --query "DarkIR 在 LOLBlur 数据集上的表现如何？" --top-k 3
```

### 5.4 使用非默认 `.env`

```cmd
python scripts\ask.py --env-file .env.local --query "DarkIR 的核心贡献是什么？"
```

## 6. 批量评测

脚本：[eval/run_questions.py](./eval/run_questions.py)

用法：

```cmd
python eval\run_questions.py
```

它会逐行读取：

```text
eval\questions.txt
```

然后逐个执行等价于：`python scripts\ask.py --query "问题"`

最后把结果写到 `eval` 文件夹下，文件名以时间命名，类似：

```text
20260612_114530.json
```

也支持这些参数：

```cmd
python eval\run_questions.py --show-debug --top-k 8
python eval\run_questions.py --stop-on-error
```

## 7. 目前还没有实现的功能

### 7.1 在线检索增强能力尚未实现

代码里当前还没有落地：

- sparse / BM25 / FTS 检索
- 多路召回
- RRF 融合
- Cross-Encoder 精排
- query rewrite / query expansion / query decomposition
- 基于意图的检索路由
- 多轮会话记忆

目前在线链路是：

```text
query -> dense retrieval -> context build -> LLM answer
```

### 7.2 PDF 解析备用方案（pymupdf）未实现

- `--parser-method pymupdf` 只是预留参数
- 当前真正可用的解析器只有 `mineru`

### 7.3 服务化接口未实现

但当前仓库还没有 FastAPI / Flask 之类的在线服务入口，主要还是 CLI 形态。

### 7.4 更完整的评测体系未实现

未实现完整的评测体系：

- Recall@k / MRR 等自动化检索指标
- dense 与 sparse 的消融实验
- rerank 消融实验
- LLM-as-a-judge
- case replay UI / dashboard
- 成本与延迟分阶段观测面板

## 8. 常见问题

### 9.1 为什么 `offline.py` 跑完了但没有 chunk

优先检查：

1. `MINERU_EXE` 是否可执行
2. `documents/output_pipeline/<pdf_stem>/auto/` 下是否真的生成了 `*_content_list_v2.json`
3. `build_manifest.py` 使用的 `PDF_ROOT` 和 `MINERU_OUTPUT_ROOT` 是否与 parse 阶段一致

### 8.2 为什么图片/表格摘要没有生效

检查：

1. `.env` 中是否设置 `VLM_ENABLED=true`
2. `VLM_API_KEY`、`VLM_BASE_URL`、`VLM_MODEL` 是否已配置
3. 运行命令时是否使用了 `--vlm-mode off`

### 8.3 为什么问答能跑但答案质量一般

这是当前实现边界，不一定是配置错误。因为在线侧现在只有 dense 检索，还没有：

- sparse 检索补充专有名词命中
- RRF 融合
- rerank 精排
- query rewrite
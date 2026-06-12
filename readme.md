# 01-offline-indexing 离线索引构建

### parse_pdfs.py：先让 MinerU 解析 PDF
```cmd
python scripts\parse_pdfs.py
```

### 扫描 PDF、找到对应 MinerU 输出、生成带 VLM 增强的 preview
--force：忽略已有 MD5 跳过逻辑，强制重新处理 PDF，并重新生成 manifest/chunks preview
--force-vlm: 用于强制绕过 VLM 摘要缓存并重新调用 VLM 生成VLM摘要

```cmd
python scripts\build_manifest.py --build-chunks-preview --force --force-vlm
```

### 把 preview 里的 chunk 做 embedding 并写入 ChromaDB
--reset 的作用是：先清空并重建 Chroma collection，再写入当前所有 chunk。也就是重新构建整个向量库

```cmd
python scripts\build_index.py --reset
```

# 02-online-query 在线问答

```cmd
python scripts\ask.py --query "DarkIR 在 LOLBlur 数据集上的表现如何？"
```

返回结果是 JSON，包含：

- `answer`：LLM 基于检索证据生成的答案
- `citations`：引用来源，包括文档名、页码、chunk 类型和图片/表格 source
- `used_chunks`：实际拼入 prompt 的证据块
- `confidence`：当前基于证据数量的粗略置信度
- `retrieval_debug_info`：仅在开启 debug 时返回

查看检索调试信息：

```cmd
python scripts\ask.py --query "DarkIR 在 LOLBlur 数据集上的表现如何？" --show-debug
```

指定召回数量：

```cmd
python scripts\ask.py --query "DarkIR 在 LOLBlur 数据集上的表现如何？" --top-k 3
```

使用非默认 `.env`：

```cmd
python scripts\ask.py --env-file .env.local --query "DarkIR 的核心贡献是什么？"
```

# 批量评测

脚本：[eval/run_questions.py](D:/t_study/python_code/myRAG/eval/run_questions.py)。

用法：

```cmd
python eval\run_questions.py
```

```text
它会逐行读取： eval\questions.txt
然后逐个执行等价于： python scripts\ask.py --query "问题"
最后把结果写到 `eval` 文件夹下，文件名类似： 20260612_114530.json
也支持这些参数：
D:\t_config\anaconda\envs\ai\python.exe eval\run_questions.py --show-debug --top-k 8
D:\t_config\anaconda\envs\ai\python.exe eval\run_questions.py --stop-on-error
```
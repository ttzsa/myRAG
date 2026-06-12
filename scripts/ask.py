# Command-line V0 online dense RAG question-answering entry point.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for online question answering."""

    parser = argparse.ArgumentParser(description="Ask a question using the V0 dense RAG pipeline.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--persist-dir", type=Path, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--embedder", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-debug", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    """Run the online QA pipeline and print a JSON response."""

    from offline_index.chroma_store import get_chroma_client, get_or_create_collection
    from offline_index.config_loader import load_config, resolve_value
    from offline_index.embedder import create_embedder
    from online_query.config import load_online_config
    from online_query.dense_retriever import DenseRetriever
    from online_query.llm_client import OpenAICompatibleChatClient
    from online_query.pipeline import OnlineQAPipeline

    args = parse_args()
    offline_config = load_config(args.env_file)
    online_config = load_online_config(args.env_file)

    persist_dir = resolve_value(args.persist_dir, offline_config.chroma.persist_dir)
    collection_name = resolve_value(args.collection, offline_config.chroma.collection)
    embedder_name = resolve_value(args.embedder, offline_config.embedding.provider)
    top_k = resolve_value(args.top_k, online_config.retrieval.top_k)

    embedder = create_embedder(embedder_name, offline_config.embedding)
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client, collection_name)
    dense_retriever = DenseRetriever(collection=collection, embedder=embedder)
    llm_client = OpenAICompatibleChatClient(
        base_url=online_config.chat.base_url,
        model=online_config.chat.model,
        api_key=online_config.chat.api_key,
        timeout=online_config.chat.timeout,
        max_retries=online_config.chat.max_retries,
    )
    pipeline = OnlineQAPipeline(
        dense_retriever=dense_retriever,
        llm_client=llm_client,
        top_k=top_k,
        max_context_chars=online_config.retrieval.max_context_chars,
    )
    result = pipeline.answer(args.query, include_debug=args.show_debug)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

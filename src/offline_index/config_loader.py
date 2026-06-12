# Loads typed application configuration from .env files for CLI scripts.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dotenv import dotenv_values, load_dotenv


T = TypeVar("T")


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem path and scan behavior configuration."""

    pdf_root: Path
    mineru_output_root: Path
    manifest_path: Path
    debug_dir: Path
    pdf_recursive: bool
    force_rebuild: bool


@dataclass(frozen=True)
class MinerUConfig:
    """MinerU CLI configuration."""

    exe: Path
    backend: str
    method: str


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB persistence and collection configuration."""

    persist_dir: Path
    collection: str


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk construction defaults."""

    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding provider configuration."""

    provider: str
    api_key: str
    base_url: str
    model: str
    batch_size: int
    timeout: float
    dimension: int | None
    mock_dimension: int


@dataclass(frozen=True)
class VLMConfig:
    """Visual-language model configuration for image/table summarization."""

    enabled: bool
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float
    max_retries: int
    cache_path: Path
    max_images_per_doc: int


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration loaded from .env."""

    paths: PathsConfig
    mineru: MinerUConfig
    chroma: ChromaConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vlm: VLMConfig


def load_config(env_file: Path = Path(".env")) -> AppConfig:
    """Load .env values into a typed AppConfig."""

    env_file = Path(env_file)
    load_dotenv(env_file, override=False)
    values = dotenv_values(env_file)
    return AppConfig(
        paths=PathsConfig(
            pdf_root=_get_path(values, "PDF_ROOT", "documents/source_documents"),
            mineru_output_root=_get_path(values, "MINERU_OUTPUT_ROOT", "documents/output_pipeline"),
            manifest_path=_get_path(values, "RAG_DOCUMENTS_PATH", "data/index/rag_documents.json"),
            debug_dir=_get_path(values, "DEBUG_DIR", "data/debug"),
            pdf_recursive=_get_bool(values, "PDF_RECURSIVE", True),
            force_rebuild=_get_bool(values, "FORCE_REBUILD", False),
        ),
        mineru=MinerUConfig(
            exe=_get_path(values, "MINERU_EXE", r"D:\t_config\anaconda\envs\ai\Scripts\mineru.exe"),
            backend=_get_str(values, "MINERU_BACKEND", "pipeline"),
            method=_get_str(values, "MINERU_METHOD", "auto"),
        ),
        chroma=ChromaConfig(
            persist_dir=_get_path(values, "CHROMA_PERSIST_DIRECTORY", "data/chroma"),
            collection=_get_str(values, "CHROMA_COLLECTION_NAME", "rag_chunks"),
        ),
        chunking=ChunkingConfig(
            chunk_size=_get_int(values, "INGEST_CHUNK_SIZE", 800),
            chunk_overlap=_get_int(values, "INGEST_CHUNK_OVERLAP", 120),
        ),
        embedding=EmbeddingConfig(
            provider=_get_str(values, "EMBEDDING_PROVIDER", "mock"),
            api_key=_get_str(values, "EMBEDDING_API_KEY", ""),
            base_url=_get_str(values, "EMBEDDING_BASE_URL", ""),
            model=_get_str(values, "EMBEDDING_MODEL", ""),
            batch_size=_get_int(values, "EMBEDDING_BATCH_SIZE", 32),
            timeout=_get_float(values, "EMBEDDING_TIMEOUT_SECONDS", 60),
            dimension=_get_optional_int(values, "EMBEDDING_DIMENSION"),
            mock_dimension=_get_int(values, "MOCK_EMBEDDING_DIMENSION", 384),
        ),
        vlm=VLMConfig(
            enabled=_get_bool(values, "VLM_ENABLED", False),
            provider=_get_str(values, "VLM_PROVIDER", "openai-compatible"),
            api_key=_get_str(values, "VLM_API_KEY", ""),
            base_url=_get_str(values, "VLM_BASE_URL", ""),
            model=_get_str(values, "VLM_MODEL", ""),
            timeout=_get_float(values, "VLM_TIMEOUT_SECONDS", 60),
            max_retries=_get_int(values, "VLM_MAX_RETRIES", 3),
            cache_path=_get_path(values, "VLM_CACHE_PATH", "data/cache/vlm_summaries.json"),
            max_images_per_doc=_get_int(values, "VLM_MAX_IMAGES_PER_DOC", 50),
        ),
    )


def resolve_value(cli_value: T | None, config_value: T) -> T:
    """Return an explicit CLI value when present, otherwise the config value."""

    return config_value if cli_value is None else cli_value


def optional_value(value: T | None) -> T | None:
    """Return a value unchanged; used to make CLI override intent explicit in scripts."""

    return value


def _get_str(values: dict, key: str, default: str) -> str:
    """Read a string value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _get_path(values: dict, key: str, default: str) -> Path:
    """Read a path value from parsed dotenv values."""

    return Path(_get_str(values, key, default))


def _get_bool(values: dict, key: str, default: bool) -> bool:
    """Read and convert a boolean value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean for {key}: {value}")


def _get_int(values: dict, key: str, default: int) -> int:
    """Read and convert an integer value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return default
    return int(str(value).strip())


def _get_optional_int(values: dict, key: str) -> int | None:
    """Read and convert an optional integer value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return None
    return int(str(value).strip())


def _get_float(values: dict, key: str, default: float) -> float:
    """Read and convert a float value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return default
    return float(str(value).strip())

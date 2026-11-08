"""Central configuration for Kuku AI Legal & Citizen Assistance.

This module defines settings only. Retrieval, token counting and answer-length
instructions must be implemented by the modules that consume these settings.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent

# Deployment environment variables take precedence over the local .env file.
load_dotenv(ROOT_DIR / ".env", override=False)


def _str_env(name: str, default: str) -> str:
    """Reject explicitly empty model names instead of failing at model load."""
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer environment variable and enforce its lower bound."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float_env(name: str, default: float) -> float:
    """Read a finite number; reject NaN and infinity."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean without treating the string 'false' as truthy."""
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    # Project location; all existing directory layouts are preserved.
    root_dir: Path = ROOT_DIR

    # repr=False prevents accidental exposure via print(settings) or logging.
    # Never log dataclasses.asdict(settings): it still contains the secret.
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip(),
        repr=False,
    )

    # Language and speech models.
    llm_model: str = field(
        default_factory=lambda: _str_env("LLM_MODEL", "openai/gpt-oss-120b")
    )
    stt_model: str = field(
        default_factory=lambda: _str_env("STT_MODEL", "whisper-large-v3-turbo")
    )

    # Embeddings. Changing the model requires rebuilding the stored index.
    # The embedding module must verify the actual dimension and input limit.
    embedding_model: str = field(
        default_factory=lambda: _str_env(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: _int_env("EMBEDDING_DIM", 384)
    )
    embedding_max_tokens: int = field(
        default_factory=lambda: _int_env("EMBEDDING_MAX_TOKENS", 512)
    )

    # Hybrid retrieval: semantic candidates + lexical candidates -> RRF.
    top_k_candidates: int = field(
        default_factory=lambda: _int_env("TOP_K_CANDIDATES", 40)
    )
    bm25_top_k: int = field(
        default_factory=lambda: _int_env("BM25_TOP_K", 40)
    )
    rerank_top_n: int = field(
        default_factory=lambda: _int_env("RERANK_TOP_N", 24)
    )
    top_k_context: int = field(
        default_factory=lambda: _int_env("TOP_K_CONTEXT", 6)
    )
    rrf_k: int = field(default_factory=lambda: _int_env("RRF_K", 60))

    # Cross-encoder reranking.
    enable_reranking: bool = field(
        default_factory=lambda: _bool_env("ENABLE_RERANKING", True)
    )
    reranker_model: str = field(
        default_factory=lambda: _str_env(
            "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
    )
    reranker_batch_size: int = field(
        default_factory=lambda: _int_env("RERANKER_BATCH_SIZE", 8)
    )

    # Expansion must retain the original query in the retrieval module.
    enable_query_expansion: bool = field(
        default_factory=lambda: _bool_env("ENABLE_QUERY_EXPANSION", True)
    )

    # Legal chunking. Count tokens with the embedding model's tokenizer.
    chunk_target_tokens: int = field(
        default_factory=lambda: _int_env("CHUNK_TARGET_TOKENS", 380)
    )
    chunk_overlap_tokens: int = field(
        default_factory=lambda: _int_env("CHUNK_OVERLAP_TOKENS", 60, minimum=0)
    )

    # Generation controls. These take effect when wired into the LLM module.
    # Sentence count is a prompt target, not a reason to cut off a qualification.
    answer_max_sentences: int = field(
        default_factory=lambda: _int_env("ANSWER_MAX_SENTENCES", 4)
    )
    llm_temperature: float = field(
        default_factory=lambda: _float_env("LLM_TEMPERATURE", 0.1)
    )
    # Completion tokens can include reasoning, depending on the selected model.
    # Keep this separate from the desired length of the visible answer.
    llm_max_tokens: int = field(
        default_factory=lambda: _int_env("LLM_MAX_TOKENS", 1024)
    )
    groq_timeout_seconds: float = field(
        default_factory=lambda: _float_env("GROQ_TIMEOUT_SECONDS", 30.0)
    )
    groq_max_retries: int = field(
        default_factory=lambda: _int_env("GROQ_MAX_RETRIES", 2, minimum=0)
    )

    def __post_init__(self) -> None:
        """Validate environment settings and direct Settings(...) overrides."""
        if not isinstance(self.root_dir, (str, Path)):
            raise ValueError("root_dir must be a path or path string")
        if not str(self.root_dir).strip():
            raise ValueError("root_dir must not be empty")
        object.__setattr__(self, "root_dir", Path(self.root_dir).expanduser().resolve())

        # Environment parsers do not run for explicitly supplied field values.
        # Validate those too, including bool (which is an int subclass in Python).
        for item in fields(self):
            value = getattr(self, item.name)
            name = item.name.upper()
            if item.type == "int":
                minimum = 0 if item.name in {
                    "chunk_overlap_tokens", "groq_max_retries"
                } else 1
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}")
            elif item.type == "float":
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} must be a finite number")
            elif item.type == "bool":
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a boolean")
            elif item.type == "str":
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                value = value.strip()
                if item.name != "groq_api_key" and not value:
                    raise ValueError(f"{name} must not be empty")
                object.__setattr__(self, item.name, value)

        if self.top_k_context > self.rerank_top_n:
            raise ValueError("TOP_K_CONTEXT cannot exceed RERANK_TOP_N")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS"
            )
        if self.chunk_target_tokens >= self.embedding_max_tokens:
            raise ValueError(
                "CHUNK_TARGET_TOKENS must be below EMBEDDING_MAX_TOKENS; "
                "the embedding module must also check prefixes, metadata "
                "and special tokens against the actual model limit"
            )
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.groq_timeout_seconds <= 0:
            raise ValueError("GROQ_TIMEOUT_SECONDS must be greater than zero")

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "processed"

    @property
    def vector_store_dir(self) -> Path:
        return self.root_dir / "vector_store"

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key)

    def require_groq_key(self) -> str:
        """Call only when initializing an API client, not during ingestion."""
        if not self.has_groq_key:
            raise ValueError("Set GROQ_API_KEY in the environment or project .env")
        return self.groq_api_key


settings = Settings()

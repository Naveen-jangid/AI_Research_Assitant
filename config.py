"""
config.py – Centralised configuration loader.

All settings are read from environment variables (with sensible defaults),
making it trivial to swap providers via a .env file or CI/CD secrets without
touching source code.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env if present (no-op in production where env vars are injected)
load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class AppConfig:
    # ── App metadata ──────────────────────────────────────────────────────────
    app_title: str = field(default_factory=lambda: _env("APP_TITLE", "AI Research Assistant"))
    max_file_size_mb: int = field(default_factory=lambda: _env_int("MAX_FILE_SIZE_MB", 50))

    # ── LLM provider selection ────────────────────────────────────────────────
    # "openai" | "anthropic" | "ollama"
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai").lower())

    # ── OpenAI settings ───────────────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))

    # ── Anthropic settings ────────────────────────────────────────────────────
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-6"))

    # ── Ollama settings ───────────────────────────────────────────────────────
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3"))

    # ── Embedding settings ────────────────────────────────────────────────────
    # "openai" | "huggingface"
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "huggingface").lower())
    hf_embedding_model: str = field(
        default_factory=lambda: _env("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # ── Vector store settings ─────────────────────────────────────────────────
    # "chroma" | "faiss"
    vector_store_type: str = field(default_factory=lambda: _env("VECTOR_STORE_TYPE", "chroma").lower())
    chroma_persist_dir: str = field(default_factory=lambda: _env("CHROMA_PERSIST_DIR", "./chroma_db"))

    # ── RAG / chunking parameters ─────────────────────────────────────────────
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    top_k_retrieval: int = field(default_factory=lambda: _env_int("TOP_K_RETRIEVAL", 5))

    # ── LLM generation parameters ─────────────────────────────────────────────
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def active_model_name(self) -> str:
        """Human-readable name of the currently active LLM."""
        mapping = {
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "ollama": self.ollama_model,
        }
        return mapping.get(self.llm_provider, "unknown")

    def validate(self) -> list[str]:
        """
        Return a list of validation error strings.
        An empty list means the configuration is valid.
        """
        errors: list[str] = []

        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

        if self.embedding_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        if self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        return errors


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this object everywhere instead of instantiating AppConfig repeatedly.
cfg = AppConfig()

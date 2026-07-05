from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Environment-backed application configuration."""

    openai_api_key: str
    google_drive_folder_ids: tuple[str, ...]
    google_application_credentials: str | None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    vector_store_dir: Path = Path("data/chroma")
    chunk_min_tokens: int = 1000
    chunk_max_tokens: int = 1500
    chunk_overlap_tokens: int = 150
    retrieval_top_k: int = 5
    embedding_batch_size: int = 64
    api_basic_auth_username: str | None = None
    api_basic_auth_password: str | None = None

    @property
    def auth_enabled(self) -> bool:
        """Enable Basic Auth only when both credentials are configured."""
        return bool(self.api_basic_auth_username and self.api_basic_auth_password)


def _split_folder_ids(value: str | None) -> tuple[str, ...]:
    """Parse comma-separated Drive folder IDs from an environment variable."""
    if not value:
        return tuple()
    return tuple(folder_id.strip() for folder_id in value.split(",") if folder_id.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once so every module sees a consistent configuration."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        google_drive_folder_ids=_split_folder_ids(os.getenv("GOOGLE_DRIVE_FOLDER_ID")),
        google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", "data/chroma")),
        chunk_min_tokens=int(os.getenv("CHUNK_MIN_TOKENS", "1000")),
        chunk_max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "1500")),
        chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "150")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        api_basic_auth_username=os.getenv("API_BASIC_AUTH_USERNAME"),
        api_basic_auth_password=os.getenv("API_BASIC_AUTH_PASSWORD"),
    )

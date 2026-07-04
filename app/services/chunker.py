from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any, Protocol

try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised only in partial local environments
    tiktoken = None  # type: ignore[assignment]

from app.services.models import DocumentChunk, DocumentPage
from app.utils.text_cleaner import clean_text
from app.utils.logger import get_logger


logger = get_logger(__name__)


class TokenEncoding(Protocol):
    def encode(self, text: str) -> list[Any]:
        ...

    def decode(self, tokens: list[Any]) -> str:
        ...


class RegexTokenEncoding:
    """Deterministic fallback used when tiktoken cannot load its BPE table."""

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)

    def encode(self, text: str) -> list[str]:
        return self._TOKEN_RE.findall(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


class TextChunker:
    def __init__(
        self,
        min_tokens: int = 1000,
        max_tokens: int = 1500,
        overlap_tokens: int = 150,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if min_tokens <= 0 or max_tokens <= 0:
            raise ValueError("Chunk token sizes must be positive")
        if min_tokens > max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = self._load_encoding(encoding_name)

    def chunk_pages(self, pages: Iterable[DocumentPage]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_index_by_file: dict[str, int] = {}

        for page in pages:
            text = clean_text(page.text)
            if not text:
                continue

            tokens = self.encoding.encode(text)
            if not tokens:
                continue

            start = 0
            while start < len(tokens):
                end = min(start + self.max_tokens, len(tokens))
                token_slice = tokens[start:end]

                if end < len(tokens) and len(token_slice) < self.min_tokens:
                    break

                chunk_text = self.encoding.decode(token_slice).strip()
                if chunk_text:
                    chunk_index = chunk_index_by_file.get(page.file_id, 0)
                    chunks.append(self._make_chunk(page, chunk_index, chunk_text))
                    chunk_index_by_file[page.file_id] = chunk_index + 1

                if end >= len(tokens):
                    break
                start = max(end - self.overlap_tokens, start + 1)

        return chunks

    @staticmethod
    def _load_encoding(encoding_name: str) -> TokenEncoding:
        if tiktoken is None:
            logger.warning("Falling back to regex tokenization because tiktoken is not installed")
            return RegexTokenEncoding()
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning("Falling back to regex tokenization because tiktoken could not load %s: %s", encoding_name, exc)
            return RegexTokenEncoding()

    @staticmethod
    def _make_chunk(page: DocumentPage, chunk_index: int, text: str) -> DocumentChunk:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        stable_key = f"{page.file_id}:{page.page_number or 0}:{chunk_index}:{content_hash}"
        chunk_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
        return DocumentChunk(
            chunk_id=chunk_id,
            file_name=page.file_name,
            file_id=page.file_id,
            source_link=page.source_link,
            page_number=page.page_number,
            chunk_index=chunk_index,
            text=text,
            content_hash=content_hash,
        )

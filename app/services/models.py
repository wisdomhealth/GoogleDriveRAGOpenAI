from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentPage:
    file_name: str
    file_id: str
    source_link: str
    page_number: int | None
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    file_name: str
    file_id: str
    source_link: str
    page_number: int | None
    chunk_index: int
    text: str
    content_hash: str

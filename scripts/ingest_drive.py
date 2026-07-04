from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.db.vector_store import ChromaVectorStore
from app.services.chunker import TextChunker
from app.services.drive_loader import GoogleDriveLoader
from app.services.embedding import OpenAIEmbeddingService
from app.services.models import DocumentPage
from app.utils.logger import configure_logging, get_logger


logger = get_logger(__name__)


async def collect_pages(loader: GoogleDriveLoader, folder_ids: tuple[str, ...]) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    async for page in loader.iter_folder_documents(folder_ids):
        pages.append(page)
    return pages


async def main() -> None:
    configure_logging()
    settings = get_settings()
    if not settings.google_drive_folder_ids:
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID is required. Use commas for multiple folders.")

    vector_store = ChromaVectorStore(settings.vector_store_dir)
    vector_store.load()

    loader = GoogleDriveLoader(settings.google_application_credentials)
    pages = await collect_pages(loader, settings.google_drive_folder_ids)
    logger.info("Extracted %s pages/documents from Google Drive", len(pages))

    chunker = TextChunker(
        min_tokens=settings.chunk_min_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunks = chunker.chunk_pages(pages)
    existing_ids = vector_store.existing_chunk_ids()
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    logger.info("Prepared %s chunks (%s new, %s existing)", len(chunks), len(new_chunks), len(chunks) - len(new_chunks))

    if not new_chunks:
        logger.info("No new chunks to embed")
        return

    embedding_service = OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    embeddings = await embedding_service.embed_texts([chunk.text for chunk in new_chunks])
    added = vector_store.add(new_chunks, embeddings)
    vector_store.save()
    logger.info("Ingestion complete. Added %s chunks.", added)


if __name__ == "__main__":
    asyncio.run(main())

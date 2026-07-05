from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from app.services.models import DocumentChunk
from app.utils.logger import get_logger


logger = get_logger(__name__)

COLLECTION_NAME = "google_drive_docs"


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot serve the requested operation."""

    pass


class ChromaVectorStore:
    """Persistent Chroma-backed vector store for document chunks."""

    def __init__(self, storage_dir: Path, collection_name: str = COLLECTION_NAME) -> None:
        """Store Chroma connection settings until load() opens the collection."""
        self.storage_dir = storage_dir
        self.collection_name = collection_name
        self.client: Any | None = None
        self.collection: Any | None = None

    @property
    def is_ready(self) -> bool:
        """Return whether the collection is loaded and contains vectors."""
        return self.collection is not None and self.collection.count() > 0

    def load(self) -> None:
        """Open or create the persistent Chroma collection."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.storage_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # Cosine distance matches the retrieval score conversion below.
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Loaded Chroma collection %s with %s records", self.collection_name, self.collection.count())

    def save(self) -> None:
        """Assert the collection exists and log its persisted record count."""
        self._require_collection()
        logger.info("Chroma collection %s persisted with %s records", self.collection_name, self.collection.count())

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> int:
        """Add new chunks and embeddings, skipping IDs already in Chroma."""
        collection = self._require_collection()
        if not chunks:
            return 0
        if len(chunks) != embeddings.shape[0]:
            raise VectorStoreError("Chunk and embedding counts do not match")

        existing_ids = self.existing_chunk_ids([chunk.chunk_id for chunk in chunks])
        new_chunks: list[DocumentChunk] = []
        new_embeddings: list[list[float]] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.chunk_id in existing_ids:
                continue
            new_chunks.append(chunk)

            # Chroma's Python API expects JSON-serializable lists, not ndarray rows.
            new_embeddings.append(np.asarray(embedding, dtype=np.float32).tolist())

        if not new_chunks:
            return 0

        collection.add(
            ids=[chunk.chunk_id for chunk in new_chunks],
            documents=[chunk.text for chunk in new_chunks],
            embeddings=new_embeddings,
            metadatas=[self._metadata_from_chunk(chunk) for chunk in new_chunks],
        )
        return len(new_chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """Search by query embedding and return normalized result dictionaries."""
        collection = self._require_collection()
        if collection.count() == 0:
            raise VectorStoreError("Vector store is empty. Run scripts/ingest_drive.py first.")

        query_vector = np.asarray(query_embedding, dtype=np.float32).tolist()
        response = collection.query(
            query_embeddings=[query_vector],
            # Never ask Chroma for more results than are available.
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
            item = dict(metadata or {})
            item["chunk_id"] = str(item.get("chunk_id") or chunk_id)
            item["text"] = document or ""
            item["distance"] = float(distance)

            # With cosine distance, a simple 1 - distance score is easier for
            # callers to read while preserving Chroma's ranking.
            item["score"] = 1.0 - float(distance)
            item["page_number"] = self._restore_page_number(item.get("page_number"))
            results.append(item)
        return results

    def existing_chunk_ids(self, chunk_ids: list[str] | None = None) -> set[str]:
        """Return stored chunk IDs, optionally limited to a candidate list."""
        collection = self._require_collection()
        if chunk_ids is None:
            result = collection.get(include=[])
        elif not chunk_ids:
            return set()
        else:
            result = collection.get(ids=chunk_ids, include=[])
        return set(result.get("ids", []))

    def _require_collection(self) -> Any:
        """Return the loaded collection or raise a clear vector-store error."""
        if self.collection is None:
            raise VectorStoreError("Vector store is not loaded")
        return self.collection

    @staticmethod
    def _metadata_from_chunk(chunk: DocumentChunk) -> dict[str, str | int]:
        """Convert chunk metadata to Chroma-compatible scalar values."""
        data = asdict(chunk)
        return {
            "file_name": str(data["file_name"]),
            "file_id": str(data["file_id"]),
            "source_link": str(data["source_link"]),
            "page_number": int(data["page_number"] or 0),
            "chunk_id": str(data["chunk_id"]),
            "chunk_index": int(data["chunk_index"]),
            "content_hash": str(data["content_hash"]),
        }

    @staticmethod
    def _restore_page_number(value: Any) -> int | None:
        """Convert Chroma's stored zero sentinel back to None."""
        if value in (None, "", 0, "0"):
            return None
        return int(value)

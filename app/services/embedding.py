from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
from openai import AsyncOpenAI


class OpenAIEmbeddingService:
    """Thin async wrapper around OpenAI embedding calls."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 64) -> None:
        """Create an OpenAI embedding client with batching settings."""
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts and return a float32 matrix in input order."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = await self.client.embeddings.create(model=self.model, input=list(batch))

            # OpenAI returns an index with each item; sort defensively so vectors
            # remain aligned with the original chunk order before storing.
            ordered = sorted(response.data, key=lambda item: item.index)
            batches.append(np.array([item.embedding for item in ordered], dtype=np.float32))
            await asyncio.sleep(0)
        return np.vstack(batches).astype(np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
from openai import AsyncOpenAI


class OpenAIEmbeddingService:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 64) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = await self.client.embeddings.create(model=self.model, input=list(batch))
            ordered = sorted(response.data, key=lambda item: item.index)
            batches.append(np.array([item.embedding for item in ordered], dtype=np.float32))
            await asyncio.sleep(0)
        return np.vstack(batches).astype(np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        embeddings = await self.embed_texts([query])
        return embeddings[0]

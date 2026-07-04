from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.vector_store import ChromaVectorStore
from app.services.embedding import OpenAIEmbeddingService
from app.services.llm import OpenAIChatService


SYSTEM_PROMPT = """You are a precise assistant answering questions using retrieved Google Drive document context.
Use only the provided context when possible. If the context is insufficient, say what is missing.
Always cite the source file names in the answer when making document-grounded claims."""


@dataclass(frozen=True)
class Source:
    file_name: str
    file_id: str
    source_link: str
    page_number: int | None
    snippet: str
    score: float


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[Source]


class RagPipeline:
    def __init__(
        self,
        embeddings: OpenAIEmbeddingService,
        llm: OpenAIChatService,
        vector_store: ChromaVectorStore,
        top_k: int = 5,
    ) -> None:
        self.embeddings = embeddings
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k

    async def answer(self, question: str) -> RagAnswer:
        query_embedding = await self.embeddings.embed_query(question)
        retrieved = self.vector_store.search(query_embedding, self.top_k)
        sources = [self._source_from_item(item) for item in retrieved]
        prompt = self._build_prompt(question, retrieved)
        answer = await self.llm.complete(SYSTEM_PROMPT, prompt)
        return RagAnswer(answer=answer, sources=sources)

    async def stream_answer(self, question: str):
        query_embedding = await self.embeddings.embed_query(question)
        retrieved = self.vector_store.search(query_embedding, self.top_k)
        prompt = self._build_prompt(question, retrieved)
        async for token in self.llm.stream(SYSTEM_PROMPT, prompt):
            yield token

    @staticmethod
    def _source_from_item(item: dict[str, Any]) -> Source:
        return Source(
            file_name=item["file_name"],
            file_id=item["file_id"],
            source_link=item.get("source_link", ""),
            page_number=item.get("page_number"),
            snippet=_snippet(item["text"]),
            score=float(item.get("score", 0.0)),
        )

    @staticmethod
    def _build_prompt(question: str, retrieved: list[dict[str, Any]]) -> str:
        context_blocks = []
        for index, item in enumerate(retrieved, start=1):
            page = f", page {item['page_number']}" if item.get("page_number") else ""
            context_blocks.append(
                f"[{index}] File: {item['file_name']} (id: {item['file_id']}{page})\n"
                f"Snippet:\n{item['text']}"
            )
        context = "\n\n---\n\n".join(context_blocks)
        return f"Question: {question}\n\nRetrieved context:\n{context}\n\nAnswer with citations by file name."


def _snippet(text: str, max_chars: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."

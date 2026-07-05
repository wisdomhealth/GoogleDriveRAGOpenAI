from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.auth import require_basic_auth
from app.db.vector_store import VectorStoreError
from app.services.rag import RagPipeline


router = APIRouter(prefix="", tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""

    question: str = Field(..., min_length=1, max_length=4000)


class SourceResponse(BaseModel):
    """Retrieved document source returned with an answer."""

    file_name: str
    file_id: str
    snippet: str
    source_link: str = ""
    page_number: int | None = None


class ChatResponse(BaseModel):
    """Non-streaming chat response payload."""

    answer: str
    sources: list[SourceResponse]


def get_rag(request: Request) -> RagPipeline:
    """Fetch the process-wide RAG pipeline initialized during app startup."""
    return request.app.state.rag


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_basic_auth)])
async def chat(payload: ChatRequest, rag: RagPipeline = Depends(get_rag)) -> ChatResponse:
    """Answer a question and return the answer plus retrieved source snippets."""
    try:
        result = await rag.answer(payload.question)
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                file_name=source.file_name,
                file_id=source.file_id,
                snippet=source.snippet,
                source_link=source.source_link,
                page_number=source.page_number,
            )
            for source in result.sources
        ],
    )


@router.post("/chat/stream", dependencies=[Depends(require_basic_auth)])
async def stream_chat(payload: ChatRequest, rag: RagPipeline = Depends(get_rag)) -> StreamingResponse:
    """Stream answer tokens as Server-Sent Events."""

    async def events():
        """Translate RAG token chunks into SSE frames."""
        try:
            async for token in rag.stream_answer(payload.question):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except VectorStoreError as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

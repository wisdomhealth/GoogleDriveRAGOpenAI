from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.config import get_settings
from app.db.vector_store import ChromaVectorStore
from app.services.embedding import OpenAIEmbeddingService
from app.services.llm import OpenAIChatService
from app.services.rag import RagPipeline
from app.utils.logger import configure_logging, get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    vector_store = ChromaVectorStore(settings.vector_store_dir)
    vector_store.load()
    embeddings = OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    llm = OpenAIChatService(api_key=settings.openai_api_key, model=settings.openai_chat_model)
    app.state.rag = RagPipeline(
        embeddings=embeddings,
        llm=llm,
        vector_store=vector_store,
        top_k=settings.retrieval_top_k,
    )
    logger.info("Application startup complete")
    yield


app = FastAPI(
    title="Google Drive RAG API",
    version="1.0.0",
    description="Ask questions over Google Drive documents with OpenAI and Chroma.",
    lifespan=lifespan,
)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

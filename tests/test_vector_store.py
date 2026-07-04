import numpy as np

from app.db.vector_store import ChromaVectorStore
from app.services.models import DocumentChunk


def make_chunk(chunk_id: str, text: str, file_name: str = "doc.txt") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        file_id=f"file-{chunk_id}",
        source_link=f"https://example.com/{chunk_id}",
        page_number=2,
        chunk_index=0,
        text=text,
        content_hash=f"hash-{chunk_id}",
    )


def test_vector_store_skips_duplicate_chunks(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path)
    store.load()
    chunk = make_chunk("chunk-1", "hello world")
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    assert store.add([chunk], embeddings) == 1
    assert store.add([chunk], embeddings) == 0
    assert store.existing_chunk_ids() == {"chunk-1"}

    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
    assert len(results) == 1
    assert results[0]["file_name"] == "doc.txt"
    assert results[0]["file_id"] == "file-chunk-1"
    assert results[0]["source_link"] == "https://example.com/chunk-1"
    assert results[0]["page_number"] == 2
    assert results[0]["text"] == "hello world"
    assert "distance" in results[0]
    assert "score" in results[0]


def test_vector_store_persists_chroma_collection(tmp_path) -> None:
    first = ChromaVectorStore(tmp_path)
    first.load()
    chunks = [
        make_chunk("chunk-1", "alpha source", file_name="alpha.txt"),
        make_chunk("chunk-2", "beta source", file_name="beta.txt"),
    ]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    assert first.add(chunks, embeddings) == 2
    first.save()

    second = ChromaVectorStore(tmp_path)
    second.load()

    assert second.existing_chunk_ids() == {"chunk-1", "chunk-2"}
    results = second.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0]["file_name"] == "beta.txt"
    assert results[0]["chunk_id"] == "chunk-2"

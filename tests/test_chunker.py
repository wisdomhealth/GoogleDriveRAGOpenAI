from app.services.chunker import TextChunker
from app.services.models import DocumentPage


def test_chunking_is_deterministic() -> None:
    text = " ".join(f"word-{index}" for index in range(220))
    page = DocumentPage(
        file_name="test.txt",
        file_id="file-1",
        source_link="https://example.com",
        page_number=None,
        text=text,
    )
    chunker = TextChunker(min_tokens=20, max_tokens=50, overlap_tokens=10)

    first = chunker.chunk_pages([page])
    second = chunker.chunk_pages([page])

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) > 1
    assert all(chunk.file_id == "file-1" for chunk in first)

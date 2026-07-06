from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

from docx import Document as DocxDocument
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

from app.services.models import DocumentPage
from app.utils.logger import get_logger
from app.utils.text_cleaner import clean_text


logger = get_logger(__name__)


SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}


@dataclass(frozen=True)
class DriveFile:
    """Google Drive file metadata needed for download and source citation."""

    file_id: str
    name: str
    mime_type: str
    web_view_link: str


class GoogleDriveLoader:
    """Read supported files from Google Drive and convert them to pages."""

    def __init__(self, drive_service: Any) -> None:
        """Use an already-authenticated Google Drive API service."""
        self._service = drive_service

    async def iter_folder_documents(self, folder_ids: Iterable[str]) -> AsyncIterator[DocumentPage]:
        """Yield extracted pages for every configured Drive folder."""
        for folder_id in folder_ids:
            async for page in self.iter_folder_document(folder_id):
                yield page

    async def iter_folder_document(self, folder_id: str) -> AsyncIterator[DocumentPage]:
        """Yield extracted pages from one Drive folder, skipping failed files."""
        files = await asyncio.to_thread(self._list_supported_files, folder_id)
        logger.info("Found %s supported files in Drive folder %s", len(files), folder_id)

        for drive_file in files:
            try:
                # Google client calls and file parsing are blocking, so run them
                # in threads to keep the async ingestion loop responsive.
                data = await asyncio.to_thread(self._download_file, drive_file.file_id)
                pages = await asyncio.to_thread(self._extract_pages, drive_file, data)
                for page in pages:
                    if page.text:
                        yield page
            except HttpError as exc:
                logger.exception("Google Drive API failed for %s (%s): %s", drive_file.name, drive_file.file_id, exc)
            except Exception as exc:
                logger.exception("Failed to process %s (%s): %s", drive_file.name, drive_file.file_id, exc)

    def _list_supported_files(self, folder_id: str) -> list[DriveFile]:
        """List non-trashed Drive files whose MIME type can be ingested."""
        mime_query = " or ".join(f"mimeType='{mime}'" for mime in SUPPORTED_MIME_TYPES)
        query = f"'{folder_id}' in parents and trashed=false and ({mime_query})"
        files: list[DriveFile] = []
        page_token: str | None = None

        while True:
            response = (
                self._service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, webViewLink)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for item in response.get("files", []):
                files.append(
                    DriveFile(
                        file_id=item["id"],
                        name=item["name"],
                        mime_type=item["mimeType"],
                        web_view_link=item.get("webViewLink", ""),
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                return files

    def _download_file(self, file_id: str) -> bytes:
        """Download a Drive file into memory as bytes."""
        request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request, chunksize=1024 * 1024)
        done = False
        while not done:
            # next_chunk advances the resumable download until Drive reports done.
            _, done = downloader.next_chunk()
        return buffer.getvalue()

    def _extract_pages(self, drive_file: DriveFile, data: bytes) -> list[DocumentPage]:
        """Dispatch file bytes to the extractor matching the Drive MIME type."""
        file_type = SUPPORTED_MIME_TYPES[drive_file.mime_type]
        if file_type == "pdf":
            return self._extract_pdf_pages(drive_file, data)
        if file_type == "docx":
            return self._extract_docx_text(drive_file, data)
        if file_type == "txt":
            return self._extract_txt_text(drive_file, data)
        raise ValueError(f"Unsupported Drive MIME type: {drive_file.mime_type}")

    def _extract_pdf_pages(self, drive_file: DriveFile, data: bytes) -> list[DocumentPage]:
        """Extract one DocumentPage per PDF page with readable text."""
        reader = PdfReader(io.BytesIO(data))
        pages: list[DocumentPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append(self._page(drive_file, text, page_number=index))
        return pages

    def _extract_docx_text(self, drive_file: DriveFile, data: bytes) -> list[DocumentPage]:
        """Extract DOCX paragraphs into one DocumentPage."""
        document = DocxDocument(io.BytesIO(data))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        text = clean_text("\n\n".join(paragraphs))
        return [self._page(drive_file, text, page_number=None)] if text else []

    def _extract_txt_text(self, drive_file: DriveFile, data: bytes) -> list[DocumentPage]:
        """Decode a UTF-8-ish text file into one DocumentPage."""
        text = data.decode("utf-8", errors="replace")
        cleaned = clean_text(text)
        return [self._page(drive_file, cleaned, page_number=None)] if cleaned else []

    @staticmethod
    def _page(drive_file: DriveFile, text: str, page_number: int | None) -> DocumentPage:
        """Build a normalized page object while preserving source metadata."""
        return DocumentPage(
            file_name=drive_file.name,
            file_id=drive_file.file_id,
            source_link=drive_file.web_view_link,
            page_number=page_number,
            text=text,
        )

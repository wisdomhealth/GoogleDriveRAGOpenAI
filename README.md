# GoogleDriveRAGOpenAI

FastAPI service for asking questions over documents stored in Google Drive. It ingests supported Drive files, chunks and embeds their text with OpenAI embeddings, stores vectors in a local Chroma database, and answers questions with source citations.

## Features

- Google Drive ingestion for PDF, DOCX, and plain text files
- OpenAI embeddings and chat completions
- Persistent local Chroma vector store
- Non-streaming and Server-Sent Events streaming chat endpoints
- Optional HTTP Basic authentication for API access
- Dockerfile and pytest test suite included

## Project Structure

```text
app/
  api/                 FastAPI routes and optional Basic Auth
  db/                  Chroma vector store wrapper
  services/            Drive loading, chunking, embeddings, LLM, RAG pipeline
  utils/               Logging and text cleaning helpers
scripts/
  ingest_drive.py      Google Drive ingestion entrypoint
tests/                 Unit tests
data/chroma/           Local Chroma persistence directory
```

## Requirements

- Python 3.11+
- OpenAI API key
- Google Cloud service account JSON file with Google Drive read access
- One or more Google Drive folder IDs shared with that service account

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file from the example and fill in your values:

```bash
cp .env.example .env
```

```bash
OPENAI_API_KEY=your-openai-api-key
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/google-service-account.json
GOOGLE_DRIVE_FOLDER_ID=drive-folder-id-1,drive-folder-id-2

# Optional model overrides
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

# Optional vector/chunking settings
VECTOR_STORE_DIR=data/chroma
CHUNK_MIN_TOKENS=1000
CHUNK_MAX_TOKENS=1500
CHUNK_OVERLAP_TOKENS=150
RETRIEVAL_TOP_K=5
EMBEDDING_BATCH_SIZE=64

# Optional API Basic Auth
API_BASIC_AUTH_USERNAME=
API_BASIC_AUTH_PASSWORD=
```

Notes:

- `GOOGLE_DRIVE_FOLDER_ID` accepts a comma-separated list.
- The service account must be able to read the Drive folders and files.
- Basic Auth is disabled unless both `API_BASIC_AUTH_USERNAME` and `API_BASIC_AUTH_PASSWORD` are set.
- `.env` and generated Chroma data are ignored by git.

## Ingest Google Drive Documents

Run ingestion before asking questions:

```bash
python scripts/ingest_drive.py
```

The script:

1. Lists supported files from the configured Drive folders.
2. Extracts text from PDF, DOCX, and TXT files.
3. Splits text into overlapping chunks.
4. Embeds new chunks with OpenAI.
5. Persists them in Chroma under `VECTOR_STORE_DIR`.

Existing chunks are skipped by chunk ID, so repeated runs only add new content.

## Run the API

```bash
uvicorn app.main:app --reload
```

Default local URL:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Chat API

### Non-streaming

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say about the onboarding process?"}'
```

Response shape:

```json
{
  "answer": "Answer text with source file names.",
  "sources": [
    {
      "file_name": "example.pdf",
      "file_id": "google-drive-file-id",
      "snippet": "Retrieved source snippet...",
      "source_link": "https://drive.google.com/...",
      "page_number": 1
    }
  ]
}
```

### Streaming

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the latest policy document."}'
```

The stream uses Server-Sent Events and emits JSON token payloads:

```text
data: {"token":"..."}

data: [DONE]
```

### With Basic Auth

If Basic Auth is enabled:

```bash
curl -u "$API_BASIC_AUTH_USERNAME:$API_BASIC_AUTH_PASSWORD" \
  -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What files mention refunds?"}'
```

## Docker

Build the image:

```bash
docker build -t google-drive-rag-openai .
```

Run the API:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$PWD/data:/app/data" \
  -v "/absolute/path/to/google-service-account.json:/credentials/service-account.json:ro" \
  google-drive-rag-openai
```

If the credentials path inside `.env` points to the host file, update it for the container mount, for example:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/credentials/service-account.json
```

To ingest inside Docker, override the command:

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD/data:/app/data" \
  -v "/absolute/path/to/google-service-account.json:/credentials/service-account.json:ro" \
  google-drive-rag-openai \
  python scripts/ingest_drive.py
```

## Tests

```bash
pytest
```

The current tests cover chunking, text cleaning, and vector store behavior.

## Troubleshooting

- `OPENAI_API_KEY is required`: set `OPENAI_API_KEY` in `.env` or the process environment.
- `GOOGLE_DRIVE_FOLDER_ID is required`: set one or more folder IDs before running ingestion.
- `GOOGLE_APPLICATION_CREDENTIALS is required for Drive ingestion`: provide an absolute path to the service account JSON file.
- `Vector store is empty`: run `python scripts/ingest_drive.py` before calling `/chat`.
- No files found during ingestion: confirm the Drive folder is shared with the service account and contains PDF, DOCX, or TXT files.

# local-rag-ai-agent

A local Retrieval-Augmented Generation (RAG) app built with **FastAPI**, **Inngest**, **Qdrant**, **Ollama**, and **LlamaIndex**.

This project ingests PDF files, splits them into chunks, generates embeddings locally with Ollama, stores them in Qdrant, and answers questions using retrieved context plus a local chat model.

## Features

- PDF ingestion with `llama-index`
- Chunking with sentence-aware splitting
- Local embeddings with Ollama
- Vector search with Qdrant
- Answer generation with a local Ollama chat model
- Inngest functions for ingestion and search workflows
- FastAPI app for serving the Inngest endpoint

## Architecture

```text
PDF -> chunking -> embeddings -> Qdrant
                           |
Question -> embedding -> vector search -> retrieved context -> Ollama -> answer
```

## Project structure

```text
.
├── custom_types.py       # Pydantic models for workflow inputs/outputs
├── data_loader.py        # PDF loading, chunking, and embedding
├── main.py               # Inngest functions + FastAPI app
├── ollama_adapter.py     # Local LLM adapter for answer generation
├── vector_db.py          # Qdrant storage wrapper
├── pyproject.toml
└── README.md
```

## Tech stack

- **FastAPI** for serving the app
- **Inngest** for workflow orchestration
- **Qdrant** for vector storage
- **Ollama** for local embeddings and generation
- **LlamaIndex** for PDF reading and chunking

## Current defaults

- **Embedding model:** `qwen3-embedding`
- **Embedding dimension:** `3072`
- **Chat model:** `qwen3.5`
- **Qdrant URL:** `http://localhost:6333`
- **Ollama URL:** `http://localhost:11434`
- **Collection name:** `docs`

## Requirements

- Python 3.14+
- Docker
- Ollama installed and running
- Qdrant running locally

## Installation

Clone the repository:

```bash
git clone https://github.com/GuilhermeCarvalho1144/local-rag-ai-agent.git
cd local-rag-ai-agent
```

Create the environment and install dependencies with `uv`:

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Start dependencies

### 1) Start Qdrant

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

Health check:

```bash
curl http://localhost:6333/healthz
```

### 2) Start Ollama

Make sure the Ollama server is running:

```bash
ollama serve
```

Pull the models used by the project:

```bash
ollama pull qwen3-embedding
ollama pull qwen3.5
```

## Environment variables

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=qwen3.5
```

> For local Ollama, the API key is not a real secret. The value `ollama` is just a placeholder used by the client code.

## Run the app

Start the FastAPI server:

```bash
uv run uvicorn main:app --reload
```

## How it works

### Ingest flow

The `rag/ingest_pdf` workflow:

1. Loads a PDF from disk
2. Extracts text pages
3. Splits text into chunks
4. Embeds the chunks with Ollama
5. Stores vectors + payloads in Qdrant

### Search flow

The `rag/search_pdf` workflow:

1. Embeds the question
2. Searches the most relevant chunks in Qdrant
3. Sends retrieved context to the local chat model
4. Returns an answer plus sources

## Example event payloads

### Ingest a PDF

```json
{
  "name": "rag/ingest_pdf",
  "data": {
    "pdf_path": "/absolute/path/to/file.pdf",
    "source_id": "my-paper"
  }
}
```

### Search a PDF

```json
{
  "name": "rag/search_pdf",
  "data": {
    "question": "What is the main contribution of the paper?",
    "top_k": 5
  }
}
```

## Data model

### Stored payload in Qdrant

Each chunk is stored with:

```json
{
  "source": "source_id",
  "text": "chunk text"
}
```

### Response types

- `RAGUpsertResult`
  - `ingested: int`

- `RAGSearchResult`
  - `context: list[str]`
  - `sources: list[str]`

- `RAGQueryResult`
  - `answer: str`
  - `sources: list[str]`
  - `num_contexts: int`

## Notes

- The project currently uses **3072-dimensional embeddings**, so your Qdrant collection must be created with the same dimension.
- If you change embedding models or dimensions, recreate the collection or use a different collection name.
- This app is designed for **local-first RAG**, so both the vector database and language model run on your machine.

## Troubleshooting

### Qdrant connection refused

Check that the container is running and mapped correctly:

```bash
docker ps
curl http://localhost:6333/healthz
```

### Vector dimension error

If Qdrant says the expected dimension does not match the embedding dimension, delete and recreate the collection with the correct size.

### Ollama model not found

Pull the required model first:

```bash
ollama pull qwen3-embedding
ollama pull qwen3.5
```

### No answer or bad answer quality

- Increase `top_k`
- Improve chunk size / overlap
- Try a stronger chat model
- Confirm the PDF text extraction is working

## Roadmap ideas

- Add a Streamlit or web UI
- Support multiple collections
- Add metadata filters
- Add citations per chunk
- Add document deletion / reindexing
- Add support for Markdown, TXT, and DOCX files
- Add evaluation and retrieval metrics

## License

This project is for educational and experimentation purposes.

## References

- [Tech With Tim video](https://www.youtube.com/watch?v=AUQJ9eeP-Ls&t=2538s)

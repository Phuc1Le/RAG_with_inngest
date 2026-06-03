# RAG Production App

A small retrieval-augmented generation app for local PDFs.

It ingests PDFs, chunks them, embeds text with `SentenceTransformer`, stores vectors in Qdrant, and answers questions using Inngest workflows.

## What this project contains

- `main.py` — FastAPI app + Inngest functions
- `data_loader.py` — PDF loading, chunking, and embedding
- `vector_db.py` — Qdrant storage and search logic
- `custom_type.py` — response data models
- `pyproject.toml` — Python dependencies and project configuration

## Prerequisites

- Python 3.14
- Docker (for local Qdrant)
- Windows PowerShell or similar terminal

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies from `pyproject.toml`:

```powershell
d:/Projects/RAGProductionApp/.venv/Scripts/python.exe -m pip install -e .
```

3. Install `sentence-transformers` if it is not already installed:

```powershell
d:/Projects/RAGProductionApp/.venv/Scripts/python.exe -m pip install sentence-transformers
```

4. Create a `.env` file in the project root and add your OpenAI API key (required for the current query inference step):

```env
OPENAI_API_KEY=your_openai_api_key
```

## Start Qdrant

Run Qdrant locally with Docker:

```powershell
docker run -d --name qdrant -v qdrant_storage:/qdrant/storage -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.2.0
```

This app connects to `http://localhost:6333` by default.

## Run the app

Start the FastAPI server:

```powershell
d:/Projects/RAGProductionApp/.venv/Scripts/python.exe -m uvicorn main:app --reload
```

## How it works

1. `rag/ingest_pdf` loads a PDF, splits text into chunks, embeds each chunk, and stores vectors in Qdrant.
2. `rag/query_pdf_ai` embeds the question, searches Qdrant, and then uses the retrieved context to generate an answer.

## Notes

- Embeddings are produced with `SentenceTransformer("all-MiniLM-L6-v2")` and stored in a 384-dimensional Qdrant collection.
- The current query answer step still uses OpenAI via `ai.openai.Adapter` in `main.py`.
- If you want to use a free local LLM for inference, replace that adapter call with a local model integration.

## Useful commands

```powershell
# Activate the venv
.\.venv\Scripts\Activate.ps1

# Start Qdrant
docker start qdrant

# Stop Qdrant
docker stop qdrant

# Run the server
python -m uvicorn main:app --reload
```

## Project structure

```text
RAGProductionApp/
├── custom_type.py
├── data_loader.py
├── main.py
├── pyproject.toml
├── README.md
├── vector_db.py
└── qdrant_storage/  # local Qdrant data files
```

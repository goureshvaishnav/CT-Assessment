# Local RAG Chatbot

Production-style, fully local RAG chatbot for PDF documents built with
Streamlit + LangChain + Ollama + Chroma.

No cloud APIs, no paid inference, and no data leaves your machine.

## Why this project

This project is designed for demos, ML labs, and portfolio use where you
need:

- reliable multi-document ingestion,
- grounded answers with citations,
- conversational follow-ups with memory,
- and a clean UI that is simple to run locally.

## Feature Highlights

- **Document ingestion**
  Upload one or more PDFs, auto-split text, embed chunks, and index in Chroma.
- **RAG retrieval**
  Retrieves top relevant chunks and injects them into the LLM prompt.
- **Conversation memory**
  Maintains recent turns to handle follow-up questions.
- **Persistent vector store**
  Chroma data is saved locally and survives app restarts.
- **Source-aware responses**
  Every answer can show source file, page number, score, and snippet.
- **Operational checks**
  In-app health panel for Ollama, model availability, and indexed chunk count.

## Tech Stack

| Layer | Choice |
|---|---|
| LLM | `mistral` via Ollama |
| Embeddings | `nomic-embed-text` via Ollama |
| Orchestration | LangChain |
| Vector DB | Chroma (persistent local storage) |
| UI | Streamlit |
| PDF parsing | PyPDF (`PyPDFLoader`) |

## System Architecture

```text
PDF Upload
  -> PDF Loader (PyPDFLoader)
  -> Recursive Text Splitter
  -> Embeddings (nomic-embed-text)
  -> Chroma Persistent Collection

Question
  -> Similarity Retrieval (top-k)
  -> Prompt Builder (context + recent history)
  -> LLM (mistral)
  -> Answer + Source Citations
```

## Quick Start

### 1) Install Ollama

- macOS / Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- Windows: install from
  [https://ollama.com/download](https://ollama.com/download)

### 2) Pull required local models

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 3) Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Start Ollama server

```bash
ollama serve
```

### 5) Start app

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

## Usage Flow

1. Open app and verify health checks in sidebar.
2. Upload one or more PDFs from **Document Ingestion**.
3. Wait for indexing summary (`indexed`, `skipped duplicates`, `errors`).
4. Ask questions in the chat box.
5. Expand **Sources** under answers to inspect citations.
6. Use **Reset DB** only when you want a full clean reindex.

## Configuration

All runtime knobs are environment-variable based.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `LLM_MODEL` | `mistral` | Generation model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local vector DB path |
| `CHROMA_COLLECTION` | `rag_docs` | Chroma collection name |
| `CHUNK_SIZE` | `800` | Chunk size for splitting |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `TOP_K_DOCS` | `5` | Retrieval top-k |
| `MAX_HISTORY` | `8` | Memory turns to include |
| `MIN_RELEVANCE_SCORE` | `0.0` | Optional retrieval score cutoff |

## GitHub-Ready Checklist

- [x] Clear project structure and single-command run flow
- [x] Source-controlled dependencies (`requirements.txt`)
- [x] Environment-driven runtime config (no hard-coded secrets)
- [x] Persistent storage isolated to `chroma_db/`
- [x] Error handling for missing Ollama/models and query failures
- [x] Documentation for setup, usage, and troubleshooting

## Production-Readiness Notes (Local-First)

This app is robust for local usage and demos:

- duplicate-safe ingestion to prevent repeated indexing noise,
- stable fallback messages for common runtime failures,
- context-grounded outputs with explicit citations,
- reset and health controls for easier operations.

For internet-facing deployment, add auth, observability, and rate limiting.

## Troubleshooting

- **No upload option visible**
  Expand the left sidebar from the top-left toggle.
- **Ollama offline**
  Run `ollama serve` and refresh.
- **Model missing**
  Run `ollama pull mistral` and/or
  `ollama pull nomic-embed-text`.
- **No answers from docs**
  Confirm chunks are indexed (`doc_chunks > 0`) and ask doc-specific queries.
- **Slow responses**
  Use fewer/lighter PDFs, reduce `TOP_K_DOCS`, or use a lighter model.
- **Need full reset**
  Use the **Reset DB** button in sidebar.

## Repository Structure

```text
.
├── app.py
├── rag_engine.py
├── requirements.txt
├── README.md
├── config.toml
└── chroma_db/
```

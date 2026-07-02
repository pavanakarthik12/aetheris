# Aetheris

Aetheris is a modular AI system foundation. This repository intentionally contains only the platform architecture needed for future cognition, memory, planning, personality, emotion, and knowledge modules.

## Architecture Overview

- `backend/`: FastAPI service boundary, centralized settings, logging, database wiring, and placeholder routers.
- `frontend/`: Next.js App Router scaffold with placeholder pages for the main application surfaces.
- `database/`: Reserved storage area for PostgreSQL-related local files and ChromaDB persistence.
- `llm/`, `embeddings/`, `memory/`, `planner/`, `reflection/`, `personality/`, `emotion/`, `knowledge_graph/`, `tools/`: Reserved module boundaries for future capabilities.

## Setup

1. Copy `.env.example` to `.env` and fill in the required values.
2. Install backend dependencies with `pip install -r requirements.txt`.
3. Install frontend dependencies from `frontend/` with `npm install`.
4. Start PostgreSQL, backend, and frontend with Docker or run the services locally.

## Project Tree

```text
Aetheris/
  backend/
    app/
  frontend/
    app/
  docs/
  database/
  memory/
  personality/
  emotion/
  planner/
  reflection/
  tools/
  knowledge_graph/
  embeddings/
  llm/
  config/
  docker/
  tests/
  scripts/
```

## Notes

- Qwen access is isolated behind the backend LLM service boundary.
- The embedding boundary is prepared for `BAAI/bge-base-en-v1.5`.
- ChromaDB persistence is configured for `database/chroma`.
- No chatbot, memory, planning, personality, or emotion logic is implemented here.
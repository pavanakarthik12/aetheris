# Aetheris

A modular AI system foundation with a FastAPI backend and Next.js frontend. Aetheris is designed for persistent memory, cognitive reasoning, and structured conversation.

## Current Phase — Phase 10: Cognitive Reasoning Engine

Aetheris is currently in **Phase 10**. The Cognitive Reasoning Engine has been implemented as a lightweight deterministic layer that analyzes every request *before* the LLM is invoked. It determines intent, complexity, required memory sources, and whether planning or clarification is needed — all without making additional LLM calls.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **Frontend** | Next.js 14+ (App Router), TypeScript |
| **LLM Provider** | Groq API (llama-3.3-70b-versatile) |
| **Embeddings** | Sentence-Transformers (BAAI/bge-base-en-v1.5) |
| **Vector Store** | ChromaDB |
| **Database** | PostgreSQL 16 (via SQLAlchemy) / SQLite (dev) |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker / Docker Compose |

---

## Capabilities

### Cognitive Reasoning Engine (Phase 10)
- **Intent Analysis** — Classifies requests into 14 semantic intent types (Mathematics, Programming, Memory Retrieval, Planning, Creative Writing, Debugging, etc.) using lightweight deterministic rules. No LLM calls for classification.
- **Complexity Classification** — Tags requests as Simple, Medium, or Complex. Simple requests bypass unnecessary stages.
- **Task Decomposition** — Splits multi-objective requests into independent internal tasks.
- **Decision Engine** — Determines which memory sources (conversation, long-term, system) are needed, whether clarification is required, and whether planning is necessary.
- **Clarification Engine** — Asks concise questions when information is missing instead of guessing.
- **Planning Engine** — Creates internal execution plans for medium and complex requests. Plans are never exposed to the user.
- **Response Verification** — Validates LLM responses before delivery (checks for empty responses, yes/no question compliance, etc.).
- **Confidence Estimation** — Internal confidence scoring; low-confidence responses prefer clarification.
- **Cognitive Trace** — Full reasoning trace visible in debug mode only. Includes intent, complexity, tasks, memory decisions, verification result, confidence, and processing time.

### Memory System (Phases 3–6)
- **Three-Layer Memory Hierarchy** — Conversation (session-scoped), Long-Term (persistent user facts), System (read-only identity).
- **Immediate Memory Processor** — Automatically evaluates and stores meaningful user statements as memories.
- **Memory Evolution** — Creates, updates, merges, and archives memories. Conflict resolution for contradicting facts.
- **Memory Evaluation** — Scores each statement for storage worthiness before persisting.
- **Semantic Search** — ChromaDB-powered vector similarity search across all long-term memories.
- **Memory Deduplication & Filtering** — Relevance filtering, deduplication, and archived-memory exclusion.

### Reflection Engine (Phase 7)
- Analyzes conversation exchanges in the background.
- Strengthens memories when new information reinforces existing facts.
- Detects contradictions and schedules conflict resolution.
- Tracks quality metrics and processing times.

### Request Routing (Phase 9)
- **Cognitive Request Router** — Central dispatch that classifies intent, routes to the correct subsystem, manages memory operations, builds context, and invokes the LLM.
- **Intent Classification** — Combines rule-based (regex) and LLM-based classification with confidence thresholds.
- **Token Budget** — Dynamically selects token limits and temperature based on intent and memory count.
- **Error Handling** — Graceful degradation for quota limits, rate limits, timeouts, and provider errors.

### Context & Prompting
- **Context Builder** — Builds memory context from retrieved memories, filtered by relevance and deduplicated.
- **Prompt Builder** — Centralized prompt templates with identity block, memory blocks, and conversation history.
- **Memory Cache** — Caches memory search results to reduce latency for repeated queries.

### API Endpoints
- `POST /api/chat` — Main chat with full reasoning pipeline
- `POST /api/session/reset` — Clear conversation memory
- `GET /health` — Service health check
- Memory CRUD endpoints
- Debug endpoints for context and hierarchy inspection

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- (Optional) Docker & Docker Compose

### Backend

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY and SECRET_KEY
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker-compose up
```

---

## Testing

```bash
pytest tests/ -v
```

Currently **151 tests** covering the request router, intent classifier, memory hierarchy, memory evolution, reflection engine, immediate memory processor, memory resolver, and the full cognitive reasoning engine.

---

## Architecture Overview

```
backend/
  app/
    main.py                    # FastAPI entrypoint
    dependencies.py            # Dependency injection wiring
    routers/
      chat.py                  # Chat API endpoint
      memory.py                # Memory CRUD
      memory_evolution.py      # Evolution management
      memory_hierarchy_debug.py
      reflection.py            # Reflection debug endpoints
      health.py                # Health check
      system.py                # System queries
      context_debug.py         # Context builder debug
    schemas/
      chat.py                  # Chat request/response models
      routing.py               # Router schemas + CognitiveTrace
      reasoning.py             # Reasoning engine schemas
      memory.py, reflection.py, evolution.py, common.py
    services/
      reasoning/               # Phase 10 — Cognitive Reasoning Engine
        pipeline.py            # Orchestrator
        intent_analyzer.py     # Module 1
        complexity_classifier.py # Module 2
        task_decomposer.py     # Module 3
        decision_engine.py     # Module 4 + 5 (Clarification)
        planning_engine.py     # Module 6
        response_verifier.py   # Module 7
        confidence_estimator.py # Module 8
        cognitive_trace.py     # Module 10
      request_router.py        # Central execution engine
      context_builder.py       # Memory context assembly
      prompt_builder.py        # Prompt templates
      intent_classifier.py     # Rule + LLM intent classification
      memory_service.py        # Long-term memory
      memory_hierarchy_service.py # 3-layer memory resolver
      memory_evolution_service.py # Memory CRUD with evolution
      memory_evaluator.py      # Storage-worthiness scoring
      memory_cache.py          # Search result cache
      chroma_service.py        # ChromaDB wrapper
      embedding_service.py     # Sentence embedding
      conversation_memory.py   # Session-scoped memory
      system_memory.py         # Read-only identity facts
      llm_service.py           # LLM provider abstraction
      reflection_service.py    # Background reflection
      immediate_memory_processor.py # Auto-save pipeline
      token_budget.py          # Dynamic token selection
      circuit_breaker.py       # Provider protection
      providers/               # Groq, OpenRouter adapters
frontend/
  app/
    chat/                      # Chat page
    page.tsx                   # Redirect to /chat
  components/
    chat/
      ChatWindow.tsx           # Chat UI component
  types/
    chat.ts                    # Frontend chat types
tests/
  test_reasoning_engine.py     # 42 tests — Phase 10
  test_request_router.py       # 14 tests
  test_intent_classifier.py    # 15 tests
  test_immediate_memory_processor.py
  test_memory_phase3.py        # 5 tests
  test_memory_phase5.py        # 2 tests
  test_memory_resolver.py      # 27 tests
  test_reflection_phase7.py    # 12 tests
```

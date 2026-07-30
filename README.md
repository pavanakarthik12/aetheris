# Aetheris

A modular AI system foundation with a FastAPI backend and Next.js frontend. Aetheris is designed for persistent memory, cognitive reasoning, context-aware conversation, and structured response generation.

**Current Phase — Phase 12.2: Conversation Intelligence & Follow-up Resolution**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **Frontend** | Next.js 15 (App Router), React 19, TypeScript |
| **LLM Provider** | Groq API (llama-3.3-70b-versatile) |
| **Embeddings** | Sentence-Transformers (BAAI/bge-base-en-v1.5) |
| **Vector Store** | ChromaDB |
| **Database** | PostgreSQL 16 (via SQLAlchemy) / SQLite (dev) |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker / Docker Compose |

---

## How Each Response Is Generated — The Request Flow

```
User Message
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Greeting Detection              │  detect_greeting() — bypasses LLM for
│    (rule-based, no LLM)            │  "hello", "hi", "hey", etc.
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Intent Classification           │  IntentClassifier — rule-based regex
│    (rule + optional LLM fallback)  │  for high-confidence intents (CREATE,
│                                     │  DELETE, UPDATE, SEARCH, WEB_SEARCH,
│                                     │  SYSTEM_QUERY), falls back to LLM
│                                     │  classification below threshold.
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Cognitive Reasoning Engine      │  ReasoningPipeline — deterministic
│    (zero LLM calls)                │  pre-processing:
│                                     │    • IntentAnalyzer (14 intent types)
│                                     │    • ComplexityClassifier (Simple/
│                                     │      Medium/Complex)
│                                     │    • TaskDecomposer (multi-objective)
│                                     │    • DecisionEngine (memory sources,
│                                     │      clarification, planning needed)
│                                     │    • PlanningEngine (execution plans)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Context Relevance Engine        │  ContextRelevanceEngine — scores
│    (Phase 12.1)                    │  each memory/conversation/search
│                                     │  result by relevance using:
│                                     │    • Chroma vector similarity (45%)
│                                     │    • Keyword overlap (20%)
│                                     │    • Topic alignment (20%)
│                                     │    • Recency (5%)
│                                     │    • Importance (10%)
│                                     │  Filters below configurable thresholds.
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Conversation Intelligence       │  ConversationIntelligenceEngine —
│    (Phase 12.2)                    │  detects follow-up references:
│                                     │    • Phrase matching (elaborate,
│                                     │      continue, why, compare, etc.)
│                                     │    • Topic resolution per intent
│                                     │    • Confidence scoring via Jaccard
│                                     │    • Keyword overlap analysis
│                                     │    • Clarification questions when
│                                     │      ambiguous (< threshold)
│                                     │  Injects: previous Q&A + resolved
│                                     │  topic into message context.
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. Memory Retrieval & Hierarchy    │  MemoryHierarchyService resolves
│    (3-layer)                       │  context from three tiers:
│                                     │    1. Conversation (session-scoped)
│                                     │    2. Long-Term (persistent, ChromaDB)
│                                     │    3. System (read-only identity)
│                                     │  + Relevance filtering per layer.
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. External Knowledge Integration  │  ExternalKnowledgeIntegrationService:
│    (when triggered)                │    • DecisionLayer checks keywords
│                                     │      (news, weather, latest, stock)
│                                     │    • SearchPipeline → Tavily API
│                                     │    • ContextIntelligenceEngine
│                                     │      cleans & deduplicates results
│                                     │    • ContextFormatter builds block
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 8. Context Building & Prompting    │  ContextBuilderService + PromptBuilder
│                                     │  assemble final prompt with:
│                                     │    • System identity block
│                                     │    • Conversation history
│                                     │    • Long-term memories
│                                     │    • External knowledge (if any)
│                                     │    • Token budget optimization
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 9. Reflection (post-response)      │  ReflectionService analyzes the
│    (background, async)             │  exchange asynchronously to:
│                                     │    • Strengthen confirmed memories
│                                     │    • Detect contradictions
│                                     │    • Track quality metrics
│                                     │  Never blocks the response.
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 10. Response Assembly & Quality    │  ResponseAssembler:
│    Check                           │    • Quality check (length,
│                                     │      false ignorance detection,
│                                     │      code completeness, URL check)
│                                     │    • Source citation injection
│                                     │    • Fallback on low quality
└─────────────────────────────────────┘
    │
    ▼
       Response to User
```

---

## Complete Feature List

### Phase 1–2: Foundation
- FastAPI application scaffold with middleware stack
- Request ID middleware (X-Request-ID header propagation)
- Global error handlers (HTTP, validation, unhandled)
- CORS configuration
- Health check endpoint
- Environment-based configuration via `.env`

### Phase 3: Memory System (CRUD)
- **MemoryService** — CRUD operations on persistent long-term memories stored in ChromaDB vector store
- **ConversationMemory** — Session-scoped in-memory store (get_recent, search, append, clear)
- **SystemMemory** — Read-only identity facts the system knows about itself
- **EmbeddingService** — Sentence embedding using Sentence-Transformers (BAAI/bge-base-en-v1.5)
- **ChromaService** — Wrapper around ChromaDB with collection management and similarity search
- Memory API endpoints (CRUD + semantic search)

### Phase 4: Prompt Engineering
- **PromptBuilder** — Centralized prompt template system
  - Identity block (who Aetheris is)
  - Memory block (relevant long-term facts)
  - Conversation block (recent exchange history)
  - External knowledge block (web search results)
  - Structured instruction formatting
- **ContextBuilder** — Builds memory context from retrieved memories, deduplication, relevance filtering
- **MemorySearchCache** — LRU cache for repeated memory search queries
- **TokenBudget** — Dynamic token limits per intent type (512 for greetings, 1024 for Q&A, 2048 for code/complex)

### Phase 5: Memory Evolution
- **MemoryEvolutionService** — Advanced memory lifecycle management:
  - Create, update, merge, and archive memories
  - Conflict resolution (e.g., changing a favorite food from "Pizza" to "Pasta" archives the old fact)
  - Automatic deduplication
- **MemoryEvaluator** — Scores statements for storage worthiness before persisting
- **ImmediateMemoryProcessor** — Auto-evaluates every user message for meaningful facts and saves them

### Phase 6: Intent Classification
- **IntentClassifier** — Hybrid classifier:
  - Rule-based (regex patterns) for high-confidence intents: CREATE_MEMORY, DELETE_MEMORY, UPDATE_MEMORY, MERGE_MEMORY, SEARCH_MEMORY, SYSTEM_QUERY, WEB_SEARCH, CONVERSATION_QUERY, NORMAL_CHAT
  - LLM-based fallback when rule confidence is low
  - Multi-intent support (a message can trigger both search and chat)

### Phase 7: Reflection Engine
- **ReflectionService** — Post-response analysis:
  - Strengthens memory when new info reinforces existing facts
  - Detects contradictions and schedules archiving
  - Tracks assistant response quality metrics
  - Persists reflection records to disk (JSON)
  - 11+ test scenarios covering corrections, achievements, contradictions, low-confidence deferral

### Phase 8: External Knowledge Integration
- **ExternalKnowledgeDecisionLayer** — Determines if a message needs web search (keyword-based)
- **SearchPipeline** — Provider-agnostic search with Tavily integration
- **ContextIntelligenceEngine** — Cleans, deduplicates, scores, and ranks search results
- **ExternalKnowledgeContextFormatter** — Formats results into structured prompt blocks
- **ExternalKnowledgeIntegrationService** — Orchestrates the full pipeline: decide → search → clean → format → inject

### Phase 9: Cognitive Request Router
- **CognitiveRequestRouter** — Central execution engine that:
  - Coordinates all subsystems
  - Routes requests by intent to the correct handler
  - Manages memory operations before/during/after processing
  - Builds context from all memory sources
  - Invokes LLM with assembled prompt
  - Assembles final response with quality checks
- **ResponseAssembler** — Quality assurance layer:
  - Empty response detection
  - False ignorance detection ("I don't know" for clearly answerable questions)
  - Code block verification for programming queries
  - Source citation management
  - Quality scoring (0.0–1.0)
- **CircuitBreaker** — Protects against provider failures
- **MetricsCollector** — Tracks processing times per subsystem
- **RouterDebugifier** — Full debug trace of every step

### Phase 10: Cognitive Reasoning Engine
- **IntentAnalyzer** — Classifies requests into 14 semantic intent types (Mathematics, Programming, Memory Retrieval, Planning, Creative Writing, Debugging, etc.) using deterministic rules — **zero LLM calls**
- **ComplexityClassifier** — Tags as Simple, Medium, or Complex; simple requests bypass unnecessary stages
- **TaskDecomposer** — Splits multi-objective requests into independent internal tasks
- **DecisionEngine** — Determines which memory sources are needed (conversation, long-term, reflection, external knowledge), whether clarification is needed, and whether planning is required
- **ClarificationEngine** — Built into DecisionEngine; asks questions when information is missing
- **PlanningEngine** — Creates internal execution plans for Medium/Complex requests
- **ResponseVerifier** — Validates LLM response quality (empty check, yes/no compliance)
- **ConfidenceEstimator** — Scores confidence (High/Medium/Low) per intent type
- **CognitiveTrace** — Full reasoning trace (intent, complexity, tasks, memory decisions, verification, confidence, timing)

### Phase 11: Memory Relevance & Context Filtering
- **ConversationContextFilter** — Classifies query types and filters conversation history by relevance to the current message
- Hidden as a dependency within the memory hierarchy

### Phase 12.1: Context Relevance Engine
- **ContextRelevanceEngine** — Multi-factor relevance scoring for all memory sources:
  - Chroma vector similarity (weight: 0.45)
  - Keyword overlap (weight: 0.20)
  - Topic alignment (weight: 0.20)
  - Recency (weight: 0.05)
  - Importance/memory strength (weight: 0.10)
  - Existing ChromaDB score (weight: 0.50, used during re-scoring)
- Configurable thresholds per source type (memory, conversation, search)
- Max entries limits per source type

### Phase 12.2: Conversation Intelligence & Follow-up Resolution
- **ConversationIntelligenceEngine** — Detects and resolves conversational references:
  - **FollowUpIntent** enum: NEW_TOPIC, CONTINUATION, CLARIFICATION, COMPARISON, FOLLOW_UP, CORRECTION, REFINEMENT, REFERENCE
  - **Phrase Detection**: 18+ regex patterns for follow-up phrases (elaborate, continue, why, compare, simplify, etc.)
  - **Topic Resolution**: Per-intent logic to identify the referenced subject
  - **Confidence Scoring**: Combined Jaccard similarity + overlap ratio with stop-word removal
  - **Clarification Questions**: Auto-generated when confidence < threshold and multiple topics exist
  - **Context Injection**: Builds a context block with previous Q&A and resolved topic
  - **Integration Point**: Runs between the Reasoning Engine and Memory Retrieval in the request router
- Config: `FOLLOWUP_CONFIDENCE_THRESHOLD=0.75`, `MAX_CONTEXT_MESSAGES=4`, `MAX_REFERENCE_DISTANCE=6`

---

## Project Structure

```
aetheris/
├── backend/
│   └── app/
│       ├── main.py                    # FastAPI entrypoint, middleware/route registration
│       ├── dependencies.py            # Dependency injection (singletons per service)
│       ├── config/
│       │   └── settings.py            # Frozen dataclass settings from .env
│       ├── routers/
│       │   ├── chat.py                # POST /api/chat — main endpoint
│       │   ├── memory.py              # Memory CRUD endpoints
│       │   ├── memory_evolution.py    # Evolution management endpoints
│       │   ├── reflection.py          # Reflection debug endpoints
│       │   ├── health.py              # GET /health
│       │   ├── system.py              # System capability queries
│       │   ├── context_debug.py       # Context builder inspection
│       │   └── memory_hierarchy_debug.py
│       ├── schemas/
│       │   ├── chat.py                # ChatRequest / ChatResponse
│       │   ├── routing.py             # RouterResult, IntentClassification, RouteStep
│       │   ├── reasoning.py           # ReasoningPlan, CognitiveTrace, SemanticIntentType
│       │   ├── memory.py              # Memory schemas
│       │   ├── reflection.py          # Reflection schemas
│       │   ├── evolution.py           # Memory evolution schemas
│       │   ├── context_intelligence.py # ScoredResultItem, StructuredContext
│       │   └── common.py              # Shared types
│       ├── services/
│       │   ├── request_router.py       # Central execution engine (1000+ lines)
│       │   ├── conversation_intelligence_engine.py  # Phase 12.2
│       │   ├── context_relevance_engine.py           # Phase 12.1
│       │   ├── conversation_context_filter.py        # Phase 11
│       │   ├── context_builder.py      # Memory context assembly
│       │   ├── prompt_builder.py       # Prompt templates
│       │   ├── intent_classifier.py    # Rule + LLM classification
│       │   ├── memory_service.py       # Long-term memory CRUD
│       │   ├── memory_hierarchy_service.py  # 3-layer resolver
│       │   ├── memory_evolution_service.py  # Evolution + conflict resolution
│       │   ├── memory_evaluator.py     # Storage-worthiness scoring
│       │   ├── memory_cache.py         # Search result LRU cache
│       │   ├── chroma_service.py       # ChromaDB wrapper
│       │   ├── embedding_service.py    # Sentence embedding
│       │   ├── conversation_memory.py  # Session-scoped
│       │   ├── system_memory.py        # Read-only identity
│       │   ├── llm_service.py          # LLM provider abstraction
│       │   ├── reflection_service.py   # Background reflection
│       │   ├── immediate_memory_processor.py  # Auto-save pipeline
│       │   ├── token_budget.py         # Dynamic token selection
│       │   ├── circuit_breaker.py      # Provider protection
│       │   ├── response_assembler.py   # Quality + sources
│       │   ├── reasoning/              # Phase 10 engine
│       │   │   ├── pipeline.py         # Orchestrator
│       │   │   ├── intent_analyzer.py
│       │   │   ├── complexity_classifier.py
│       │   │   ├── task_decomposer.py
│       │   │   ├── decision_engine.py
│       │   │   ├── planning_engine.py
│       │   │   ├── response_verifier.py
│       │   │   ├── confidence_estimator.py
│       │   │   └── cognitive_trace.py
│       │   ├── external_knowledge/
│       │   │   ├── integration_service.py  # Orchestrator
│       │   │   ├── decision_layer.py       # Trigger decisions
│       │   │   ├── search_pipeline.py      # Search abstraction
│       │   │   ├── base_provider.py        # Provider interface
│       │   │   ├── tavily_provider.py      # Tavily implementation
│       │   │   ├── provider_manager.py     # Provider registry
│       │   │   ├── context_intelligence.py # Result cleaning
│       │   │   └── context_formatter.py    # Output formatting
│       │   └── providers/
│       │       └── groq_provider.py    # Groq LLM adapter
│       └── middleware/
│           ├── error_handlers.py       # Global exception handlers
│           └── request_id.py           # X-Request-ID middleware
├── frontend/
│   ├── app/
│   │   ├── chat/page.tsx              # Chat page
│   │   └── page.tsx                    # Redirects to /chat
│   ├── components/
│   │   └── chat/
│   │       ├── ChatWindow.tsx
│   │       ├── ChatHeader.tsx
│   │       ├── MessageInput.tsx
│   │       ├── SendButton.tsx
│   │       └── LoadingIndicator.tsx
│   └── types/chat.ts
├── tests/
│   ├── test_conversation_intelligence_engine.py  # 26 tests (Phase 12.2)
│   ├── test_context_relevance_engine.py          # 34 tests (Phase 12.1)
│   ├── test_reasoning_engine.py                  # 42 tests (Phase 10)
│   ├── test_request_router.py                    # 14 tests
│   ├── test_intent_classifier.py                 # 22 tests
│   ├── test_immediate_memory_processor.py        # 14 tests
│   ├── test_memory_phase3.py                     # 5 tests
│   ├── test_memory_phase5.py                     # 2 tests
│   ├── test_memory_resolver.py                   # 27 tests
│   ├── test_reflection_phase7.py                 # 12 tests
│   ├── test_response_assembler.py                # 40+ tests
│   ├── test_llm_truncation_fix.py                # 18 tests
│   ├── test_external_knowledge.py                # 12 tests
│   ├── test_external_knowledge_integration.py    # 22 tests
│   ├── test_context_intelligence.py              # 20 tests
│   ├── test_search_pipeline.py                   # 6 tests
│   └── test_*.py                                 # 387 total
├── .env                         # Live configuration (gitignored)
├── .env.example                 # Configuration template
├── docker-compose.yml           # PostgreSQL + backend
├── requirements.txt             # Python dependencies
└── phase7context.txt            # Phase 12.2 specification document
```

---

## Component Deep-Dive

### CognitiveRequestRouter (`request_router.py`)
The brain of the system. A single `route()` method orchestrates the entire pipeline:
1. Greeting detection (fast path, no LLM)
2. Intent classification (rule-based → LLM fallback)
3. Reasoning pipeline (intent, complexity, tasks, decisions)
4. Conversation Intelligence (follow-up detection & context injection)
5. Memory retrieval via hierarchy (conversation → long-term → system)
6. External knowledge integration (if triggered by reasoning)
7. Context building + prompt assembly
8. LLM invocation with dynamic token budget
9. Response assembly with quality checks
10. Background reflection (fire-and-forget)
11. Debug trace collection

### ContextRelevanceEngine
Scores each memory/conversation/search item against the current query. The final score is a weighted combination: `chroma * 0.45 + keyword * 0.20 + topic * 0.20 + recency * 0.05 + importance * 0.10`. Items below threshold are discarded, and results are capped at configurable limits.

### ConversationIntelligenceEngine
Runs after the reasoning pipeline but before memory retrieval. Takes the user message + recent conversation history and:
1. Checks if the message matches a follow-up phrase pattern (elaborate, continue, why, compare, etc.)
2. Determines the intent type (continuation, follow-up, clarification, comparison, correction, refinement, reference)
3. Resolves the topic by scanning conversation history based on intent-type rules
4. Assigns confidence using Jaccard similarity + keyword overlap
5. If confidence < threshold and multiple topics exist, returns a clarification question
6. Otherwise, builds a context block with previous Q&A and resolved topic to inject into the message

### ReasoningPipeline
A deterministic pre-processing layer that runs **before** any LLM call. It:
- Analyzes the semantic intent of the message (14 types)
- Classifies complexity (Simple messages skip most stages)
- Decomposes multi-objective requests into tasks
- Decides which memory sources are needed
- Plans execution for complex requests
- Verifies the final LLM response

### ExternalKnowledgeIntegrationService
Triggered when keywords like "latest", "news", "weather", "stock" are detected. Executes a Tavily web search, passes results through the ContextIntelligenceEngine for deduplication and scoring, formats them as structured context, and injects into the prompt alongside an instruction telling the LLM to prioritize these results.

### ReflectionService
Runs asynchronously after each response. It:
- Analyzes the user message for memory-worthy facts
- Checks if new information contradicts existing memories (archives old ones)
- Strengthens memories when information is reinforced
- Persists reflection records to disk as JSON
- Never blocks the response path

### ResponseAssembler
The final quality gate before delivering a response. It:
- Checks for empty or whitespace-only responses
- Detects false ignorance (LLM saying "I don't know" when it should know)
- Verifies code blocks exist for programming queries
- Detects URLs in responses
- Injects source citations from external knowledge
- Computes overall quality score and attaches metadata

---

## Configuration

All configuration is in `.env` (or `.env.example` as template):

```ini
# LLM Provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# Conversation Intelligence Engine (Phase 12.2)
FOLLOWUP_CONFIDENCE_THRESHOLD=0.75
MAX_CONTEXT_MESSAGES=4
MAX_REFERENCE_DISTANCE=6

# Context Relevance Engine (Phase 12.1)
MEMORY_RELEVANCE_THRESHOLD=0.55
CONVERSATION_RELEVANCE_THRESHOLD=0.40
SEARCH_RELEVANCE_THRESHOLD=0.55
MAX_MEMORY_ENTRIES=5
MAX_CONVERSATION_ENTRIES=3
MAX_SEARCH_RESULTS=5

# Embeddings
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# ChromaDB
CHROMA_DB_PATH=./database/chroma

# External Knowledge
TAVILY_API_KEY=your_key

# Database
DATABASE_URL=sqlite:///./database/aetheris.db
```

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- (Optional) Docker & Docker Compose

### Backend
```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY, TAVILY_API_KEY, and SECRET_KEY
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

**387 tests** across 16 test files, covering:
- Conversation Intelligence Engine: 26 tests
- Context Relevance Engine: 34 tests
- Reasoning Engine: 42 tests
- Request Router: 14 tests
- Intent Classifier: 22 tests
- Immediate Memory Processor: 14 tests
- Memory CRUD: 5 tests
- Memory Evolution: 2 tests
- Memory Resolver: 27 tests
- Reflection Engine: 12 tests
- Response Assembler: 40+ tests
- External Knowledge Integration: 34 tests
- Context Intelligence: 20 tests
- Search Pipeline: 6 tests
- LLM Truncation Fix: 18 tests

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Main chat — full pipeline with reasoning, memory, CI, context, LLM |
| POST | `/api/session/reset` | Clear conversation memory |
| GET | `/health` | Service health check |
| POST | `/api/memory` | Save a memory |
| GET | `/api/memory` | List all memories |
| GET | `/api/memory/search` | Semantic memory search |
| DELETE | `/api/memory/{id}` | Delete a memory |
| POST | `/api/memory/update` | Update a memory |
| POST | `/api/memory/merge` | Merge memories |
| GET | `/api/debug/context` | Debug context builder state |
| GET | `/api/debug/hierarchy` | Debug memory hierarchy state |
| GET | `/api/reflection` | List reflection records |

---

## What Still Needs to Be Implemented

Based on the project structure (empty directories at root) and missing features:

### Planned but not started:
- **Knowledge Graph** (`knowledge_graph/` directory exists, empty) — Entity extraction and relationship mapping
- **Emotion/Personality System** (`emotion/`, `personality/` directories exist, empty) — Affective computing and personality-driven response style
- **Planner / Tool Use** (`planner/`, `tools/` directories exist, empty) — Autonomous tool calling and task execution
- **LLM Orchestration** (`llm/` directory exists, empty) — Multi-model routing, fallback chains, model selection
- **Memory Visualization** (`memory/` directory exists, empty) — Memory graph browser
- **Embedding Management** (`embeddings/` directory exists, empty) — Custom embedding fine-tuning
- **Frontend Dashboard** — Dashboard, Memory Browser, Settings, Developer pages are placeholders
- **Frontend State Management** — No state management library; plain useState for chat

### Improvements needed:
- **Session ID**: Frontend doesn't send `session_id` — conversation memory is session-based but sessions are never differentiated
- **Database ORM**: `DatabaseService` exists but has no ORM entities defined
- **Security**: API keys are exposed in `.env` (should use environment variables in production)
- **Caching**: No Redis or distributed cache for multi-process deployments

---

## Architecture Decisions

- **No Pydantic-Settings**: Settings are loaded via a manual `.env` parser into a frozen dataclass (avoids Pydantic dependency for config)
- **Deterministic Reasoning**: The reasoning engine uses zero LLM calls — all intent analysis, complexity classification, and decisions are rule-based for speed and reliability
- **Stateless Services**: All services are stateless singletons managed via `lru_cache` in `dependencies.py` — state lives in ChromaDB and conversation memory
- **Session-Scoped Memory**: Conversation memory is a process-wide singleton — no user/session isolation in the current implementation
- **Middle Insertion**: Phase 12.2 was inserted between the reasoning pipeline and memory retrieval without changing existing component interfaces

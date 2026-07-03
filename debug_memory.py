"""
Aetheris Memory System Diagnostic Script
Run from the project root:
    .venv/bin/python debug_memory.py
"""
import os
import sys

# Make sure imports resolve from the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

print("=" * 60)
print("AETHERIS MEMORY DIAGNOSTIC")
print("=" * 60)

# ── 1. Package check ─────────────────────────────────────────
print("\n[1] PACKAGE CHECK")
missing = []
for pkg in ["chromadb", "sentence_transformers", "fastapi", "pydantic"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK  {pkg} == {ver}")
    except ImportError as e:
        print(f"  MISSING  {pkg}: {e}")
        missing.append(pkg)

if missing:
    print(f"\nSTOP: install missing packages first: {missing}")
    sys.exit(1)

# ── 2. Settings check ────────────────────────────────────────
print("\n[2] SETTINGS")
from backend.app.config.settings import get_settings
s = get_settings()
print(f"  chroma_db_path  : {s.chroma_db_path!r}")
print(f"  embedding_model : {s.embedding_model!r}")
print(f"  llm_provider    : {s.llm_provider!r}")
print(f"  llm_model       : {s.llm_model!r}")
print(f"  qwen_base_url   : {s.qwen_base_url!r}")
# Never print the API key value
print(f"  qwen_api_key    : {'SET (sk-or-v1-...)' if s.qwen_api_key.startswith('sk-or-v1-') else ('SET' if s.qwen_api_key else 'NOT SET')}")

from pathlib import Path
chroma_abs = (Path(PROJECT_ROOT) / s.chroma_db_path).resolve()
print(f"  chroma resolved : {chroma_abs}")
print(f"  chroma exists   : {chroma_abs.exists()}")

# ── 3. Embedding service check ───────────────────────────────
print("\n[3] EMBEDDING SERVICE")
try:
    from backend.app.services.embedding_service import EmbeddingService
    emb = EmbeddingService(s)
    test_text = "I am building Aetheris."
    print(f"  Embedding text  : {test_text!r}")
    vector = emb.embed_text(test_text)
    print(f"  Embedding dim   : {len(vector)}")
    print(f"  First 5 values  : {[round(v, 6) for v in vector[:5]]}")
    print("  STATUS: OK")
except Exception as e:
    print(f"  STATUS: FAILED — {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 4. ChromaDB service check ────────────────────────────────
print("\n[4] CHROMADB SERVICE")
try:
    from backend.app.services.chroma_service import ChromaService, COLLECTION_NAME
    chroma = ChromaService(s)
    col = chroma._get_collection()
    count_before = col.count()
    print(f"  Collection name : {COLLECTION_NAME!r}")
    print(f"  Storage path    : {chroma.storage_path}")
    print(f"  Records before  : {count_before}")
    print("  STATUS: OK")
except Exception as e:
    print(f"  STATUS: FAILED — {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 5. save_memory() x3 ─────────────────────────────────────
print("\n[5] save_memory() — WRITING 3 TEST MEMORIES")
from backend.app.services.memory_service import MemoryService

mem_svc = MemoryService(embedding_service=emb, chroma_service=chroma)

test_memories = [
    "I'm building Aetheris.",
    "My favorite programming language is Java.",
    "I'm preparing for software engineering interviews.",
]

saved_ids = []
for text in test_memories:
    try:
        result = mem_svc.save_memory(memory_text=text)
        saved_ids.append(result["memory_id"])
        print(f"  Memory Saved")
        print(f"    Memory ID   : {result['memory_id']}")
        print(f"    Memory Text : {text!r}")
        print(f"    Created At  : {result['created_at']}")
        print()
    except Exception as e:
        print(f"  FAILED to save {text!r}: {e}")
        import traceback; traceback.print_exc()

count_after = chroma._get_collection().count()
print(f"  Records after save : {count_after}")

if count_after == count_before:
    print("  BUG: count did not increase — memories were NOT persisted!")
elif count_after < count_before + len(test_memories):
    print(f"  WARNING: expected {count_before + len(test_memories)} records, got {count_after}")
else:
    print("  All memories persisted successfully.")

# ── 6. search_memory() x3 ───────────────────────────────────
print("\n[6] search_memory() — QUERYING 3 TIMES")
queries = [
    "What project am I building?",
    "What language do I like?",
    "What am I preparing for?",
]

for query in queries:
    print(f"\n  Query: {query!r}")
    try:
        results = mem_svc.search_memory(query=query, top_k=5)
        if not results:
            print("    NO RESULTS RETURNED")
        for i, r in enumerate(results, 1):
            print(f"    [{i}] score={r['score']:.4f} | text={r['document']!r}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()

# ── 7. list_memories() ───────────────────────────────────────
print("\n[7] list_memories()")
try:
    all_mems = mem_svc.list_memories()
    print(f"  Total stored: {len(all_mems)}")
    for m in all_mems:
        print(f"    id={m['id']} | text={m['document']!r}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

# ── 8. Context builder check ─────────────────────────────────
print("\n[8] CONTEXT BUILDER — final prompt check")
try:
    from backend.app.services.context_builder import ContextBuilderService, _MIN_SCORE
    cb = ContextBuilderService()

    # Use last query results for the prompt demo
    demo_query = "What project am I building?"
    demo_results = mem_svc.search_memory(query=demo_query, top_k=5)
    memory_context = cb.build_memory_context(demo_results)
    system_prompt = cb.build_system_prompt()

    print(f"\n  MIN_SCORE filter   : {_MIN_SCORE}")
    print(f"  Raw results count  : {len(demo_results)}")
    for r in demo_results:
        kept = r['score'] >= _MIN_SCORE
        print(f"    score={r['score']:.4f} {'KEPT' if kept else 'DROPPED (below threshold)'} | {r['document']!r}")

    print(f"\n  Memory context empty: {memory_context == ''}")
    print()
    print("  ── EXACT PROMPT SENT TO LLM ──────────────────────────")
    final_prompt_user_turn = (
        f"{memory_context.strip()}\n\nCurrent User Message:\n{demo_query.strip()}"
        if memory_context.strip()
        else demo_query.strip()
    )
    print(system_prompt)
    print()
    print(final_prompt_user_turn)
    print("  ───────────────────────────────────────────────────────")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

# ── 9. Cleanup saved test memories ───────────────────────────
print("\n[9] CLEANUP — deleting test memories")
for mid in saved_ids:
    try:
        mem_svc.delete_memory(mid)
        print(f"  Deleted {mid}")
    except Exception as e:
        print(f"  Could not delete {mid}: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

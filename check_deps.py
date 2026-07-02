missing = []
for pkg in ["fastapi", "chromadb", "sentence_transformers", "pydantic", "uvicorn"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  MISSING: {pkg}")
        missing.append(pkg)

print()
if missing:
    print("Need to install:", missing)
else:
    print("All packages OK")

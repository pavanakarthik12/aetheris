"""Quick test script for Tavily API connectivity."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.services.external_knowledge.tavily_provider import TavilyProvider


async def test():
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        try:
            for line in open(".env"):
                if line.startswith("TAVILY_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip("\"").strip("'")
                    break
        except FileNotFoundError:
            pass

    if not key:
        print("NO API KEY FOUND - set TAVILY_API_KEY in .env")
        return

    print(f"Using key: {key[:8]}...{key[-4:]}")
    provider = TavilyProvider(api_key=key)

    print("\nHealth check...")
    healthy = await provider.health_check()
    print(f"Health: {'OK' if healthy else 'FAILED'}")

    print("\nSearching for 'latest AI news'...")
    result = await provider.search("latest AI news", max_results=3)
    print(f"Results count: {result.total_results}")
    answer = result.answer[:200] if result.answer else "(none)"
    print(f"Answer: {answer}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    print(f"Provider: {result.provider}")

    if result.results:
        print("\nResults:")
        for i, r in enumerate(result.results[:3], 1):
            print(f"  {i}. {r.title}")
            print(f"     URL: {r.url}")
            print(f"     Score: {r.score}")
            print(f"     Snippet: {r.snippet[:100]}...")
    else:
        print("\nNo results returned - check API key and network")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(test())

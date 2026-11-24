from ddgs import DDGS

try:
    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                "India 2023 Cricket World Cup winner",
                max_results=5
            )
        )

    print("\nSUCCESS")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('title')}")
        print(f"URL: {result.get('href')}")
        print(f"Snippet: {result.get('body')}")

except Exception as e:
    print("\nFAILED")
    print(type(e).__name__)
    print(e)
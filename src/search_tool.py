import html
import json
import re
import urllib.parse
import urllib.request
from ddgs import DDGS


def _clean_html_snippet(text: str) -> str:
    """Strip HTML tags and unescape entities from snippets."""
    clean = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(clean).strip()


def _search_wikipedia(query: str, max_results: int = 5) -> list:
    """
    Fallback search using Wikipedia's public MediaWiki search API.
    Highly reliable, never blocks cloud server IPs, and returns factual summaries.
    """
    results = []
    try:
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}&format=json&utf8=1"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgenticFactChecker/2.0 (web-verification-agent)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("query", {}).get("search", [])
            for item in items[:max_results]:
                title = item.get("title", "")
                snippet = _clean_html_snippet(item.get("snippet", ""))
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append({
                    "title": title,
                    "url": page_url,
                    "snippet": snippet,
                })
    except Exception as e:
        print(f"[WIKIPEDIA FALLBACK ERROR] {e}")
    return results


def search_web(query: str, max_results: int = 10) -> list:
    """
    Searches the web for factual evidence.
    Primary: DuckDuckGo Search (DDGS).
    Fallback: Wikipedia Search API if DDGS fails or returns 0 results.
    """
    results = []

    # 1. Primary: DuckDuckGo search
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                })
    except Exception as e:
        print(f"[DDGS SEARCH NOTICE] {e}. Trying fallback sources...")

    # 2. Fallback / Augment: Wikipedia if primary returned no results
    if not results:
        print(f"[SEARCH FALLBACK] Querying Wikipedia for '{query}'...")
        wiki_results = _search_wikipedia(query, max_results=max_results)
        results.extend(wiki_results)

    return results

# tools/web_search.py
# ============================================================
#  Web Search Tool — Tavily API
#
#  When the Hybrid Retriever returns low confidence (no matching
#  internal data), the agent falls back here.
#
#  Tavily is purpose-built for LLM pipelines:
#  - Returns clean, structured results (not raw HTML)
#  - Filters out ads, navigation, boilerplate
#  - Has a "medical" domain bias in professional plans
#
#  Flow:
#  1. Rewrite user query to be search-engine-friendly
#  2. Call Tavily with max_results=3
#  3. Extract and clean the result text
#  4. Return as a list of context strings
# ============================================================

import re
from tavily import TavilyClient
from utils.config import config
from utils.logger import logger


def _rewrite_query_for_search(query: str) -> str:
    """
    Add medical context to vague queries so search returns
    more relevant medical results.

    Example:
        "fever and cough" → "fever and cough symptoms causes treatment medical"
    """
    medical_suffix = "symptoms causes diagnosis treatment medical information"
    return f"{query} {medical_suffix}"


def _clean_text(text: str) -> str:
    """
    Remove excessive whitespace and noise from web result text.
    """
    # Collapse multiple newlines/spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def web_search(query: str, num_results: int = None) -> list[dict]:
    """
    Search the web via Tavily and return clean results.

    Args:
        query:       Original user query
        num_results: Number of results to fetch (default from config)

    Returns:
        List of dicts:
        {
            "title":   "Page title",
            "url":     "https://...",
            "content": "Cleaned text snippet",
            "score":   0.85  (Tavily's relevance score)
        }
    """
    if num_results is None:
        num_results = config.WEB_SEARCH_RESULTS

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    search_query = _rewrite_query_for_search(query)

    logger.info(f"Web search triggered for: '{query}'")
    logger.debug(f"Rewritten query: '{search_query}'")

    try:
        response = client.search(
            query=search_query,
            max_results=num_results,
            search_depth="advanced",          # deeper search, more relevant
            include_answer=True,              # Tavily's own summary
            include_raw_content=False,        # we don't need full HTML
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

    results = []

    # Include Tavily's own synthesised answer if available
    if response.get("answer"):
        results.append({
            "title":   "Web Search Summary",
            "url":     "",
            "content": _clean_text(response["answer"]),
            "score":   1.0,
        })

    # Include individual page results
    for item in response.get("results", []):
        content = item.get("content", "") or item.get("snippet", "")
        if not content:
            continue
        results.append({
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "content": _clean_text(content),
            "score":   float(item.get("score", 0.5)),
        })

    logger.info(f"Web search returned {len(results)} results")
    return results


def format_web_context(results: list[dict]) -> str:
    """
    Convert Tavily results into a single context string
    that can be inserted into the LLM prompt.

    Args:
        results: Output of web_search()

    Returns:
        Formatted multi-source context string
    """
    if not results:
        return "No web search results available."

    parts = []
    for i, r in enumerate(results, 1):
        source = f"[Source {i}]"
        if r.get("title"):
            source += f" {r['title']}"
        if r.get("url"):
            source += f" ({r['url']})"
        parts.append(f"{source}\n{r['content']}")

    return "\n\n".join(parts)


if __name__ == "__main__":
    # Quick test
    results = web_search("itchy skin rash causes")
    context = format_web_context(results)
    print(context[:500])

# retriever/bm25.py
# ============================================================
#  BM25 Keyword Retriever
#
#  BM25 (Best Match 25) is a classical TF-IDF variant that
#  finds documents containing the exact same keywords as the
#  query. It's fast, deterministic, and great at rare terms
#  like specific disease names or drug names.
#
#  Think of it as "ctrl+F on steroids with term weighting".
#
#  Why use BM25 alongside vectors?
#  - Vector search is great for paraphrases & semantics
#  - BM25 is great for exact term matching (e.g. "metformin")
#  - Together they're stronger than either alone → Hybrid RAG
# ============================================================

import re
from rank_bm25 import BM25Okapi
from utils.logger import logger
from utils.config import config


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercase everything, split on non-alphanumeric chars.

    Example:
        "Fever, chills and cough" → ["fever", "chills", "and", "cough"]
    """
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t]  # remove empty strings


class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi with our document format.

    Usage:
        retriever = BM25Retriever(documents)
        results = retriever.retrieve("fever and cough", top_k=5)
    """

    def __init__(self, documents: list[dict]):
        """
        Build BM25 index from documents.

        Args:
            documents: List of dicts with at least a 'text' key
                       (output from data_processor.build_documents)
        """
        self.documents = documents

        # Tokenize all document texts
        # BM25Okapi expects a list-of-list-of-tokens
        tokenized_corpus = [_tokenize(doc["text"]) for doc in documents]

        # Build the index — this is the O(n) indexing step
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve top-k documents by BM25 score.

        Args:
            query: Natural language query (e.g. "fever and cough")
            top_k: Number of results to return (default from config)

        Returns:
            List of dicts, each containing:
                - 'doc':   Original document dict
                - 'score': Raw BM25 score (unnormalised)
                - 'normalised_score': Score scaled 0-1 relative to this query
        """
        if top_k is None:
            top_k = config.TOP_K

        # Tokenize query with the same tokenizer used for indexing
        query_tokens = _tokenize(query)

        if not query_tokens:
            logger.warning("BM25: empty query after tokenisation — returning empty results")
            return []

        # BM25 gives a score array (one value per document)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Normalise scores to [0, 1] so we can blend with vector scores
        max_score = max(scores[i] for i in top_indices) if top_indices else 1.0
        max_score = max_score if max_score > 0 else 1.0  # avoid divide-by-zero

        results = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score <= 0:
                continue  # BM25 returns 0 for docs with no matching tokens
            results.append({
                "doc":              self.documents[idx],
                "score":            raw_score,
                "normalised_score": raw_score / max_score,  # relative to this query's best
            })

        logger.debug(f"BM25 returned {len(results)} results for query: '{query}'")
        return results


if __name__ == "__main__":
    # Quick smoke test — run: python retriever/bm25.py
    from utils.data_processor import load_documents
    docs = load_documents()
    retriever = BM25Retriever(docs)
    results = retriever.retrieve("fever and cough", top_k=3)
    for r in results:
        print(f"Score: {r['normalised_score']:.3f} | {r['doc']['name']}")

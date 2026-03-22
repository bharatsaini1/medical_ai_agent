# retriever/hybrid.py
# ============================================================
#  Hybrid Retriever — BM25 + Vector Search
#
#  Why hybrid?
#  ┌─────────────────────────────────────────────────────┐
#  │  Query: "fever with joint pain"                     │
#  │                                                     │
#  │  BM25:   Finds docs containing "fever", "joint"    │
#  │          "pain" — good for exact medical terms      │
#  │                                                     │
#  │  Vector: Finds docs about "arthritis", "flu",      │
#  │          "chikungunya" — semantically close        │
#  │                                                     │
#  │  Hybrid: Union of both, weighted and re-ranked     │
#  └─────────────────────────────────────────────────────┘
#
#  Scoring formula:
#  final_score = (BM25_WEIGHT × bm25_score) + (VECTOR_WEIGHT × vector_score)
#  Both weights default to 0.5 (equal blend).
# ============================================================

from utils.config import config
from utils.logger import logger
from retriever.bm25 import BM25Retriever
from retriever.vector import VectorRetriever
from utils.reranker import rerank


class HybridRetriever:
    """
    Combines BM25 and vector retrieval with configurable weights.

    Usage:
        hr = HybridRetriever(documents)
        hr.setup(documents)           # build BM25, index Pinecone
        results, confidence = hr.retrieve("itchy skin rash")
    """

    def __init__(self):
        self.bm25_retriever:   BM25Retriever   = None
        self.vector_retriever: VectorRetriever = None
        self.is_ready = False

    def setup(self, documents: list[dict]):
        """
        Initialise both sub-retrievers.
        Call this once at startup (not per request).

        Args:
            documents: Processed docs from data_processor
        """
        logger.info("Setting up Hybrid Retriever...")

        # BM25 builds its index in-memory instantly
        self.bm25_retriever = BM25Retriever(documents)

        # Vector retriever connects to Pinecone & indexes if needed
        self.vector_retriever = VectorRetriever()
        self.vector_retriever.index_documents(documents)

        self.is_ready = True
        logger.info("Hybrid Retriever is ready")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        bm25_weight: float = None,
        vector_weight: float = None,
    ) -> tuple[list[dict], float]:
        """
        Run both retrievers, merge results, return top-k with confidence.

        Args:
            query:         User query string
            top_k:         Number of results to return
            bm25_weight:   Weight for BM25 scores (0.0 – 1.0)
            vector_weight: Weight for vector scores (0.0 – 1.0)

        Returns:
            (results, confidence_score) where:
            - results is a list of dicts (doc + score)
            - confidence_score is the top result's hybrid score [0, 1]
        """
        if not self.is_ready:
            raise RuntimeError("Call setup(documents) before retrieve()")

        if top_k is None:
            top_k = config.TOP_K
        if bm25_weight is None:
            bm25_weight = config.BM25_WEIGHT
        if vector_weight is None:
            vector_weight = config.VECTOR_WEIGHT

        # Fetch results from both retrievers (fetch 2× top_k to have more to blend)
        fetch_k = top_k * 2

        bm25_results   = self.bm25_retriever.retrieve(query,   top_k=fetch_k)
        vector_results = self.vector_retriever.retrieve(query, top_k=fetch_k)

        # Build lookup tables: disease_code → result
        bm25_map   = {r["doc"]["disease_code"]: r for r in bm25_results}
        vector_map = {r["doc"]["disease_code"]: r for r in vector_results}

        # Union of all document IDs seen by either retriever
        all_ids = set(bm25_map.keys()) | set(vector_map.keys())

        merged = []
        for doc_id in all_ids:
            bm25_score   = bm25_map[doc_id]["normalised_score"]   if doc_id in bm25_map   else 0.0
            vector_score = vector_map[doc_id]["normalised_score"] if doc_id in vector_map else 0.0

            # Hybrid score: weighted average
            hybrid_score = (bm25_weight * bm25_score) + (vector_weight * vector_score)

            # Use whichever retriever found this doc (prefer vector for richer metadata)
            doc = (vector_map.get(doc_id) or bm25_map[doc_id])["doc"]

            merged.append({
                "doc":           doc,
                "score":         hybrid_score,
                "bm25_score":    bm25_score,
                "vector_score":  vector_score,
            })

        # Sort descending by hybrid score
        merged.sort(key=lambda x: x["score"], reverse=True)

        # Rerank the top candidates with lightweight cross-signal scoring
        merged = rerank(query, merged)
        top_results = merged[:top_k]

        # Confidence = top result's hybrid score
        confidence = top_results[0]["score"] if top_results else 0.0

        logger.info(
            f"Hybrid retrieval: {len(top_results)} results | "
            f"confidence={confidence:.3f} | query='{query}'"
        )

        return top_results, confidence

    def is_confident(self, confidence: float) -> bool:
        """
        Returns True if confidence is above the configured threshold.
        If False, the agent should fall back to web search.
        """
        return confidence >= config.CONFIDENCE_THRESHOLD


if __name__ == "__main__":
    from utils.data_processor import load_documents
    docs = load_documents()
    hr = HybridRetriever()
    hr.setup(docs)
    results, conf = hr.retrieve("itchy skin blisters")
    print(f"\nConfidence: {conf:.3f} | Confident: {hr.is_confident(conf)}")
    for r in results[:3]:
        print(f"  {r['score']:.3f} — {r['doc']['name']}")

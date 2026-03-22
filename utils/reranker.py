# utils/reranker.py
# ============================================================
#  Result Reranker
#
#  After hybrid retrieval returns top-K candidates, reranking
#  applies a second-pass score to sort them more accurately.
#
#  Why rerank?
#  BM25 + vector give a rough shortlist. Reranking applies
#  more nuanced signals:
#    1. Symptom overlap count (how many query symptoms appear?)
#    2. Query length penalty (prefer specific matches)
#    3. Exact name match bonus (e.g. query mentions "scabies")
#
#  This is a lightweight cross-encoder alternative — no model
#  needed, runs in microseconds.
# ============================================================

import re


def _extract_symptom_words(text: str) -> set[str]:
    """Tokenise text into a set of lowercase words."""
    return set(re.split(r"[^a-z]+", text.lower())) - {"", "and", "or", "the", "a", "an", "with", "in", "of"}


def rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Re-score and re-sort retrieval results using lightweight signals.

    Args:
        query:   Original user query string
        results: List of hybrid retrieval results (each has 'doc' and 'score')

    Returns:
        Reranked list (same dicts, 'score' updated, new 'rerank_score' added)
    """
    query_words = _extract_symptom_words(query)

    for r in results:
        doc = r["doc"]
        base_score = r["score"]

        # Signal 1 — Symptom overlap
        # Count how many query words appear in the document's symptom list
        symptom_words = _extract_symptom_words(doc.get("symptoms", ""))
        overlap = len(query_words & symptom_words)
        overlap_bonus = min(overlap * 0.05, 0.25)  # cap at +0.25

        # Signal 2 — Exact disease name match in query
        # e.g. user types "I think I have scabies"
        disease_name_words = _extract_symptom_words(doc.get("name", ""))
        name_overlap = len(query_words & disease_name_words)
        name_bonus = 0.15 if name_overlap >= 1 else 0.0

        # Signal 3 — Treatment keyword bonus
        # Some users mention treatments ("metformin", "insulin")
        treatment_words = _extract_symptom_words(doc.get("treatments", ""))
        treatment_overlap = len(query_words & treatment_words)
        treatment_bonus = min(treatment_overlap * 0.03, 0.10)

        # Final rerank score
        rerank_score = base_score + overlap_bonus + name_bonus + treatment_bonus

        r["rerank_score"]   = rerank_score
        r["overlap_count"]  = overlap
        r["score"]          = rerank_score  # update score for downstream use

    # Sort by rerank score descending
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results


if __name__ == "__main__":
    # Quick demo
    mock_results = [
        {"doc": {"name": "Scabies", "symptoms": "intense itching night small blisters", "treatments": "scabicides"}, "score": 0.60},
        {"doc": {"name": "Eczema",  "symptoms": "dry itchy skin rashes blisters",       "treatments": "moisturizers"}, "score": 0.58},
        {"doc": {"name": "Psoriasis","symptoms": "red patches dry cracked skin itching", "treatments": "topical steroids"}, "score": 0.55},
    ]
    query = "intense itching at night with small blisters on hands"
    reranked = rerank(query, mock_results)
    print("After reranking:")
    for r in reranked:
        print(f"  {r['doc']['name']}: {r['score']:.3f} (overlap={r['overlap_count']})")

# utils/evaluator.py
# ============================================================
#  Evaluation Metrics for Hybrid Retrieval
#
#  Metrics implemented:
#  1. Precision@K  — of the top-K results, how many are relevant?
#  2. Recall@K     — of all relevant docs, how many did we find in top-K?
#  3. MRR          — Mean Reciprocal Rank (how high is the first correct hit?)
#  4. Hit Rate@K   — did ANY correct answer appear in top-K?
#
#  Usage:
#    evaluator = RetrievalEvaluator(hybrid_retriever)
#    results = evaluator.evaluate(test_cases)
#    evaluator.print_report(results)
# ============================================================

from dataclasses import dataclass, field
from utils.logger import logger


@dataclass
class TestCase:
    """A single evaluation test case."""
    query:              str                   # user query
    relevant_disease:   str                   # expected disease name (partial match OK)
    description:        str = ""              # human-readable description


@dataclass
class EvalResult:
    """Evaluation results for a single test case."""
    query:              str
    expected:           str
    retrieved:          list[str]             # names of retrieved docs
    hit:                bool                  # was expected found in top-K?
    rank:               int                   # 1-indexed rank of first correct hit (0 = not found)
    precision_at_k:     float
    recall_at_k:        float
    reciprocal_rank:    float


class RetrievalEvaluator:
    """
    Evaluates Hybrid Retrieval quality against a labelled test set.

    Args:
        retriever: An initialised HybridRetriever instance
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def _is_match(self, retrieved_name: str, expected: str) -> bool:
        """Fuzzy match: expected string is a substring of retrieved name (case-insensitive)."""
        return expected.lower() in retrieved_name.lower()

    def evaluate_single(self, test_case: TestCase, k: int = 5) -> EvalResult:
        """Run retrieval for one test case and compute metrics."""
        results, confidence = self.retriever.retrieve(test_case.query, top_k=k)
        retrieved_names = [r["doc"]["name"] for r in results]

        # Find first correct hit
        rank = 0
        for i, name in enumerate(retrieved_names, 1):
            if self._is_match(name, test_case.relevant_disease):
                rank = i
                break

        hit = rank > 0
        # Precision@K: correct hits / K
        correct_count = sum(1 for n in retrieved_names if self._is_match(n, test_case.relevant_disease))
        precision_at_k = correct_count / k if k > 0 else 0.0
        # Recall@K: we assume 1 ground truth doc → recall = 1 if hit else 0
        recall_at_k = 1.0 if hit else 0.0
        # MRR: 1/rank if found, else 0
        reciprocal_rank = 1.0 / rank if rank > 0 else 0.0

        return EvalResult(
            query=test_case.query,
            expected=test_case.relevant_disease,
            retrieved=retrieved_names,
            hit=hit,
            rank=rank,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            reciprocal_rank=reciprocal_rank,
        )

    def evaluate(self, test_cases: list[TestCase], k: int = 5) -> list[EvalResult]:
        """Evaluate all test cases and return individual results."""
        logger.info(f"Evaluating {len(test_cases)} test cases at K={k}")
        results = []
        for tc in test_cases:
            result = self.evaluate_single(tc, k=k)
            results.append(result)
            status = "✓" if result.hit else "✗"
            logger.debug(
                f"{status} '{tc.query[:50]}' → expected '{tc.relevant_disease}' "
                f"| rank={result.rank} | P@{k}={result.precision_at_k:.2f}"
            )
        return results

    def aggregate(self, results: list[EvalResult], k: int = 5) -> dict:
        """Compute aggregate metrics across all test cases."""
        n = len(results)
        if n == 0:
            return {}

        hit_rate     = sum(r.hit for r in results) / n
        mean_prec    = sum(r.precision_at_k for r in results) / n
        mean_recall  = sum(r.recall_at_k for r in results) / n
        mrr          = sum(r.reciprocal_rank for r in results) / n
        avg_rank     = sum(r.rank for r in results if r.rank > 0) / max(sum(r.hit for r in results), 1)

        return {
            f"hit_rate@{k}":      round(hit_rate, 3),
            f"precision@{k}":     round(mean_prec, 3),
            f"recall@{k}":        round(mean_recall, 3),
            "mrr":                round(mrr, 3),
            "avg_rank_when_hit":  round(avg_rank, 2),
            "total_cases":        n,
            "hits":               sum(r.hit for r in results),
        }

    def print_report(self, results: list[EvalResult], k: int = 5):
        """Print a formatted evaluation report to stdout."""
        agg = self.aggregate(results, k=k)
        print("\n" + "=" * 60)
        print("  RETRIEVAL EVALUATION REPORT")
        print("=" * 60)
        for key, val in agg.items():
            print(f"  {key:<25} {val}")
        print("-" * 60)
        print(f"  {'QUERY':<40} {'EXPECTED':<20} RESULT")
        print("-" * 60)
        for r in results:
            status = "✓ HIT" if r.hit else "✗ MISS"
            rank_str = f"rank {r.rank}" if r.hit else "not found"
            print(f"  {r.query[:38]:<40} {r.expected[:18]:<20} {status} ({rank_str})")
        print("=" * 60)


# ── Built-in test cases matching your dataset ───────────────
DEFAULT_TEST_CASES = [
    TestCase("intense itching at night small blisters", "Scabies"),
    TestCase("cloudy eyes excessive tearing child", "Congenital Glaucoma"),
    TestCase("avoidance restriction certain foods weight loss", "ARFID"),
    TestCase("abdominal pain bleeding organ", "Injury to Internal Organ"),
    TestCase("itchy skin hands feet", "Gestational Cholestasis"),
    TestCase("blurred vision, shortness of breath, cloudy eyes", "Congenital Glaucoma"),
    TestCase("fatigue weight gain cold sensitivity dry skin", "Hypothyroidism"),
    TestCase("persistent sadness loss of interest sleep problems", "Depression"),
    TestCase("heartburn acid reflux chest pain", "GERD"),
    TestCase("joint pain stiffness swelling decreased motion", "Arthritis"),
]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.data_processor import load_documents
    from retriever.hybrid import HybridRetriever

    docs = load_documents()
    hr = HybridRetriever()
    hr.setup(docs)

    evaluator = RetrievalEvaluator(hr)
    results = evaluator.evaluate(DEFAULT_TEST_CASES, k=5)
    evaluator.print_report(results, k=5)

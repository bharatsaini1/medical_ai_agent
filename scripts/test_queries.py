#!/usr/bin/env python3
# scripts/test_queries.py
# ============================================================
#  Test the API with sample medical queries
#  Run: python scripts/test_queries.py
#  (Server must be running: python api/app.py)
# ============================================================

import os
import sys
import json
import requests

BASE_URL = "http://localhost:5000"

# ── Sample queries covering different scenarios ──────────────
TEST_QUERIES = [
    # Clear symptom queries (should get HIGH confidence from database)
    {
        "category": "Skin Condition",
        "query": "I have intense itching especially at night and small blisters on my skin",
        "expected_disease": "Scabies",
    },
    {
        "category": "Eye Condition",
        "query": "My child has cloudy eyes, excessive tearing and sensitivity to light",
        "expected_disease": "Congenital Glaucoma",
    },
    {
        "category": "Metabolic",
        "query": "I feel very tired, I am gaining weight, always cold and my skin is dry",
        "expected_disease": "Hypothyroidism",
    },
    {
        "category": "Digestive",
        "query": "I have severe heartburn, acid coming up my throat and chest pain after eating",
        "expected_disease": "GERD",
    },
    {
        "category": "Mental Health",
        "query": "I feel sad all the time, lost interest in things I used to enjoy and have trouble sleeping",
        "expected_disease": "Depression",
    },
    # Vague queries (may trigger web search fallback)
    {
        "category": "Vague Query",
        "query": "my back hurts when I breathe",
        "expected_disease": "unknown — may fall back to web",
    },
    # Complex multi-symptom query
    {
        "category": "Complex",
        "query": "fever with joint pain and a rash spreading across my face and arms",
        "expected_disease": "Could be multiple conditions",
    },
]


def run_tests():
    print("=" * 70)
    print("  Medical AI Agent — Test Query Runner")
    print("=" * 70)

    # Check server health first
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        health = r.json()
        print(f"\n✓ Server healthy | Model: {health['model']} | Dataset: {health['dataset']}")
    except Exception as e:
        print(f"\n✗ Server not reachable: {e}")
        print("  Make sure the server is running: python api/app.py")
        sys.exit(1)

    print()

    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"─" * 70)
        print(f"Test {i}/{len(TEST_QUERIES)} [{test['category']}]")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected_disease']}")
        print()

        try:
            r = requests.post(
                f"{BASE_URL}/chat",
                json={"query": test["query"], "session_id": f"test_{i}"},
                timeout=30,
            )
            data = r.json()

            print(f"Confidence: {data.get('confidence_pct', 0)}%")
            print(f"Source: {data.get('source', 'unknown')}")
            print(f"Web search used: {data.get('used_web_search', False)}")
            print(f"Processing time: {data.get('processing_time', 0)}s")

            top_matches = data.get("top_matches", [])
            if top_matches:
                print("Top matches:")
                for m in top_matches[:3]:
                    print(f"  - {m['name']} (score: {m['score']:.3f})")

            print("\nResponse (truncated):")
            resp = data.get("response", "")
            print(resp[:400] + "..." if len(resp) > 400 else resp)

        except Exception as e:
            print(f"✗ Request failed: {e}")

        print()

    print("=" * 70)
    print("Test run complete!")


if __name__ == "__main__":
    run_tests()

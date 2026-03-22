# utils/query_rewriter.py
# ============================================================
#  Query Rewriting Utility
#
#  Why rewrite queries?
#  Users often type casually:
#    "my back hurts and I feel tired"
#  But retrieval works better with:
#    "back pain fatigue symptoms causes"
#
#  We use two strategies:
#  1. Rule-based expansion  — fast, no API call needed
#  2. LLM rewriting         — slower, but handles complex cases
#
#  The agent uses LLM rewriting (in agent.py).
#  This module provides the rule-based version as a utility
#  that runs BEFORE the LLM call to pre-clean queries.
# ============================================================

import re

# Medical abbreviation expansions
ABBREVIATIONS = {
    "bp":    "blood pressure",
    "hr":    "heart rate",
    "sob":   "shortness of breath",
    "gi":    "gastrointestinal",
    "uti":   "urinary tract infection",
    "ent":   "ear nose throat",
    "uri":   "upper respiratory infection",
    "lbp":   "lower back pain",
    "ha":    "headache",
    "n/v":   "nausea vomiting",
    "n/v/d": "nausea vomiting diarrhea",
    "fever": "fever high temperature",
}

# Filler phrases that add noise to retrieval
NOISE_PHRASES = [
    r"\bi have\b",
    r"\bi am having\b",
    r"\bi am\b",
    r"\bmy\b",
    r"\bplease help\b",
    r"\bcan you tell me\b",
    r"\bwhat is wrong with\b",
    r"\bi think i have\b",
    r"\bi feel\b",
    r"\bfor the past \d+ (days?|weeks?|months?)\b",
    r"\bsince \w+\b",
]


def expand_abbreviations(text: str) -> str:
    """Replace known medical abbreviations with full terms."""
    words = text.lower().split()
    expanded = []
    for w in words:
        clean = w.strip(".,;:!?")
        if clean in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[clean])
        else:
            expanded.append(w)
    return " ".join(expanded)


def remove_noise(text: str) -> str:
    """Remove filler phrases that don't help retrieval."""
    for pattern in NOISE_PHRASES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def rule_based_rewrite(query: str) -> str:
    """
    Fast, deterministic query cleaning.
    Used as a pre-processing step before BM25/vector search.

    Example:
        "I have been feeling very tired and my back hurts a lot"
        → "feeling tired back hurts"
    """
    query = expand_abbreviations(query)
    query = remove_noise(query)
    return query


if __name__ == "__main__":
    tests = [
        "I have been having n/v/d for 3 days",
        "my bp is high and I have a bad ha",
        "can you tell me what is wrong with me, I feel sob",
        "I am having fever and body aches since Monday",
    ]
    for t in tests:
        print(f"Original: {t}")
        print(f"Rewritten: {rule_based_rewrite(t)}")
        print()

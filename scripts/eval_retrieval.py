"""
Retrieval evaluation harness for the guideline vector store.

Runs a small labeled set of clinical queries against the live pgvector index
and reports:
  - Hit rate @ k: did at least one retrieved chunk contain an expected keyword?
  - Guideline routing accuracy: did the hits actually come from the expected
    source document, given multiple guidelines now share the same store?
  - Confidence calibration: whether RELEVANCE_DISTANCE_THRESHOLD in graph.py
    actually separates true matches from true negatives, using a set of
    deliberately out-of-domain queries the system should refuse to answer
    confidently (see the Clinical Safety fallback in evaluate_node).

This is a manual, ad hoc evaluation tool, not an automated test suite —
run it after re-ingesting guidelines or changing the embedding model/chunking
to sanity-check retrieval quality, and use its distance numbers to recalibrate
RELEVANCE_DISTANCE_THRESHOLD.

Usage:
    python scripts/eval_retrieval.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import AsyncSessionLocal
from app.services.vector_search import search_guideline_chunks
from app.services.graph import RELEVANCE_DISTANCE_THRESHOLD

# Each case represents a realistic rag_node query for a patient profile.
# `expect_keywords`: hit if ANY appear (case-insensitive) in ANY retrieved chunk.
# `expect_guideline_contains`: if set, the chunk(s) containing a keyword hit
#   must also come from a guideline title containing this substring —
#   catches cross-document routing errors (e.g. hypertension case pulling
#   cardiovascular-guideline chunks that happen to share vocabulary).
RETRIEVAL_CASES = [
    {
        "name": "Stage 2 hypertension + diabetes",
        "query": "cardiovascular risk hypertension blood pressure 162/98 mmHg diabetes",
        "expect_keywords": ["140", "160", "antihypertensive", "blood pressure"],
        "expect_guideline_contains": "hypertension",
    },
    {
        "name": "Elevated BP, no comorbidities",
        "query": "cardiovascular risk hypertension blood pressure 135/85 mmHg",
        "expect_keywords": ["130", "lifestyle", "elevated", "blood pressure"],
        "expect_guideline_contains": "hypertension",
    },
    {
        "name": "General CVD risk stratification",
        "query": "cardiovascular risk assessment total risk approach prevention",
        "expect_keywords": ["risk", "cardiovascular", "prevention"],
        "expect_guideline_contains": "cardiovascular",
    },
    {
        "name": "Smoking + lipid risk factors",
        "query": "cardiovascular risk tobacco smoking cholesterol lipid lowering",
        "expect_keywords": ["tobacco", "smoking", "lipid", "cholesterol"],
        "expect_guideline_contains": "cardiovascular",
    },
    {
        "name": "Diabetes as a CVD risk multiplier",
        "query": "cardiovascular risk diabetes mellitus multiple risk factor intervention",
        "expect_keywords": ["diabetes", "risk factor"],
        "expect_guideline_contains": None,  # legitimately could come from either doc
    },
]

# Queries the system should NOT treat as confidently answerable — used to
# calibrate RELEVANCE_DISTANCE_THRESHOLD against true negatives.
OUT_OF_DOMAIN_CASES = [
    {"name": "Unrelated: pediatric asthma dosing", "query": "pediatric asthma inhaler corticosteroid dosing schedule"},
    {"name": "Unrelated: cooking recipe", "query": "how to bake a chocolate cake from scratch"},
    {"name": "Unrelated: software licensing", "query": "open source software license compliance requirements"},
]


def contains_any(text: str, keywords) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


async def run_case(db, case, k=3):
    chunks = await search_guideline_chunks(db, case["query"], limit=k)
    hit_chunks = [c for c in chunks if contains_any(c["content"], case["expect_keywords"])]
    hit = bool(hit_chunks)

    routing_ok = True
    if hit and case.get("expect_guideline_contains"):
        expected = case["expect_guideline_contains"].lower()
        routing_ok = any(expected in (c["guideline_title"] or "").lower() for c in hit_chunks)

    min_distance = min((c["distance"] for c in chunks), default=None)
    evidence_sufficient = min_distance is not None and min_distance <= RELEVANCE_DISTANCE_THRESHOLD

    return {
        "name": case["name"],
        "hit": hit,
        "routing_ok": routing_ok,
        "min_distance": min_distance,
        "evidence_sufficient": evidence_sufficient,
        "top_sources": [(c["guideline_title"], round(c["distance"], 4)) for c in chunks],
    }


async def run_negative_case(db, case, k=3):
    chunks = await search_guideline_chunks(db, case["query"], limit=k)
    min_distance = min((c["distance"] for c in chunks), default=None)
    evidence_sufficient = min_distance is not None and min_distance <= RELEVANCE_DISTANCE_THRESHOLD
    return {
        "name": case["name"],
        "min_distance": min_distance,
        # correctly refused = NOT flagged as sufficient evidence
        "correctly_refused": not evidence_sufficient,
        "top_sources": [(c["guideline_title"], round(c["distance"], 4)) for c in chunks],
    }


async def main():
    async with AsyncSessionLocal() as db:
        print("=" * 70)
        print(" RETRIEVAL EVALUATION — in-domain clinical queries")
        print("=" * 70)
        results = [await run_case(db, c) for c in RETRIEVAL_CASES]
        for r in results:
            status = "HIT " if r["hit"] else "MISS"
            routing = "" if r["routing_ok"] else "  [WRONG GUIDELINE]"
            evidence = "sufficient" if r["evidence_sufficient"] else "INSUFFICIENT"
            print(f"[{status}] {r['name']}{routing}")
            print(f"        min_distance={r['min_distance']:.4f}  evidence={evidence}")
            for title, dist in r["top_sources"]:
                print(f"          - {title} (distance={dist})")

        hit_rate = sum(r["hit"] for r in results) / len(results)
        routing_rate = sum(r["routing_ok"] for r in results if r["hit"]) / max(1, sum(r["hit"] for r in results))
        print(f"\nHit rate: {hit_rate:.0%}  |  Correct-guideline routing rate (of hits): {routing_rate:.0%}")

        print("\n" + "=" * 70)
        print(" CONFIDENCE CALIBRATION — out-of-domain queries (should be refused)")
        print("=" * 70)
        neg_results = [await run_negative_case(db, c) for c in OUT_OF_DOMAIN_CASES]
        for r in neg_results:
            status = "REFUSED " if r["correctly_refused"] else "FALSE POSITIVE"
            print(f"[{status}] {r['name']}  min_distance={r['min_distance']:.4f}")
            for title, dist in r["top_sources"]:
                print(f"          - {title} (distance={dist})")

        refusal_rate = sum(r["correctly_refused"] for r in neg_results) / len(neg_results)
        print(f"\nCorrect-refusal rate: {refusal_rate:.0%}")
        print(f"\nCurrent RELEVANCE_DISTANCE_THRESHOLD = {RELEVANCE_DISTANCE_THRESHOLD}")
        print("If hit rate is low, or false positives appear above, adjust this threshold")
        print("in app/services/graph.py using the min_distance values printed here.")


if __name__ == "__main__":
    asyncio.run(main())

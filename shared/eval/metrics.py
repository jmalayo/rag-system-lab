from __future__ import annotations

import re
import statistics

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def is_chunk_correct(chunk: dict, question: dict) -> bool:

    valid_docs = [d.strip() for d in question["source_doc"].split(";")]

    if chunk.get("doc_id") not in valid_docs:
        return False

    chunk_text = _normalize(chunk.get("text", ""))

    return any(_normalize(span) in chunk_text for span in question["gold_spans"])

def recall_at_k(results: dict[str, list[dict]], questions: list[dict], k: int) -> float:
    
    hits = 0

    for q in questions:
        retrieved = results.get(q["id"], [])[:k]

        if any(is_chunk_correct(c, q) for c in retrieved):
            hits += 1

    return hits / len(questions) if questions else 0.0

def mrr(results: dict[str, list[dict]], questions: list[dict], k: int | None = None) -> float:
    
    reciprocal_ranks = []

    for q in questions:

        retrieved = results.get(q["id"], [])

        if k is not None:
            retrieved = retrieved[:k]

        rank = next(
            (i + 1 for i, c in enumerate(retrieved) if is_chunk_correct(c, q)),
            None
        )

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def percentile(values: list[float], p: float) -> float:

    if not values:
        return 0.0

    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))

    return ordered[idx]

def latency_summary(latencies_ms: list[float]) -> dict:

    return {
        "p50_ms": round(percentile(latencies_ms, 50), 1),
        "p95_ms": round(percentile(latencies_ms, 95), 1),
        "mean_ms": round(statistics.mean(latencies_ms), 1) if latencies_ms else 0.0,
    }

def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 42):

    import random

    if not values:
        return (0.0, 0.0)

    rng = random.Random(seed)
    means = []
    n = len(values)

    for _ in range(n_resamples):
        sample = [
            values[rng.randrange(n)] for _ in range(n)
        ]

        means.append(sum(sample) / n)

    means.sort()

    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1

    return (round(means[lo_idx], 4), round(means[hi_idx], 4))

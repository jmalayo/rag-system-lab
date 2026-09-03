import json
import time
from pathlib import Path

import logging

import numpy as np
import requests

from shared.eval.data import load_questions
from shared.eval.metrics import bootstrap_ci, is_chunk_correct, latency_summary, mrr, recall_at_k
from shared.ingest import (
    build_qdrant_client,
    chunk_documents,
    corpus_hash,
    index_chunks,
    load_corpus,
    qualified_collection_name
)
from shared.retrieval import dense_search
from shared.settings import settings
from shared.tracking import log_metrics, tracked_run

OVERLAP_FRACS = [0.0, 0.10, 0.25]
SAFETY_MARGIN = 0.85
K_MAX = 10
COLLECTION = qualified_collection_name("exp_chunking_dynamic")

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent

def run_config(docs, questions, chunk_size: int, overlap_frac: int):

    overlap = int(chunk_size * overlap_frac)

    client = build_qdrant_client()
    chunks = chunk_documents(docs, chunk_size, overlap)
    n_indexed = index_chunks(
        client,
        chunks,
        COLLECTION,
        {
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "corpus_hash": corpus_hash(docs),
            "experiment": "exp_chunking",
        },
    )

    results = {}
    latencies = []

    for q in questions:
        t0 = time.perf_counter()

        results[q["id"]] = dense_search(client, COLLECTION, q["question"], K_MAX)

        latencies.append((time.perf_counter() - t0) * 1000) # ms

    per_question_hit5 = [
        1.0 if any(is_chunk_correct(c, q) 
            for c in results[q["id"]][:5]) 
                else 0.0 
                    for q in questions
    ]

    ci_low, ci_high = bootstrap_ci(per_question_hit5)

    row = {
        "chunk_size": chunk_size,
        "overlap_frac": overlap_frac,
        "overlap_tokens": overlap,
        "n_chunks": n_indexed,
        "recall@1": round(recall_at_k(results, questions, 1), 4),
        "recall@3": round(recall_at_k(results, questions, 3), 4),
        "recall@5": round(recall_at_k(results, questions, 5), 4),
        "recall@5_ci95": [ci_low, ci_high],
        "recall@10": round(recall_at_k(results, questions, 10), 4),
        "mrr@10": round(mrr(results, questions, 10), 4),
        **latency_summary(latencies),
    }

    return row

def count_tokens(text: str) -> int:

    resp = requests.post(f"{settings.text_embedder_url}/tokenize", json={"inputs": text})
    resp.raise_for_status()

    return len(resp.json()[0])

def evaluate_chunks_limits(sample_text: str) -> list[int]:

    info_embedder = requests.get(f"{settings.text_embedder_url}/info").json()
    cap_tokens = info_embedder["max_input_length"] * SAFETY_MARGIN

    sizes = []
    exponent = 7

    while True:
        candidate = 2 ** exponent

        chunks_tokens = np.array([
            count_tokens(sample_text[i:i+candidate])
                for i in range(0, len(sample_text), candidate)
        ])

        p95_tokens = np.percentile(chunks_tokens, 95)

        if candidate >= len(sample_text) or p95_tokens > cap_tokens:
            break

        sizes.append(candidate)

        exponent += 1

    lo_candidate, hi_candidate = (sizes[-1] if sizes else 0), candidate

    while hi_candidate - lo_candidate > 1:
        mid = (lo_candidate + hi_candidate) // 2

        chunks_tokens = np.array([
            count_tokens(sample_text[i:i+mid])
                for i in range(0, len(sample_text), mid)
        ])

        p95_tokens = np.percentile(chunks_tokens, 95)

        if p95_tokens <= cap_tokens:
            lo_candidate = mid
        else:
            hi_candidate = mid

    if lo_candidate and lo_candidate not in sizes:
        sizes.append(lo_candidate)

    return sizes

def main():

    docs = load_corpus()
    questions = load_questions()

    chunk_sizes = evaluate_chunks_limits("".join(d.text for d in docs))

    logger.info(f"chunk sizes dinámicos: {chunk_sizes}")

    rows = []

    for chunk_size in chunk_sizes:

        for overlap_frac in OVERLAP_FRACS:

            with tracked_run(
                "chunking_dynamic",
                f"cs{chunk_size}_ov{int(overlap_frac*100)}",
                {
                    "chunk_size": chunk_size, 
                    "overlap_frac": overlap_frac
                }

            ) as run:

                logger.info(f"Running config: chunk_size={chunk_size} overlap_frac={overlap_frac}")

                row = run_config(docs, questions, chunk_size, overlap_frac)

                log_metrics(row)

                rows.append(row)

                print(
                    f"chunk_size={chunk_size:>4} overlap={overlap_frac:.0%} "
                    f"-> recall@5={row['recall@5']:.3f} mrr@10={row['mrr@10']:.3f} boostrap_ci={row['recall@5_ci95']}"
                    f"p50={row['p50_ms']}ms"
                )

    rows.sort(key=lambda r: (r["recall@5"], r["mrr@10"]), reverse=True)
    best = rows[0]

    (OUT_DIR / "results" / "best_config.json").write_text(
        json.dumps(
            {
                "chunk_size": best["chunk_size"], 
                "chunk_overlap": best["overlap_tokens"]
            }, indent=2
        ),
        encoding="utf-8",
    )

    print(f"\nMejor config: chunk_size={best['chunk_size']} overlap={best['overlap_frac']:.0%} "
          f"(recall@5={best['recall@5']:.3f}) -> guardado en 01-chunking/best_config.json")

if __name__ == "__main__":
    main()

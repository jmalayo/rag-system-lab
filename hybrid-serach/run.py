import json
import sys
import time
from pathlib import Path

import logging

from shared.eval.data import load_questions
from shared.eval.metrics import bootstrap_ci, is_chunk_correct, latency_summary, mrr, recall_at_k
from shared.ingest import (
    build_qdrant_client,
    chunk_documents,
    corpus_hash,
    get_current_config,
    index_chunks,
    load_corpus,
    qualified_collection_name,
)
from shared.retrieval import BM25Index, dense_search, fetch_all_chunks, reciprocal_rank_fusion
from shared.tracking import log_metrics, tracked_run

K_MAX = 10
COLLECTION = qualified_collection_name("exp_hybrid_search")
OUT_DIR = Path(__file__).resolve().parent

BEST_CHUNKING = json.loads(
    (OUT_DIR.parent / "chunking" / "best_config.json").read_text()
)

logger = logging.getLogger(__name__)

def evaluate(name: str, results: dict, questions: list, latencies: list) -> dict:

    per_q_hit5 = [
        1.0 if any(is_chunk_correct(c, q) for c in results[q["id"]][:5]) else 0.0
        for q in questions
    ]
    
    ci_low, ci_high = bootstrap_ci(per_q_hit5)
    return {
        "method": name,
        "recall@1": round(recall_at_k(results, questions, 1), 4),
        "recall@3": round(recall_at_k(results, questions, 3), 4),
        "recall@5": round(recall_at_k(results, questions, 5), 4),
        "recall@5_ci95": [ci_low, ci_high],
        "recall@10": round(recall_at_k(results, questions, 10), 4),
        "mrr@10": round(mrr(results, questions, 10), 4),
        **latency_summary(latencies),
    }

def main():

    docs = load_corpus()
    questions = load_questions()

    client = build_qdrant_client()

    expected_config = {
        **BEST_CHUNKING,
        "corpus_hash": corpus_hash(docs),
    }

    current = get_current_config(client, COLLECTION)

    config_matches = current is not None and all(
        current.get(k) == v
            for k, v in expected_config.items()
    )

    if config_matches:
        logger.info(f"Reusing existing collection {COLLECTION} (config y corpus sin cambios)")

    else:

        if not BEST_CHUNKING:
            raise ValueError("BEST_CHUNKING is not defined; please run 01-chunking first")

        logger.info(f"Indexing chunks with config: {expected_config}")

        chunks = chunk_documents(docs, **BEST_CHUNKING)

        index_chunks(
            client,
            chunks,
            COLLECTION,
            {
                **expected_config,
                "experiment": "exp_chunking"
            }
        )

    bm25 = BM25Index(
        fetch_all_chunks(client, COLLECTION)
    )

    dense_results, bm25_results, hybrid_results = {}, {}, {}
    dense_lat, bm25_lat, hybrid_lat = [], [], []

    for q in questions:

        logger.info(f"Evaluating question {q['id']}")

        logger.info(f"Dense search for question {q['id']}")
        t0 = time.perf_counter()

        d = dense_search(client, COLLECTION, q["question"], K_MAX)

        dense_lat.append((time.perf_counter() - t0) * 1000)

        dense_results[q["id"]] = d

        logger.info(f"BM25 search for question {q['id']}")
        t0 = time.perf_counter()

        b = bm25.search(q["question"], K_MAX)

        bm25_lat.append((time.perf_counter() - t0) * 1000)

        bm25_results[q["id"]] = b

        logger.info(f"Reciprocal rank fusion for question {q['id']}")

        t0 = time.perf_counter()

        h = reciprocal_rank_fusion([d, b], top_k=K_MAX)

        hybrid_lat.append((time.perf_counter() - t0) * 1000 + dense_lat[-1] + bm25_lat[-1])
        
        hybrid_results[q["id"]] = h

    rows = [
        evaluate("dense", dense_results, questions, dense_lat),
        evaluate("bm25", bm25_results, questions, bm25_lat),
        evaluate("hybrid_rrf", hybrid_results, questions, hybrid_lat),
    ]

    with tracked_run("02-hybrid-search", "sweep", BEST_CHUNKING):
        for row in rows:
            log_metrics({f"{row['method']}_{k}": v for k, v in row.items()})

    rows.sort(key=lambda r: r["recall@5"], reverse=True)
    best = rows[0]

    (OUT_DIR / "best_method.json").write_text(json.dumps({"method": best["method"]}, indent=2))

    dense_row = next(r for r in rows if r["method"] == "dense")
    delta = best["recall@5"] - dense_row["recall@5"]

    adopted = (
        best["method"] 
            if (best["method"] != "dense" or delta > 0) 
            else "dense"
    )

    results = {
        "chunking_config": BEST_CHUNKING,
        "n_questions": len(questions),
        "rows": rows,
        "best_method": best["method"],
        "dense_recall@5": dense_row["recall@5"],
        "delta_recall@5": round(delta, 4),
        "latency_delta_p50_ms": round(best["p50_ms"] - dense_row["p50_ms"], 1),
        "adopted_method": adopted,
        "negative_finding": adopted == "dense" and best["method"] != "dense",
    }
    
    (OUT_DIR / "results.json").write_text(
        json.dumps(
            results, 
            indent=2, 
            ensure_ascii=False), 
            encoding="utf-8"
    )

    for r in rows:
        print(f"{r['method']:>10}: recall@5={r['recall@5']:.3f} mrr@10={r['mrr@10']:.3f} "
              f"p50={r['p50_ms']}ms")

    print(f"\nMejor método: {best['method']} -> guardado en 02-hybrid-search/best_method.json")


if __name__ == "__main__":
    main()

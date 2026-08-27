import json
import time
from pathlib import Path

from shared.eval.data import load_questions
from shared.eval.metrics import bootstrap_ci, groundedness_rate, hallucination_rate, latency_summary
from shared.ingest import build_qdrant_client, chunk_documents, index_chunks, load_corpus
from shared.llm import generate
from shared.retrieval import BM25Index, dense_search, fetch_all_chunks, reciprocal_rank_fusion, rerank
from shared.tracking import get_best_run, log_metrics, tracked_run

from experiments.evaluation.prompts import ANSWER_PROMPT, GROUNDEDNESS_PROMPT, RELEVANCE_PROMPT

from experiments.reranking.run import base_search
from qdrant_client import QdrantClient

TOP_K_CONTEXT = 5
CANDIDATE_POOL = 20
COLLECTION = "evaluation"
OUT_DIR = Path(__file__).resolve().parent

_best_chunking_run = get_best_run("chunking")

BEST_CHUNKING = {
    "chunk_size": int(_best_chunking_run["metrics.chunk_size"]),
    "chunk_overlap": int(_best_chunking_run["metrics.overlap_tokens"]),
}
BEST_METHOD = get_best_run("hybrid-search")["params.method"]
USE_RERANKER = get_best_run("reranking")["params.use_reranker"] == "True"

def retrieve_context(client, bm25, question: str) -> list[dict]:

    candidates = base_search(client, bm25, question, CANDIDATE_POOL)

    if USE_RERANKER:
        return rerank(question, candidates, TOP_K_CONTEXT)

    return candidates[:TOP_K_CONTEXT]

def judge_yes(prompt: str) -> bool:
    reply = generate(prompt, max_new_tokens=5).strip().upper()
    return reply.startswith("S")


def main():
    docs = load_corpus()
    questions = load_questions()

    client = build_qdrant_client()

    chunks = chunk_documents(docs, BEST_CHUNKING["chunk_size"], BEST_CHUNKING["chunk_overlap"])

    index_chunks(client, chunks, COLLECTION, BEST_CHUNKING)

    bm25 = BM25Index(
        fetch_all_chunks(client, COLLECTION)
    )

    grounded_verdicts, relevant_verdicts, gen_latencies = [], [], []
    per_question_rows = []

    for i, q in enumerate(questions, 1):

        context_chunks = retrieve_context(client, bm25, q["question"])
        
        context_text = "\n\n".join(
            f"[{c['doc_id']}] {c['text']}"  
                for c in context_chunks
        )

        t0 = time.perf_counter()

        answer = generate(
            ANSWER_PROMPT.format(
                context=context_text, 
                question=q["question"]
            )
        )
        
        gen_latencies.append((time.perf_counter() - t0) * 1000)

        grounded = judge_yes(GROUNDEDNESS_PROMPT.format(context=context_text, answer=answer))
        relevant = judge_yes(RELEVANCE_PROMPT.format(question=q["question"], answer=answer))

        grounded_verdicts.append(grounded)
        relevant_verdicts.append(relevant)
        per_question_rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": answer,
                "grounded": grounded,
                "relevant": relevant,
            }
        )
        print(f"[{i}/{len(questions)}] {q['id']} grounded={grounded} relevant={relevant}")

    g_rate = groundedness_rate(grounded_verdicts)
    h_rate = hallucination_rate(grounded_verdicts)
    r_rate = groundedness_rate(relevant_verdicts)  # mismo cálculo, distinto verdict list
    g_ci = bootstrap_ci([1.0 if v else 0.0 for v in grounded_verdicts])
    r_ci = bootstrap_ci([1.0 if v else 0.0 for v in relevant_verdicts])
    lat = latency_summary(gen_latencies)

    with tracked_run(
        "evaluation",
        "groundedness",
        {
            "base_method": BEST_METHOD,
            "use_reranker": USE_RERANKER,
            "llm_backend": __import__("shared.settings", fromlist=["settings"]).settings.llm_backend,
            **BEST_CHUNKING,
        },
    ):
        log_metrics(
            {
                "groundedness_rate": g_rate,
                "hallucination_rate": h_rate,
                "answer_relevance_rate": r_rate,
                **lat,
            }
        )

    (OUT_DIR / "per_question_results.json").write_text(
        json.dumps(per_question_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = {
        "chunking_config": BEST_CHUNKING,
        "base_method": BEST_METHOD,
        "use_reranker": USE_RERANKER,
        "n_questions": len(questions),
        "groundedness_rate": round(g_rate, 4),
        "groundedness_rate_ci95": [round(g_ci[0], 4), round(g_ci[1], 4)],
        "hallucination_rate": round(h_rate, 4),
        "answer_relevance_rate": round(r_rate, 4),
        "answer_relevance_rate_ci95": [round(r_ci[0], 4), round(r_ci[1], 4)],
        **lat,
    }

    (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results" / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\ngroundedness_rate={g_rate:.3f} hallucination_rate={h_rate:.3f} "
        f"answer_relevance_rate={r_rate:.3f} -> guardado en evaluation/results/results.json"
    )


if __name__ == "__main__":
    main()

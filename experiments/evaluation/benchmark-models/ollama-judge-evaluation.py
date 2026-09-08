import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from shared.eval.data import load_questions
from shared.eval.metrics import latency_summary
from shared.ingest import build_qdrant_client, chunk_documents, index_chunks, load_corpus
from shared.llm import generate
from shared.retrieval import BM25Index, fetch_all_chunks, rerank
from shared.settings import settings
from shared.tracking import get_best_run

from experiments.evaluation.prompts import ANSWER_PROMPT, GROUNDEDNESS_PROMPT, RELEVANCE_PROMPT
from experiments.evaluation.run import BEST_CHUNKING
from experiments.reranking.run import base_search

OUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"

COLLECTION = "exp_judge_benchmark"
POOL_SIZE = 20
TOP_K = 5

BEST_METHOD = get_best_run("hybrid-search")["params.method"]
USE_RERANKER = get_best_run("reranking")["params.use_reranker"] == "True"

BASE_MODEL = settings.llm_model_base
JUDGE_MODEL = settings.llm_model_judge

JUDGES = {
    "deepseek": JUDGE_MODEL,
    "llama": BASE_MODEL
}

ANSWER_MODEL = BASE_MODEL

VERDICT_RE = re.compile(r"TRUE|FALSE", re.IGNORECASE)

def parse_verdict(raw: str) -> tuple[bool, bool]:

    match = VERDICT_RE.search(raw)

    if not match:
        print(f"  [!] Veredicto no reconocido, se toma como FALSE: {raw[:80]!r}", flush=True)
        return False, False

    return match.group().upper() == "TRUE", True


def retrieve_context(client, bm25: BM25Index, question: str) -> list[dict]:

    candidates = base_search(client, bm25, question, collection=COLLECTION, k=POOL_SIZE)

    if USE_RERANKER:
        return rerank(question, candidates, TOP_K)

    return candidates[:TOP_K]

def build_answers(client, bm25, questions: list[dict], answers_path: Path) -> list[dict]:

    rows = []

    for i, q in enumerate(questions, start=1):

        context_chunks = retrieve_context(client, bm25, q["question"])
        context_text = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in context_chunks)

        t0 = time.perf_counter()
        answer = generate(
            ANSWER_PROMPT.format(context=context_text, question=q["question"]),
            model=ANSWER_MODEL,
            num_predict=1500,
            repeat_penalty=1.3
        )
        elapsed = time.perf_counter() - t0

        print(f"  [{i}/{len(questions)}] {q['id']} ({ANSWER_MODEL}): {elapsed:.1f}s", flush=True)

        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "context": context_text,
                "answer": answer,
            }
        )

        pd.DataFrame(rows).to_csv(
            answers_path, 
            index=False,
            encoding="utf-8",
            sep=";",
        )

    return rows

def judge_answers(rows: list[dict], judge_model: str) -> tuple[list[dict], list[float]]:

    verdicts, latencies = [], []

    for i, row in enumerate(rows, start=1):

        t0 = time.perf_counter()

        grounded_raw = generate(
            GROUNDEDNESS_PROMPT.format(context=row["context"], answer=row["answer"]),
            model=judge_model,
            num_predict=3000,
            repeat_penalty=1.3,
        )
        relevant_raw = generate(
            RELEVANCE_PROMPT.format(question=row["question"], answer=row["answer"]),
            model=judge_model,
            num_predict=3000,
            repeat_penalty=1.3,
        )

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000)

        print(f"  [{i}/{len(rows)}] {row['id']} ({judge_model}): {elapsed:.1f}s", flush=True)

        grounded_verdict, grounded_valid = parse_verdict(grounded_raw)
        relevant_verdict, relevant_valid = parse_verdict(relevant_raw)

        verdicts.append(
            {
                "id": row["id"],
                "grounded": grounded_verdict,
                "relevant": relevant_verdict,
                "asw_valido": grounded_valid and relevant_valid,
            }
        )

    return verdicts, latencies


def summarize(judge_name: str, judge_model: str, verdicts: list[dict], latencies: list[float]) -> dict:

    grounded = [v["grounded"] for v in verdicts]
    relevant = [v["relevant"] for v in verdicts]
    g_rate = sum(grounded) / len(grounded) if grounded else 0.0
    r_rate = sum(relevant) / len(relevant) if relevant else 0.0

    mode = "self-judging" if judge_model == ANSWER_MODEL else "cross-judging"

    return {
        "judge": judge_name,
        "judge_model": judge_model,
        "mode": mode,
        "groundedness_rate": round(g_rate, 4),
        "hallucination_rate": round(1 - g_rate, 4),
        "answer_relevance_rate": round(r_rate, 4),
        **latency_summary(latencies),
    }


def agreement_rate(verdicts_a: list[dict], verdicts_b: list[dict], key: str) -> float:

    if not verdicts_a:
        return 0.0

    matches = 0

    for a, b in zip(verdicts_a, verdicts_b):
        if a[key] == b[key]:
            matches += 1

    return round(matches / len(verdicts_a), 4)

def main():

    questions = load_questions()

    exp = "exp_6"

    answers_path = RESULTS_DIR / f"base_answers_{exp}.csv"

    reuse_answers = False

    if reuse_answers:
        print(f"Reusando respuestas base cacheadas en {answers_path}", flush=True)

        rows = pd.read_csv(answers_path, encoding="utf-8", sep=";").to_dict("records")

    else:
        docs = load_corpus()

        client = build_qdrant_client()
        chunks = chunk_documents(docs, BEST_CHUNKING["chunk_size"], BEST_CHUNKING["chunk_overlap"])

        index_chunks(client, chunks, COLLECTION, BEST_CHUNKING)

        bm25 = BM25Index(
            fetch_all_chunks(client, COLLECTION)
        )

        answers_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Generando {len(questions)} respuestas base con {ANSWER_MODEL}...", flush=True)
        
        rows = build_answers(client, bm25, questions, answers_path)

    summaries = []
    verdicts_by_judge = {}

    for judge_name, judge_model in JUDGES.items():

        print(f"Juzgando respuestas con {judge_name} ({judge_model})...", flush=True)

        verdicts, latencies = judge_answers(rows, judge_model)

        verdicts_by_judge[judge_name] = verdicts

        summaries.append(summarize(judge_name, judge_model, verdicts, latencies))

    agreement = {
        "groundedness": agreement_rate(verdicts_by_judge["llama"], verdicts_by_judge["deepseek"], "grounded"),
        "relevance": agreement_rate(verdicts_by_judge["llama"], verdicts_by_judge["deepseek"], "relevant"),
    }

    comparison_df = pd.DataFrame(rows)[["id", "question", "answer"]]

    for judge_name, verdicts in verdicts_by_judge.items():

        judge_df = pd.DataFrame(verdicts).rename(
            columns={
                "grounded": f"grounded_{judge_name}",
                "relevant": f"relevant_{judge_name}",
                "asw_valido": f"{judge_name}_asw_valido",
            }
        )

        comparison_df = comparison_df.merge(judge_df, on="id")

    comparison_df.to_csv(
        RESULTS_DIR / f"per_question_judge_verdicts_{exp}.csv", 
        index=False,
        encoding="utf-8",
        sep=";",
    )

    summary_df = pd.DataFrame(summaries)
    summary_df.insert(0, "base_model", ANSWER_MODEL)
    summary_df.insert(1, "n_questions", len(questions))

    summary_df.to_csv(
        RESULTS_DIR / f"results_summary_{exp}.csv", 
        index=False, 
        encoding="utf-8", 
        sep=";",
    )

    header = f"{'Modo':<16}{'Modelo juez':<22}{'Grounded':>10}{'Halluc.':>10}{'Relevant':>10}{'p50 ms':>10}"

    print("\n" + header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s['mode']:<16}{s['judge_model']:<22}{s['groundedness_rate']:>10.1%}"
            f"{s['hallucination_rate']:>10.1%}{s['answer_relevance_rate']:>10.1%}{s['p50_ms']:>10.1f}"
        )

    print("-" * len(header))
    print(
        f"Acuerdo entre jueces (mismo set de respuestas) -> "
        f"groundedness: {agreement['groundedness']:.1%} | relevance: {agreement['relevance']:.1%}"
    )

if __name__ == "__main__":
    main()

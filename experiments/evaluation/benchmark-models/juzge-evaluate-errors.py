import re
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluation.prompts import GROUNDEDNESS_PROMPT, RELEVANCE_PROMPT
from shared.settings import settings

RESULTS_DIR = Path(__file__).resolve().parent / "results"

BASE_MODEL = settings.llm_model_base
JUDGE_MODEL = settings.llm_model_judge

JUDGES = {
    "deepseek": JUDGE_MODEL,
    "llama": BASE_MODEL,
}

NUM_PREDICT = 3000
REPEAT_PENALTY = 1.3

VERDICT_RE = re.compile(r"TRUE|FALSE", re.IGNORECASE)

def generate(prompt: str, model: str) -> dict:

    resp = requests.post(
        f"{settings.llm_host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
                "num_thread": 8,
                "num_predict": NUM_PREDICT,
                "repeat_penalty": REPEAT_PENALTY,
            },
        },
        timeout=180,
    )

    resp.raise_for_status()

    return resp.json()

def is_valid(raw: str) -> bool:
    return bool(VERDICT_RE.search(raw or ""))

def diagnose(answers_by_id: dict, judge_model: str) -> dict:

    grounded = generate(
        GROUNDEDNESS_PROMPT.format(context=answers_by_id["context"], answer=answers_by_id["answer"]),
        judge_model,
    )
    relevant = generate(
        RELEVANCE_PROMPT.format(question=answers_by_id["question"], answer=answers_by_id["answer"]),
        judge_model,
    )

    grounded_response = grounded.get("response") or ""
    relevant_response = relevant.get("response") or ""

    return {
        "question": answers_by_id["id"],
        "grounded_fails": "valido" if is_valid(grounded_response) else f"({grounded.get('done_reason')})", 
        "relevant_fails": "valido" if is_valid(relevant_response) else f"({relevant.get('done_reason')})",
        "thinking_grounded_chars": f"{len(grounded.get('thinking') or '')}",
        "thinking_relevant_chars": f"{len(relevant.get('thinking') or '')}",
    }

def main():

    run_dir = RESULTS_DIR / "exp_5" / "llama_base"
    judge_name = "deepseek"

    answers_path = run_dir / "base_answers.csv"
    verdicts_path = run_dir / "per_question_judge_verdicts.csv"

    answers_by_id = {
        row["id"]: row
            for row in pd.read_csv(
                answers_path,
                encoding="utf-8",
                sep=";"
            ).to_dict("records")
    }

    verdicts = pd.read_csv(verdicts_path, encoding="utf-8", sep=";")

    valid_col = f"{judge_name}_asw_valido"

    failed_ids = verdicts[verdicts[valid_col] == False]["id"].tolist()

    if not failed_ids:
        print(f"Ninguna fila inválida para {judge_name} en {run_dir}.")

        return

    print(f"Reproduciendo {len(failed_ids)} preguntas de {judge_name} ({JUDGES[judge_name]}) en {run_dir}: {failed_ids}\n")

    rows_report = []

    for i, qid in enumerate(failed_ids, start=1):

        print(f"[{i} / {len(failed_ids)}] procesando pregunta {qid}", flush=True)

        rows_report.append(
            diagnose(answers_by_id[qid], JUDGES[judge_name])
        )

    report_df = pd.DataFrame(rows_report)

    print("\n" + report_df.to_string(index=False))

    report_df.to_csv(
        run_dir / f"judge_errors_{judge_name}_{NUM_PREDICT}.csv",
        index=False,
        encoding="utf-8",
        sep=";",
    )

if __name__ == "__main__":
    main()

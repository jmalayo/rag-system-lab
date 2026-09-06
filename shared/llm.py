from __future__ import annotations

import requests
from shared.settings import settings

import pandas as pd
from transformers import AutoTokenizer

from shared.eval.data import load_questions

def generate(
    prompt: str,
    model: str | None = None,
    num_predict: int = 500,
    repeat_penalty: float = 1.0,
    stop: list[str] | None = None,
) -> str:

    if settings.llm_backend is None:
        raise ValueError("LLM backend is not set, please set RAG_LLM_BACKEND in your environment.")

    options = {
        "temperature": 0.0,
        "num_ctx": 4096,
        "num_thread": 8,
        "num_predict": num_predict
    }

    if stop:
        options["stop"] = stop

    if repeat_penalty:
        options["repeat_penalty"] = repeat_penalty

    resp = requests.post(
        f"{settings.llm_host}/api/generate",
        json={
            "model": model or settings.llm_model_base,
            "prompt": prompt,
            "stream": False,
            "options": options,
        },
        timeout=120,
    )
    
    resp.raise_for_status()

    return resp.json()["response"].strip()

def get_length_questions():

    questions = load_questions()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    
    rows = []

    for q in questions:

        rows.append({
            "id": q["id"],
            "question_tokens": len(tokenizer.encode(q["question"])),
            "expected_answer_tokens": len(tokenizer.encode(q["expected_answer"]))
        })

    df = pd.DataFrame(rows)

    print(f"Máximo número de tokens - Questions: {df['question_tokens'].max()}")
    print(f"Máximo número de tokens - Expected Answer: {df['expected_answer_tokens'].max()}")
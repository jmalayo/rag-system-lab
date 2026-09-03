from __future__ import annotations

import requests
from shared.settings import settings

import pandas as pd
from transformers import AutoTokenizer

from shared.eval.data import load_questions

def generate(prompt: str, max_new_tokens: int = 256) -> str:

    if settings.llm_backend is None:
        raise ValueError("LLM backend is not set, please set RAG_LLM_BACKEND in your environment.")

    resp = requests.post(
        f"{settings.llm_host}/api/generate",
        json={
            "model": settings.llm_model, 
            "prompt": prompt, 
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_thread": 8,
                "num_predict": max_new_tokens,
            }    
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

    print(df)
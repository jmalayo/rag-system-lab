from shared.eval.data import load_questions
from shared.eval.metrics import is_chunk_correct
from shared.ingest import build_qdrant_client, qualified_collection_name
from shared.retrieval import dense_search

import pandas as pd

from pathlib import Path

BASELINE_COLLECTION = qualified_collection_name("exp_chunking_dynamic")
TABLES_COLLECTION = qualified_collection_name("exp_chunking_dynamic_tables")

SCRIPT_DIR = Path(__file__).resolve().parent

def validate_payload(client):

    records, _ = client.scroll(
        collection_name=TABLES_COLLECTION,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    with_tables = [
        r for r in records 
            if r.payload.get("tables")
    ]

    evidence_tables: list[dict] = []

    for rw in with_tables:

        evidence = {
            "chunk_id": rw.payload["chunk_id"],
            "doc_id": rw.payload["doc_id"],
            "text": rw.payload["text"],
            "tables_lenght": len(rw.payload["tables"]),
            "tables": [tb["text_content"] for tb in rw.payload["tables"]]
        }

        evidence_tables.append(evidence)

    df = pd.DataFrame(evidence_tables).explode("tables", ignore_index=True)

    df.to_csv(
        SCRIPT_DIR / "results" / "tables_payload_exp_chunking.csv", 
        sep=";", 
        encoding="utf-8",
        index=False
    )

    print(df)

def validate_target_questions(client):

    questions = load_questions()
    rows = []

    for q in questions:

        for label, coll in [("baseline", BASELINE_COLLECTION), ("con_tablas", TABLES_COLLECTION)]:
            
            results = dense_search(client, coll, q["question"], 5)

            hit = False
            rank = None
            chunk_id = None
            tables_count = 0

            for i, c in enumerate(results):
                if is_chunk_correct(c, q):
                    hit = True
                    rank = i + 1
                    chunk_id = c.get("chunk_id")
                    tables_count = len(c.get("tables", []))

                    break

            print(f" {label}: hit@5={hit}")

            if hit:
                print(f" -> rank {rank}, chunk_id={chunk_id}, tablas_en_payload={tables_count}")

            rows.append({
                "question_id": q["id"],
                "collection": label,
                "hit": hit,
                "rank": rank,
                "chunk_id": chunk_id,
                "tables_count": tables_count
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        SCRIPT_DIR / "results" / "coll_evaluation_exp_chunking.csv", 
        sep=";", 
        encoding="utf-8",
        index=False
    )

def main():

    qdrant_client = build_qdrant_client()

    # validate_payload(qdrant_client)
    validate_target_questions(qdrant_client)

if __name__ == "__main__":
    main()

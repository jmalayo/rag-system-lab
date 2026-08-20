from __future__ import annotations

import math
import re
from collections import Counter

from qdrant_client import QdrantClient

from shared.ingest import get_embedder

def dense_search(client: QdrantClient, collection_name: str, query: str, k: int) -> list[dict]:
    
    embedder = get_embedder()
    vector = embedder.encode(query, normalize_embeddings=True).tolist()
    hits = client.query_points(
        collection_name=collection_name, 
        query=vector, 
        limit=k
    ).points

    return [
        {
            **h.payload, 
            "score": h.score
        } for h in hits
    ]

def fetch_all_chunks(client: QdrantClient, collection_name: str) -> list[dict]:

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=None,
        limit=1000000,
        with_payload=True
    )

    return [
        p.payload 
            for p in points
    ]

class BM25Index:

    def __init__(
        self, 
        chunks: list[dict], 
        k1: float = 1.5, 
        b: float = 0.75
    ):

        self._chunks = chunks
        self._k1 = k1
        self._b = b

        doc_tokens = [
            self._tokenize(c.get("text", "")) 
                for c in chunks
        ]

        self._doc_len = [
            len(toks) 
                for toks in doc_tokens
        ]

        self._avgdl = sum(self._doc_len) / len(self._doc_len) if doc_tokens else 0.0

        self._term_freqs = [Counter(toks) for toks in doc_tokens]

        df: Counter = Counter()
        for toks in doc_tokens:
            df.update(set(toks))

        n = len(chunks)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _tokenize(text: str) -> list[str]:
        
        return text.lower().split()
        
    def search(self, query: str, k: int) -> list[dict]:

        if not self._chunks:
            return []

        query_terms = self._tokenize(query)
        scores = [0.0] * len(self._chunks)

        for i, (tf, dl) in enumerate(zip(self._term_freqs, self._doc_len)):

            score = 0.0
            for term in query_terms:

                freq = tf.get(term)
                if not freq:
                    continue

                idf = self._idf.get(term, 0.0)
                norm = self._k1 * (1 - self._b + self._b * dl / self._avgdl) if self._avgdl else self._k1
                score += idf * (freq * (self._k1 + 1)) / (freq + norm)

            scores[i] = score

        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {**self._chunks[i], "score": scores[i]}
            for i in ranked
        ]


def reciprocal_rank_fusion(
    rankings: list[list[dict]], 
    k_constant: int = 60, 
    top_k: int | None = None
) -> list[dict]:

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_constant + rank + 1)
            payloads[cid] = chunk

    fused_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    if top_k:
        fused_ids = fused_ids[:top_k]

    return [
        {
            **payloads[cid], 
            "rrf_score": scores[cid]
        } for cid in fused_ids
    ]

from __future__ import annotations

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

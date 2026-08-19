from __future__ import annotations

import re
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from shared.settings import settings

logger = logging.getLogger(__name__)

FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n\n?", re.DOTALL)

@dataclass
class SourceDoc:
    doc_id: str  # p.ej. "qdrant/hybrid-search.md" -- coincide con source_doc en questions.jsonl
    library: str
    text: str

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    library: str
    text: str
    chunk_index: int

def validate_chunk(text, model_tokenizer, max_tokens):

    tokens = model_tokenizer.encode(text)

    if len(tokens) > max_tokens:
        logger.warning(f"Chunk too long: {len(tokens)} tokens")

        return False

    logger.info(f"Chunk is valid: {len(tokens)} tokens")

    return True

def load_corpus(corpus_dir: str | None = None) -> list[SourceDoc]:

    root = Path(corpus_dir or settings.corpus_dir)
    docs = []

    for path in sorted(root.rglob("*.md")):

        raw = path.read_text(encoding="utf-8")
        body = FRONT_MATTER_RE.sub("", raw, count=1)

        library = path.parent.name
        doc_id = f"{library}/{path.name}"
        
        docs.append(
            SourceDoc(
                doc_id=doc_id, 
                library=library, 
                text=body.strip()
            )
        )

    return docs


def chunk_documents(docs: list[SourceDoc], chunk_size: int, chunk_overlap: int) -> list[Chunk]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    embedder = get_embedder()

    chunks: list[Chunk] = []

    for doc in docs:

        pieces = splitter.split_text(doc.text)

        for i, piece in enumerate(pieces):
            
            if not validate_chunk(piece, embedder.tokenizer, embedder.max_seq_length):
                continue

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}#{i}")),
                    doc_id=doc.doc_id,
                    library=doc.library,
                    text=piece,
                    chunk_index=i,
                )
            )
            
    return chunks
    
def get_embedder():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def build_qdrant_client(local_path: str | None = None) -> QdrantClient:

    if settings.qdrant_mode == "server":
        return QdrantClient(
            host=settings.qdrant_host, 
            port=settings.qdrant_port
        )
        
    return (
        QdrantClient(path=local_path) 
            if local_path else QdrantClient(":memory:")
    )


def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    collection_name: str,
    batch_size: int = 64,
) -> int:

    embedder = get_embedder()
    dim = embedder.get_sentence_embedding_dimension()

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dim, 
            distance=models.Distance.COSINE
        ),
    )

    total = 0
    for start in range(0, len(chunks), batch_size):
        
        batch = chunks[start : start + batch_size]
        vectors = embedder.encode(
            [
                c.text for c in batch
            ], 
            normalize_embeddings=True,
            show_progress_bar=True
        )

        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector=vector.tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "library": c.library,
                    "chunk_index": c.chunk_index,
                },
            )
            for c, vector in zip(batch, vectors)
        ]

        client.upsert(
            collection_name=collection_name, 
            points=points
        )

        total += len(points)

    return total


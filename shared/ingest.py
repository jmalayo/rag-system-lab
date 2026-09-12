from __future__ import annotations

import functools
import hashlib
import re
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from transformers import AutoConfig, AutoTokenizer

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
    tables: list[dict] = field(default_factory=list)

def validate_chunk(text, model_tokenizer, max_tokens):

    tokens = model_tokenizer.encode(text, add_special_tokens=False, truncation=True)

    return model_tokenizer.decode(tokens)

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

def corpus_hash(docs: list[SourceDoc]) -> str:

    data = (
        "".join(
                f"{d.doc_id}:{d.text}" 
                    for d in sorted(docs, key=lambda d: d.doc_id)
            )
        )
    
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def chunk_documents(
    docs: list[SourceDoc], 
    tables: list[dict], 
    chunk_size: int, 
    chunk_overlap: int
) -> list[Chunk]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    embedder = get_embedder()

    chunks: list[Chunk] = []

    for doc in docs:

        doc_tables = [
            t for t in tables
                if t["doc_id"] == doc.doc_id
        ]

        marked_text = doc.text

        for idx, t in enumerate(doc_tables):
            marked_text = marked_text.replace(t["text_content"], f"[[TABLE:{idx}]]")
    
        marked_text = re.sub(r"\n+(\[\[TABLE:\d+\]\])", r" \1", marked_text)
        marked_text = re.sub(r"(\[\[TABLE:\d+\]\])\n+", r"\1 ", marked_text)

        splits = splitter.split_text(marked_text)

        pending_tables = []

        for i, split in enumerate(splits):

            matches = re.findall(r"\[\[TABLE:(\d+)\]\]", split)

            tables_idx = []

            for m in matches:
                tables_idx.append(int(m))

            piece_tables = [doc_tables[m] for m in tables_idx]

            text = re.sub(r"\[\[TABLE:\d+\]\]", "", split).strip()

            if not text:
                pending_tables.extend(piece_tables)
                
                continue

            validated_piece = validate_chunk(text, embedder.tokenizer, embedder.max_seq_length)

            chunks.append(
                Chunk(
                    chunk_index=i,
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}#{i}")),
                    doc_id=doc.doc_id,
                    library=doc.library,
                    text=validated_piece,
                    tables=pending_tables + piece_tables,
                )
            )

            pending_tables = []

        if pending_tables:

            doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]

            if doc_chunks:
                doc_chunks[-1].tables.extend(pending_tables)
            else:
                logger.warning(
                    f"{doc.doc_id}: el doc quedo compuesto solo por tablas, "
                    "sin ningun chunk de texto al cual adjuntarlas"
                )

    return chunks
    
class RemoteEmbedder:

    def __init__(self, url: str, model_name: str):
        self._url = url.rstrip("/")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = requests.get(f"{self._url}/info").json()["max_input_length"]
        self._dim = AutoConfig.from_pretrained(model_name).hidden_size

    def get_dimension(self):
        return self._dim

    _MAX_CLIENT_BATCH = 32

    def encode(self, texts, normalize_embeddings=False):

        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)

        embeddings = []

        for start in range(0, len(inputs), self._MAX_CLIENT_BATCH):
            
            batch = inputs[start : start + self._MAX_CLIENT_BATCH]

            resp = requests.post(
                f"{self._url}/embed",
                json={
                    "inputs": batch, 
                    "truncate": True
                }
            )

            resp.raise_for_status()

            embeddings.extend(resp.json())

        vectors = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)

        return vectors[0] if single else vectors


@functools.lru_cache(maxsize=1)
def get_embedder():

    if settings.text_embedder_mode == "server":
        return RemoteEmbedder(
            url=settings.text_embedder_url,
            model_name=f"sentence-transformers/{settings.embedding_model}"
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name_or_path=settings.embedding_model, 
        device="cpu"
    )

def qualified_collection_name(base_name: str) -> str:

    model_tag = settings.embedding_model.replace("/", "--")

    return f"{base_name}__{model_tag}"

def get_current_config(client: QdrantClient, collection_name: str) -> dict | None:

    if not client.collection_exists(collection_name):
        return None

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        return None

    return {
        k: v 
            for k, v in points[0].payload.items() 
                if k not in ["chunk_id", "text", "doc_id", "library", "chunk_index", "tables"]
    }

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
    config: dict,
    batch_size: int = 64,
) -> int:

    embedder = get_embedder()
    dim = embedder.get_dimension()

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
            normalize_embeddings=True
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
                    "tables": c.tables,
                    **config,
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

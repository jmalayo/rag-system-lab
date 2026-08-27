# rag-system-lab

Laboratorio de experimentos para un sistema RAG (Retrieval-Augmented Generation) en español, sobre un corpus técnico propio (`music-tagger`). Cada carpeta es un experimento incremental — chunking, luego hybrid search — que se evalúa contra un set fijo de 34 preguntas con respuestas conocidas (gold spans, `shared/eval/questions.jsonl`), y cada uno documenta su resultado en su propio README.

## Stack

- **Qdrant** — vector store para la búsqueda densa (`docker-compose.yml`, puerto `6333`).
- **Text Embeddings Inference (HuggingFace)** — sirve el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` como servicio HTTP (`docker-compose.yml`, puerto `1010`).
- **MLflow** — tracking de cada corrida de cada experimento (`mlflow/mlruns`).

Ambos servicios corren vía `docker-compose up` y se reutilizan entre experimentos.

### Modelo de embeddings (descarga previa requerida)

El servicio `embedder` corre con `HF_HUB_OFFLINE=1` (no descarga nada en runtime), así que el modelo debe existir en `./embedder_cache/` **antes** de levantar el stack:

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --local-dir ./embedder_cache/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2 \
  --include "*.safetensors" "*.json" "sentencepiece.bpe.model" "tokenizer*" "1_Pooling/*" "2_Dense/*"
```

**Problema:** sin `--include`, el comando descarga *todos* los formatos del repo (pytorch, tensorflow, onnx x6, openvino x2) — **4.4 GB** — cuando TEI solo usa el safetensors + tokenizer. Con el filtro, baja a **~470 MB**.

## Matemática usada en la evaluación

Todo vive en `shared/eval/metrics.py` (métricas) y `shared/retrieval.py` (scoring).

- **Recall@k / MRR@k** — métricas estándar de IR: si el chunk correcto aparece entre los primeros k resultados (recall), y en qué posición (MRR, recíproco del rank).
- **Bootstrap CI** (`bootstrap_ci`) — remuestreo con reemplazo (1000 veces) sobre los aciertos por pregunta para estimar un intervalo de confianza del 95% del recall, sin asumir una distribución normal.
- **BM25 / Okapi BM25** (`BM25Index`) — el término `log(1 + (n - freq + 0.5) / (freq + 0.5))` es el **IDF (Inverse Document Frequency)** de la fórmula Okapi BM25: penaliza términos que aparecen en muchos documentos, con logaritmo para que el efecto se aplane a medida que la frecuencia crece.
- **RRF / Reciprocal Rank Fusion** (`reciprocal_rank_fusion`) — combina rankings distintos sumando `1/(k+rank+1)` por lista, en vez de sumar scores en escalas distintas (ver `experiments/hybrid-search/README.md`).
- **Cross-encoder reranking** (`rerank`) — a diferencia de dense/BM25, que puntúan pregunta y chunk por separado, el cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) procesa el par `(pregunta, chunk)` junto en un solo forward pass y devuelve un score de relevancia directo — más caro por candidato, pero más preciso para reordenar un pool ya reducido.



## Experimentos realizados

- **Experimento 1 (chunking)** — resuelto. Reporte en `experiments/chunking/EXP_CHUNKING_REPORT.md`, resultado del barrido en `experiments/chunking/results/best_config.json`. Mejor config: `chunk_size=512`, `overlap=25%` → `recall@5=0.471`.
- **Experimento 2 (hybrid search)** — resuelto. Reporte en `experiments/hybrid-search/EXP_RRF_REPORT.md`, resultados en `experiments/hybrid-search/results/results.json`. Mejor método: `hybrid_rrf` → `recall@5=0.529` (vs. `dense=0.471`, delta `+0.059`).
- **Experimento 3 (reranking)** — resuelto. Reporte en `experiments/reranking/EXP_RERANK_REPORT.md`, resultados en `experiments/reranking/results/results.json`. Cross-encoder sobre `hybrid_rrf`: `recall@5` 0.529→0.588 (+0.059), `mrr@5` 0.328→0.393 (+0.065), a costa de +957.6ms en p95 — se adopta el reranker.

La carpeta se renombró de `hybrid-serach` a `hybrid-search` (typo corregido) y los resultados de cada experimento ahora viven en su propia subcarpeta `results/` en vez de sueltos junto al `run.py`.

## Estructura

- `shared/` — código común: ingesta/chunking, retrieval (dense, BM25, RRF, rerank), tracking, settings, y el set de evaluación (`shared/eval`).
- `experiments/chunking/` — experimento 1: barrido de `chunk_size`/`overlap`.
- `experiments/hybrid-search/` — experimento 2: dense vs. BM25 vs. fusión híbrida (RRF).
- `experiments/reranking/` — experimento 3: cross-encoder sobre el mejor retriever de la etapa anterior.


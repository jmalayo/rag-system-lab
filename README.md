# rag-system

Laboratorio de experimentos para un sistema RAG (Retrieval-Augmented Generation) en español, sobre un corpus técnico propio (`music-tagger`). Cada carpeta es un experimento incremental — chunking, luego hybrid search — que se evalúa contra un set fijo de preguntas con respuestas conocidas (gold spans), y cada uno documenta su resultado en su propio README.

## Stack

- **Qdrant** — vector store para la búsqueda densa (`docker-compose.yml`, puerto `6333`).
- **Text Embeddings Inference (HuggingFace)** — sirve el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` como servicio HTTP (`docker-compose.yml`, puerto `1010`).
- **MLflow** — tracking de cada corrida de cada experimento (`mlflow/mlruns`).

Ambos servicios corren vía `docker-compose up` y se reutilizan entre experimentos.

## Estructura

- `shared/` — código común: ingesta/chunking, retrieval (dense, BM25, RRF), tracking, settings, y el set de evaluación (`shared/eval`).
- `chunking/` — experimento 1: barrido de `chunk_size`/`overlap`.
- `hybrid-serach/` — experimento 2: dense vs. BM25 vs. fusión híbrida (RRF).

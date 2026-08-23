# Experimento 2 — Hybrid Search (dense + BM25 vía RRF)

| Tracking: los 3 métodos quedaron alojados en MLflow (experimento `rag-system-eval-02-hybrid-search`), un run por método.

Se partió de la mejor config de chunking del experimento anterior (`chunk_size=512`, `overlap=25%`, ver `[chunking/best_config.json](../chunking/best_config.json)`) y se compararon tres métodos de recuperación sobre el mismo set de 34 preguntas con gold spans: búsqueda densa sola, BM25 sola, y la fusión de ambas por Reciprocal Rank Fusion (RRF).

## Resultado

**Mejor método:** `hybrid_rrf` → `recall@5=0.529` (vs. `dense=0.471`, delta `+0.059`)

Guardado en `[best_method.json](./best_method.json)`.

```
hybrid_rrf: recall@5=0.529 mrr@10=0.347 p50=17.0ms
      bm25: recall@5=0.500 mrr@10=0.346 p50=0.4ms
     dense: recall@5=0.471 mrr@10=0.336 p50=16.7ms
```

RRF combinando dense + BM25 supera a cada método por separado. Lo notable es que **BM25 solo queda muy cerca de dense** (0.500 vs 0.471) siendo **~40x más rápido** (p50 0.4ms vs 16.7ms, corre en memoria sin llamar al embedder). El híbrido paga el costo de ambas búsquedas (p50 17.0ms, dominado por el embedding) a cambio de +5.9 puntos de recall@5 sobre dense solo.

## Cómo se evaluó

Con la colección ya indexada (o reusada si el `chunk_size`/`overlap`/corpus no cambiaron), por cada pregunta se corren los tres métodos pidiendo los 10 resultados más cercanos:

- **dense**: embedding de la pregunta + búsqueda por similitud en Qdrant (`dense_search`).
- **bm25**: índice invertido en memoria sobre todos los chunks de la colección (`BM25Index`, k1=1.5, b=0.75), construido on-the-fly en cada corrida, scoreando solo los términos que la query comparte con cada chunk.
- **hybrid_rrf**: fusión de los rankings dense y bm25 por Reciprocal Rank Fusion (`reciprocal_rank_fusion`, k=60).

Un chunk se marca "correcto" igual que en el experimento de chunking: viene del documento fuente correcto **y** contiene el texto exacto del gold span.

## Métricas

Mismas que en `[chunking/README.md](../chunking/README.md)` (recall@k, mrr@k, bootstrap CI, p50/p95 de latencia) — no se repiten acá.

Vale la pena notar los intervalos de confianza al 95% de `recall@5`: dense `[0.294, 0.647]`, bm25 `[0.353, 0.676]`, hybrid_rrf `[0.353, 0.706]` — se solapan bastante entre los tres, esperable con solo 34 preguntas. La diferencia de +5.9 puntos del híbrido sobre dense es la mejor estimación puntual, pero no es una victoria aplastante estadísticamente; el punto más sólido es que BM25 (barato) rinde a la par de dense (caro), no que el híbrido sea claramente superior.

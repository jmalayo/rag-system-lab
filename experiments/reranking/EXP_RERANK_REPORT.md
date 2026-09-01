# Experimento 3 — Reranking (cross-encoder)

| Tracking: los 2 candidatos quedaron alojados en MLflow (experimento `rag-system-eval-reranking`), un run por candidato (`baseline` / `reranked`).

Se partió del mejor chunking (`chunk_size=512`, `overlap=128`, experimento 1) y del mejor retriever (`hybrid_rrf`, experimento 2) para comparar dos variantes sobre el mismo set de 34 preguntas con gold spans: los 5 resultados que entrega `hybrid_rrf` tal cual (**baseline**), contra esos mismos candidatos reordenados por un cross-encoder antes de quedarse con los 5 finales (**reranked**).

## Resultado

**Se adopta el reranker:** `recall@5` sube de `0.529` a `0.588` (delta `+0.059`), `mrr@5` sube de `0.328` a `0.393` (delta `+0.065`), a costa de `+22.7ms` en p95.

```
baseline: recall@5=0.529 mrr@5=0.328 p50=13.4ms p95=23.1ms
reranked: recall@5=0.588 mrr@5=0.393 p50=34.4ms p95=45.8ms
```

Guardado en `[results.json](./results/results.json)`.

El reranker gana en las dos métricas de calidad, no solo en recall: `mrr@5` sube más en términos relativos (+19.7%) que `recall@5` (+11.1%), consistente con lo que se espera de un cross-encoder — no solo mete el chunk correcto en el top 5 con más frecuencia, sino que además lo sube de posición dentro del top 5. El costo es una latencia **~2.6x mayor** en p50 (13.4ms → 34.4ms) y **~2x mayor** en p95 (23.1ms → 45.8ms).

## Cómo se evaluó

Por cada pregunta:

1. **Pool de candidatos**: `hybrid_rrf` (dense + BM25 vía RRF, k=60) trae los 20 candidatos más relevantes (`POOL_SIZE=20`) — el mismo pool alimenta a las dos variantes, así que la comparación aísla el efecto de reordenar, no el de una recuperación distinta.
2. **baseline**: se recortan los primeros 5 candidatos del pool tal cual los entregó `hybrid_rrf` (`TOP_K=5`).
3. **reranked**: un `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) puntúa los 20 pares `(pregunta, texto_del_chunk)` — 2 lotes de `batch_size=16` por pregunta — y se conservan los 5 con mayor score.

Un chunk se marca "correcto" igual que en los experimentos 1 y 2: viene del documento fuente correcto **y** contiene el texto exacto del gold span.

## Métricas

Intervalos de confianza al 95% de `recall@5`: baseline `[0.382, 0.706]`, reranked `[0.412, 0.735]` — se solapan bastante, esperable con solo 34 preguntas; con este volumen de datos no es una victoria estadísticamente aplastante. Lo que sí es consistente: el reranker mejora tanto el punto estimado de `recall@5` como el de `mrr@5` al mismo tiempo, y el rango completo del intervalo se desplaza hacia arriba (el peor caso del reranked, 0.412, ya supera al peor caso del baseline, 0.382).

Sobre la latencia: de los `34.4ms` de p50 en `reranked`, aproximadamente `21.0ms` (34.4 − 13.4) son el cross-encoder puro — la recuperación (`hybrid_rrf`) es prácticamente el mismo costo que en baseline. El cross-encoder domina ~61% del tiempo total por pregunta; es el costo real de la decisión, no ruido de medición.
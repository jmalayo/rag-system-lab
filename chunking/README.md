# Experimento 1 — Chunking

| Tracking: los 9 runs quedaron alojados en MLflow, un run por combinación de `chunk_size`/`overlap`.

Se probaron distintas combinaciones de `chunk_size` (128, 256 y 512) y `overlap` (0%, 10% y 25%). En total, fueron 9 configuraciones evaluadas sobre un conjunto fijo de preguntas con respuestas conocidas (gold spans).

## Resultado

**Mejor config:** `chunk_size=512`**,** `overlap=25%` → `recall@5=0.471`

Guardado en `[best_config.json](./best_config.json)`.

```
chunk_size= 128 overlap=0%  -> recall@5=0.088 mrr@10=0.058
chunk_size= 128 overlap=10% -> recall@5=0.118 mrr@10=0.087
chunk_size= 128 overlap=25% -> recall@5=0.176 mrr@10=0.109
chunk_size= 256 overlap=0%  -> recall@5=0.324 mrr@10=0.239
chunk_size= 256 overlap=10% -> recall@5=0.324 mrr@10=0.271
chunk_size= 256 overlap=25% -> recall@5=0.324 mrr@10=0.268
chunk_size= 512 overlap=0%  -> recall@5=0.441 mrr@10=0.332
chunk_size= 512 overlap=10% -> recall@5=0.441 mrr@10=0.332
chunk_size= 512 overlap=25% -> recall@5=0.471 mrr@10=0.336
```

Tendencia clara: chunks más grandes ganan. Chunks de 128 caracteres cortan las respuestas a la mitad y el embedding pierde contexto; a 512 caracteres el chunk alcanza a contener la respuesta completa con más frecuencia. El overlap ayuda poco por sí solo, pero suma cuando el chunk ya es grande.

## Cómo se evaluó

Por cada config: se trocea el corpus, se indexa en Qdrant, y para cada pregunta del set se hace una búsqueda densa (embedding + similitud) pidiendo los 10 resultados más cercanos. Un chunk se marca "correcto" si viene del documento fuente correcto **y** contiene el texto exacto de la respuesta esperada (gold span).

Modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (`shared/settings.py`), con límite de **128 tokens**. Un chunk que supera ese límite se trunca en silencio antes de generar el embedding (`validate_chunk` en `shared/ingest.py`, vía `truncation=True`) — verificado debuggeando. Esto importa porque `chunk_size` se mide en caracteres, no en tokens: un chunk de 512 caracteres puede superar los 128 tokens del modelo, y en ese caso el embedding solo representa la parte truncada, no el chunk completo.

## Métricas (conceptual, sin fórmulas)

- **recall@k**: de todas las preguntas, ¿en qué fracción el chunk correcto apareció entre los primeros k resultados? Es la métrica principal: mide si el sistema *encuentra* la información, sin importar en qué posición.
- **mrr@k (mean reciprocal rank)**: además de encontrarlo, ¿qué tan arriba aparece? Si el chunk correcto es el resultado #1, suma casi el máximo; si aparece en el puesto #5, suma poco; si no aparece, suma cero. Es recall ponderado por posición — penaliza que la respuesta esté "enterrada".
- **bootstrap CI (intervalo de confianza)**: el set de preguntas es chico (34), así que un solo número de recall puede ser ruido. Se re-muestrea el set de preguntas al azar miles de veces y se recalcula recall en cada muestra; el rango donde cae el 95% de esas muestras es el intervalo de confianza. Un intervalo ancho significa "con este poco de datos, no confíes demasiado en el número exacto".
- **p50 (latencia)**: mediana del tiempo de respuesta por búsqueda. Sirve para chequear que la config ganadora no sea impráctica en producción, no para elegir la config.

Los intervalos de confianza se solapan bastante entre 256 y 512 caracteres, así que la diferencia no es enorme con este volumen de preguntas — pero 512/25% es consistentemente la mejor punta en recall y mrr, así que es el punto de partida razonable para el siguiente experimento. Vale notar que la ganancia ocurre pese a la truncación a 128 tokens del embedder, no porque el modelo esté viendo el chunk completo.
# Experimento 1 — Chunking 

La corrida original probó `chunk_size` fijos (128, 256, 512) sin verificar si entraban en el límite real del embedder. **512 caracteres no entra**: el embedder (`paraphrase-multilingual-MiniLM-L12-v2` vía TEI) tiene `max_input_length=128` tokens, y un chunk de 512 caracteres mide en promedio **158.6 tokens, hasta 201 en el peor caso** (medido con `/tokenize` real de TEI) — muy por encima del límite. `validate_chunk()` (`shared/ingest.py`) trunca en silencio antes de generar el embedding, así que el "ganador" de 512/25% (`recall@5=0.471`) se midió sobre chunks que en realidad **perdían ~19% de su contenido en promedio, hasta ~36% en el peor caso**, cortado en un punto arbitrario del tokenizer, no en un límite de oración/párrafo controlado.

El barrido dinámico (más abajo) reemplaza esto: en vez de adivinar `chunk_size`, lo deriva midiendo tokens reales contra el embedder, con margen de seguridad, así que ningún candidato evaluado excede el límite. **El nuevo ganador (`chunk_size=277`) es el número confiable** — el de 512 queda documentado abajo solo como referencia histórica de por qué no sirve.

## Corrida original (chunk_size fijo, sin validar contra el límite del embedder)

| Tracking: los 9 runs quedaron alojados en MLflow, un run por combinación de `chunk_size`/`overlap`.

Se probaron distintas combinaciones de `chunk_size` (128, 256 y 512) y `overlap` (0%, 10% y 25%). En total, fueron 9 configuraciones evaluadas sobre un conjunto fijo de preguntas con respuestas conocidas (gold spans).

### Resultado

**Mejor config:** `chunk_size=512`**,** `overlap=25%` → `recall@5=0.471` — **inválida, ver aviso arriba**.

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

Tendencia observada en su momento: chunks más grandes ganan. Chunks de 128 caracteres cortan las respuestas a la mitad y el embedding pierde contexto; a 512 caracteres el chunk alcanza a contener la respuesta completa con más frecuencia — pero, como se documenta arriba, ese "contener más" no se traduce en "embeddear más": el chunk de 512 se trunca antes de vectorizarse.

## Barrido dinámico — chunk_size derivado del límite real del embedder

Script: `experiments/chunking/run.py::evaluate_chunks_limits()`. Experimento MLflow: `chunking_dynamic`. Colección Qdrant: `exp_chunking_dynamic`.

### Fórmula del límite seguro

```
cap_tokens = max_input_length(embedder) * SAFETY_MARGIN
SAFETY_MARGIN = 0.85
```

Con `max_input_length=128` (medido vía `GET /info` de TEI), `cap_tokens = 108.8`. Un `chunk_size` (en caracteres) se acepta solo si el **p95** de tokens, medido tokenizando el corpus completo en ventanas no solapadas de ese tamaño (`POST /tokenize` real, no una estimación char/token), queda por debajo de `cap_tokens`. Se usa p95 y no el promedio porque lo que importa es el peor caso típico, no el caso típico — un chunk_size que en promedio entra pero cuyo 5% más denso (headers, código, etc.) se pasa del límite igual trunca contenido en ese 5%.

Búsqueda: potencias de 2 desde `128` hasta que el p95 supera el cap, más un binary search final para afinar el techo exacto en vez de saltarlo (detalle en `evaluate_chunks_limits()`). Resultado de esta corrida: candidatos `[128, 256, 277]` — muy por debajo del `512` que se probaba a ciegas antes.

### El overlap (25%) no afecta este margen

`chunk_overlap` controla cuánto texto se repite **entre** chunks consecutivos (`RecursiveCharacterTextSplitter`), pero ningún chunk individual crece más allá de `chunk_size` caracteres por tener overlap — el splitter sigue cortando cada pieza al mismo tope. Por eso el cálculo del cap (arriba) se hace en función de `chunk_size` únicamente: es válido para cualquier `overlap_frac` de la lista, incluido 25%, sin necesidad de repetir la medición de tokens por cada overlap.

### Resultado

**Mejor config:** `chunk_size=277`**,** `overlap=25%` → `recall@5=0.382`, `mrr@10=0.303`, CI95 `[0.235, 0.559]`. Guardado en `[best_config.json](./best_config.json)`.

```
chunk_size= 128 overlap=0%  -> recall@5=0.088 mrr@10=0.058 CI95=[0.000, 0.176]
chunk_size= 128 overlap=10% -> recall@5=0.118 mrr@10=0.087 CI95=[0.029, 0.235]
chunk_size= 128 overlap=25% -> recall@5=0.176 mrr@10=0.109 CI95=[0.059, 0.294]
chunk_size= 256 overlap=0%  -> recall@5=0.324 mrr@10=0.239 CI95=[0.176, 0.500]
chunk_size= 256 overlap=10% -> recall@5=0.324 mrr@10=0.271 CI95=[0.176, 0.471]
chunk_size= 256 overlap=25% -> recall@5=0.324 mrr@10=0.268 CI95=[0.176, 0.471]
chunk_size= 277 overlap=0%  -> recall@5=0.353 mrr@10=0.274 CI95=[0.206, 0.529]
chunk_size= 277 overlap=10% -> recall@5=0.353 mrr@10=0.271 CI95=[0.206, 0.529]
chunk_size= 277 overlap=25% -> recall@5=0.382 mrr@10=0.303 CI95=[0.235, 0.559]
```

### Comparación: mejor config anterior (inválida) vs. mejor config dinámica (válida)


|                                 | `chunk_size=512, overlap=25%` (anterior)                                       | `chunk_size=277, overlap=25%` (dinámico)         |
| ------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------ |
| recall@5                        | 0.471                                                                          | 0.382                                            |
| mrr@10                          | 0.336                                                                          | 0.303                                            |
| ¿excede el límite del embedder? | Sí — avg 158.6 tok, max 201 tok sobre cap real de 128                          | No — p95 verificado ≤ 108.8 tok (cap con margen) |
| ¿qué se embeddea realmente?     | ~81% del chunk en promedio (resto truncado en silencio, corte arbitrario)      | El chunk completo                                |
| confiabilidad del número        | Baja — mide una config que no se puede desplegar tal cual sin seguir truncando | Alta — validado contra el servicio real          |


El problema de fondo entre ambos: el número de 512 es más alto, pero no es comparable — mide un chunk que en producción nunca se embeddea completo, así que parte de esa "ganancia" es artefacto de qué mitad del chunk sobrevive al corte del tokenizer, no de contexto real disponible para el modelo. El de 277 es más bajo pero es el número real que se puede confiar y desplegar sin truncación oculta. 256 y 277 siguen sin distinguirse con significancia estadística entre sí (CIs muy solapados); 128 sí es claramente peor que ambos.

## Cómo se evaluó

Por cada config: se trocea el corpus, se indexa en Qdrant, y para cada pregunta del set se hace una búsqueda densa (embedding + similitud) pidiendo los 10 resultados más cercanos. Un chunk se marca "correcto" si viene del documento fuente correcto **y** contiene el texto exacto de la respuesta esperada (gold span).

Modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (`shared/settings.py`), con límite de **128 tokens**. Un chunk que supera ese límite se trunca en silencio antes de generar el embedding (`validate_chunk` en `shared/ingest.py`, vía `truncation=True`) — verificado debuggeando. Esto importa porque `chunk_size` se mide en caracteres, no en tokens: un chunk de 512 caracteres puede superar los 128 tokens del modelo, y en ese caso el embedding solo representa la parte truncada, no el chunk completo. El barrido dinámico existe justamente para no tener que adivinar este límite: lo mide contra el servicio real antes de correr el experimento.

## Métricas (conceptual, sin fórmulas)

- **recall@k**: de todas las preguntas, ¿en qué fracción el chunk correcto apareció entre los primeros k resultados? Es la métrica principal: mide si el sistema *encuentra* la información, sin importar en qué posición.
- **mrr@k (mean reciprocal rank)**: además de encontrarlo, ¿qué tan arriba aparece? Si el chunk correcto es el resultado #1, suma casi el máximo; si aparece en el puesto #5, suma poco; si no aparece, suma cero. Es recall ponderado por posición — penaliza que la respuesta esté "enterrada".
- **bootstrap CI (intervalo de confianza)**: el set de preguntas es chico (34), así que un solo número de recall puede ser ruido. Se re-muestrea el set de preguntas al azar miles de veces y se recalcula recall en cada muestra; el rango donde cae el 95% de esas muestras es el intervalo de confianza. Un intervalo ancho significa "con este poco de datos, no confíes demasiado en el número exacto".
- **p50 (latencia)**: mediana del tiempo de respuesta por búsqueda. Sirve para chequear que la config ganadora no sea impráctica en producción, no para elegir la config.

Los intervalos de confianza se solapan bastante entre 256 y 277 caracteres en el barrido dinámico, así que esa diferencia puntual no es significativa con este volumen de preguntas — pero 277/25% es consistentemente la mejor punta en recall y mrr dentro de los candidatos válidos, así que es el punto de partida real para el siguiente experimento (reemplaza al 512/25% de la corrida original, que queda descartado por la razón documentada arriba).
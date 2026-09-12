# Experimento 1 — Chunking

La fase 3 (`dinámico + tablas`) fue la última corrida del experimento — sobre la base validada en fase 2 contra el embedder, mantuvo el criterio de límite de tokens y recalibró el cap sobre el corpus sin tablas markdown. Así cerró dos problemas que arrastraba desde fase 1.

1. El problema de elegir `chunk_size` (`128, 256, 512`) sin validar el límite del embedder era silencioso. `paraphrase-multilingual-MiniLM-L12-v2` vía TEI establece `max_input_length=128` tokens, mientras que el chunk de 512 caracteres ganador alcanzaba **158.6 tokens de media y 201 en el peor caso**, medidos mediante `/tokenize` real de TEI. `validate_chunk()` (`shared/ingest.py`) lo truncaba antes del embedding, por lo que el `512/25%` de fase 1 (`recall@5=0.471`) representaba chunks con una pérdida de **~19% de contenido promedio y ~36% máxima**, determinada por cortes arbitrarios del tokenizer.
2. Trocear el texto sin reconocer las tablas markdown como unidad, cortándolas entre chunks o dejándolas inline sin que el LLM llegara a leerlas completas.

El experimento tuvo estas 3 fases.

1. **Corrida original** — barrido de `chunk_size` fijos (128/256/512) y `overlap` (0%/10%/25%) sin verificar contra el límite del embedder; el "ganador" (512/25%) resultó inválido por la truncación escrita anteriormente.
2. **Barrido dinámico** — reemplaza la elección a ciegas, derivando `chunk_size` (candidato final `277`) al medir tokens reales contra el embedder (`/tokenize` de TEI, con margen de seguridad), así que ningún candidato evaluado excede el límite.
3. **★ Dinámico + tablas** — la fase final, con el mismo criterio de límite de tokens que la fase 2 pero recalibrado sobre el corpus sin tablas markdown (candidato final `284`) y con las tablas extraídas del embedding y adjuntas aparte en el payload — la config recomendada y la que cierra el problema de la fase 1.

El overlap (25%) tampoco afecta este análisis en ninguna de las 3 fases. `chunk_overlap` controla cuánto texto se repite **entre** chunks consecutivos (`RecursiveCharacterTextSplitter`), pero ningún chunk individual crece más allá de `chunk_size` caracteres por tener overlap — el splitter sigue cortando cada pieza al mismo tope, así que el límite de tokens depende únicamente de `chunk_size` y vale igual para las tres corridas de este experimento.

Los archivos `results/*.json` (`best_config.json`) tampoco son estables entre fases — `run.py` siempre escribe en la misma ruta, así que cada corrida sobreescribe el resultado de la anterior. La config real de cada fase se respalda en su experimento de MLflow, uno distinto por fase: `chunking` (fase 1), `chunking_dynamic` (fase 2) y `exp_chunking_dynamic_tables` (fase 3).

## Cómo se evaluó (misma forma de evaluación densa para las 3 fases)

Por cada config, en cualquiera de las 3 fases: se trocea el corpus, se indexa en Qdrant, y para cada pregunta del set se hace una búsqueda densa (embedding + similitud) pidiendo los 10 resultados más cercanos. Un chunk se marca "correcto" si viene del documento fuente correcto **y** contiene el texto exacto de la respuesta esperada (gold span).

Modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (`shared/settings.py`), el mismo en las 3 fases, con límite de **128 tokens**. Un chunk que supera ese límite se trunca en silencio antes de generar el embedding (`validate_chunk` en `shared/ingest.py`, vía `truncation=True`) — verificado debuggeando. Esto importa porque `chunk_size` se mide en caracteres, no en tokens: un chunk de 512 caracteres puede superar los 128 tokens del modelo, y en ese caso el embedding solo representa la parte truncada, no el chunk completo. Fase 1 ignoraba este límite; fases 2 y 3 lo miden contra el servicio real antes de correr el experimento, en vez de adivinarlo.

## Métricas (conceptual, sin fórmulas)

- **recall@k**: de todas las preguntas, ¿en qué fracción el chunk correcto apareció entre los primeros k resultados? Es la métrica principal: mide si el sistema *encuentra* la información, sin importar en qué posición.
- **mrr@k (mean reciprocal rank)**: además de encontrarlo, ¿qué tan arriba aparece? Si el chunk correcto es el resultado #1, suma casi el máximo; si aparece en el puesto #5, suma poco; si no aparece, suma cero. Es recall ponderado por posición — penaliza que la respuesta esté "enterrada".
- **bootstrap CI (intervalo de confianza)**: el set de preguntas es chico (34), así que un solo número de recall puede ser ruido. Se re-muestrea el set de preguntas al azar miles de veces y se recalcula recall en cada muestra; el rango donde cae el 95% de esas muestras es el intervalo de confianza. Un intervalo ancho significa "con este poco de datos, no confíes demasiado en el número exacto".
- **p50 (latencia)**: mediana del tiempo de respuesta por búsqueda. Sirve para chequear que la config ganadora no sea impráctica en producción, no para elegir la config.

> **Pendiente** — si fase 3 sigue sin mejorar los chunks que le llegan al LLM, una acción a evaluar es trocear el texto sin el formato markdown (negrita, headers, etc.) y reservar el markdown únicamente para el payload de las tablas, ya que el markdown no aporta significado real al LLM que consume el chunk y solo suma tokens adicionales.


## Fase 1 (chunk_size fijo, sin validar contra el límite del embedder)

Script: `experiments/chunking/run.py`.  
Experimento MLflow: `chunking`.  
Colección Qdrant: `exp_chunking`.

Se probaron distintas combinaciones de `chunk_size` (128, 256 y 512) y `overlap` (0%, 10% y 25%). En total, fueron 9 configuraciones evaluadas sobre un conjunto fijo de preguntas con respuestas conocidas (gold spans).

### Resultado

**Mejor config:** `chunk_size=512`**,** `overlap=25%` → `recall@5=0.471, mrr@10=0.336`

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

Tendencia observada: chunks mayores contenían más veces la respuesta completa y conservaban más contexto (`128` < `512`), pero **512 no podía aprovechar ese contenido adicional**, ya que se truncaba antes de la vectorización.

## Fase 2 — Barrido dinámico (chunk_size derivado del límite real del embedder)

Script: `experiments/chunking/run.py::evaluate_chunks_limits()`.  
Experimento MLflow: `chunking_dynamic`.  
Colección Qdrant: `exp_chunking_dynamic`.

### Fórmula del límite seguro

```
cap_tokens = max_input_length(embedder) * SAFETY_MARGIN
SAFETY_MARGIN = 0.85
```

Con `max_input_length=128` (medido vía `GET /info` de TEI), `cap_tokens = 108.8`. Un `chunk_size` (en caracteres) se acepta solo si el **p95** de tokens, medido tokenizando el corpus completo en ventanas no solapadas de ese tamaño (`POST /tokenize` real, no una estimación char/token), queda por debajo de `cap_tokens`. Se usa p95 y no el promedio porque lo que importa es el peor caso típico, no el caso típico — un chunk_size que en promedio entra pero cuyo 5% más denso (headers, código, etc.) se pasa del límite igual trunca contenido en ese 5%.

Búsqueda: potencias de 2 desde `128` hasta que el p95 supera el cap, más un binary search final para afinar el techo exacto en vez de saltarlo (detalle en `evaluate_chunks_limits()`). Resultado de esta corrida: candidatos `[128, 256, 277]` — muy por debajo del `512` que se probaba a ciegas antes.

### Resultado

**Mejor config:** `chunk_size=277`**,** `overlap=25%` → `recall@5=0.382`, `mrr@10=0.303`, CI95 `[0.235, 0.559]`. Guardado en su momento en `results/best_config.json`.

> Ese archivo ya no respalda este número (ver nota introductoria sobre `results/*.json`) — quedó sobreescrito por la corrida de fase 3. El transcript completo de esta corrida sigue abajo como evidencia de estos números.

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

Los intervalos de confianza se solapan bastante entre 256 y 277 caracteres, así que esa diferencia puntual no es significativa con este volumen de preguntas — pero 277/25% es consistentemente la mejor punta en recall y mrr dentro de los candidatos válidos, así que es el punto de partida real para la siguiente fase (reemplaza al 512/25% de fase 1, descartado por la razón documentada arriba).

## Fase 3 — Tablas separadas del embedding, adjuntas al payload

La evaluación manual de las respuestas del judge (`experiments/evaluation/benchmark-models/results/exp_6/llama_base/base_answers.csv`) señaló errores en `q002`, `q010` y `q015`. Revisando el `context` real que recibió el LLM en esa corrida:

- `q015` es el caso limpio: el contexto recuperado termina literalmente en *"...la importancia en el Random Forest queda:"* — la tabla que completaba esa frase con el ranking de features nunca llegó, y el modelo responde *"No lo sé según el contexto proporcionado"*. El gold span de esta pregunta (`questions.jsonl`) es directamente una fila de esa tabla.
- `q002` también depende de una fila de tabla como gold span comparando cifras exactas entre dos documentos — el modelo da una respuesta con números puntuales que no se puede verificar sin esa tabla completa.
- `q010` entra en la misma revisión, pero su contexto (mismo CSV) ya traía sus 3 `gold_spans` como prosa fuera de cualquier tabla — su falla no se explica por este mecanismo (ver más abajo).

Causa raíz: `chunk_documents()` (`shared/ingest.py`) cortaba el texto por caracteres sin ningún criterio que reconociera una tabla markdown como unidad — podía dejarla inline dentro de un chunk (ruido en el embedding, ver `## Barrido dinámico` arriba) o partirla a la mitad entre dos chunks, perdiendo la relación fila/columna necesaria para que el LLM la lea.

### Qué se implementó

- `get_tables_from_docs()` (`experiments/chunking/run.py`): parte `doc.text` en bloques separados por línea en blanco (`re.split(r"\n{2,}", ...)`) y valida cada bloque con `markdown.markdown(block, extensions=["tables"])` — si el HTML resultante contiene `<table`, ese bloque (substring literal del texto original, garantizado por venir del mismo `re.split`) se guarda como tabla.
  - Se descartaron dos alternativas antes de llegar a esta: `unstructured.partition.md` reconstruye el texto de una tabla aplanando todas sus celdas con espacios (`TableBlock.iter_elements`, `unstructured==0.27.5`, `.../site-packages/unstructured/partition/html/parser.py:554-556`), perdiendo la sintaxis markdown — imposible de volver a ubicar en el texto original. Una regex de fila fija (`\|(.*?)\|.*?\|.*?\|`) truncaba silenciosamente cualquier columna más allá de la 3ra.
- `chunk_documents()` (`shared/ingest.py`): cada tabla del doc se reemplaza por un marcador atómico `[[TABLE:idx]]`, insertado **inline** dentro del párrafo que la rodeaba (justificación de este diseño más abajo). Por cada `split`, el marcador se detecta y se saca del texto antes de `validate_chunk` (nunca se embebe), y la tabla completa se adjunta a un campo nuevo, `Chunk.tables` — separado de `Chunk.text`.
- **Caso borde manejado:** si un `split` queda compuesto solo por marcador(es) de tabla (sin texto propio alrededor), no se crea un chunk vacío — sus tablas se acarrean (`pending_tables`) al próximo chunk con texto real del mismo doc. Sin este fix el texto embebido llegaba vacío (`""`) y el servidor de embeddings (`text-embeddings-inference`) rechazaba el batch completo (`413`, `"inputs cannot be empty"`), bloqueando toda la corrida.
- `is_chunk_correct()` (`shared/eval/metrics.py`): ahora concatena `chunk["text"]` con el `text_content` de cada tabla en `chunk["tables"]` antes de buscar los `gold_spans` — una pregunta cuya respuesta viva solo en una tabla adjunta cuenta como acierto.

### Qué se hizo en Qdrant

Colección separada del baseline, para no pisar los números que ya usa el resto de este reporte: `exp_chunking_dynamic_tables__paraphrase-multilingual-MiniLM-L12-v2` (antes: `exp_chunking_dynamic`). El payload de cada punto mantiene el esquema de siempre (`chunk_id`, `text`, `doc_id`, `library`, `chunk_index`) y suma `tables: list[dict]` — cada entrada con `text_content` (markdown crudo de esa tabla) y `doc_id`. Nunca participa del vector, solo viaja en el payload.

Validado directo contra Qdrant con `experiments/chunking/results/validate_tables_payload.py` (transcript completo: [tables_validation.log](./results/tables_validation.log)), sobre el estado final de la colección (`chunk_size=284, overlap=25%`, 215 puntos totales):

- **215 puntos totales, 10 con ≥ 1 tabla adjunta en el payload — exactamente las 10 tablas únicas del corpus, sin duplicados** (9 en `music-tagger/music-tagger-benchmark.md`, 1 en `music-tagger/tagger-music-genesis.md`).
- Verificado con muestras reales: ningún punto quedó con `tables` poblado y `text=""` (confirma el fix de `pending_tables`); el `text_content` de las tablas trae el markdown crudo intacto (pipes y fila separadora incluidos), no una versión aplanada.

> **Nota:** una primera medición de este experimento daba **13** puntos con tabla para las mismas 10 tablas del corpus — la razón por la que el marcador se inserta inline y no aislado (justificación más abajo) evita justamente esa duplicación.

### Justificación del uso del marcador inline

Aislar el marcador como un mini-párrafo (`\n\n[[TABLE:idx]]\n\n`) puede hacer que `RecursiveCharacterTextSplitter` lo trate como un chunk independiente, ya que comienza separando por `"\n\n"`. Si ese bloque cabe completo en la ventana de `chunk_overlap`, puede aparecer nuevamente en el siguiente chunk, duplicando la referencia a la tabla adjunta. Por eso se mantiene inline, como parte del párrafo, usando `marked_text.replace(t["text_content"], f"[[TABLE:{idx}]]")` y colapsando los saltos de línea a su alrededor

```python
marked_text = re.sub(r"\n+(\[\[TABLE:\d+\]\])", r" \1", marked_text)
marked_text = re.sub(r"(\[\[TABLE:\d+\]\])\n+", r"\1 ", marked_text)
```

Al no quedar aislado, el marcador nunca cae solo en el borde de `chunk_size`, así que el overlap no lo arrastra dos veces. Verificado contra las 10 tablas del corpus: **0 de 10 aparecen en más de un chunk**.

### Fix: falso negativo en `is_chunk_correct` por markdown crudo

Al revisar por qué una pregunta con `gold_span` dentro del corpus (`q019`) daba 0 chunks correctos, se encontró que `_normalize()` (`shared/eval/metrics.py`) solo colapsaba espacios en blanco, sin tocar sintaxis markdown. El `gold_span` estaba escrito tal como se lee (renderizado), pero `chunk.text` guarda el markdown crudo, con los `**` de negrita todavía como caracteres literales en medio de la oración:

```
gold_span:  'verificado 2026-08-18 contra el código real (`features/config.py`): el sr actual es 22050, no 44100'
chunk.text: '**verificado 2026-08-18 contra el código real (**`features/config.py`**): el sr actual es 22050, no 44100.**'
```

Los `**` insertados por la negrita rompen la continuidad de caracteres que necesita el chequeo de substring (`_normalize(span) in chunk_text`), aunque el chunk sí contenga la información correcta — un falso negativo silencioso que subestima `recall@5`/`mrr@10`.

Se evaluó usar `unstructured.partition.md` para limpiar el markdown antes de comparar, pero normaliza de más para este uso: además de sacar la negrita, también saca los backticks de código inline y aplana los `|` de las tablas — y el `gold_span` de este mismo caso sí espera conservar los backticks (`features/config.py`). Se optó por un regex quirúrgico que solo saca `**`/`__`, dejando todo lo demás (backticks, tablas, links) intacto:

```python
def _normalize(text: str) -> str:
    text = re.sub(r"(\*\*|__)", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()
```

### Resultado (`chunk_size` dinámico + tablas)

Script: `experiments/chunking/run.py`.   
Experimento MLflow: `exp_chunking_dynamic_tables`. 

```
chunk_size= 128 overlap=0%  -> recall@5=0.147 mrr@10=0.103 CI95=[0.0294, 0.2647]
chunk_size= 128 overlap=10% -> recall@5=0.176 mrr@10=0.132 CI95=[0.0588, 0.2941]
chunk_size= 128 overlap=25% -> recall@5=0.235 mrr@10=0.149 CI95=[0.1176, 0.3824]
chunk_size= 256 overlap=0%  -> recall@5=0.412 mrr@10=0.281 CI95=[0.2647, 0.5882]
chunk_size= 256 overlap=10% -> recall@5=0.382 mrr@10=0.313 CI95=[0.2353, 0.5588]
chunk_size= 256 overlap=25% -> recall@5=0.382 mrr@10=0.310 CI95=[0.2353, 0.5588]
chunk_size= 284 overlap=0%  -> recall@5=0.412 mrr@10=0.310 CI95=[0.2647, 0.5882]
chunk_size= 284 overlap=10% -> recall@5=0.412 mrr@10=0.310 CI95=[0.2647, 0.5882]
chunk_size= 284 overlap=25% -> recall@5=0.441 mrr@10=0.336 CI95=[0.2647, 0.6176]
```

**Mejor config:** `chunk_size=284, overlap=25%` → `recall@5=0.441`, `mrr@10=0.336`, CI95 `[0.2647, 0.6176]`. Guardado en [results/best_config.json](./results/best_config.json). Estos números ya incorporan las dos correcciones vistas anteriormente (duplicación de tablas + falso negativo en `is_chunk_correct`) — antes de aplicarlas, esta misma config daba `recall@5=0.412`, `mrr@10=0.295`.

`evaluate_chunks_limits()` corre sobre `"".join(text_without_tables)` en vez del texto completo — al no medir tokens de líneas densas en pipes (que tokenizan más denso por carácter que prosa normal), el p95 de tokens baja, y el candidato máximo que entra bajo el mismo `cap_tokens=108.8` sube de 277 a 284 caracteres. No es un ajuste manual: es consecuencia directa de calibrar sobre un corpus sin tablas.

### Impacto puntual en las 3 preguntas que motivaron el cambio

`hit@5` (¿el chunk correcto aparece entre los primeros 5 resultados de `dense_search`?) para `q002`/`q010`/`q015`, comparando la colección baseline (`exp_chunking_dynamic`, sin mecanismo de tablas) contra esta (`exp_chunking_dynamic_tables`, con las dos correcciones aplicadas) — datos completos (las 34 preguntas, no solo estas 3) en [coll_evaluation_exp_chunking.csv](./results/coll_evaluation_exp_chunking.csv):


| pregunta | gold span depende de tabla        | hit@5 baseline | hit@5 con tablas                  |
| -------- | --------------------------------- | -------------- | --------------------------------- |
| q015     | Sí — fila de ranking de features  | No             | **Sí** (rank 1, `tables_count=1`) |
| q002     | Sí — fila de tabla comparativa    | No             | No                                |
| q010     | No — sus 3 `gold_spans` son prosa | Sí (rank 4)    | Sí (rank 5)                       |


`q015` mejora de forma directa y verificable: el mismo chunk que ya se recuperaba por su texto ahora trae la tabla completa adjunta en el payload, resolviendo el corte de contexto que causaba el "No lo sé" original. Antes de corregir la duplicación de tablas, este mismo hit aparecía en el rank 5 (con la tabla ya adjunta, pero repartida entre dos chunks candidatos); al dejar de fragmentar el chunk relevante, el chunk queda con más contexto propio y sube al rank 1 — la corrección de ingesta no solo evitó el duplicado, mejoró la calidad del embedding de ese chunk.

`q002` sigue sin resolverse a nivel de recall@5 — limitación abierta, no ocultada: el mecanismo solo adjunta la tabla al chunk que **ya** entra en el top 5 por su propio texto; no cambia el ranking de ningún chunk. Si el chunk correcto no se recupera por su contenido (como acá), tener la tabla en el payload no lo hace aparecer.

`q010` no dependía de una tabla — su falla original no se explica por este mecanismo, es un problema distinto (de generación del LLM sobre un contexto que ya estaba completo, no de recuperación).

## Comparación: mejor config anterior (inválida) vs. mejor config dinámica (válida) vs. dinámica + tablas


|                                 | `chunk_size=512, overlap=25%` (anterior)                                       | `chunk_size=277, overlap=25%` (dinámico)         | `chunk_size=284, overlap=25%` (dinámico + tablas)                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| recall@5                        | 0.471                                                                          | 0.382                                            | 0.441                                                                                                                      |
| mrr@10                          | 0.336                                                                          | 0.303                                            | 0.336                                                                                                                      |
| ¿excede el límite del embedder? | Sí — avg 158.6 tok, max 201 tok sobre cap real de 128                          | No — p95 verificado ≤ 108.8 tok (cap con margen) | No — mismo cap, calibrado sobre texto sin tablas                                                                           |
| ¿qué se embeddea realmente?     | ~81% del chunk en promedio (resto truncado en silencio, corte arbitrario)      | El chunk completo                                | El chunk completo, sin el markdown de sus tablas                                                                           |
| ¿tablas markdown separadas?     | No — vivían inline en el texto embebido                                        | No — vivían inline en el texto embebido          | Sí — extraídas y adjuntas como payload aparte (`chunk.tables`), nunca entran al embedding                                  |
| confiabilidad del número        | Baja — mide una config que no se puede desplegar tal cual sin seguir truncando | Alta — validado contra el servicio real          | Alta — validado contra el servicio real y contra Qdrant directo ([tables_validation.log](./results/tables_validation.log)) |


**Resumen:** `512` sobreestima su rendimiento al evaluarse con truncación, por lo que `277` es el candidato válido sin truncación oculta. `256` y `277` no muestran diferencias significativas y `128` resulta inferior. `284` corresponde a una calibración independiente sobre el corpus sin tablas, por lo que no debe compararse directamente con `277`.
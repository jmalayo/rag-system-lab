# Módulo de Evaluación

| Sin MLflow actualmente (a diferencia de los experimentos 1-3) : standalone (`experiments/evaluation/benchmark-models/ollama-judge-evaluation.py`)

La etapa LLM del pipeline principal (`experiments/evaluation/run.py`) sigue pausada (`groundedness_rate`/`hallucination_rate` en `shared/eval/metrics.py` = `NotImplementedError`). **Antes de implementarlas, se evaluó si Llama puede juzgar sus propias respuestas sin introducir self-enhancement bias.** Para ello, se compararon dos esquemas sobre las mismas 34 respuestas de Llama: **auto-juicio (Llama → Llama)** vs. **juicio cruzado (Llama → DeepSeek-R1)**.  
**DeepSeek-R1-Distill-Qwen-7B** se eligió como segundo juez por su linaje distinto a Llama, evitando medir el mismo sesgo dos veces; además, corre en Ollama sin nueva integración y cabe junto a Llama en los 8 GB de VRAM. Verificado: `deepseek-r1:7b` (Q4_K_M) = **4.7 GB**, dentro del rango de 4–5 GB solicitado.

## Setup

- **Fix:** se incorporó `llm_backend`/`llm_model` en `shared/settings.py` y añadir `model: str | None = None` a `generate()` para permitir selección por llamada.
- **Fix:** se añadió `collection: str = COLLECTION` a `base_search` y usar `collection="exp_judge_benchmark"` en el benchmark, sin alterar el default. El cambio corrige el uso forzado de `exp_reranking` (inexistente; la colección real es `reranking`) sin importar quién invoque la función.
- Cualquier **Ollama nativo (**`systemd`**)** que ocupe `11434` debe detenerse antes de levantar el contenedor, para asegurar que las peticiones lleguen a la instancia de Docker. Tras suspender/reanudar el host, verificar también red y DNS (`NetworkSettings.Networks`), ya que el healthcheck puede seguir OK aunque exista pérdida de conectividad. Los modelos validados con `docker exec ollama ollama list`.
- **Fix:** se configuró el runtime NVIDIA en Docker (`nvidia-ctk runtime configure --runtime=docker`), reiniciar el daemon y declarar `deploy.resources.reservations.devices` con `driver: nvidia`. Se añadieron `OLLAMA_MAX_LOADED_MODELS=1` y `OLLAMA_KEEP_ALIVE=30s` para no mantener ambos modelos en VRAM. El problema se identificaba porque `ollama ps` mostraba `100% CPU` y el arranque reportaba `total_vram="0 B"`: el toolkit estaba instalado, pero no registrado en Docker.
  ```yaml
  ## docker-compose.yml 
  # servicio ollama
  ollama:
    environment:
      OLLAMA_MAX_LOADED_MODELS: "1"
      OLLAMA_KEEP_ALIVE: "30s"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # servicio para descargar el juez
  ollama-pull-deepseek:
    restart: no
    image: ollama/ollama:latest
    container_name: deepseek-pull
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: ["ollama"]
    command: ["pull", "deepseek-r1:7b"]
    environment:
      OLLAMA_HOST: "ollama:11434"
  ```
- Script [ollama-judge-evaluation.py](experiments/evaluation/benchmark-models/ollama-judge-evaluation.py) inserta la raíz del repo en `sys.path`; requiere correrse con la raíz como `cwd`. Las respuestas base se cachean en `results/base_answers_exp_N.csv` apenas se generan. Salida en pandas/CSV, separador `;`.
- `generate()` (`shared/llm.py`) para las llamadas de juicio usa `repeat_penalty=1.3` (evita que quede repitiendo el mismo token en loop).
- **Fix:** `num_predict` de `judge_answers()` subido de 1500 a 3000. Con 1500, DeepSeek gastaba todo el presupuesto en su bloque de razonamiento oculto (`thinking`) antes de escribir el veredicto, dejando `response` vacío (`done_reason: "length"`) — 26.5% de los juicios (9/34) DeepSeek autojuzgándose (`exp_5/deepseek_base/results_summary.csv`) y 17.6% (6/34) juzgando a Llama (`exp_5/llama_base/results_summary.csv`). Con 3000 el fallo bajó a la mitad en ambas direcciones: 11.8% (`exp_6/deepseek_base/results_summary.csv`) y 8.8% (`exp_6/llama_base/results_summary.csv`). Estos casos quedan marcados con columnas `{judge}_asw_valido` en `per_question_judge_verdicts.csv`, y son reproducibles con [juzge-evaluate-errors.py](experiments/evaluation/benchmark-models/juzge-evaluate-errors.py).

  Detalle de las 6 preguntas que fallaban con 1500 (base Llama, juzgadas por DeepSeek — [reporte_num_predict_1500_vs_3000_llama_base.csv](./benchmark-models/results/reporte_num_predict_1500_vs_3000_llama_base.csv)):

  | Pregunta | Falla con 1500 | Con 3000 | Resuelto |
  |---|---|---|---|
  | q002 | relevant | válido | sí |
  | q012 | grounded | grounded (thinking 6624→13269) | no |
  | q015 | relevant | válido | sí |
  | q023 | grounded | válido | sí |
  | q025 | grounded | grounded (thinking 5947→11521) | no |
  | q030 | grounded | grounded (thinking 6811→13499) | no |

  Las 3 que no resuelven escalan su `thinking` casi proporcional al presupuesto (~2x con el doble de `num_predict`) — es necesidad real de más tokens, no un tope arbitrario, así que se documentan como límite conocido del juez en vez de seguir subiendo `num_predict`. Confirmado con la corrida completa (`34/34`, `exp_6/llama_base/results_summary.csv`): fallo 6/34 (17.6%) con `1500` baja a exactamente 3/34 (8.8%) con `3000`, groundedness limpio DeepSeek **80.6%** (n=31).

  Mismo patrón del otro lado (base DeepSeek, autojuzgándose — [reporte_num_predict_1500_vs_3000_deepseek_base.csv](./benchmark-models/results/reporte_num_predict_1500_vs_3000_deepseek_base.csv)):

  | Pregunta | Falla con 1500 | Con 3000 | Resuelto |
  |---|---|---|---|
  | q004 | válido | válido | sí |
  | q005 | grounded | válido | sí |
  | q009 | grounded | válido | sí |
  | q014 | grounded | válido | sí |
  | q015 | grounded + relevant | válido | sí |
  | q018 | grounded | grounded (thinking 7292→15862) | no |
  | q019 | relevant | grounded (thinking 3411→14206) | no |
  | q020 | relevant | válido | sí |
  | q023 | grounded | grounded (thinking 6156→12503) | no |

  6 de 9 se resuelven en esta reproducción puntual, las 3 que no (`q018`, `q019`, `q023`) repiten el mismo escalado de `thinking` casi 2x — mismo límite conocido que en `llama_base`. La corrida completa (`34/34`, `exp_6/deepseek_base/results_summary.csv`) confirma la reducción en magnitud (4/34, 11.8%) pero no coincide pregunta por pregunta con esta lista puntual (`q005, q009, q018, q023` en la corrida completa) — no-determinismo del backend con `num_thread=8` hace que el mismo prompt no siempre falle en la misma pregunta entre corridas, aunque la tasa agregada sea consistente.

## Prompt engineering — `GROUNDEDNESS_PROMPT`/`RELEVANCE_PROMPT`

- Palabra clave del veredicto: `SÍ`/`NO` → `TRUE`/`FALSE`. Motivo: `"SÍ".startswith("SI")` es `False` en Python (tilde ≠ sin tilde); todo acierto se contaba como error.
- **Se invirtió** el orden del prompt: de **Contexto/Respuesta → pregunta de evaluación** a **pregunta de evaluación → Contexto/Respuesta**. Con el orden original, `llama3.2:3b` devolvía `FALSE` sistemáticamente, incluso cuando la respuesta era una copia literal del contexto. Con la pregunta primero, razonaba correctamente. **DeepSeek no mostró sensibilidad al orden.**
- **Se simplificó** la instrucción de formato de `"No expliques tu razonamiento. Responde ÚNICAMENTE con la palabra completa TRUE o FALSE, sin abreviar, sin puntuación ni texto adicional"` a `"Responde TRUE o FALSE"`. La versión elaborada sesgaba a `llama3.2:3b` hacia `FALSE` independientemente del contenido, incluso en el mismo caso de control, produciendo un resultado incorrecto. Con la instrucción simple, **DeepSeek mantuvo conclusiones correctas**: razonó brevemente y terminó en `TRUE`/`FALSE`. La versión elaborada no aportaba valor al segundo juez y perjudicaba al primero.
- Se reemplazó el parseo `raw.startswith("SI")`, que estaba anclado al inicio y era sensible a mayúsculas/tildes, por `parse_verdict()` con `re.search(r"TRUE|FALSE", raw, re.IGNORECASE)`, que buscaba `TRUE`/`FALSE` en cualquier parte del texto. Esto permitió tolerar razonamientos previos del modelo sin exigir que la respuesta fuera una única palabra exacta al inicio. Si no se encontraba ningún veredicto, se contabilizaba como `FALSE` y se registraba en logs; ocurrió en una fracción pequeña de llamadas, principalmente con DeepSeek cuando el razonamiento no terminaba a tiempo.

```diff
 GROUNDEDNESS_PROMPT = """
+¿Cada afirmación de la "Respuesta a evaluar" de abajo está directamente respaldada por el Contexto de abajo? Responde TRUE o FALSE.
+
 Contexto: {context}

 Respuesta a evaluar: {answer}
-
-¿Cada afirmación de la "Respuesta a evaluar" está directamente respaldada por el Contexto de arriba? Responde con exactamente una palabra: SÍ o NO.
 """

 RELEVANCE_PROMPT = """
-Pregunta: {question}
+¿La "Respuesta a evaluar" de abajo realmente aborda la Pregunta de abajo, sin importar si el contenido es correcto? Responde TRUE o FALSE.

-Respuesta: {answer}
+Pregunta: {question}

-¿La Respuesta realmente aborda la Pregunta (sin importar si es correcta)? Responde con exactamente una palabra: SÍ o NO.
+Respuesta a evaluar: {answer}
 """
```

## Resultado (`exp_4`, 34 preguntas, respuestas de `llama3.2:3b`)

```
                judge_model             mode  grounded  halluc.  relevant    p50 ms
llama           llama3.2:3b     self-judging     2.9%    97.1%     32.4%    2182.4
deepseek        deepseek-r1:7b  cross-judging    76.5%    23.5%     73.5%   12773.3
```

`exp_4/llama_base/results_summary.csv` · `exp_4/llama_base/per_question_judge_verdicts.csv`. Acuerdo entre jueces: **26.5%** en groundedness, **52.9%** en relevancia — bajo, y esta vez es señal real, no un artefacto de instrumentación: los dos jueces reparten juicios positivos y negativos (no es "los dos dicen que no" como en corridas anteriores). *(Corrida con `num_predict=1500`; no auditada todavía con `_asw_valido` — pendiente.)*

Llama, juzgándose a sí mismo, es mucho más estricto que DeepSeek juzgándolo desde afuera (2.9% vs. 76.5% en groundedness). Con una sola dirección de cruce no se podía aislar si es **self-enhancement bias invertido** (auto-crítica) o simplemente que Llama es un juez más duro en general — hacía falta la dirección opuesta.

## Resultado (`exp_6`, 34 preguntas, respuestas de `deepseek-r1:7b`)

```
                judge_model             mode  grounded  halluc.  relevant    p50 ms
deepseek        deepseek-r1:7b  self-judging     70.6%    29.4%     82.4%   13078.1
llama           llama3.2:3b    cross-judging      8.8%    91.2%     41.2%    2201.2
```

`exp_6/deepseek_base/results_summary.csv` · `exp_6/deepseek_base/per_question_judge_verdicts.csv`. Filtrando las filas inválidas (`_asw_valido=False`, ver fix de `num_predict` arriba), el número limpio sube: DeepSeek self-judging **80.0%** (n=30), Llama cross-judging **9.1%** (n=33). Las 4 filas que siguen inválidas incluso con `num_predict=3000` están diagnosticadas en `exp_6/deepseek_base/judge_errors_deepseek_3000.csv`.

**Lectura final**: con las dos direcciones completas, el patrón se sostiene sin importar quién escribió la respuesta — Llama se mantiene duro (2.9% autojuzgándose, 8.8-9.1% juzgando a DeepSeek) y DeepSeek se mantiene permisivo (76.5% juzgando a Llama, 70.6-80.0% autojuzgándose). La brecha entre jueces (~3-9% vs. ~70-80%) es muchísimo mayor que la brecha por autor dentro de cada juez, lo que descarta el self-enhancement bias como explicación principal: **Llama es sistemáticamente un juez más estricto que DeepSeek**, independientemente de quién generó la respuesta evaluada.
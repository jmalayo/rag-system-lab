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
- `generate()` (`shared/llm.py`) para las llamadas de juicio usa `num_predict=1500` (DeepSeek-R1 antepone razonamiento visible antes de la respuesta; con poco margen se corta sin concluir) y `repeat_penalty=1.3` (evita que quede repitiendo el mismo token en loop).



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

`[results_summary_exp_4.csv](./benchmark-models/results/results_summary_exp_4.csv)`. Acuerdo entre jueces: **26.5%** en groundedness, **52.9%** en relevancia — bajo, y esta vez es señal real, no un artefacto de instrumentación: los dos jueces reparten juicios positivos y negativos (no es "los dos dicen que no" como en corridas anteriores).

**Lectura**: Llama, juzgándose a sí mismo, es mucho más estricto que DeepSeek juzgándolo desde afuera (2.9% vs. 76.5% en groundedness). Es la comparación que buscaba este benchmark, pero con una sola dirección de cruce (Llama nunca juzga respuestas de DeepSeek) no se puede aislar todavía si es **self-enhancement bias invertido** (auto-crítica) o simplemente que Llama es un juez más duro/menos capaz en general, sin importar de quién sea la respuesta — haría falta una tercera corrida (Llama juzgando respuestas generadas por otro modelo) para separar ambas explicaciones.
ANSWER_PROMPT = """
Responde la pregunta usando ÚNICAMENTE el contexto de abajo. Si el contexto no contiene la respuesta, di "No lo sé según el contexto proporcionado."

Contexto: {context}
Pregunta: {question}

Responde en español, de forma concisa (2-3 oraciones):
"""

GROUNDEDNESS_PROMPT = """
Contexto: {context}

Respuesta a evaluar: {answer}

¿Cada afirmación de la "Respuesta a evaluar" está directamente respaldada por el Contexto de arriba? Responde con exactamente una palabra: SÍ o NO.
"""

RELEVANCE_PROMPT = """
Pregunta: {question}

Respuesta: {answer}

¿La Respuesta realmente aborda la Pregunta (sin importar si es correcta)? Responde con exactamente una palabra: SÍ o NO.
"""

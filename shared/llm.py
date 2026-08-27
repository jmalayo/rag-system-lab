from __future__ import annotations

import requests

def _generate_ollama(prompt: str, max_new_tokens: int = 256) -> str:
    return "Hola"

def generate(prompt: str, max_new_tokens: int = 256) -> str:
    return _generate_ollama(prompt, max_new_tokens)

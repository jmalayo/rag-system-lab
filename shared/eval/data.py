from __future__ import annotations

import json
from pathlib import Path

from shared.settings import settings

def load_questions(path: str | None = None) -> list[dict]:

    p = Path(path or settings.questions_path)

    questions = []

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line:
            questions.append(json.loads(line))
            
    return questions

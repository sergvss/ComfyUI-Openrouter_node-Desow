"""Разбор ответа судьи геометрии (Furnish 2.5): JSON {"ok": bool, "problems": "..."} -> (ok, problems).

Зачем: генератор иногда достраивает окна и двери или сдвигает стены. Судья сравнивает пустую комнату с результатом;
при ok=false граф один раз генерирует заново, передав problems текстом. Неразборчивый ответ судьи (ошибка API,
нет JSON) считается ok=True: судья - страховка и не должен ронять прогон или запускать лишний платный повтор.

Модуль без ComfyUI - тестируется отдельно.
"""

import json
import re


def parse_verdict(raw):
    """Текст судьи -> (ok: bool, problems: str)."""
    if not isinstance(raw, str):
        return True, ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return True, ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return True, ""
    if not isinstance(d, dict) or not isinstance(d.get("ok"), bool):
        return True, ""
    problems = str(d.get("problems") or "").strip()
    return d["ok"], ("" if d["ok"] else problems)

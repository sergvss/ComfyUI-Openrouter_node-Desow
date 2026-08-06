# ВЕНДОРЕННЫЙ КОД. Источники: desow/layout_openings_match.py (parse_scan_openings,
# _normalize_kind, _normalize_wall) и desow/plan2d/extract.py (extract_json_object) -
# синхронизировать при правках. Двойное ведение осознанное (README, «Ноды Desow»).
"""Разбор ответа детектора проёмов сканера (выход `output_text_openings`).

Формат детектора (воркфлоу `segments.json`, нода OpenRouter #269):

    {"openings":[{"type":"window","wall":"back","position_on_wall":"single",
                  "confidence":0.9}],
     "summary":{"windows_count":1,"doors_count":0,"total_count":1},
     "room":{"shape":"rectangle","confidence_shape":0.85,
             "type":"living_room","confidence_type":0.9}}

Дополнительно принимается обёртка `{"openings_data": {...}}` (так проёмы лежат в
записи скана на бэкенде) и голый список — обратная совместимость со старыми
записями `image_scan_results`.

Мержу нужны только `(тип, стена)`; `position_on_wall` не используется — сканер
не измеряет геометрию, позиции приходят от VLM-экстрактора (см. merge.py).

Пустой/непарсимый вход НЕ фатален: сканер — приор, а не обязательный источник.
Нода отдаёт план по одной экстракции и пишет причину в debug.
"""
from __future__ import annotations

import json
import re

# Допустимые базовые kinds (после нормализации compound types).
_VALID_KINDS = frozenset({"window", "door"})
_VALID_WALLS = frozenset({"front", "back", "left", "right"})

# floor_to_ceiling_window — отдельный effective kind: занимает всю высоту стены
# и значимо отличается от обычного окна для композиции комнаты.
_FLOOR_TO_CEILING = "floor_to_ceiling_window"

# Front-стена — за камерой; сканер её физически не видит, поэтому низкая
# уверенность там означает галлюцинацию VLM. Порог и мотивация — из источника.
_FRONT_OPENING_MIN_CONFIDENCE = 0.7

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def extract_json_object(text):
    """Достаёт JSON-объект из ответа модели (терпимо к ```json fence и преамбуле)."""
    t = (text or "").strip()
    match = _FENCE_RE.search(t)
    if match:
        t = match.group(1)
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            t = t[i:j + 1]
    return json.loads(t)


def normalize_wall(wall):
    """`'front_wall'` -> `'front'`. Lowercase + strip суффикс `_wall`."""
    if not wall:
        return ""
    s = str(wall).strip().lower()
    if s.endswith("_wall"):
        s = s[: -len("_wall")]
    return s


def normalize_kind(kind):
    """Тип проёма -> базовый `window` / `door` либо `floor_to_ceiling_window`.

    Сканер возвращает compound-типы (`arched_window`, `sliding_door`): всё, что
    оканчивается на window/door, сворачивается в базовый тип; неизвестное -> `''`
    (отфильтруется по whitelist).
    """
    if not kind:
        return ""
    s = str(kind).strip().lower()
    if not s:
        return ""
    if s in _VALID_KINDS:
        return s
    if s == _FLOOR_TO_CEILING:
        return _FLOOR_TO_CEILING
    if s.endswith("_window") or s.endswith("window"):
        return "window"
    if s.endswith("_door") or s.endswith("door"):
        return "door"
    return ""


def _raw_openings(payload):
    """Список проёмов из любого из поддерживаемых конвертов."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    nested = payload.get("openings_data")
    if isinstance(nested, dict):
        return nested.get("openings")
    if isinstance(nested, list):
        return nested
    return payload.get("openings")


def parse_scanner_openings(text):
    """Текст детектора -> `(openings, room, notes)`.

    `openings` — элементы `{kind, wall, quantity, display_type}` (тот же вид, что
    у `parse_scan_openings` на бэкенде; дальше их разворачивает
    `merge.scanner_openings_from_scan`). `room` — блок сканера с формой и типом
    комнаты (справочно, в debug). `notes` — что отброшено и почему.
    """
    notes = []
    if not (text or "").strip():
        return [], {}, ["сканер: вход пуст — состав проёмов только от VLM"]
    try:
        payload = extract_json_object(text)
    except (ValueError, TypeError) as exc:
        return [], {}, ["сканер: не разобран JSON (%s) — состав проёмов только от VLM" % exc]

    raw = _raw_openings(payload)
    if not isinstance(raw, list):
        return [], {}, ["сканер: в ответе нет списка openings — состав проёмов только от VLM"]

    out = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            notes.append("сканер[%d]: не объект -> выброшен" % i)
            continue
        original_type = entry.get("type")
        kind = normalize_kind(original_type)
        wall = normalize_wall(entry.get("wall"))
        if kind not in (_VALID_KINDS | {_FLOOR_TO_CEILING}) or wall not in _VALID_WALLS:
            # Сюда попадает и wall='unknown' — привязать проём к стене нечем.
            notes.append("сканер[%d]: %r на стене %r -> выброшен" % (i, original_type, entry.get("wall")))
            continue
        if wall == "front" and "confidence" in entry:
            # Явно указанная низкая уверенность на front-стене -> галлюцинация.
            # Отсутствие поля (legacy-формат) фильтр не включает.
            raw_conf = entry.get("confidence")
            try:
                confidence = float(raw_conf) if raw_conf is not None else None
            except (TypeError, ValueError):
                confidence = None
            if confidence is not None and confidence < _FRONT_OPENING_MIN_CONFIDENCE:
                notes.append(
                    "сканер[%d]: %s на front с confidence=%.2f < %.2f -> выброшен"
                    % (i, kind, confidence, _FRONT_OPENING_MIN_CONFIDENCE)
                )
                continue
        out.append({
            "kind": kind,
            "wall": wall,
            "quantity": 1,
            "display_type": str(original_type) if original_type else kind,
        })

    room = payload.get("room") if isinstance(payload, dict) else None
    return out, room if isinstance(room, dict) else {}, notes

# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/furnish.py - синхронизировать при правках.
# Двойное ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
#
# Отличия от источника (каждое — с причиной, менять только вместе с источником):
#   1. Нет pydantic: предмет проверяет `schema_lite.validate_furniture_item`
#      (построчный порт `FurnitureItem`).
#   2. Нет `plan2d.errors`: терминальная ошибка — `FurnishError` с тем же кодом
#      `plan_furnish_invalid_json`, чтобы причина в логах читалась одинаково.
#   3. `place_furniture` СИНХРОННАЯ и принимает callable `complete(messages) -> str`
#      вместо async-клиента: у ноды ComfyUI нет event loop, а сеть вынесена в
#      `openrouter_api.chat_complete` (общий http-стек репозитория). Тесту при этом
#      достаточно подставить обычную функцию.
#   4. `build_messages` знает про `style_hint` и `seed` — их у бэкенда нет.
#      Это входы ноды (пожелания стиля и вариативность/обход кеша); оба уезжают
#      ОТДЕЛЬНЫМИ помеченными секциями user-сообщения, системный промпт с
#      правилами эргономики не трогают.
"""Фаза B: расстановка мебели LLM + детерминированный валидатор (порт `hybrid-proto/furnish.py`).

Паттерн из ресерча (Architect-Ant / Co-Layout): модель предлагает размещение в
JSON -> валидатор кодом (containment, пересечения, дуга и подход двери, полоса
перед окном, зазор техники от коробки) -> список нарушений возвращается модели
ре-промптом (не более `MAX_FURNISH_RETRIES`) -> финальный JSON.

Судья расстановки — код, а не вторая модель: это требование приёмки (детерминизм
и объяснимость нарушений).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .geometry import room_bbox
from .schema_lite import PlanDataError, validate_furniture_item
from .validate import validate_furniture

PLAN_FURNISH_MODEL = "google/gemini-3.6-flash"
MAX_FURNISH_RETRIES = 3

# Допуск на центр предмета за габаритом комнаты. Предмет, вылезающий за стену, —
# штатное нарушение: его ловит validate_furniture и возвращает модели ре-промптом.
# Здесь отсекается только бессмыслица (центр в сотне метров от комнаты), которая
# иначе доехала бы до рендера: после исчерпания ре-промптов возвращается последняя
# расстановка КАК ЕСТЬ, вместе с нарушениями.
CENTER_TOLERANCE_DW = 2.0

# Потолок на пожелания стиля. Вход ноды сквозной, а в прод-описаниях дизайна
# лежат простыни на тысячи символов про архитектуру помещения — в промпте
# расстановки они перевешивают правила эргономики. Режем и помечаем в debug.
STYLE_HINT_LIMIT = 600

CODE_FURNISH_INVALID = "plan_furnish_invalid_json"

FURNISH_SYSTEM_PROMPT = """You are an interior architect placing furniture on a 2D floor plan.

INPUT: a JSON with the room footprint and openings. Units: door-widths, "dw" (1 dw = 0.85 m).
Coordinates: x to the right (0 = left wall), y downward (0 = back wall). Walls: back=top, front=bottom.

TASK: furnish this {ROOM_TYPE}. Return ONLY a JSON array "furniture" (no fences, no comments):
[{"kind": "...", "center_dw": [x, y], "size_m": [w, d], "rotation": 0|90|180|270}, ...]

Allowed kinds: bed_double, bed, wardrobe, dresser, side_table, desk, chair, armchair, rug, plant, tv_stand.
size_m is the footprint in METERS before rotation (w along x, d along y at rotation 0).
rotation 0 means the headboard/back faces the back wall (top). 90 = rotated clockwise 90 degrees
(headboard faces right wall), 180 = faces front, 270 = faces left.

Allowed kinds (extra, for non-bedroom rooms): sofa, table, chair, kitchen_run, sink, hob, fridge.

ERGONOMIC RULES (hard requirements, validated by code):
- Everything fully inside the room polygon; large case furniture (bed, wardrobe, dresser, desk) against a wall.
- Do NOT block the door swing arc: keep a quarter-circle of radius = door width free at the door.
- Keep a clear approach zone in front of every door: rectangle of the door width, extending 0.7 m beyond the swing arc into the room (rug is allowed there).
- Furniture with opening fronts (wardrobe, fridge, dresser) must stay >= 0.4 m away from any door frame.
- Keep a 0.5 m strip in front of every window (incl. floor-to-ceiling windows) free of tall furniture (wardrobe, fridge, kitchen_run); low furniture and rug are fine.
- Keep passages >= 0.7 m between furniture pieces and between furniture and opposite walls.
- Bed accessible from two long sides (>= 0.5 m each side) where possible; side tables flank the headboard.
- Typical sizes: bed_double 1.6x2.05, wardrobe 0.6 deep x 1.6-2.4 wide, side_table 0.45x0.45, dresser 0.5x1.0, sofa 0.9x2.2, fridge 0.65x0.65.
- "partitions" in the input are interior wall stubs (solid walls inside the room, wall thickness = 0.25 dw centered on their axis). Never place anything overlapping them (not even a rug), and keep the 0.7 m passage at a partition's free end clear.
"""

_ARRAY_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class FurnishError(ValueError):
    """Расстановка не получена. `code` — машинная причина (как у бэкенда)."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


def parse_furniture(text: str, plan: Optional[dict] = None) -> list[dict]:
    """Ответ модели -> список валидных предметов.

    Один битый предмет не хоронит всю расстановку: невалидные элементы
    отбрасываются, ошибка поднимается только если валидных не осталось ни одного.
    Схема проверяет сам предмет (конечные числа, размер в метрах), а `plan` —
    привязку к комнате: центр должен лежать в её габарите с допуском.

    Raises:
        FurnishError(CODE_FURNISH_INVALID): не JSON-массив или все элементы битые.
    """
    t = (text or "").strip()
    m = _ARRAY_RE.search(t)
    if m:
        t = m.group(1).strip()
    i, j = t.find("["), t.rfind("]")
    if i != -1 and j > i:
        t = t[i:j + 1]
    try:
        data = json.loads(t)
    except (ValueError, TypeError) as exc:
        raise FurnishError(CODE_FURNISH_INVALID, "not a JSON array: %s" % exc) from exc
    if isinstance(data, dict):
        data = data.get("furniture", [])
    if not isinstance(data, list):
        raise FurnishError(CODE_FURNISH_INVALID, "expected list, got %s" % type(data).__name__)

    bounds = _center_bounds(plan)
    items: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            item = validate_furniture_item(entry)
        except PlanDataError as exc:
            logging.debug("desow_plan: skipping invalid furniture item %s: %s", entry, exc)
            continue
        if bounds is not None and not _center_inside(item["center_dw"], bounds):
            logging.debug("desow_plan: skipping furniture item outside the room: %s", item)
            continue
        items.append(item)
    if not items:
        raise FurnishError(CODE_FURNISH_INVALID, "no valid furniture items in response")
    return items


def _center_bounds(plan: Optional[dict]) -> Optional[tuple[float, float, float, float]]:
    """Габарит комнаты, расширенный допуском. None — план не передан (юнит-вызов)."""
    if not isinstance(plan, dict) or not isinstance(plan.get("room"), dict):
        return None
    try:
        x0, y0, x1, y1 = room_bbox(plan["room"])
    except (KeyError, TypeError, ValueError):
        return None
    t = CENTER_TOLERANCE_DW
    return x0 - t, y0 - t, x1 + t, y1 + t


def _center_inside(center, bounds) -> bool:
    x0, y0, x1, y1 = bounds
    return x0 <= center[0] <= x1 and y0 <= center[1] <= y1


def build_messages(plan: dict, room_type: str, style_hint: str = "", seed: int = 0) -> list[dict]:
    """Стартовый диалог расстановки: системный промпт + план без мебели.

    `style_hint` и `seed` — добавка ноды (у бэкенда их нет). Обе уезжают
    отдельными помеченными секциями ПОСЛЕ Room JSON: правила эргономики живут в
    системном промпте и смешивать их с пожеланиями пользователя нельзя —
    свободный текст рядом с правилом читается моделью как его отмена.
    """
    # `camera` в промпт не едет: это якорь ракурса для картиночной модели, а не
    # данные расстановки. Пустив её в Room JSON, мы бы молча поменяли вход
    # размещающей модели (и её раскладки) ради поля, которое ей нечем применить.
    room_only = {k: v for k, v in plan.items() if k not in ("furniture", "camera")}
    user = ["Room JSON:\n" + json.dumps(room_only, ensure_ascii=False)]
    hint = (style_hint or "").strip()[:STYLE_HINT_LIMIT]
    if hint:
        user.append(
            "STYLE PREFERENCES (affect WHICH pieces you choose and their proportions; "
            "the ergonomic rules above stay in force):\n" + hint
        )
    if seed:
        user.append(
            "LAYOUT VARIANT ID: %d. Runs with different ids must differ in arrangement, "
            "runs with the same id must repeat it." % int(seed)
        )
    return [
        {
            "role": "system",
            "content": FURNISH_SYSTEM_PROMPT.replace(
                "{ROOM_TYPE}", (room_type or "room").upper().replace("_", " ")
            ),
        },
        {"role": "user", "content": "\n\n".join(user)},
    ]


def place_furniture(
    complete,
    plan: dict,
    room_type: str,
    *,
    max_retries: int = MAX_FURNISH_RETRIES,
    style_hint: str = "",
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Расстановка с циклом ре-промптов. Возврат: `(furniture, meta)`.

    `complete(messages) -> str` — один вызов текстовой модели (сеть живёт вне
    пакета). meta: `{retries, violations, calls, violations_by_attempt}`.
    Оставшиеся после лимита попыток нарушения — не повод отказать в генерации
    (планка качества «правдоподобно, не идеально»); они уезжают в meta для
    наблюдаемости.
    """
    messages = build_messages(plan, room_type, style_hint, seed)
    # violations_by_attempt — сверх бэкендового meta: у ноды нет логов сервиса,
    # а разбираться, почему расстановка «почти валидная», надо по debug-выходу.
    meta = {"retries": 0, "violations": [], "calls": 0, "violations_by_attempt": []}

    furniture: Optional[list[dict]] = None
    text = ""
    last_error: Optional[FurnishError] = None
    for attempt in range(max_retries + 1):
        text = complete(messages)
        meta["calls"] += 1
        try:
            candidate = parse_furniture(text, plan)
        except FurnishError as exc:
            last_error = exc
            meta["violations_by_attempt"].append(["ответ не разобран: %s" % exc])
            if furniture is not None:
                # Уже есть рабочая расстановка с прошлой итерации — на ней и стоим.
                break
            if attempt >= max_retries:
                raise
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Your reply was not a valid JSON array. Return ONLY the JSON array of furniture."},
            ]
            meta["retries"] = attempt + 1
            continue

        furniture = candidate
        errs = validate_furniture(plan, furniture)
        meta["violations"] = errs
        meta["violations_by_attempt"].append(list(errs))
        if not errs or attempt >= max_retries:
            break
        meta["retries"] = attempt + 1
        messages += [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "Validator found violations, fix them and return the corrected full JSON array only:\n- "
                + "\n- ".join(errs),
            },
        ]

    if furniture is None:   # недостижимо: последняя неудачная попытка уже подняла ошибку
        raise last_error or FurnishError(CODE_FURNISH_INVALID, "no furniture produced")
    return furniture, meta

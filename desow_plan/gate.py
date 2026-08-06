# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/gate.py - синхронизировать при правках.
# Отличие от источника: только импорт схемы (schema -> schema_lite). Двойное
# ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
"""Гейт проёмов: комната обязана иметь дверь и окно (требование приёмки Фазы 1),
и ни один проём не должен налезать на другой или на угол.

Сканер видит только то, что попало в кадр; front-стена находится за камерой,
поэтому на части боевых кадров (например scan_379) итоговый набор проёмов
оставался вообще без двери. Промптовое правило «нарисуй дверь, если её не видно»
срабатывало непредсказуемо, поэтому условие исполняет код:

- нет ни одной двери -> дверь во front-стену у угла (0.2 м от угла до косяка,
  петли у того же угла, открывание внутрь);
- нет ни одного окна -> окно по центру front-стены.

Обе вставки идут через общий `geometry.place_opening`, то есть считаются с уже
стоящими проёмами. Перед ними работает `resolve_opening_conflicts`: он разводит
то, что пришло от VLM и мержа (наложения, проём впритык к углу, проём шире
стены). Порядок именно такой — гейт должен видеть уже вычищенную стену, иначе
свободные интервалы считаются по мусорной картине.

Гейт работает ПОСЛЕ мержа и ДО расстановки мебели: валидатор расстановки должен
видеть дугу вставленной двери и полосу перед вставленным окном.
"""
from __future__ import annotations

from .geometry import clamp, occupied_spans, place_opening, usable_spans, wall_span
from .schema_lite import (
    DEFAULT_WIDTH_DW,
    DOOR_TYPES,
    MIN_CORNER_CLEARANCE_DW,
    MIN_OPENING_WIDTH_DW,
    WINDOW_TYPES,
)

GATE_WALL = "front"
MIN_WINDOW_WIDTH_DW = 0.8   # окно уже 0.68 м рисовать бессмысленно


def resolve_opening_conflicts(plan: dict) -> list[str]:
    """Разводит проёмы, налезающие друг на друга, на угол или на край стены.

    Источник таких проёмов — не только деградация мержа: модель тоже присылает
    проём впритык к углу (боевой кадр e4 серии v83: passage вплотную к углу
    «открыл» его на чертеже). Рендер клэмпит проём внутрь стены, но простенка не
    гарантирует, и картинка расходилась бы с сохранённым planjson.

    По каждой стене слева направо: проём клэмпится в стену с простенком
    MIN_CORNER_CLEARANCE от угла, при наложении на предыдущий сдвигается вправо,
    при нехватке места сужается, а если сужать некуда — выбрасывается (план без
    проёма честнее плана с проёмом внахлёст). Проёмы, которые ни с чем не
    конфликтуют, не двигаются и не меняют ширину.

    Возвращает пометки вида `moved:door/front`, `narrowed:window/back 1.60->1.20`,
    `dropped:passage/left` — они уходят в meta плана и в debug ноды.
    """
    notes: list[str] = []
    openings = plan.get("openings") or []
    room = plan.get("room") or {}
    by_wall: dict = {}
    for op in openings:
        by_wall.setdefault(op.get("wall"), []).append(op)

    dropped: set[int] = set()
    for wall, items in by_wall.items():
        if not isinstance(wall, str):
            continue   # ребро полигона по индексу: длину стены здесь не резолвим
        start, end = wall_span(room, wall)
        cursor = start + MIN_CORNER_CLEARANCE_DW
        for op in sorted(items, key=lambda o: float(o.get("offset_dw", 0))):
            width = float(op.get("width_dw", 0))
            available = end - MIN_CORNER_CLEARANCE_DW - cursor
            if available < min(width, MIN_OPENING_WIDTH_DW):
                dropped.add(id(op))
                notes.append("dropped:%s/%s" % (op.get("type"), wall))
                continue
            if width > available:
                notes.append("narrowed:%s/%s %.2f->%.2f" % (op.get("type"), wall, width, available))
                width = available
            offset = float(op.get("offset_dw", 0))
            low, high = cursor + width / 2, end - MIN_CORNER_CLEARANCE_DW - width / 2
            # Допуск в 1e-3 dw (0.85 мм) — меньше пикселя чертежа. Без него каждый
            # проём, поставленный ровно по норме и округлённый до 3 знаков, ловил
            # бы «сдвиг» на своём же округлении и засорял пометки.
            if offset < low - 1e-3 or offset > high + 1e-3:
                moved = clamp(offset, low, high)
                notes.append("moved:%s/%s %.2f->%.2f" % (op.get("type"), wall, offset, moved))
                offset = moved
            op["offset_dw"] = round(offset, 3)
            op["width_dw"] = round(width, 3)
            cursor = offset + width / 2 + MIN_CORNER_CLEARANCE_DW

    if dropped:
        plan["openings"] = [op for op in openings if id(op) not in dropped]
    return notes


def ensure_door_and_window(plan: dict) -> list[str]:
    """Дополняет `plan['openings']` недостающими дверью/окном. Возвращает пометки.

    Пометки уходят в meta плана: `door_inserted` / `window_inserted` (вставили),
    `door_gate_skipped` / `window_gate_skipped` (на front-стене не нашлось места —
    например, она целиком занята панорамным остеклением).
    """
    notes: list[str] = []
    openings = plan.setdefault("openings", [])
    present = {op.get("type") for op in openings}
    room = plan["room"]
    wall_start, wall_end = wall_span(room, GATE_WALL)

    if not (present & DOOR_TYPES):
        width = DEFAULT_WIDTH_DW["door"]
        spans = usable_spans(wall_start, wall_end, occupied_spans(openings, GATE_WALL))
        placement = place_opening(spans, width, wall_start, wall_end, anchor="corner")
        if placement is None:
            notes.append("door_gate_skipped")
        else:
            offset, eff_width, hinge = placement
            openings.append({
                "type": "door",
                "wall": GATE_WALL,
                "offset_dw": round(offset, 3),
                "width_dw": round(eff_width, 3),
                "swing": {"hinge": hinge, "direction": "in"},
            })
            notes.append("door_inserted")

    present = {op.get("type") for op in openings}
    if not (present & WINDOW_TYPES):
        width = DEFAULT_WIDTH_DW["window"]
        spans = usable_spans(wall_start, wall_end, occupied_spans(openings, GATE_WALL))
        placement = place_opening(
            spans, width, wall_start, wall_end, anchor="center", min_width=MIN_WINDOW_WIDTH_DW
        )
        if placement is None:
            notes.append("window_gate_skipped")
        else:
            offset, eff_width, _side = placement
            openings.append({
                "type": "window",
                "wall": GATE_WALL,
                "offset_dw": round(offset, 3),
                "width_dw": round(eff_width, 3),
            })
            notes.append("window_inserted")

    return notes

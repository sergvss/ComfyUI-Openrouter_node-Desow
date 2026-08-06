# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/gate.py - синхронизировать при правках.
# Отличие от источника: только импорт схемы (schema -> schema_lite). Двойное
# ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
"""Гейт проёмов: комната обязана иметь дверь и окно (требование приёмки Фазы 1).

Сканер видит только то, что попало в кадр; front-стена находится за камерой,
поэтому на части боевых кадров (например scan_379) итоговый набор проёмов
оставался вообще без двери. Промптовое правило «нарисуй дверь, если её не видно»
срабатывало непредсказуемо, поэтому условие исполняет код:

- нет ни одной двери -> дверь во front-стену у угла (0.2 м от угла до косяка,
  петли у того же угла, открывание внутрь);
- нет ни одного окна -> окно по центру front-стены.

Гейт работает ПОСЛЕ мержа и ДО расстановки мебели: валидатор расстановки должен
видеть дугу вставленной двери и полосу перед вставленным окном.
"""
from __future__ import annotations

from .geometry import free_spans, wall_span
from .schema_lite import DEFAULT_WIDTH_DW, DOOR_TYPES, DW_M, WINDOW_TYPES

GATE_WALL = "front"
DOOR_CORNER_CLEARANCE_M = 0.2
DOOR_CORNER_CLEARANCE_DW = DOOR_CORNER_CLEARANCE_M / DW_M
MIN_WINDOW_WIDTH_DW = 0.8   # окно уже 0.68 м рисовать бессмысленно


def _occupied_spans(plan: dict, wall: str) -> list[tuple[float, float]]:
    """Занятые проёмами отрезки указанной стены в глобальных offset-координатах."""
    spans: list[tuple[float, float]] = []
    for op in plan.get("openings", []):
        if op.get("wall") != wall:
            continue
        off, w = float(op.get("offset_dw", 0)), float(op.get("width_dw", 0))
        spans.append((off - w / 2, off + w / 2))
    return spans


def _place_near_corner(spans: list[tuple[float, float]], width: float, wall_start: float, wall_end: float):
    """Центр проёма шириной width у ближайшего к углу свободного места.

    Возврат: `(offset, hinge)` или None, если ни один свободный интервал не вмещает
    проём. hinge — «left»/«right» по тому углу, к которому проём прижат.
    """
    fitting = [(a, b) for a, b in spans if b - a >= width - 1e-6]
    if not fitting:
        return None
    # Ближайший к любому из углов интервал; из двух вариантов берём тот, чей
    # свободный край ближе к своему углу.
    best = min(fitting, key=lambda s: min(s[0] - wall_start, wall_end - s[1]))
    a, b = best
    if (a - wall_start) <= (wall_end - b):
        offset = a + min(DOOR_CORNER_CLEARANCE_DW, max(0.0, (b - a) - width)) + width / 2
        return offset, "left"
    offset = b - min(DOOR_CORNER_CLEARANCE_DW, max(0.0, (b - a) - width)) - width / 2
    return offset, "right"


def _place_centered(spans: list[tuple[float, float]], width: float, wall_start: float, wall_end: float):
    """Центр проёма по центру стены; при коллизии — центр самого широкого свободного
    интервала, при необходимости с сужением проёма. Возврат: `(offset, width)` или None."""
    center = (wall_start + wall_end) / 2
    for a, b in spans:
        if a <= center - width / 2 + 1e-6 and center + width / 2 <= b + 1e-6:
            return center, width
    widest = max(spans, key=lambda s: s[1] - s[0], default=None)
    if widest is None:
        return None
    span_len = widest[1] - widest[0]
    if span_len < MIN_WINDOW_WIDTH_DW:
        return None
    return (widest[0] + widest[1]) / 2, min(width, span_len)


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
        spans = free_spans(wall_start, wall_end, _occupied_spans(plan, GATE_WALL))
        placement = _place_near_corner(spans, width, wall_start, wall_end)
        if placement is None:
            notes.append("door_gate_skipped")
        else:
            offset, hinge = placement
            openings.append({
                "type": "door",
                "wall": GATE_WALL,
                "offset_dw": round(offset, 3),
                "width_dw": width,
                "swing": {"hinge": hinge, "direction": "in"},
            })
            notes.append("door_inserted")

    present = {op.get("type") for op in openings}
    if not (present & WINDOW_TYPES):
        width = DEFAULT_WIDTH_DW["window"]
        spans = free_spans(wall_start, wall_end, _occupied_spans(plan, GATE_WALL))
        placement = _place_centered(spans, width, wall_start, wall_end)
        if placement is None:
            notes.append("window_gate_skipped")
        else:
            offset, eff_width = placement
            openings.append({
                "type": "window",
                "wall": GATE_WALL,
                "offset_dw": round(offset, 3),
                "width_dw": round(eff_width, 3),
            })
            notes.append("window_inserted")

    return notes

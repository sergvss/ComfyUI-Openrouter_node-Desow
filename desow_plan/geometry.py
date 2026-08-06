# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/geometry.py - синхронизировать при правках.
# Отличие от источника: только импорт схемы (schema -> schema_lite). Двойное
# ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
"""Геометрия плана в координатах dw (порт `hybrid-proto/render_plan.py`).

Общая основа для рендера, валидаторов и гейта проёмов: полигон комнаты, резолв
именованной стены в ребро полигона, прямоугольник простенка, отрезок проёма.
Отдельный модуль (а не внутренности рендера) — чтобы валидаторы не тянули Pillow.
"""
from __future__ import annotations

import math

from .schema_lite import WALL_T_DW

# Единичные векторы направлений «внутрь комнаты» для стен-якорей простенков.
PART_VEC = {"front": (0.0, 1.0), "back": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def room_polygon_dw(room: dict) -> list[tuple[float, float]]:
    """Полигон пола в dw, y вниз (back сверху). Для rectangle строим из width/depth."""
    if room.get("shape") == "l_shape" and room.get("polygon_dw"):
        return [(float(x), float(y)) for x, y in room["polygon_dw"]]
    w = float(room["width_dw"])
    d = float(room["depth_dw"])
    return [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]


def room_bbox(room: dict) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) габарита комнаты в dw."""
    poly = room_polygon_dw(room)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def inside_polygon(poly: list, x: float, y: float) -> bool:
    """Ray casting: точка внутри полигона."""
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def _wall_candidates(room: dict, poly: list, wall: str) -> list[tuple[int, int, float, float]]:
    """Рёбра полигона, относящиеся к именованной стене.

    Возврат: список (i, j, global_start, global_end), где global_* — координата
    вдоль стены в глобальной системе (back/front — по x, left/right — по y).
    """
    n = len(poly)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    out: list[tuple[int, int, float, float]] = []
    for i in range(n):
        j = (i + 1) % n
        (ax, ay), (bx, by) = poly[i], poly[j]
        horizontal = abs(ay - by) < 1e-9
        if wall in ("back", "front") and horizontal:
            # Для L-формы «своя» сторона определяется половиной габарита:
            # back — горизонтальные рёбра верхней половины, front — нижней.
            side_ok = (
                (min(ay, by) < (min(ys) + max(ys)) / 2)
                if wall == "back"
                else (min(ay, by) >= (min(ys) + max(ys)) / 2)
            )
            if side_ok:
                out.append((i, j, min(ax, bx), max(ax, bx)))
        elif wall in ("left", "right") and not horizontal:
            side_ok = (ax < (min(xs) + max(xs)) / 2) if wall == "left" else (ax >= (min(xs) + max(xs)) / 2)
            if side_ok:
                out.append((i, j, min(ay, by), max(ay, by)))
    return out


def wall_edge(room: dict, wall, poly: list, offset_dw: float | None = None) -> tuple[int, int, float]:
    """Ребро полигона (i -> j) для стены + локальный offset вдоль ребра.

    Для прямоугольника — фиксированная карта. Для l_shape именованная стена может
    состоять из НЕСКОЛЬКИХ рёбер (два back-сегмента у L): выбираем ребро, в чей
    глобальный диапазон попадает offset, и пересчитываем offset в локальный.
    """
    n = len(poly)
    if isinstance(wall, int):
        return wall, (wall + 1) % n, float(offset_dw or 0)
    if room.get("shape") != "l_shape" or n == 4:
        names = {"back": (0, 1), "right": (1, 2), "front": (2, 3), "left": (3, 0)}
        i, j = names[wall]
        return i, j, float(offset_dw or 0)
    off = float(offset_dw or 0)
    candidates = _wall_candidates(room, poly, wall)
    if not candidates:
        raise ValueError(f"нет ребра для стены {wall}")
    # Ребро, содержащее глобальный offset; иначе ближайшее.
    best = min(
        candidates,
        key=lambda c: 0 if c[2] - 1e-6 <= off <= c[3] + 1e-6 else min(abs(off - c[2]), abs(off - c[3])),
    )
    i, j, g0, _ = best
    return i, j, off - g0


def wall_span(room: dict, wall: str) -> tuple[float, float]:
    """Глобальный диапазон стены (start, end) в dw — для гейта проёмов.

    Для l_shape берём самый длинный сегмент этой стены: гейт вставляет дверь/окно
    в реальную сплошную стену, а не в «виртуальную» суммарную длину.
    """
    poly = room_polygon_dw(room)
    if room.get("shape") == "l_shape" and len(poly) > 4:
        candidates = _wall_candidates(room, poly, wall)
        if candidates:
            best = max(candidates, key=lambda c: c[3] - c[2])
            return best[2], best[3]
    x0, y0, x1, y1 = room_bbox(room)
    if wall in ("back", "front"):
        return x0, x1
    return y0, y1


def wall_params(poly: list, i: int, j: int):
    """(start, unit_vector, length) ребра в dw c конвенцией начала отсчёта offset_dw:
    back/front — левый угол, left/right — back-угол."""
    ax, ay = poly[i]
    bx, by = poly[j]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    if abs(ux) > abs(uy):   # горизонтальная стена: начало — меньший x
        if ux < 0:
            ax, ay, ux, uy = bx, by, -ux, -uy
    else:                   # вертикальная стена: начало — меньший y (back)
        if uy < 0:
            ax, ay, ux, uy = bx, by, -ux, -uy
    return (ax, ay), (ux, uy), length


def partition_rect_dw(data: dict, p: dict) -> tuple[float, float, float, float]:
    """Прямоугольник внутреннего простенка (x0, y0, x1, y1) в dw.

    Привязанный (attach = стена): ось перпендикулярна стене, растёт внутрь комнаты
    от внутренней грани; offset_dw — от начала стены (конвенция как у openings).
    Свободностоящий (attach = "free"): ось от start_dw вдоль direction.
    Толщина = толщине стены (WALL_T_DW), ось по центру толщины.
    """
    poly = room_polygon_dw(data["room"])
    xs = [q[0] for q in poly]
    ys = [q[1] for q in poly]
    t = WALL_T_DW
    length = float(p["length_dw"])
    attach = p.get("attach", "free")
    if attach == "free":
        sx, sy = (float(v) for v in p["start_dw"])
        dx, dy = PART_VEC[p["direction"]]
    else:  # от внутренней грани стены внутрь комнаты (bbox достаточно: стены осепараллельны)
        off = float(p["offset_dw"])
        if attach == "back":
            sx, sy, (dx, dy) = min(xs) + off, min(ys), PART_VEC["front"]
        elif attach == "front":
            sx, sy, (dx, dy) = min(xs) + off, max(ys), PART_VEC["back"]
        elif attach == "left":
            sx, sy, (dx, dy) = min(xs), min(ys) + off, PART_VEC["right"]
        else:  # right
            sx, sy, (dx, dy) = max(xs), min(ys) + off, PART_VEC["left"]
    ex, ey = sx + dx * length, sy + dy * length
    if abs(dx) > 0:  # горизонтальная ось: толщина по y
        return min(sx, ex), sy - t / 2, max(sx, ex), sy + t / 2
    return sx - t / 2, min(sy, ey), sx + t / 2, max(sy, ey)


def opening_span(data: dict, op: dict) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Отрезок проёма (c0, c1) и внутренняя нормаль (в комнату) в координатах dw."""
    room = data["room"]
    poly = room_polygon_dw(room)
    i, j, local_off = wall_edge(room, op["wall"], poly, op.get("offset_dw"))
    (sx, sy), (ux, uy), ln = wall_params(poly, i, j)
    w = float(op["width_dw"])
    a = max(0.0, local_off - w / 2)
    b = min(ln, local_off + w / 2)
    c0 = (sx + ux * a, sy + uy * a)
    c1 = (sx + ux * b, sy + uy * b)
    # Внутренняя нормаль: кандидат и проверка точкой внутри полигона.
    nx, ny = -uy, ux
    mx, my = (c0[0] + c1[0]) / 2 + nx * 0.05, (c0[1] + c1[1]) / 2 + ny * 0.05
    if not inside_polygon(poly, mx, my):
        nx, ny = uy, -ux
    return c0, c1, (nx, ny)


def free_spans(
    wall_start: float, wall_end: float, occupied: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Свободные интервалы стены после вычитания занятых проёмами отрезков."""
    spans: list[tuple[float, float]] = []
    cursor = wall_start
    for a, b in sorted(occupied):
        a, b = max(a, wall_start), min(b, wall_end)
        if b <= cursor:
            continue
        if a > cursor:
            spans.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < wall_end:
        spans.append((cursor, wall_end))
    return spans

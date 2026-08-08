# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/validate.py - синхронизировать при правках.
# Отличие от источника: только импорт схемы (schema -> schema_lite). Двойное
# ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
"""Детерминированные проверки плана (порт `hybrid-proto/check.py` + `furnish.py`).

Два уровня:
1. `validate_structure` — инварианты схемы: проёмы в пределах стен, без взаимных
   пересечений, дверь со swing, простенок не перекрывает комнату.
2. `validate_furniture` — эргономика расстановки: containment, попарные
   пересечения, дуга и подход двери, полоса перед окном, зазор 0.4 м от коробки
   для открывающейся техники, простенок как препятствие.

Нарушения возвращаются списком человекочитаемых строк: `validate_structure` —
в meta плана (рендер клэмпит проёмы, план всё равно строится),
`validate_furniture` — обратно в LLM ре-промптом (паттерн Architect-Ant).
"""
from __future__ import annotations

import math

from .geometry import (
    PART_VEC,
    inside_polygon,
    opening_span,
    partition_rect_dw,
    room_bbox,
    room_polygon_dw,
)
from .schema_lite import (
    CONTACT_TOL_DW,
    DOOR_APPROACH_M,
    DOOR_FRAME_CLEAR_M,
    DW_M,
    OPENING_FRONT_KINDS,
    PASSAGE_CLEAR_M,
    TALL_KINDS,
    WALL_LEN_KEYS,
    WINDOW_STRIP_M,
)

PASSAGE_CLEAR_DW = PASSAGE_CLEAR_M / DW_M


def rect_of(f: dict) -> tuple[float, float, float, float]:
    """bbox предмета мебели в dw с учётом поворота."""
    w_dw, d_dw = f["size_m"][0] / DW_M, f["size_m"][1] / DW_M
    if int(f.get("rotation", 0)) % 180 == 90:
        w_dw, d_dw = d_dw, w_dw
    cx, cy = f["center_dw"]
    return (cx - w_dw / 2, cy - d_dw / 2, cx + w_dw / 2, cy + d_dw / 2)


def _rects_overlap(a, b, tol: float = CONTACT_TOL_DW) -> bool:
    """Пересекаются ли прямоугольники ГЛУБЖЕ допуска касания.

    Допуск общий для всех потребителей (простенки, попарные пересечения, зоны
    окна и двери): везде вопрос один — «предметы стоят рядом или один залез на
    другой». Касание и наложение в миллиметры — цена округления координат
    моделью, а не ошибка расстановки (см. CONTACT_TOL_M в схеме).
    """
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


def _rect_seg_dist(rect, p0, p1, samples: int = 24) -> float:
    """Мин. расстояние прямоугольник-отрезок (сэмплированием отрезка)."""
    x0, y0, x1, y1 = rect
    best = float("inf")
    for i in range(samples + 1):
        t = i / samples
        px_ = p0[0] + (p1[0] - p0[0]) * t
        py_ = p0[1] + (p1[1] - p0[1]) * t
        qx = min(max(px_, x0), x1)
        qy = min(max(py_, y0), y1)
        best = min(best, math.hypot(qx - px_, qy - py_))
    return best


def validate_partitions(data: dict) -> list[str]:
    """Простенок не перекрывает комнату полностью (проход >= 0.7 м от каждого
    свободного конца по оси), концы внутри комнаты."""
    errs: list[str] = []
    room = data.get("room", {})
    bx0, by0, bx1, by1 = room_bbox(room)
    for p in data.get("partitions", []):
        attach = p.get("attach")
        length = float(p.get("length_dw", 0))
        if length <= 0:
            errs.append(f"partition ({attach}): неположительная длина {length}")
            continue
        if attach == "free":
            vec = PART_VEC.get(p.get("direction"))
            start = p.get("start_dw")
            if vec is None or not start:
                errs.append("free-partition без start_dw/direction")
                continue
            sx, sy = float(start[0]), float(start[1])
            ex, ey = sx + vec[0] * length, sy + vec[1] * length
            free_ends = [((sx, sy), (-vec[0], -vec[1])), ((ex, ey), vec)]
        elif attach in PART_VEC:
            off = float(p.get("offset_dw", 0))
            if attach == "back":
                s, vec = (bx0 + off, by0), PART_VEC["front"]
            elif attach == "front":
                s, vec = (bx0 + off, by1), PART_VEC["back"]
            elif attach == "left":
                s, vec = (bx0, by0 + off), PART_VEC["right"]
            else:
                s, vec = (bx1, by0 + off), PART_VEC["left"]
            ex, ey = s[0] + vec[0] * length, s[1] + vec[1] * length
            free_ends = [((ex, ey), vec)]
        else:
            errs.append(f"partition: неизвестный attach {attach}")
            continue
        for (px_, py_), (dx_, dy_) in free_ends:
            if not (bx0 - 1e-6 <= px_ <= bx1 + 1e-6 and by0 - 1e-6 <= py_ <= by1 + 1e-6):
                errs.append(f"partition ({attach}): конец [{px_:.2f},{py_:.2f}] вне комнаты")
                continue
            if dx_ > 0:
                dist = bx1 - px_
            elif dx_ < 0:
                dist = px_ - bx0
            elif dy_ > 0:
                dist = by1 - py_
            else:
                dist = py_ - by0
            if dist < PASSAGE_CLEAR_DW - 1e-6:
                errs.append(
                    f"partition ({attach}): проход {dist * DW_M:.2f} м < {PASSAGE_CLEAR_M} м от свободного конца"
                )
    return errs


def validate_structure(data: dict) -> list[str]:
    """Инварианты схемы. Возвращает список нарушений (пусто = ок)."""
    errs: list[str] = []
    room = data.get("room", {})
    if room.get("shape") not in ("rectangle", "l_shape"):
        errs.append(f"room.shape неизвестен: {room.get('shape')}")
    if room.get("shape") == "l_shape" and len(room.get("polygon_dw") or []) < 6:
        errs.append("l_shape без polygon_dw из 6 вершин")
    by_wall: dict[str, list[tuple[float, float]]] = {}
    for op in data.get("openings", []):
        wall = op.get("wall")
        off, w = float(op.get("offset_dw", -1)), float(op.get("width_dw", 0))
        if room.get("shape") == "rectangle" and wall in WALL_LEN_KEYS:
            wl = float(room[WALL_LEN_KEYS[wall]])
            if off - w / 2 < -1e-6 or off + w / 2 > wl + 1e-6:
                errs.append(
                    f"{op.get('type')} на {wall}: [{off - w / 2:.2f}..{off + w / 2:.2f}] вне стены 0..{wl}"
                )
        if op.get("type") in ("door", "double_door") and not op.get("swing"):
            errs.append(f"дверь на {wall} без swing")
        by_wall.setdefault(str(wall), []).append((off - w / 2, off + w / 2))
    for wall, spans in by_wall.items():
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            if b[0] < a[1] - 1e-6:
                errs.append(f"проёмы на {wall} пересекаются: {a} и {b}")
    errs += validate_partitions(data)
    return errs


def validate_furniture(room_data: dict, furniture: list) -> list[str]:
    """Containment, попарные пересечения, дуга двери, полоса перед окном,
    подход к двери, зазор техники от коробки, простенки как препятствия."""
    errs: list[str] = []
    room = room_data["room"]
    W, D = float(room["width_dw"]), float(room["depth_dw"])
    # Для l_shape containment проверяем по полигону: bbox недостаточно (есть вырез).
    poly = None
    if room.get("shape") == "l_shape" and room.get("polygon_dw"):
        poly = room_polygon_dw(room)

    rects: list[tuple[str, float, float, float, float]] = []
    for f in furniture:
        x0, y0, x1, y1 = rect_of(f)
        if poly is not None:
            # Углы вжимаются внутрь на допуск касания: предмет, вставший вплотную
            # к грани полигона, проверяется как стоящий внутри неё.
            eps = min(CONTACT_TOL_DW, (x1 - x0) / 2, (y1 - y0) / 2)
            corners = [(x0 + eps, y0 + eps), (x1 - eps, y0 + eps), (x1 - eps, y1 - eps), (x0 + eps, y1 - eps)]
            if not all(inside_polygon(poly, cx_, cy_) for cx_, cy_ in corners):
                errs.append(
                    f"{f['kind']} выходит за пределы L-полигона комнаты: "
                    f"[{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}]"
                )
        else:
            # Симметрично по всем четырём границам. Печатаем ещё и величину
            # выхода: округление до сотых в dw само по себе прячет сантиметры и
            # делает сообщение неправдоподобным («3.80 при ширине 3.8»).
            out_dw = max(-x0, -y0, x1 - W, y1 - D)
            if out_dw > CONTACT_TOL_DW:
                errs.append(
                    f"{f['kind']} выходит за пределы комнаты на {out_dw * DW_M * 1000:.0f} мм: "
                    f"[{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}] room {W}x{D}"
                )
        rects.append((f["kind"], x0, y0, x1, y1))

    # Простенки — препятствия: никакая мебель (включая ковёр) их не пересекает.
    for prt in room_data.get("partitions", []) or []:
        pr = partition_rect_dw(room_data, prt)
        for kind, x0, y0, x1, y1 in rects:
            if _rects_overlap(pr, (x0, y0, x1, y1)):
                errs.append(f"{kind} пересекает внутренний простенок ({prt.get('attach')})")

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            k1, ax0, ay0, ax1, ay1 = rects[i]
            k2, bx0, by0, bx1, by1 = rects[j]
            if "rug" in (k1, k2):   # ковёр может лежать под мебелью
                continue
            if _rects_overlap((ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1)):
                errs.append(f"пересечение {k1} и {k2}")

    # Дуга двери: петля — центр дуги, радиус = ширина проёма.
    for op in room_data.get("openings", []):
        if op["type"] not in ("door", "double_door"):
            continue
        off, w = float(op["offset_dw"]), float(op["width_dw"])
        hinge = (op.get("swing") or {}).get("hinge", "back")
        wall = op["wall"]
        if wall == "left":
            hp = (0.0, off - w / 2 if hinge == "back" else off + w / 2)
        elif wall == "right":
            hp = (W, off - w / 2 if hinge == "back" else off + w / 2)
        elif wall == "back":
            hp = (off - w / 2 if hinge == "left" else off + w / 2, 0.0)
        else:
            hp = (off - w / 2 if hinge == "left" else off + w / 2, D)
        for kind, x0, y0, x1, y1 in rects:
            if kind == "rug":
                continue
            qx = min(max(hp[0], x0), x1)
            qy = min(max(hp[1], y0), y1)
            dist = math.hypot(qx - hp[0], qy - hp[1])
            if dist < w - 1e-6:
                errs.append(
                    f"{kind} перекрывает дугу двери на {wall} (расст {dist:.2f} < {w:.2f} dw)"
                )

    # Полоса перед окном, подход к двери, зазор техники от коробки.
    for op in room_data.get("openings", []):
        nt = op["type"]
        c0, c1, n_in = opening_span(room_data, op)
        if nt in ("window", "floor_to_ceiling_window"):
            strip_d = WINDOW_STRIP_M / DW_M
            zone = _zone_from_span(c0, c1, n_in, strip_d)
            for kind, x0, y0, x1, y1 in rects:
                if kind in TALL_KINDS and _rects_overlap(zone, (x0, y0, x1, y1)):
                    errs.append(
                        f"{kind} перекрывает полосу {WINDOW_STRIP_M} м перед окном на {op['wall']}"
                    )
        elif nt in ("door", "double_door"):
            w = float(op["width_dw"])
            zone = _zone_from_span(c0, c1, n_in, w + DOOR_APPROACH_M / DW_M)
            for kind, x0, y0, x1, y1 in rects:
                if kind != "rug" and _rects_overlap(zone, (x0, y0, x1, y1)):
                    errs.append(
                        f"{kind} перекрывает подход к двери на {op['wall']} ({DOOR_APPROACH_M} м за дугой)"
                    )
            clear = DOOR_FRAME_CLEAR_M / DW_M
            for kind, x0, y0, x1, y1 in rects:
                if kind in OPENING_FRONT_KINDS:
                    d = _rect_seg_dist((x0, y0, x1, y1), c0, c1)
                    if d < clear - 1e-6:
                        errs.append(
                            f"{kind} ближе {DOOR_FRAME_CLEAR_M} м к коробке двери на {op['wall']} "
                            f"(расст {d * DW_M:.2f} м)"
                        )
    return errs


def _zone_from_span(c0, c1, n_in, depth: float) -> tuple[float, float, float, float]:
    """Прямоугольная зона: отрезок проёма, вытянутый внутрь комнаты на depth."""
    zx0 = min(c0[0], c1[0]) + (n_in[0] * depth if n_in[0] < 0 else 0)
    zy0 = min(c0[1], c1[1]) + (n_in[1] * depth if n_in[1] < 0 else 0)
    zx1 = max(c0[0], c1[0]) + (n_in[0] * depth if n_in[0] > 0 else 0)
    zy1 = max(c0[1], c1[1]) + (n_in[1] * depth if n_in[1] > 0 else 0)
    return zx0, zy0, zx1, zy1

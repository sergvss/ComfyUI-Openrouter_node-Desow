# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/render.py - синхронизировать при правках.
# Отличие от источника: только импорт схемы (schema -> schema_lite). Двойное
# ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
"""Детерминированный рендер 2D-плана из структурного JSON (порт `hybrid-proto/render_plan.py`).

Графстандарт (замерен пипеткой по эталону промптового пути v35):
- холст 1152x928, лист белый (255), пол светло-серый (237), стены сплошной чёрный (0);
- толщина стены = 1/4 ширины двери (эталон: проём 148 px, стена 37 px);
- дверь: разрыв в стене + полотно (тонкий белый прямоугольник с контуром) + четвертьдуга
  (символ один на все `DOOR_TYPES` — они различаются смыслом, а не нотацией); проём
  шире `DOUBLE_LEAF_THRESHOLD_M` — две створки от краёв к центру и две дуги навстречу;
- окно: разрыв + линии по граням стены + двойной штрих + торцевые перемычки;
- проход (passage): чистый разрыв в цвет пола;
- мебель: белая заливка, тонкий чёрный контур, без текста;
- камера (только `draw_camera=True`): тёмный ромб + полупрозрачный оранжевый
  сектор обзора — единственный цветной элемент листа.

Замкнутость контура, равнотолщинность стен и поля листа гарантированы построением,
а не промптом: рендер рисует ровно то, что есть в JSON.
"""
from __future__ import annotations

import io
import math

from PIL import Image, ImageChops, ImageDraw

from .geometry import (
    clamp,
    inside_polygon,
    partition_rect_dw,
    room_polygon_dw,
    wall_edge,
    wall_params,
    wall_span,
)
from .schema_lite import DOOR_TYPES, DW_M, WALL_T_DW, is_double_leaf

# --- константы графстандарта (в финальных пикселях) ---
CANVAS = (1152, 928)
MARGIN = {"left": 104, "right": 104, "top": 79, "bottom": 71}
PAGE = 255
FLOOR = 237
INK = 0
SS = 4          # суперсэмплинг для гладких дуг
THIN = 2        # тонкая линия в финальных px
LEAF_T = 7      # толщина дверного полотна в px

# --- маркер камеры (рисуется только в camera-версии листа) ---
# Канон приёма «камера на плане» для Nano Banana: контрастная точка стояния плюс
# явный указатель направления (docs/design/CAMERA_ON_PLAN_RESEARCH.md). У нас это
# иконка камеры + широкий сектор обзора; сектор полупрозрачный, чтобы мебель
# читалась сквозь него.
CAM_SECTOR_RGB = (245, 165, 122)
CAM_SECTOR_ALPHA = 90       # ~35%: заливка видна, но не спорит с чертежом
CAM_FOV_DEG = 75.0          # раскрытие сектора обзора
CAM_DOT_DW = 0.11           # полудиагональ ромба (legacy-маркер)
# Иконка камеры: корпус со скруглением + выступающий объектив; ось объектива и есть
# направление взгляда. Габарит 0.47 x 0.46 dw (~0.4 м) — вдвое крупнее прежнего
# ромба: узнаваемость силуэта требует места, а на листе это по-прежнему мелкий знак.
CAM_ICON_INSET_DW = 0.03    # зазор от внутренней грани стены до задней стенки корпуса
# Стиль знака (вердикт пользователя, бенч 16): корпус слегка вытянут вдоль
# взгляда, углы ОСТРЫЕ (без скруглений), вокруг всего знака контрастная белая
# обводка 2px финального листа - знак читается на любом фоне (сектор, стена,
# дуга двери).
CAM_ICON_BODY_DW = 0.38     # корпус вдоль взгляда (вытянут)
CAM_ICON_WIDTH_DW = 0.42    # корпус поперёк взгляда
CAM_ICON_LENS_DW = 0.16       # вылет объектива за корпус
CAM_ICON_LENS_W_NEAR_DW = 0.14  # объектив-раструб: ширина у корпуса...
CAM_ICON_LENS_W_FAR_DW = 0.30   # ...и на срезе (трапеция, расширяется по взгляду)
CAM_ICON_BORDER_PX = 2      # обводка в пикселях ФИНАЛЬНОГО листа
# Направление взгляда НА ЛИСТЕ (не в мире): y вниз, поэтому "up" — это -y.
CAM_DIR_VEC = {"up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
# "camera_icon" — иконка + сектор (дефолт); "sector" и "dot" — прежний ромб
# с сектором и без него, оставлены для сравнения и старых вызовов.
CAMERA_STYLES = ("camera_icon", "sector", "dot")
DIAMOND_STYLES = ("sector", "dot")


def render_plan(
    data: dict,
    *,
    with_furniture: bool = True,
    draw_camera: bool = False,
    camera_style: str = "camera_icon",
) -> tuple[bytes, dict]:
    """Рисует план и возвращает (png_bytes, meta).

    meta: масштаб px/dw, толщина стены в px, габарит комнаты в dw и в метрах,
    флаг `scale_fallback` (в комнате нет двери — дверная линейка недоступна).

    `draw_camera` добавляет маркер точки съёмки (блок `camera` плана). Без него
    лист остаётся строго трёхтоновым и БАЙТ-В-БАЙТ прежним: на чистом плане
    расставляется мебель, и любой цветной пиксель на нём был бы помехой.
    """
    room = data["room"]
    poly = room_polygon_dw(room)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    w_dw, d_dw = max(xs) - min(xs), max(ys) - min(ys)

    # Масштаб px/dw: вписываем интерьер + стены в контент-бокс листа.
    # Дополнительно резервируем место под НАРУЖНЫЕ дуги дверей (swing=out),
    # чтобы дуга не съедала поле листа.
    t_dw = WALL_T_DW
    ext = {"left": 0.0, "right": 0.0, "back": 0.0, "front": 0.0}
    for op in data.get("openings", []):
        if op.get("type") in DOOR_TYPES and (op.get("swing") or {}).get("direction") == "out":
            wall = op.get("wall")
            if wall in ext:
                r = float(op.get("width_dw", 1.0))
                if is_double_leaf(op):   # створка вдвое уже проёма — и дуга тоже
                    r = r / 2
                ext[wall] = max(ext[wall], max(0.0, r - t_dw))
    box_w = CANVAS[0] - MARGIN["left"] - MARGIN["right"]
    box_h = CANVAS[1] - MARGIN["top"] - MARGIN["bottom"]
    span_w = w_dw + 2 * t_dw + ext["left"] + ext["right"]
    span_h = d_dw + 2 * t_dw + ext["back"] + ext["front"]
    s = min(box_w / span_w, box_h / span_h)   # px за dw (финальные px)

    total_w, total_h = span_w * s, span_h * s
    ox = MARGIN["left"] + (box_w - total_w) / 2 + (ext["left"] + t_dw) * s - min(xs) * s
    oy = MARGIN["top"] + (box_h - total_h) / 2 + (ext["back"] + t_dw) * s - min(ys) * s

    def P(x_dw, y_dw):  # dw -> суперсэмпл-px
        return ((ox + x_dw * s) * SS, (oy + y_dw * s) * SS)

    img = Image.new("L", (CANVAS[0] * SS, CANVAS[1] * SS), PAGE)
    dr = ImageDraw.Draw(img)
    t = t_dw * s * SS
    thin = THIN * SS
    n_pts = len(poly)

    # 1) Стены: чёрная полоса толщиной t наружу от каждого ребра; на выпуклых
    # углах продлеваем на t, чтобы угол был залит; пол рисуется поверх.
    convex = []
    for k in range(n_pts):
        x0, y0 = poly[(k - 1) % n_pts]
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n_pts]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        convex.append(cross > 0)   # полигон по часовой в y-вниз координатах

    edge_geo = []   # (start_pt, u, n_out, length) в dw — для резолва нормали проёма
    for k in range(n_pts):
        (ax, ay), (bx, by) = poly[k], poly[(k + 1) % n_pts]
        ux, uy = bx - ax, by - ay
        ln = math.hypot(ux, uy)
        ux, uy = ux / ln, uy / ln
        nx, ny = -uy, ux
        mx, my = (ax + bx) / 2 + nx * 0.01, (ay + by) / 2 + ny * 0.01
        if inside_polygon(poly, mx, my):
            nx, ny = uy, -ux
        edge_geo.append(((ax, ay), (ux, uy), (nx, ny), ln))
        ext_a = t_dw if convex[k] else 0.0
        ext_b = t_dw if convex[(k + 1) % n_pts] else 0.0
        pts_dw = [
            (ax - ux * ext_a, ay - uy * ext_a),
            (bx + ux * ext_b, by + uy * ext_b),
            (bx + ux * ext_b + nx * t_dw, by + uy * ext_b + ny * t_dw),
            (ax - ux * ext_a + nx * t_dw, ay - uy * ext_a + ny * t_dw),
        ]
        dr.polygon([P(x, y) for x, y in pts_dw], fill=INK)

    # 2) Пол
    dr.polygon([P(x, y) for x, y in poly], fill=FLOOR)

    # 2b) Внутренние простенки: чёрный блок толщиной стены ПОВЕРХ пола;
    # контур комнаты не меняется, пол остаётся с обеих сторон.
    for prt in data.get("partitions", []):
        px0, py0, px1, py1 = partition_rect_dw(data, prt)
        dr.polygon([P(px0, py0), P(px1, py0), P(px1, py1), P(px0, py1)], fill=INK)

    # 3) Проёмы
    for op in data.get("openings", []):
        i, j, local_off = wall_edge(room, op["wall"], poly, op.get("offset_dw"))
        (sx, sy), (ux, uy), ln = wall_params(poly, i, j)
        nx, ny = None, None
        for (a, u, n_, l_) in edge_geo:
            if abs(l_ - ln) < 1e-9 and (abs(u[0]) == abs(ux) and abs(u[1]) == abs(uy)):
                # то же ребро? сверим, что start лежит на нём
                if (
                    abs((sx - a[0]) * u[1] - (sy - a[1]) * u[0]) < 1e-6
                    and -1e-6 <= (sx - a[0]) * u[0] + (sy - a[1]) * u[1] <= l_ + 1e-6
                ):
                    nx, ny = n_
                    break
        if nx is None:
            raise ValueError(f"edge not found for opening {op}")

        off = local_off
        half = float(op["width_dw"]) / 2
        off = max(half, min(ln - half, off))   # клэмп в стену (инвариант кодом)
        c0 = (sx + ux * (off - half), sy + uy * (off - half))
        c1 = (sx + ux * (off + half), sy + uy * (off + half))

        gap_fill = FLOOR if op["type"] == "passage" else PAGE
        is_window = op["type"] in ("window", "floor_to_ceiling_window")
        gap_pts = [
            c0, c1,
            (c1[0] + nx * t_dw, c1[1] + ny * t_dw),
            (c0[0] + nx * t_dw, c0[1] + ny * t_dw),
        ]
        dr.polygon([P(x, y) for x, y in gap_pts], fill=gap_fill)

        if is_window:   # window и floor_to_ceiling_window рисуются одинаково
            def line_dw(p, q, width=thin):
                dr.line([P(*p), P(*q)], fill=INK, width=int(width))

            line_dw(c0, c1)                                                     # внутренняя грань
            line_dw((c0[0] + nx * t_dw, c0[1] + ny * t_dw),
                    (c1[0] + nx * t_dw, c1[1] + ny * t_dw))                     # внешняя грань
            line_dw(c0, (c0[0] + nx * t_dw, c0[1] + ny * t_dw))                 # торец
            line_dw(c1, (c1[0] + nx * t_dw, c1[1] + ny * t_dw))                 # торец
            for frac in (0.37, 0.63):                                           # двойной штрих
                line_dw((c0[0] + nx * t_dw * frac, c0[1] + ny * t_dw * frac),
                        (c1[0] + nx * t_dw * frac, c1[1] + ny * t_dw * frac))
        elif op["type"] in DOOR_TYPES:
            swing = op.get("swing") or {}
            direction = swing.get("direction", "in")
            hinge = swing.get("hinge")
            # Петли на конце c0 (начало стены: back/левый угол) или c1.
            hinge_at_c0 = hinge not in ("front", "right")
            din = (-nx, -ny) if direction != "out" else (nx, ny)

            def draw_leaf_arc(hp, other, width_dw_leaf):
                lx, ly = din
                tip = (hp[0] + lx * width_dw_leaf, hp[1] + ly * width_dw_leaf)
                leaf_t_dw = LEAF_T / s
                ovx, ovy = other[0] - hp[0], other[1] - hp[1]
                ol = math.hypot(ovx, ovy)
                ovx, ovy = ovx / ol, ovy / ol
                leaf_pts = [
                    hp, tip,
                    (tip[0] + ovx * leaf_t_dw, tip[1] + ovy * leaf_t_dw),
                    (hp[0] + ovx * leaf_t_dw, hp[1] + ovy * leaf_t_dw),
                ]
                dr.polygon([P(x, y) for x, y in leaf_pts], fill=PAGE,
                           outline=INK, width=max(1, thin // 2))
                r = width_dw_leaf * s * SS
                cx, cy = P(*hp)
                bbox = [cx - r, cy - r, cx + r, cy + r]
                a_other = math.degrees(math.atan2(other[1] - hp[1], other[0] - hp[0]))
                a_tip = math.degrees(math.atan2(tip[1] - hp[1], tip[0] - hp[0]))
                a0, a1 = a_other % 360, a_tip % 360
                if (a1 - a0) % 360 <= 180:
                    start, end = a0, a1
                else:
                    start, end = a1, a0
                dr.arc(bbox, start=start, end=end, fill=INK, width=max(1, thin // 2))

            if not is_double_leaf(op):   # одна створка на всю ширину проёма
                hp = c0 if hinge_at_c0 else c1
                other = c1 if hinge_at_c0 else c0
                draw_leaf_arc(hp, other, half * 2)
            else:   # две створки от краёв, дуги встречаются в центре
                mid = ((c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2)
                draw_leaf_arc(c0, mid, half)
                draw_leaf_arc(c1, mid, half)

    # 4) Мебель
    if with_furniture:
        for f in data.get("furniture", []):
            _draw_furniture(dr, P, s, f)

    # 5) Камера: маркер ракурса поверх готового чертежа.
    if draw_camera:
        img = _paint_camera(img, data, poly, P, s, style=camera_style)

    final = img.resize(CANVAS, Image.LANCZOS)
    buf = io.BytesIO()
    final.save(buf, format="PNG")
    has_door = any(o.get("type") in DOOR_TYPES for o in data.get("openings", []))
    meta = {
        "px_per_dw": round(s, 3),
        "wall_px": round(t_dw * s, 2),
        "room_dw": [round(w_dw, 2), round(d_dw, 2)],
        "room_m": [round(w_dw * DW_M, 2), round(d_dw * DW_M, 2)],
        "scale_fallback": not has_door,
    }
    return buf.getvalue(), meta


def _paint_camera(img, data: dict, poly: list, P, s: float, *, style: str = "camera_icon"):
    """Накладывает маркер камеры на готовый лист и возвращает RGB-изображение.

    Сектор заливается ТОЛЬКО по чистому полу (маска «пиксель ровно цвета пола»):
    так он ложится ПОД мебель и под символы проёмов — те остаются нетронутыми и
    читаются сквозь заливку, а не тонут в ней. Маркер рисуется последним:
    его не должен закрыть предмет, стоящий у стены.

    Работает по блоку `camera` плана; блока нет (план построен до его появления) —
    берётся конвенция camera-relative: центр front-стены, взгляд в глубину листа.
    """
    room = data["room"]
    cam = data.get("camera") or {}
    wall = cam.get("wall") if cam.get("wall") in ("back", "front", "left", "right") else "front"
    try:
        position = clamp(float(cam.get("position", 0.5)), 0.0, 1.0)
    except (TypeError, ValueError):
        position = 0.5
    dx, dy = CAM_DIR_VEC.get(cam.get("direction"), CAM_DIR_VEC["up"])

    # Точка стояния: доля длины стены -> точка на её ВНУТРЕННЕЙ грани (полигон
    # комнаты и есть внутренняя грань — стены рисуются наружу от него).
    start, end = wall_span(room, wall)
    wall_coord = _dodge_door_arc(data, wall, start, end, start + position * (end - start))
    i, j, local = wall_edge(room, wall, poly, wall_coord)
    (ax, ay), (ux, uy), ln = wall_params(poly, i, j)
    local = clamp(local, 0.0, ln)
    # Точка на внутренней грани стены, от неё маркер растёт внутрь комнаты.
    base = (ax + ux * local, ay + uy * local)
    diamond = style in DIAMOND_STYLES
    # Апекс сектора: у ромба — его центр (сдвиг внутрь на радиус, иначе половина
    # знака тонет в чёрной стене), у иконки — срез объектива, чтобы обзор
    # расходился именно от него, а не из-под корпуса.
    apex_dw = CAM_DOT_DW if diamond else (
        CAM_ICON_INSET_DW + CAM_ICON_BODY_DW + CAM_ICON_LENS_DW
    )
    cx, cy = base[0] + dx * apex_dw, base[1] + dy * apex_dw

    out = img.convert("RGB")
    if style != "dot":
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        # Радиус заведомо больше комнаты: лишнее срежет маска пола, зато сектор
        # гарантированно дотягивается до дальней стены при любой форме.
        reach = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        # Направление ВЗГЛЯДА (оптическая ось) — биссектриса лучей к углам
        # back-стены, а не середина сектора покрытия: сектор несимметричен из-за
        # боковых проёмов и скашивал иконку на фронтальных кадрах (вердикт
        # пользователя, bedroom бенча 16А). Для камеры в центре взгляд выходит
        # «прямо», для съёмки в угол (fin463) — честный наклон.
        gaze = math.degrees(math.atan2(dy, dx))
        if room.get("shape") == "rectangle" and wall == "front":
            try:
                width_dw = float(room["width_dw"])
                left = math.atan2(0.0 - base[1], 0.0 - base[0])
                right = math.atan2(0.0 - base[1], width_dw - base[0])
                gaze = math.degrees((left + right) / 2)
                dx, dy = math.cos(math.radians(gaze)), math.sin(math.radians(gaze))
            except (KeyError, TypeError, ValueError):
                pass
        if not diamond:
            cx, cy = base[0] + dx * apex_dw, base[1] + dy * apex_dw
        a0, a1 = _sector_angles(data, room, wall, (cx, cy), gaze)
        px, py = P(cx, cy)
        r = reach * s * SS
        if a0 < 0:      # pieslice ждёт неотрицательные углы по часовой стрелке
            a0, a1 = a0 + 360, a1 + 360
        sector = Image.new("L", img.size, 0)
        ImageDraw.Draw(sector).pieslice(
            [px - r, py - r, px + r, py + r], start=a0, end=a1, fill=CAM_SECTOR_ALPHA
        )
        floor_only = img.point(lambda v: 255 if v == FLOOR else 0)
        out.paste(CAM_SECTOR_RGB, mask=ImageChops.multiply(sector, floor_only))

    if diamond:
        r = CAM_DOT_DW
        ImageDraw.Draw(out).polygon(
            [P(cx, cy - r), P(cx + r, cy), P(cx, cy + r), P(cx - r, cy)], fill=(INK, INK, INK)
        )
    else:
        _draw_camera_icon(out, P, s, base, (dx, dy))
    return out


# Зазор знака камеры от дверного проёма вдоль стены: полкорпуса знака + просвет,
# чтобы дуга открывания не прочерчивала иконку насквозь.
CAM_DOOR_DODGE_DW = 0.5


def _dodge_door_arc(data: dict, wall: str, start: float, end: float, coord: float) -> float:
    """Сдвиг ЗНАКА камеры вдоль стены из-под дверного полотна и дуги.

    Данные не трогаются — позиция съёмки в plan_json остаётся честной (реальная
    дверь и камера легитимно совпадают: снимали из проёма). Двигается только
    рисунок знака в `_cam`-варианте: дуга прочерчивала иконку насквозь (аудит
    архитектора 2026-08-10, frame13/frame14). Знак уходит за ближний косяк;
    некуда (дверь во всю стену) — остаётся на месте.
    """
    for op in data.get("openings") or []:
        if op.get("wall") != wall or op.get("type") not in ("door", "double_door"):
            continue
        try:
            lo = float(op["offset_dw"]) - float(op["width_dw"]) / 2
            hi = float(op["offset_dw"]) + float(op["width_dw"]) / 2
        except (KeyError, TypeError, ValueError):
            continue
        if lo - CAM_DOOR_DODGE_DW < coord < hi + CAM_DOOR_DODGE_DW:
            candidates = [c for c in (lo - CAM_DOOR_DODGE_DW, hi + CAM_DOOR_DODGE_DW)
                          if start + 0.2 <= c <= end - 0.2]
            if candidates:
                return min(candidates, key=lambda c: abs(c - coord))
    return coord


def _sector_angles(data: dict, room: dict, cam_wall: str, apex, view: float):
    """Углы сектора обзора (a0, a1) в градусах листа.

    Раскрыв считается по составу плана, а не фиксированной константой (вердикт
    пользователя, бенч 16): в кадр гарантированно попала ВСЯ back-стена (наша
    конвенция) и каждый проём на боковых стенах, который экстрактор реально
    видел, — сектор обязан накрывать их все. Лучи тянутся от точки съёмки к
    углам back-стены и к дальним косякам боковых проёмов; вставки гейта
    (`inserted`) не в счёт — их на фото не было. Ширина зажимается в
    [CAM_FOV_DEG, 165°]: уже дефолтного конуса обзор не бывает (объектив
    телефона), шире 165° — вырожденный веер на всю комнату.

    Не rectangle или камера не на front — прежний фиксированный конус.
    """
    half = CAM_FOV_DEG / 2
    if room.get("shape") != "rectangle" or cam_wall != "front":
        return view - half, view + half
    try:
        width = float(room["width_dw"])
        depth = float(room["depth_dw"])
    except (KeyError, TypeError, ValueError):
        return view - half, view + half
    cx, cy = apex
    targets = [(0.0, 0.0), (width, 0.0)]                 # углы back-стены
    for op in data.get("openings") or []:
        if op.get("inserted") or op.get("wall") not in ("left", "right"):
            continue
        try:
            lo = float(op["offset_dw"]) - float(op["width_dw"]) / 2
            hi = float(op["offset_dw"]) + float(op["width_dw"]) / 2
        except (KeyError, TypeError, ValueError):
            continue
        x = 0.0 if op["wall"] == "left" else width
        targets += [(x, clamp(lo, 0.0, depth)), (x, clamp(hi, 0.0, depth))]
    deltas = []
    for tx, ty in targets:
        if math.hypot(tx - cx, ty - cy) < 1e-6:
            continue
        ang = math.degrees(math.atan2(ty - cy, tx - cx))
        deltas.append((ang - view + 180.0) % 360.0 - 180.0)
    if not deltas:
        return view - half, view + half
    # Раскрыв СИММЕТРИЧЕН вокруг оси взгляда (вердикт пользователя: как у
    # настоящего объектива — полууглы влево и вправо равны). Ширина берётся по
    # самому далёкому от оси видимому таргету, чтобы всё попало в клин.
    margin = 3.0                                          # чтобы косяк не лежал на краю заливки
    sym_half = max(half, min(max(abs(d) for d in deltas) + margin, 82.5))
    return view - sym_half, view + sym_half


def _draw_camera_icon(out, P, s: float, base, direction) -> None:
    """Иконка камеры в точке съёмки: корпус со скруглением + объектив по взгляду.

    Направление произвольное (объектив смотрит в середину сектора обзора —
    вердикт пользователя, бенч 16): иконка рисуется осевой на отдельном
    прозрачном слое (объектив кладётся первым, корпус поверх — стык не виден),
    поворачивается на угол взгляда и сажается задней стенкой на `base` — точку
    на ВНУТРЕННЕЙ грани стены, от которой растёт внутрь комнаты.
    """
    fx, fy = direction
    px = s * SS                           # dw -> суперсэмпл-пиксели
    border = CAM_ICON_BORDER_PX * SS     # 2px финального листа до даунскейла
    length = CAM_ICON_INSET_DW + CAM_ICON_BODY_DW + CAM_ICON_LENS_DW
    pad = int(math.ceil(length * px + border)) + 4   # запас на любой поворот
    side = pad * 2
    icon = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    ink = (INK, INK, INK, 255)
    white = (PAGE, PAGE, PAGE, 255)
    ax, ay = float(pad), float(pad)                # якорь = задняя стенка корпуса

    def fbox(f0: float, f1: float, half_w: float, grow: float = 0.0):
        return [ax + f0 * px - grow, ay - half_w * px - grow,
                ax + f1 * px + grow, ay + half_w * px + grow]

    def lens_poly(grow: float = 0.0):
        # Объектив-раструб (вердикт пользователя): трапеция от корпуса наружу,
        # узкая у тела, широкая на срезе - читается как «смотрит туда».
        f0 = body_end * px - grow
        f1 = (body_end + CAM_ICON_LENS_DW) * px + grow
        near = CAM_ICON_LENS_W_NEAR_DW / 2 * px + grow
        far = CAM_ICON_LENS_W_FAR_DW / 2 * px + grow
        return [(ax + f0, ay - near), (ax + f1, ay - far),
                (ax + f1, ay + far), (ax + f0, ay + near)]

    body_end = CAM_ICON_INSET_DW + CAM_ICON_BODY_DW
    body = (CAM_ICON_INSET_DW, body_end, CAM_ICON_WIDTH_DW / 2)
    # Сначала белый силуэт, раздутый на ширину обводки, затем чёрное тело -
    # получается контрастная окантовка вокруг всего знака. Углы острые.
    draw.polygon(lens_poly(grow=border), fill=white)
    draw.rectangle(fbox(*body, grow=border), fill=white)
    draw.polygon(lens_poly(), fill=ink)
    draw.rectangle(fbox(*body), fill=ink)
    # PIL поворачивает против часовой в визуальных осях; у листа y растёт вниз,
    # поэтому вектор (fx, fy) достигается углом atan2(-fy, fx). Поворот вокруг
    # якоря — канвас с запасом не даёт срезать углы.
    icon = icon.rotate(math.degrees(math.atan2(-fy, fx)), resample=Image.BICUBIC,
                       center=(ax, ay))
    bx, by = P(base[0], base[1])
    out.paste(icon, (int(round(bx - ax)), int(round(by - ay))), icon)


def _draw_furniture(dr, P, s, f: dict) -> None:
    """Глиф мебели: белая заливка + тонкий контур. center_dw в координатах комнаты."""
    kind = f["kind"]
    cx, cy = f["center_dw"]
    w_m, h_m = f["size_m"]
    rot = int(f.get("rotation", 0)) % 360
    w_dw, h_dw = w_m / DW_M, h_m / DW_M
    if rot in (90, 270):
        w_dw, h_dw = h_dw, w_dw
    x0, y0, x1, y1 = cx - w_dw / 2, cy - h_dw / 2, cx + w_dw / 2, cy + h_dw / 2
    thin = THIN * SS

    def rect(a, b, c, d, fill=PAGE):
        dr.rectangle([P(a, b), P(c, d)], fill=fill, outline=INK, width=thin // 2)

    def line(a, b, c, d):
        dr.line([P(a, b), P(c, d)], fill=INK, width=thin // 2)

    if kind in ("bed", "bed_double"):
        rect(x0, y0, x1, y1)
        ph = 0.55 / DW_M   # глубина зоны подушек ~55 см
        if rot == 0:
            line(x0, y0 + ph, x1, y0 + ph)
            if kind == "bed_double":
                line((x0 + x1) / 2, y0, (x0 + x1) / 2, y0 + ph)
        elif rot == 180:
            line(x0, y1 - ph, x1, y1 - ph)
            if kind == "bed_double":
                line((x0 + x1) / 2, y1 - ph, (x0 + x1) / 2, y1)
        elif rot == 90:
            line(x1 - ph, y0, x1 - ph, y1)
            if kind == "bed_double":
                line(x1 - ph, (y0 + y1) / 2, x1, (y0 + y1) / 2)
        else:
            line(x0 + ph, y0, x0 + ph, y1)
            if kind == "bed_double":
                line(x0, (y0 + y1) / 2, x0 + ph, (y0 + y1) / 2)
    elif kind == "sofa":
        rect(x0, y0, x1, y1)
        bh = 0.25 / DW_M   # спинка
        if rot == 0:
            line(x0, y0 + bh, x1, y0 + bh)
        elif rot == 180:
            line(x0, y1 - bh, x1, y1 - bh)
        elif rot == 90:
            line(x1 - bh, y0, x1 - bh, y1)
        else:
            line(x0 + bh, y0, x0 + bh, y1)
    elif kind in ("wardrobe", "dresser", "tv_stand", "kitchen_run", "desk", "fridge"):
        rect(x0, y0, x1, y1)
        if kind == "wardrobe":
            line(x0, y0, x1, y1)
    elif kind == "rug":
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=thin // 2)
    elif kind in ("sink", "hob"):
        rect(x0, y0, x1, y1)
        dr.ellipse([P(x0 + 0.05, y0 + 0.05), P(x1 - 0.05, y1 - 0.05)],
                   outline=INK, width=thin // 2)
    elif kind == "plant":
        dr.ellipse([P(x0, y0), P(x1, y1)], fill=PAGE, outline=INK, width=thin // 2)
    else:
        # table / side_table / chair / armchair и любой незнакомый глиф — простой блок.
        rect(x0, y0, x1, y1)

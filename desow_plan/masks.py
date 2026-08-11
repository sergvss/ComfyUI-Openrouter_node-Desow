# КАНОНИЧЕСКИЙ КОД. Пары в бэкенде нет: масочная геометрия исполняется только
# воркфлоу (нода DesowPlanRender). Методика отработана на бенче 16А 2026-08-10.
"""Масочная опора геометрии: сегментация gemini-2.5-flash -> измерение проёмов.

Конвейер (сумма знаний сессии 2026-08-10, принят пользователем на 3 кадрах):
1. Сегментация отдаёт маски пола/стен/проёмов (box_2d в 0-1000 + PNG-маска).
2. Из ВЕРХНЕЙ границы маски пола фитом трёх прямых извлекаются линии плинтусов
   (левая, задняя, правая) - опора из сотен пикселей вместо четырёх точек.
3. Углы пола = пересечения прямых -> гомография пола (та же конструкция, что
   в ручном эталоне камеры: точки схода + известная ширина back-стены).
4. Косяки проёма = вертикали крайних столбцов его маски, ОПУЩЕННЫЕ на линию
   плинтуса своей стены (окно пола не касается - его маску нельзя проецировать
   напрямую: нижняя кромка это подоконник). Точки пола -> гомография -> глубины.
5. Масштаб глубины - линейка-проём: обычная дверь = ровно 1 dw; двустворчатая/
   балконная/окно - шириной из экстракции. Маска, обрезанная краем кадра,
   линейкой быть не может и сама не перемеряется (видимая часть != проём).

Всё fail-soft: любой сбой любого шага -> пометка в notes, соответствующий
проём остаётся с экстракторской геометрией. Состав проёмов маски НЕ меняют -
только offset/width уже известных (сопоставление по типу и стене).
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

# Рабочий растр разбора масок. Координаты дальше нормируются в 0..1, поэтому
# точный аспект фото не нужен; 4:3 близок к типичным кадрам.
GRID_W, GRID_H = 1200, 900
# Насколько крайний столбец маски может отстоять от края растра, чтобы проём
# ещё считался целиком видимым (иначе - «обрезан кадром»).
CLIP_MARGIN_PX = 4
# Нижний кламп ширины перемеренного проёма (dw): шумная маска не должна
# схлопнуть проём в щель.
MIN_MEASURED_WIDTH_DW = 0.35
# Средняя ошибка фита трёх плинтусных прямых (px растра): на честных
# прямоугольных кадрах бенча 3-12px; L-образный пол (lroom, 23.6px) и
# зеркальная стена (fin463, 22.4px) ломают опору - там перемер только вредит.
# ВНИМАНИЕ: порог откалиброван под растр GRID_W x GRID_H - менять их вместе.
MAX_FIT_ERR_PX = 15.0
# Верхний порог сопоставления измеренного проёма с проёмом плана (dw по
# центрам): дальше - это другой объект (двери в зеркале fin463 давали матч
# с расстоянием 2.1 и портили офсет), измерение не применяется.
MATCH_TOL_DW = 2.0
# Перемер УТОЧНЯЕТ экстракцию, а не переворачивает её: расхождение ширины
# больше этой доли значит, что маска видит лишь часть проёма (шторы, мебель,
# острый угол) - живой пример: занавешенное окно living_room, маска дала 43%
# ширины и пользователь измерение отверг.
MAX_WIDTH_CHANGE_RATIO = 0.35

_OPENING_WORDS = ("door", "window", "passage")


def _line(p, q):
    l = np.cross(p, q)
    n = np.hypot(l[0], l[1])
    return l / n if n > 1e-9 else None


def build_floor_h_inv(pts: dict):
    """Обратная гомография пола из 4 опорных точек (порт eval_jambs)."""
    need = ("back_left", "back_right", "left_base", "right_base")
    if any(not isinstance(pts.get(k), (list, tuple)) or len(pts[k]) != 2 for k in need):
        return None
    bl, br, lb, rb = (np.array([float(pts[k][0]), float(pts[k][1]), 1.0]) for k in need)
    left_line, right_line, back_line = _line(bl, lb), _line(br, rb), _line(bl, br)
    if left_line is None or right_line is None or back_line is None:
        return None
    V = np.cross(left_line, right_line)
    if np.hypot(V[0], V[1]) < 1e-9:
        return None
    if abs(V[2]) > 1e-12:
        Vn = V / V[2]
        vB = np.cross(back_line, np.array([0.0, 1.0, -Vn[1]]))
    else:
        vB = np.array([1.0, 0.0, 0.0]) if abs(back_line[0]) < 1e-9 else np.cross(
            back_line, np.array([0.0, 1.0, -bl[1]]))
    num = np.cross(br, bl)
    den = np.cross(br, vB)
    idx = int(np.argmax(np.abs(den)))
    if abs(den[idx]) < 1e-12:
        return None
    s = -num[idx] / den[idx]
    H = np.column_stack([s * vB, V, bl])
    if abs(np.linalg.det(H)) < 1e-12:
        return None
    return np.linalg.inv(H)


def floor_xy(Hinv, pt):
    v = Hinv @ np.array([float(pt[0]), float(pt[1]), 1.0])
    if abs(v[2]) < 1e-12:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def parse_segmentation(text: str):
    """Список сегментов из ответа модели (фенсы/обвязка терпимы). None - мусор."""
    if not text:
        return None
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e <= s:
        return None
    try:
        items = json.loads(text[s:e + 1], strict=False)
    except (ValueError, TypeError):
        return None
    return items if isinstance(items, list) else None


def mask_to_grid(item):
    """PNG-маска сегмента, растянутая в рабочий растр. None при мусоре.

    box_2d за пределами 0..1000 (модель так делает) клампится к растру, а не
    роняет разбор: одна кривая маска не должна отменять измерение остальных.
    """
    try:
        y0, x0, y1, x1 = [float(v) / 1000.0 for v in item["box_2d"]]
        m64 = item["mask"].split("base64,")[-1]
        m = Image.open(io.BytesIO(base64.b64decode(m64))).convert("L")
        bx0 = max(0, min(GRID_W - 1, int(x0 * GRID_W)))
        by0 = max(0, min(GRID_H - 1, int(y0 * GRID_H)))
        bx1 = max(bx0 + 1, min(GRID_W, int(x1 * GRID_W)))
        by1 = max(by0 + 1, min(GRID_H, int(y1 * GRID_H)))
        m = m.resize((bx1 - bx0, by1 - by0))
        full = np.zeros((GRID_H, GRID_W), bool)
        full[by0:by1, bx0:bx1] = np.array(m) > 127
        return full
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _seg_fit(xs, ys, cum, a, b):
    """МНК-прямая по сегменту [a:b) через префикс-суммы + средний |остаток|.

    Закрытая формула вместо lstsq: полный перебор разбиений звал lstsq ~10^5
    раз и стоил ~5 секунд на кадр; суммы дают тот же фит за миллисекунды.
    """
    n = b - a
    sx = cum["x"][b] - cum["x"][a]
    sy = cum["y"][b] - cum["y"][a]
    sxx = cum["xx"][b] - cum["xx"][a]
    sxy = cum["xy"][b] - cum["xy"][a]
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return None
    k = (n * sxy - sx * sy) / den
    c = (sy - k * sx) / n
    err = float(np.abs(ys[a:b] - (k * xs[a:b] + c)).mean())
    return float(k), float(c), err


def floor_support(floor: np.ndarray):
    """Опора из маски пола: 3 прямых плинтусов + углы. None при деградации."""
    xs, ys = [], []
    for x in range(0, GRID_W, 4):
        col = np.where(floor[:, x])[0]
        if len(col) > 5:
            xs.append(x)
            ys.append(col.min())
    xs, ys = np.array(xs, float), np.array(ys, float)
    n = len(xs)
    if n < 30:
        return None
    cum = {key: np.concatenate([[0.0], np.cumsum(vals)])
           for key, vals in (("x", xs), ("y", ys), ("xx", xs * xs), ("xy", xs * ys))}

    def total(i, j):
        parts = [_seg_fit(xs, ys, cum, a, b) for a, b in ((0, i), (i, j), (j, n))]
        if any(p is None for p in parts):
            return None
        return sum(p[2] for p in parts), [(p[0], p[1]) for p in parts]

    # Перебор разбиений в два прохода: грубая сетка (шаг 4) + точная доводка
    # вокруг лучшей пары - вместо полного O(n^2) перебора каждой пары.
    best, best_ij = None, None
    for i in range(5, n - 10, 4):
        for j in range(i + 5, n - 5, 4):
            cand = total(i, j)
            if cand and (best is None or cand[0] < best[0]):
                best, best_ij = cand, (i, j)
    if best is None:
        return None
    bi, bj = best_ij
    for i in range(max(5, bi - 4), min(n - 10, bi + 5)):
        for j in range(max(i + 5, bj - 4), min(n - 5, bj + 5)):
            cand = total(i, j)
            if cand and cand[0] < best[0]:
                best = cand
    err, fits = best
    (kl, bl), (kb, bb), (kr, br) = fits
    if abs(kl - kb) < 1e-9 or abs(kr - kb) < 1e-9:
        return None
    xL = (bb - bl) / (kl - kb)
    xR = (bb - br) / (kr - kb)
    if not (0 <= xL < xR <= GRID_W):
        return None
    pts = {"back_left": [xL / GRID_W, (kb * xL + bb) / GRID_H],
           "back_right": [xR / GRID_W, (kb * xR + bb) / GRID_H],
           "left_base": [xs[2] / GRID_W, (kl * xs[2] + bl) / GRID_H],
           "right_base": [xs[-3] / GRID_W, (kr * xs[-3] + br) / GRID_H]}
    lines = {"left": (kl, bl), "back": (kb, bb), "right": (kr, br),
             "xL": xL, "xR": xR}
    return pts, err, lines


def _opening_ground(mask: np.ndarray, lines: dict):
    """Стена проёма + точки его косяков на линии плинтуса + флаг обрезки."""
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    x_lo, x_hi = int(xs.min()), int(xs.max())
    clipped = x_lo <= CLIP_MARGIN_PX or x_hi >= GRID_W - CLIP_MARGIN_PX - 1  # край кадра
    x_mid = (x_lo + x_hi) / 2
    if x_mid < lines["xL"]:
        wall, (k, b) = "left", lines["left"]
    elif x_mid > lines["xR"]:
        wall, (k, b) = "right", lines["right"]
    else:
        wall, (k, b) = "back", lines["back"]
    pa = [x_lo / GRID_W, (k * x_lo + b) / GRID_H]
    pb = [x_hi / GRID_W, (k * x_hi + b) / GRID_H]
    return wall, pa, pb, clipped


def _kind_from_label(label: str) -> str:
    label = label.lower()
    if "balcony" in label or "garden" in label or "terrace" in label:
        return "balcony_door"
    if "door" in label:
        return "door"
    if "window" in label:
        return "window"
    return "passage"


# Пороги детектора диагонального кадра (съёмка в угол комнаты, видны 2 стены
# из 4). Калибровка по 15 кадрам бенча 18 (2026-08-11): честные фронтальные
# (включая узкие пеналы fin379/fin380) не проходят ГЕОМЕТРИЧЕСКИЙ критерий;
# ловушки (зеркало fin463/frame13, L-пол lroom) проходят его, но отсекаются
# МАСОЧНЫМ (их back-стена видна широко). Оба критерия обязаны совпасть.
DIAG_BACK_SLOPE = 0.12        # |наклон| среднего сегмента фита: горизонтали нет
DIAG_SIDE_SLOPE_MIN = 0.05    # крайние сегменты «наклонены всерьёз» (не шум фита)
DIAG_BACK_AREA_MAX = 0.08     # маска back-стены < 8% кадра (только у диагонали)
DIAG_SIDE_AREA_MIN = 0.03     # одна из боковых < 3% кадра - её и не видно


def diagnose_diagonal(seg_text: str):
    """Детектор кадра «в угол»: None (не диагональ/нет данных) или сторона
    невидимой боковой стены ('left'|'right') - камера стоит в углу с этой
    стороны, взгляд по диагонали в видимый угол."""
    items = parse_segmentation(seg_text)
    if not items:
        return None
    floor_item = next((i for i in items
                       if "floor" in str(i.get("label", "")).lower()), None)
    floor = mask_to_grid(floor_item) if floor_item else None
    sup = floor_support(floor) if floor is not None else None
    if sup is None:
        return None
    _pts, _err, lines = sup
    kl, kb, kr = lines["left"][0], lines["back"][0], lines["right"][0]
    slanted = abs(kb) > DIAG_BACK_SLOPE or (
        kl * kr > 0 and abs(kl) > DIAG_SIDE_SLOPE_MIN and abs(kr) > DIAG_SIDE_SLOPE_MIN)
    if not slanted:
        return None
    area = {"back": 0.0, "left": 0.0, "right": 0.0}
    for it in items:
        lab = str(it.get("label", "")).lower()
        # Только сегменты самих стен: «door on the right wall» тоже содержит
        # слово wall, но это проём - в площадь стены не идёт.
        if "wall" not in lab or any(w in lab for w in _OPENING_WORDS):
            continue
        side = ("back" if "back" in lab else "left" if "left" in lab
                else "right" if "right" in lab else None)
        if side:
            m = mask_to_grid(it)
            if m is not None:
                area[side] += float(m.sum()) / (GRID_W * GRID_H)
    if area["back"] >= DIAG_BACK_AREA_MAX:
        return None
    if area["left"] < DIAG_SIDE_AREA_MIN <= area["right"]:
        return "left"
    if area["right"] < DIAG_SIDE_AREA_MIN <= area["left"]:
        return "right"
    return None


def measure_openings_from_masks(seg_text: str, plan: dict) -> list[str]:
    """Перемеряет offset/width проёмов плана по маскам сегментации.

    Мутирует проёмы плана: сопоставление по типу+стене, ближайший по центру
    в пределах MATCH_TOL_DW, каждый проём плана перемеряется не больше
    одного раза. Состав не меняет. Возвращает пометки для debug.
    """
    notes: list[str] = []
    items = parse_segmentation(seg_text)
    if not items:
        return ["segmentation: нечитаемый ответ, пропуск"]
    room = plan.get("room") or {}
    # Масочная опора считает 3 прямых плинтуса прямоугольной комнаты; на
    # l_shape геометрия другая - честно пропускаем.
    if room.get("shape") == "l_shape":
        return ["segmentation: l_shape, пропуск"]
    try:
        width_dw = float(room["width_dw"])
    except (KeyError, TypeError, ValueError):
        return ["segmentation: нет ширины комнаты, пропуск"]

    floor_item = next((i for i in items
                       if "floor" in str(i.get("label", "")).lower()), None)
    if floor_item is None:
        return ["segmentation: нет маски пола, пропуск"]
    floor = mask_to_grid(floor_item)
    if floor is None:
        return ["segmentation: маска пола нечитаема, пропуск"]
    sup = floor_support(floor)
    if sup is None:
        return ["segmentation: опора из маски пола не построилась, пропуск"]
    pts, err, lines = sup
    if err > MAX_FIT_ERR_PX:
        return ["segmentation: опора шумная (fit %.1fpx > %.0f) - пол не "
                "прямоугольный или зеркало, пропуск" % (err, MAX_FIT_ERR_PX)]
    Hinv = build_floor_h_inv(pts)
    if Hinv is None:
        return ["segmentation: гомография не построилась, пропуск"]
    notes.append("segmentation: опора ok (fit %.1fpx)" % err)

    measured = []
    for it in items:
        label = str(it.get("label", ""))
        if not any(w in label.lower() for w in _OPENING_WORDS):
            continue
        mask = mask_to_grid(it)
        if mask is None:
            continue
        ground = _opening_ground(mask, lines)
        if ground is None:
            continue
        wall, pa, pb, clipped = ground
        fa, fb = floor_xy(Hinv, pa), floor_xy(Hinv, pb)
        if fa is None or fb is None:
            continue
        measured.append({"kind": _kind_from_label(label), "wall": wall,
                         "fa": fa, "fb": fb, "clipped": clipped, "label": label})

    def ext_width(wall, kinds):
        for op in plan.get("openings") or []:
            if op.get("wall") == wall and op.get("type") in kinds:
                try:
                    return float(op["width_dw"]) or None
                except (KeyError, TypeError, ValueError):
                    return None
        return None

    scale, ruler = None, None
    for cand in ("door", "balcony_door", "window"):
        for m in measured:
            if m["kind"] != cand or m["wall"] == "back" or m["clipped"]:
                continue
            span = abs(m["fb"][1] - m["fa"][1])
            if cand == "door":
                w = ext_width(m["wall"], ("double_door", "balcony_door")) or 1.0
            elif cand == "balcony_door":
                w = ext_width(m["wall"], ("balcony_door", "double_door"))
            else:
                w = ext_width(m["wall"], ("window", "floor_to_ceiling_window"))
            if w and span > 1e-6:
                scale, ruler = span / w, m["label"]
                break
        if scale:
            break
    if ruler:
        notes.append("segmentation: линейка «%s»" % ruler)

    kind_match = {
        "door": ("door", "double_door", "balcony_door"),
        "balcony_door": ("balcony_door", "double_door", "door"),
        "window": ("window", "floor_to_ceiling_window"),
        "passage": ("passage",),
    }
    applied_ids = set()   # каждый проём плана перемеряется не больше одного раза
    for m in measured:
        if m["clipped"]:
            notes.append("segmentation: %s/%s обрезан кадром - не перемеряю"
                         % (m["kind"], m["wall"]))
            continue
        if m["wall"] == "back":
            lo, hi = sorted((m["fa"][0], m["fb"][0]))
            offset, width = (lo + hi) / 2 * width_dw, (hi - lo) * width_dw
        else:
            if not scale:
                notes.append("segmentation: нет линейки глубины - %s/%s пропущен"
                             % (m["kind"], m["wall"]))
                continue
            d = sorted((m["fa"][1] / scale, m["fb"][1] / scale))
            offset, width = (d[0] + d[1]) / 2, d[1] - d[0]
        width = max(width, MIN_MEASURED_WIDTH_DW)
        if width > width_dw * 1.2:
            notes.append("segmentation: %s/%s ширина %.2f неправдоподобна - пропущен"
                         % (m["kind"], m["wall"], width))
            continue
        target = None
        for op in plan.get("openings") or []:
            if (id(op) in applied_ids or op.get("wall") != m["wall"]
                    or op.get("type") not in kind_match.get(m["kind"], ())):
                continue
            try:
                d_c = abs(float(op["offset_dw"]) - offset)
            except (KeyError, TypeError, ValueError):
                continue
            if d_c <= MATCH_TOL_DW and (target is None or d_c < target[0]):
                target = (d_c, op)
        if target is None:
            notes.append("segmentation: %s/%s (%.2f) не сопоставлен - пропущен"
                         % (m["kind"], m["wall"], offset))
            continue
        op = target[1]
        try:
            old_w = float(op["width_dw"])
        except (KeyError, TypeError, ValueError):
            old_w = 0.0
        if old_w > 1e-6 and abs(width - old_w) / old_w > MAX_WIDTH_CHANGE_RATIO:
            notes.append("segmentation: %s/%s маска %.2f против экстракции %.2f - "
                         "проём перекрыт, не перемеряю" % (m["kind"], m["wall"], width, old_w))
            continue
        notes.append("segmentation: %s/%s %.2f/%.2f -> %.2f/%.2f"
                     % (op.get("type"), m["wall"], float(op["offset_dw"]),
                        float(op["width_dw"]), offset, width))
        op["offset_dw"] = round(offset, 3)
        op["width_dw"] = round(width, 3)
        applied_ids.add(id(op))
    return notes

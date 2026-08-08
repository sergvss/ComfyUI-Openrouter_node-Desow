# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/schema.py - синхронизировать при правках.
# Двойное ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
"""Схема структурного JSON 2D-плана комнаты — те же правила, но без pydantic.

Отличия от источника (`plan2d/schema.py`) — осознанные, каждое с причиной:

1. **Нет pydantic.** На машинах ComfyUI он не гарантирован, а тянуть зависимость
   ради одной ноды нельзя. Правила выражены функциями на голых dict; сами
   правила (границы, конечность чисел, обязательность полей) — построчный порт.
2. **`FurnitureItem` — функцией, а не моделью.** Расстановка мебели переехала с
   бэкенда в ноду `DesowPlanFurnish`, поэтому проверка предмета нужна и здесь:
   `validate_furniture_item` — построчный порт `FurnitureItem` (снап поворота к
   0/90/180/270, конечные числа, размер в разумных метрах). Отличие от источника
   только в способе сигнализации: `PlanDataError` вместо `ValidationError`.
   `validate_plan` по-прежнему игнорирует ключ `furniture` во входе — план
   строится пустым, мебель кладётся поверх отдельным шагом.
3. **Частично битый вход не роняет план.** У бэкенда pydantic отвергает ответ
   целиком (там есть второй прогон VLM и fail-loud до списания кредитов). У ноды
   ретрая нет, а уронить скан она не имеет права, поэтому:
   - `room` невалиден -> `PlanDataError` (без габарита рисовать нечего);
   - `shape` не из словаря или полигон L-формы битый -> откат в `rectangle`;
   - отдельный битый проём / простенок -> выбрасывается, остальные остаются.
   Каждое такое решение попадает в `notes` и дальше в debug-выход ноды.

Единица измерения — ширина дверного полотна `dw` (1 dw = 0.85 м).
Конвенция ориентации (camera-relative): `back` — стена в глубине кадра (на плане
СВЕРХУ), `front` — за камерой (СНИЗУ), `left`/`right` — как видит камера;
координаты плана: x вправо (0 = левая стена), y вниз (0 = back-стена).
"""
from __future__ import annotations

import math

# ── Константы графстандарта и единиц ─────────────────────────────────
DW_M = 0.85           # 1 dw = 0.85 м
WALL_T_DW = 0.25      # толщина стены = четверть ширины двери

DOOR_TYPES = frozenset({"door", "double_door"})
WINDOW_TYPES = frozenset({"window", "floor_to_ceiling_window"})
OPENING_TYPES = frozenset(DOOR_TYPES | WINDOW_TYPES | {"passage"})

WALL_NAMES = frozenset({"back", "front", "left", "right"})
SHAPES = frozenset({"rectangle", "l_shape"})
ATTACH_NAMES = frozenset({"back", "front", "left", "right", "free"})
SWING_DIRECTIONS = frozenset({"in", "out"})

# Камера: где стоит точка съёмки и куда смотрит НА ЛИСТЕ (не в мире).
CAMERA_DIRECTIONS = frozenset({"up", "down", "left", "right"})
# Символ маркера: тёмная точка-ромб + полупрозрачный оранжевый сектор обзора.
# Поле нужно потребителю плана — по нему он пишет легенду в промпт («the dot marks
# the camera, the orange wedge marks what it sees», docs/design/CAMERA_ON_PLAN_RESEARCH.md).
CAMERA_MARKERS = frozenset({"orange_sector"})
# Конвенция плана camera-relative: камера у front-стены (нижняя кромка листа),
# смотрит в глубину кадра. Пока константа, но живёт данными: положение камеры
# на стене будет задавать пользователь.
DEFAULT_CAMERA = {"wall": "front", "position": 0.5, "direction": "up", "marker": "orange_sector"}

# Ширина проёма по умолчанию, когда позицию неоткуда взять (fallback мержа и гейт).
DEFAULT_WIDTH_DW = {"door": 1.0, "window": 1.6, "passage": 1.0}

# Длина стены прямоугольной комнаты по имени стены.
WALL_LEN_KEYS = {"back": "width_dw", "front": "width_dw", "left": "depth_dw", "right": "depth_dw"}

# Ролевые группы мебели для эргономических правил (см. validate.validate_furniture).
TALL_KINDS = frozenset({"wardrobe", "fridge", "kitchen_run"})
OPENING_FRONT_KINDS = frozenset({"wardrobe", "fridge", "dresser"})

# Нормы эргономики (метры). Значения — те же, что у бэкенда: по ним считает
# validate.py и о них же написано в системном промпте расстановки.
WINDOW_STRIP_M = 0.5       # свободная полоса перед окном
DOOR_APPROACH_M = 0.7      # подход к двери за дугой
DOOR_FRAME_CLEAR_M = 0.4   # зазор открывающейся техники от коробки двери
PASSAGE_CLEAR_M = 0.7      # проход у свободного конца простенка

# Допуск «касание = не пересечение» для проверок расстановки (validate_furniture).
# Мебель ВПЛОТНУЮ к стене — норма жизни, но выразить её точно модель не может:
# центр она округляет до 0.01 dw (8.5 мм), а размер даёт в метрах, и перевод в dw
# (деление на 0.85) даёт бесконечную дробь. Шкаф 0.6 м у стены x=3.8 при центре
# 3.45 вылезает на 2.5 мм — с прежним допуском 1e-6 dw (0.85 микрона) это
# считалось выходом за пределы комнаты, съедало все ре-промпты и печаталось как
# «[..,3.80,..] room 3.8» (виноват %.2f). 1 см принимает флеш-постановку и всё
# ещё ловит настоящий выход: 2 см — уже нарушение.
CONTACT_TOL_M = 0.01
CONTACT_TOL_DW = CONTACT_TOL_M / DW_M

# Предельный след одного предмета мебели. 10 м — заведомо больше любого реального
# (самый крупный глиф словаря — kitchen_run ~4 м); всё, что больше, — галлюцинация.
MAX_FURNITURE_SIZE_M = 10.0

# Минимальный простенок: от угла комнаты до косяка и между соседними проёмами.
# Та же норма 0.2 м, по которой гейт ставит дверь у угла. Нужна не только для
# красоты: проём впритык к углу «открывает» угол на чертеже (стены смыкаются
# ровно там, где вырезан проём), а два проёма без простенка сливаются в один.
MIN_CORNER_CLEARANCE_M = 0.2
MIN_CORNER_CLEARANCE_DW = MIN_CORNER_CLEARANCE_M / DW_M
# Уже 0.6 м проём сужать бессмысленно — такой проём выбрасывается целиком.
MIN_OPENING_WIDTH_DW = 0.7

# Верхняя граница любой длины в схеме: 30 dw = 25.5 м. Жилая комната такого
# размера не встречается — значение за границей означает галлюцинацию модели
# (перепутанные единицы, «5900» вместо 5.9). Пропустить её нельзя: масштаб
# рендера считается от габарита комнаты, и один такой прогон даёт чертёж, где
# комната вырождается в точку.
MAX_LENGTH_DW = 30.0

# Допуски проверки полигона L-комнаты (см. Room._check_polygon в источнике).
_POLYGON_ORIGIN_TOL = 0.05
_POLYGON_BBOX_ABS_TOL = 0.5
_POLYGON_BBOX_REL_TOL = 0.1


class PlanDataError(ValueError):
    """Данные плана непригодны для построения (фатально: рисовать нечего)."""


# ── Примитивы проверки ───────────────────────────────────────────────

def _number(value, name):
    """Конечное число из числа или числовой строки.

    Строки принимаются намеренно: pydantic в lax-режиме (как на бэкенде) сам
    приводит `"3.8"` к `3.8`, и VLM время от времени присылает именно строки.
    `bool` числом не считается: `True` вместо ширины — мусор, а не 1.0.
    """
    if isinstance(value, bool) or value is None:
        raise PlanDataError("%s: ожидалось число, получено %r" % (name, value))
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise PlanDataError("%s: не число: %r" % (name, value[:40]))
    if not isinstance(value, (int, float)):
        raise PlanDataError("%s: ожидалось число, получено %s" % (name, type(value).__name__))
    value = float(value)
    if not math.isfinite(value):
        raise PlanDataError("%s: не конечное число (%r)" % (name, value))
    return value


def _length(value, name, *, minimum=0.0, allow_min=False, maximum=MAX_LENGTH_DW):
    """Длина в dw в допустимом диапазоне (границы — из pydantic-полей источника)."""
    number = _number(value, name)
    if allow_min:
        ok = minimum <= number <= maximum
    else:
        ok = minimum < number <= maximum
    if not ok:
        raise PlanDataError(
            "%s=%g вне диапазона %s%g..%g dw" % (name, number, "" if allow_min else ">", minimum, maximum)
        )
    return number


# ── Комната ──────────────────────────────────────────────────────────

def _polygon(raw, room, notes):
    """Полигон L-комнаты. Любая проблема -> None (откат в rectangle) + пометка.

    Полигон — источник истины контура для рендера, а мусорная вершина (`5900`,
    `NaN`) вырождает масштаб чертежа: рендер вписывает габарит полигона в лист,
    и настоящая комната схлопывается в точку. Габарит сверяется с width/depth
    мягко: расхождение в пределах 0.5 dw или 10% — оценка «на глаз».
    """
    if not isinstance(raw, (list, tuple)) or len(raw) < 6:
        notes.append("polygon_dw: нужно не меньше 6 вершин -> откат в rectangle")
        return None
    points = []
    for i, vertex in enumerate(raw):
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            notes.append("polygon_dw[%d]: не пара [x, y] -> откат в rectangle" % i)
            return None
        try:
            x = _number(vertex[0], "polygon_dw[%d].x" % i)
            y = _number(vertex[1], "polygon_dw[%d].y" % i)
        except PlanDataError as exc:
            notes.append("%s -> откат в rectangle" % exc)
            return None
        if not (-_POLYGON_ORIGIN_TOL <= x <= MAX_LENGTH_DW and -_POLYGON_ORIGIN_TOL <= y <= MAX_LENGTH_DW):
            notes.append("polygon_dw[%d]=[%g, %g] вне 0..%g dw -> откат в rectangle" % (i, x, y, MAX_LENGTH_DW))
            return None
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    if span_x <= 0 or span_y <= 0:
        notes.append("polygon_dw вырожден (нулевая ширина или глубина) -> откат в rectangle")
        return None
    for key, span, declared in (("width_dw", span_x, room["width_dw"]), ("depth_dw", span_y, room["depth_dw"])):
        if abs(span - declared) > max(_POLYGON_BBOX_ABS_TOL, _POLYGON_BBOX_REL_TOL * declared):
            notes.append(
                "polygon_dw: габарит %s=%.2f расходится с заявленным %g -> откат в rectangle"
                % (key, span, declared)
            )
            return None
    return points


def validate_room(raw, notes):
    """Контур комнаты -> канонический dict. Ошибка здесь фатальна для плана."""
    if not isinstance(raw, dict):
        raise PlanDataError("room: ожидался объект, получено %s" % type(raw).__name__)
    room = {
        "shape": "rectangle",
        "width_dw": _length(raw.get("width_dw"), "room.width_dw"),
        "depth_dw": _length(raw.get("depth_dw"), "room.depth_dw"),
    }
    shape = raw.get("shape") or "rectangle"
    if shape not in SHAPES:
        # Источник (pydantic Literal) отверг бы весь план. Здесь мягче: 'square'
        # и подобное от модели не мешают нарисовать прямоугольник по width/depth.
        notes.append("room.shape=%r не из словаря -> rectangle" % (shape,))
        shape = "rectangle"
    if shape == "l_shape":
        polygon = _polygon(raw.get("polygon_dw"), room, notes)
        if polygon is not None:
            room["shape"] = "l_shape"
            room["polygon_dw"] = [list(p) for p in polygon]
    return room


# ── Проёмы ───────────────────────────────────────────────────────────

def _swing(raw, notes, label):
    """Открывание двери. Битые значения чинятся дефолтами источника, не роняют проём."""
    hinge, direction = "left", "in"
    if isinstance(raw, dict):
        if raw.get("hinge") in WALL_NAMES:
            hinge = raw["hinge"]
        elif raw.get("hinge") is not None:
            notes.append("%s.swing.hinge=%r не из словаря -> left" % (label, raw.get("hinge")))
        if raw.get("direction") in SWING_DIRECTIONS:
            direction = raw["direction"]
        elif raw.get("direction") is not None:
            notes.append("%s.swing.direction=%r не из словаря -> in" % (label, raw.get("direction")))
    return {"hinge": hinge, "direction": direction}


def validate_openings(raw, notes):
    """Список проёмов. Битый проём выбрасывается с пометкой, остальные остаются."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        notes.append("openings: ожидался список, получено %s -> пусто" % type(raw).__name__)
        return []
    out = []
    for i, entry in enumerate(raw):
        label = "openings[%d]" % i
        if not isinstance(entry, dict):
            notes.append("%s: не объект -> выброшен" % label)
            continue
        kind = entry.get("type")
        if kind not in OPENING_TYPES:
            notes.append("%s: тип %r не из словаря -> выброшен" % (label, kind))
            continue
        wall = entry.get("wall")
        # Для l_shape именованная стена бывает неоднозначной — допускается индекс
        # ребра полигона (та же поблажка, что WallRef в источнике).
        if isinstance(wall, bool) or not (wall in WALL_NAMES or isinstance(wall, int)):
            notes.append("%s: стена %r не из словаря -> выброшен" % (label, wall))
            continue
        try:
            opening = {
                "type": kind,
                "wall": wall,
                "offset_dw": _length(entry.get("offset_dw"), label + ".offset_dw", allow_min=True),
                "width_dw": _length(entry.get("width_dw"), label + ".width_dw"),
            }
        except PlanDataError as exc:
            notes.append("%s -> выброшен" % exc)
            continue
        if kind in DOOR_TYPES and entry.get("swing") is not None:
            opening["swing"] = _swing(entry.get("swing"), notes, label)
        if entry.get("confidence") is not None:
            try:
                opening["confidence"] = _number(entry.get("confidence"), label + ".confidence")
            except PlanDataError:
                pass   # confidence — справочное поле, на геометрию не влияет
        out.append(opening)
    return out


# ── Камера ───────────────────────────────────────────────────────────

def validate_camera(raw, notes):
    """Точка съёмки -> канонический dict. Битые значения чинятся дефолтом.

    Камера — не декорация, а часть контракта плана: по ней рендер ставит маркер
    ракурса, а потребитель плана пишет легенду в промпт. Поэтому блок есть у
    КАЖДОГО плана, даже если во входе его не было.
    """
    camera = dict(DEFAULT_CAMERA)
    if raw is None:
        return camera
    if not isinstance(raw, dict):
        notes.append("camera: ожидался объект, получено %s -> дефолт" % type(raw).__name__)
        return camera
    if raw.get("wall") in WALL_NAMES:
        camera["wall"] = raw["wall"]
    elif raw.get("wall") is not None:
        notes.append("camera.wall=%r не из словаря -> front" % (raw.get("wall"),))
    if raw.get("position") is not None:
        try:
            # Доля длины стены, а не dw: камера ездит по стене любой длины.
            camera["position"] = _length(
                raw.get("position"), "camera.position", minimum=0.0, allow_min=True, maximum=1.0
            )
        except PlanDataError as exc:
            notes.append("%s -> %g" % (exc, DEFAULT_CAMERA["position"]))
    if raw.get("direction") in CAMERA_DIRECTIONS:
        camera["direction"] = raw["direction"]
    elif raw.get("direction") is not None:
        notes.append("camera.direction=%r не из словаря -> up" % (raw.get("direction"),))
    if raw.get("marker") is not None and raw.get("marker") not in CAMERA_MARKERS:
        notes.append("camera.marker=%r не из словаря -> %s" % (raw.get("marker"), DEFAULT_CAMERA["marker"]))
    return camera


# ── Простенки ────────────────────────────────────────────────────────

def validate_partitions(raw, notes):
    """Внутренние простенки. Битый простенок выбрасывается с пометкой.

    Простенок рисуется чёрным блоком поверх пола: мусорная координата (NaN,
    `4100` вместо `4.1`) закрасила бы весь лист.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        notes.append("partitions: ожидался список, получено %s -> пусто" % type(raw).__name__)
        return []
    out = []
    for i, entry in enumerate(raw):
        label = "partitions[%d]" % i
        if not isinstance(entry, dict):
            notes.append("%s: не объект -> выброшен" % label)
            continue
        attach = entry.get("attach")
        if attach not in ATTACH_NAMES:
            notes.append("%s: attach=%r не из словаря -> выброшен" % (label, attach))
            continue
        try:
            partition = {
                "attach": attach,
                "length_dw": _length(entry.get("length_dw"), label + ".length_dw"),
            }
            if attach == "free":
                start = entry.get("start_dw")
                if not isinstance(start, (list, tuple)) or len(start) != 2:
                    raise PlanDataError("%s: free-простенку нужен start_dw [x, y]" % label)
                if entry.get("direction") not in WALL_NAMES:
                    raise PlanDataError("%s: free-простенку нужен direction" % label)
                partition["start_dw"] = [
                    _length(start[0], label + ".start_dw.x", minimum=-_POLYGON_ORIGIN_TOL, allow_min=True),
                    _length(start[1], label + ".start_dw.y", minimum=-_POLYGON_ORIGIN_TOL, allow_min=True),
                ]
                partition["direction"] = entry["direction"]
            else:
                partition["offset_dw"] = _length(
                    entry.get("offset_dw"), label + ".offset_dw", allow_min=True
                )
        except PlanDataError as exc:
            notes.append("%s -> выброшен" % exc)
            continue
        out.append(partition)
    return out


# ── Мебель ───────────────────────────────────────────────────────────

def _pair(raw, name):
    """Пара чисел `[a, b]` (порт `tuple[float, float]` из источника)."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise PlanDataError("%s: ожидалась пара [a, b], получено %r" % (name, raw))
    return _number(raw[0], "%s[0]" % name), _number(raw[1], "%s[1]" % name)


def validate_furniture_item(raw):
    """Предмет мебели -> канонический dict. Порт `schema.FurnitureItem`.

    Raises:
        PlanDataError: предмет непригоден (вызывающий его выбрасывает, остальные
        предметы расстановки при этом остаются — см. `furnish.parse_furniture`).
    """
    if not isinstance(raw, dict):
        raise PlanDataError("furniture: ожидался объект, получено %s" % type(raw).__name__)
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise PlanDataError("furniture.kind: ожидалась непустая строка, получено %r" % (kind,))
    kind = kind.strip()

    center = _pair(raw.get("center_dw"), "%s.center_dw" % kind)
    size = _pair(raw.get("size_m"), "%s.size_m" % kind)
    # Конечность чисел проверил _number: pydantic в источнике принимает nan/inf
    # как float, поэтому там она вынесена в отдельный валидатор, здесь входит
    # в примитив. Итог тот же: NaN до рендера не доезжает и не роняет Pillow.
    for side in size:
        if not 0 < side <= MAX_FURNITURE_SIZE_M:
            raise PlanDataError(
                "%s: size_m %g вне диапазона (0, %g] м" % (kind, side, MAX_FURNITURE_SIZE_M)
            )

    # Модель иногда шлёт 45 / 360 / -90. Рендер и валидатор различают только
    # 0/90/180/270, поэтому приводим к ближайшему из них, а не браковываем
    # весь ответ из-за одного предмета.
    rotation = _number(raw.get("rotation", 0), "%s.rotation" % kind)
    rotation = int(round(rotation / 90.0)) * 90 % 360

    return {"kind": kind, "center_dw": [center[0], center[1]],
            "size_m": [size[0], size[1]], "rotation": rotation}


# ── Корень ───────────────────────────────────────────────────────────

def validate_plan(raw):
    """Сырой разобранный JSON -> `(канонический план, пометки)`.

    Raises:
        PlanDataError: план непригоден (не объект, нет/битая комната).
    """
    if not isinstance(raw, dict):
        raise PlanDataError("план: ожидался объект, получено %s" % type(raw).__name__)
    notes = []
    plan = {
        "room": validate_room(raw.get("room"), notes),
        "openings": validate_openings(raw.get("openings"), notes),
        "camera": validate_camera(raw.get("camera"), notes),
    }
    partitions = validate_partitions(raw.get("partitions"), notes)
    if partitions:
        plan["partitions"] = partitions
    if raw.get("furniture"):
        # Нода строит пустой план; мебель — фаза B на бэкенде.
        notes.append("furniture: %d предмет(ов) во входе проигнорировано" % len(raw["furniture"]))
    return plan, notes

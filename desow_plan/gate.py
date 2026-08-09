# КАНОНИЧЕСКИЙ КОД. Пары в бэкенде больше нет: `plan2d/gate.py` удалён вместе со
# вторым конвейером построения плана - гейт исполняет только воркфлоу (нода
# DesowPlanRender). Синхронизировать с бэкендом нужно лишь схему (schema_lite.py).
"""Гейт проёмов: комната обязана иметь вход и окно (требование приёмки Фазы 1),
и ни один проём не должен налезать на другой или на угол.

Сканер видит только то, что попало в кадр; front-стена находится за камерой,
поэтому на части боевых кадров (например scan_379) итоговый набор проёмов
оставался вообще без двери. Промптовое правило «нарисуй дверь, если её не видно»
срабатывало непредсказуемо, поэтому условие исполняет код:

- нет ни одного ВХОДА -> дверь у угла (0.2 м от угла до косяка, петли у того же
  угла, открывание внутрь);
- нет ни одного окна -> окно по центру front-стены.

Вход — это `door`, `double_door` или `passage` (`ENTRANCE_TYPES`): проход в
соседнее помещение входом является, а балконная дверь (`balcony_door`) — нет,
она ведёт наружу. Кадр fin424 (панорамное остекление с балконной дверью) до
этого различения выглядел для гейта как комната с дверью и оставался без входа.

Стена входа: сначала front (она за камерой, вход обычно там и не виден), а если
на ней нет места — ближайшая глухая стена, затем остальные. Место считается по
`usable_spans`, поэтому «занята» здесь означает физически, а не по числу записей:
панорамное остекление во всю front-стену свободных интервалов не оставляет.
Отрезок стены под точкой съёмки тоже считается занятым (`_camera_keepout`):
вставленный ровно под знаком камеры проём читается как «дверь через камеру».

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
    CAMERA_KEEPOUT_DW,
    DEFAULT_WIDTH_DW,
    DW_M,
    ENTRANCE_TYPES,
    MIN_CORNER_CLEARANCE_DW,
    WINDOW_TYPES,
    min_kept_width_dw,
)

GATE_WALL = "front"
# Куда уходит вход, если на front-стене места нет. Сначала соседние стены (они
# ближе к камере и к настоящему входу), back — последней: она в глубине кадра,
# её видно, и вход, которого там не видели, наименее правдоподобен.
FALLBACK_WALLS = ("left", "right", "back")
MIN_WINDOW_WIDTH_DW = 0.8   # окно уже 0.68 м рисовать бессмысленно
# Простенок «в обрез» для узкой стены: 0.1 м вместо обычных 0.2. Норму 0.2 м
# держим сколько можем, но в коридоре шириной 1.15 м (кадр frame14) выбор стоит
# так: либо вход с уменьшенным простенком там, где он на самом деле есть, либо
# нормативный простенок и дверь на длинной глухой стене. Первое честнее.
TIGHT_CORNER_CLEARANCE_DW = 0.1 / DW_M


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
            # Нижняя граница сужения — своя у каждого типа: окно доживает до
            # 0.3 м, дверь только до 0.7 м (уже — не дверь, а щель).
            if available < min(width, min_kept_width_dw(op.get("type", ""))):
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


def _hinge(wall: str, side: str) -> str:
    """Петля у того угла, к которому проём прижат.

    На горизонтальных стенах углы называются `left`/`right` (как и сам `side`), на
    вертикальных — `back`/`front`: ближний к началу стены конец это back-угол.
    Та же конвенция, что у дефолтных позиций мержа.
    """
    if wall in ("left", "right"):
        return "back" if side == "left" else "front"
    return side


def _camera_keepout(plan: dict, wall: str, start: float, end: float) -> list:
    """Занятый отрезок стены под точкой съёмки (пусто, если камера не на этой стене).

    Камера стоит на стене данными плана, а гейт вставляет проёмы в свободное
    место той же стены. Совпадение позиций даёт чертёж, где дверь прорезана прямо
    сквозь знак камеры. Отдаём этот отрезок как «занятый» — дальше его учитывает
    общий `usable_spans`, и никакой особой логики размещению не нужно.
    """
    camera = plan.get("camera") or {}
    if (camera.get("wall") or GATE_WALL) != wall:
        return []
    try:
        position = clamp(float(camera.get("position", 0.5)), 0.0, 1.0)
    except (TypeError, ValueError):
        position = 0.5
    centre = start + position * (end - start)
    return [(centre - CAMERA_KEEPOUT_DW / 2, centre + CAMERA_KEEPOUT_DW / 2)]


def _entrance_walls(openings: list) -> list[str]:
    """Порядок стен-кандидатов для входной двери: front, глухие, затем занятые.

    «Глухая» = на стене нет ни одного проёма. Порядок внутри группы фиксирован
    (`FALLBACK_WALLS`), чтобы план оставался воспроизводимым: две одинаково
    свободные стены не должны меняться местами от порядка проёмов во входе.
    """
    busy = {op.get("wall") for op in openings}
    rest = sorted(FALLBACK_WALLS, key=lambda w: (w in busy, FALLBACK_WALLS.index(w)))
    return [GATE_WALL] + rest


def ensure_door_and_window(plan: dict) -> list[str]:
    """Дополняет `plan['openings']` недостающими входом/окном. Возвращает пометки.

    Пометки уходят в meta плана: `door_inserted` (вход встал на front-стену),
    `door_inserted:<стена>` (front занят, вход ушёл на другую стену),
    `window_inserted`, а также `door_gate_skipped` / `window_gate_skipped` —
    места не нашлось нигде / на front-стене.
    """
    notes: list[str] = []
    openings = plan.setdefault("openings", [])
    present = {op.get("type") for op in openings}
    room = plan["room"]
    wall_start, wall_end = wall_span(room, GATE_WALL)

    if not (present & ENTRANCE_TYPES):
        want = DEFAULT_WIDTH_DW["door"]
        narrow = min_kept_width_dw("door")
        # Стену меняем ПОСЛЕДНЕЙ. Сначала на текущей пробуем норму, затем дверь
        # минимальной ширины с простенком в обрез: узкая комната (коридор,
        # гардеробная) не повод уводить вход на длинную глухую стену, где его
        # никто не видел. Приоритет front сохраняется целиком.
        for wall in _entrance_walls(openings):
            start, end = wall_span(room, wall)
            busy = occupied_spans(openings, wall)
            keepout = _camera_keepout(plan, wall, start, end)
            # Попытки по убыванию строгости: норма -> простенок в обрез -> (только
            # на front) без обхода знака камеры. Последняя ступень для комнат вроде
            # коридора frame14: там знак камеры занимает середину единственной
            # короткой стены, и обходить его негде. Знак — косметика чертежа, а
            # стена входа — геометрия; снимали как раз из этого проёма.
            attempts = [
                (MIN_CORNER_CLEARANCE_DW, busy + keepout),
                (TIGHT_CORNER_CLEARANCE_DW, busy + keepout),
            ]
            if wall == GATE_WALL:
                attempts.append((TIGHT_CORNER_CLEARANCE_DW, busy))
            placement, under_camera = None, False
            for step, (clearance, occupied) in enumerate(attempts):
                spans = usable_spans(start, end, occupied, clearance)
                placement = place_opening(
                    spans, want, start, end, anchor="corner", min_width=narrow
                )
                if placement is not None:
                    under_camera = step == 2
                    break
            if placement is None:
                continue
            offset, eff_width, side = placement
            openings.append({
                "type": "door",
                "wall": wall,
                "offset_dw": round(offset, 3),
                "width_dw": round(eff_width, 3),
                "swing": {"hinge": _hinge(wall, side), "direction": "in"},
            })
            notes.append("door_inserted" if wall == GATE_WALL else "door_inserted:%s" % wall)
            if eff_width < want - 1e-6:
                notes.append("door_narrowed:%.2f" % eff_width)
            if under_camera:
                notes.append("door_under_camera")
            break
        else:
            notes.append("door_gate_skipped")

    present = {op.get("type") for op in openings}
    if not (present & WINDOW_TYPES):
        # Точка съёмки окну не мешает: окно живёт в толще стены, а знак камеры
        # стоит внутри комнаты у её грани — они соседи, а не наложение. Обходит
        # камеру только дверь: её полотно и дуга выметают именно то место.
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

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
- нет ни одного окна -> окно по центру дальней от камеры боковой стены
  (front — только когда обе боковые заняты).

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
# Простенка «в обрез» больше нет (аудит архитектора 2026-08-10): 0.2 м от угла —
# жёсткий закон; в тесной стене сужается сама дверь (до 0.7 м) и отступает обход
# знака камеры, а не норма.


def enforce_min_opening_widths(plan: dict) -> list[str]:
    """Проём уже физического минимума расширяется до него (аудит архитектора).

    `min_kept_width_dw` (дверь/проход 0.7 м, окно 0.3 м) раньше применялся
    только при вынужденном сужении из-за тесноты — проём, которому сужаться не
    пришлось, проходил без проверки (fin424: балконная дверь 0.64 м от
    экстрактора). Дверей уже 0.7 м не бывает: раз модель выдала меньше, она
    занизила ширину, и честнее дотянуть до минимума (симметрично, offset на
    месте), чем оставить нежилую щель. Наложения после расширения разводит
    штатный `resolve_opening_conflicts` — вызывать ДО него.
    """
    notes: list[str] = []
    for op in plan.get("openings") or []:
        kind = op.get("type")
        try:
            width = float(op["width_dw"])
        except (KeyError, TypeError, ValueError):
            continue
        floor_w = min_kept_width_dw(kind or "")
        if width < floor_w - 1e-9:
            notes.append("widened:%s/%s %.2f->%.2f" % (kind, op.get("wall"), width, floor_w))
            op["width_dw"] = round(floor_w, 3)
    return notes


# Минимальный простенок бокового проёма со стороны камеры. Конец боковой стены
# у камеры в кадр не попадает (он за нижней кромкой), поэтому расстояние от
# проёма до него модель ГАДАЕТ — и на кадре lroom систематически занижала,
# рисуя проход впритык к невидимому краю. Ширina проёма и его дальний косяк
# видимы и не трогаются — сдвигается только положение вдоль стены.
MIN_FRONT_PIER_DW = 0.8


def keep_side_front_pier(plan: dict) -> list[str]:
    """Боковой проём не ближе MIN_FRONT_PIER_DW к невидимому front-концу стены.

    Работает только по проёмам боковых стен (left/right) и только когда проём
    залез в последние MIN_FRONT_PIER_DW стены — тогда он сдвигается к back ровно
    настолько, чтобы простенок восстановился. Вызывается ДО развода конфликтов:
    сдвиг может создать наложение, и его разведёт штатный механизм.
    """
    notes: list[str] = []
    room = plan.get("room") or {}
    if room.get("shape") != "rectangle":
        return notes
    for op in plan.get("openings") or []:
        if op.get("wall") not in ("left", "right"):
            continue
        try:
            start, end = wall_span(room, op["wall"])
            offset = float(op["offset_dw"])
            half = float(op["width_dw"]) / 2
        except (KeyError, TypeError, ValueError):
            continue
        limit = end - MIN_FRONT_PIER_DW - half
        if offset > limit and limit > start + half:
            notes.append("side_pier:%s/%s %.2f->%.2f" % (op.get("type"), op["wall"],
                                                         offset, limit))
            op["offset_dw"] = round(limit, 3)
    return notes


def snap_front_door_to_camera(plan: dict) -> list[str]:
    """Дверь экстрактора на front-стене встаёт в позицию камеры.

    Front-стена в кадре не видна (она за камерой; в лучшем случае — краем, как
    дверное полотно frame13), поэтому offset такой двери — догадка модели,
    дрожащая между прогонами. Вердикт пользователя: «камера прям в двери
    стоит» — снимают, как правило, от входа. Код ставит дверь под камеру;
    вставки гейта (`inserted`) не трогаются — их угол выбран правилом осознанно,
    и позиция камеры им не указ. Вызывается ПОСЛЕ камера-пробы и ДО развода
    конфликтов (наложения после переноса разводятся штатно).
    """
    notes: list[str] = []
    room = plan.get("room") or {}
    camera = plan.get("camera") or {}
    if (camera.get("wall") or GATE_WALL) != GATE_WALL:
        return notes
    try:
        start, end = wall_span(room, GATE_WALL)
    except (KeyError, TypeError, ValueError):
        return notes
    try:
        position = clamp(float(camera.get("position", 0.5)), 0.0, 1.0)
    except (TypeError, ValueError):
        return notes
    centre = start + position * (end - start)
    for op in plan.get("openings") or []:
        if (op.get("type") in ("door", "double_door") and op.get("wall") == GATE_WALL
                and not op.get("inserted")):
            try:
                half = float(op.get("width_dw", 0)) / 2
                old = float(op.get("offset_dw", 0))
            except (TypeError, ValueError):
                continue
            low = start + MIN_CORNER_CLEARANCE_DW + half
            high = end - MIN_CORNER_CLEARANCE_DW - half
            if low > high:
                continue
            new = clamp(centre, low, high)
            if abs(new - old) > 1e-3:
                op["offset_dw"] = round(new, 3)
                notes.append("front_door_to_camera:%.2f->%.2f" % (old, new))
    return notes


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


def _camera_position(plan: dict) -> float:
    """Позиция камеры вдоль front-стены (0..1), 0.5 при мусоре в данных."""
    camera = plan.get("camera") or {}
    try:
        return clamp(float(camera.get("position", 0.5)), 0.0, 1.0)
    except (TypeError, ValueError):
        return 0.5


def _window_walls(plan: dict) -> list[str]:
    """Порядок стен-кандидатов для вставного окна: боковая стена НАПРОТИВ
    бокового входа (не глухая), затем front, затем та же боковая даже глухой.

    Калибровка по вердиктам пользователя (бенч 16) + экспертиза floorplan-expert
    (2026-08-11, база знаний в .claude/agents/floorplan-expert.md):
    - вход жилой комнаты почти всегда ведёт из внутреннего коридора, наружная
      стена с окном — напротив него; правило распространено с прохода на ВСЁ
      семейство входов (door/double_door/passage) — кейс спальни v85, где дверь
      right не триггерила боковую вставку и окно ложно уезжало на front;
    - противоположная боковая берётся сразу только если VLM НЕ объявил её
      глухой: «глухая видимая стена с окном» противоречит фотографии (fin380);
      глухой вариант остаётся крайним запасным перед window_gate_skipped
      (для вымышленного окна «глухая» = «проёма не видно», не запрет);
    - back в кандидаты не входит: это самая просматриваемая стена кадра, и
      отсутствие её в solid_walls — слабое свидетельство (VLM мог не заметить),
      тогда как front не видел никто по построению.
    """
    entrances = {op.get("wall") for op in (plan.get("openings") or [])
                 if op.get("type") in ENTRANCE_TYPES and op.get("wall") in ("left", "right")}
    if len(entrances) == 1:
        opposite = "left" if entrances == {"right"} else "right"
        if opposite in (plan.get("solid_walls") or ()):
            return [GATE_WALL, opposite]
        return [opposite, GATE_WALL]
    return [GATE_WALL]


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
    `window_inserted:<стена>` (окно на боковой), `window_inserted` (обе боковые
    заняты, окно на front), `solid_wall_opened:<стена>` (окно встало на стену из
    `solid_walls` — метка глухости снята), а также `door_gate_skipped` /
    `window_gate_skipped` — места не нашлось ни на одной стене.
    """
    notes: list[str] = []
    openings = plan.setdefault("openings", [])
    present = {op.get("type") for op in openings}
    room = plan["room"]

    if not (present & ENTRANCE_TYPES):
        want = DEFAULT_WIDTH_DW["door"]
        narrow = min_kept_width_dw("door")
        # Угол для вставной двери на front выбирается по составу комнаты, а не
        # «какой свободный отрезок ближе» (двери скакали по углам от дрожания
        # экстракции, бенч 15↔16). Калибровка по кадрам frame12/frame13:
        # 1) сторона единственной глухой боковой стены — реальный вход обычно
        #    рядом с ней (frame13: зеркальный шкаф у входа);
        # 2) иначе сторона, чья боковая стена БЕЗ проёмов, — вход не делают
        #    вплотную к остеклению (frame12: балконная дверь справа -> вход слева);
        # 3) иначе дальний от камеры угол — его в кадре не видно, дверь там
        #    правдоподобнее всего.
        solid = set(plan.get("solid_walls") or [])
        busy = {op.get("wall") for op in openings}
        if ("left" in solid) != ("right" in solid):
            preferred = "left" if "left" in solid else "right"
        elif ("left" in busy) != ("right" in busy):
            preferred = "right" if "left" in busy else "left"
        else:
            preferred = "left" if _camera_position(plan) >= 0.5 else "right"
        # Стену меняем ПОСЛЕДНЕЙ: узкая комната (коридор, гардеробная) не повод
        # уводить вход на длинную глухую стену, где его никто не видел.
        # Приоритет front сохраняется целиком. Иерархия внутри стены (вердикт
        # пользователя по аудиту архитектора, 2026-08-10): простенок 0.2 м —
        # ЖЁСТКИЙ закон и не ужимается никогда; не хватает места — сужается сама
        # дверь (до 0.7 м); обход ЗНАКА камеры — косметика чертежа и отступает
        # первым: дверь под камерой легитимна (снимали из проёма), а слипание
        # знака с дугой на чертеже разводит рендер.
        for wall in _entrance_walls(openings):
            start, end = wall_span(room, wall)
            busy = occupied_spans(openings, wall)
            keepout = _camera_keepout(plan, wall, start, end)
            attempts = [(MIN_CORNER_CLEARANCE_DW, busy + keepout)]
            if wall == GATE_WALL:
                attempts.append((MIN_CORNER_CLEARANCE_DW, busy))
            placement, under_camera = None, False
            fallback = None   # лучший вариант с чужим углом (нормальный простенок)
            for step, (clearance, occupied) in enumerate(attempts):
                spans = usable_spans(start, end, occupied, clearance)
                last_attempt = step == len(attempts) - 1
                cand = place_opening(
                    spans, want, start, end, anchor="corner", min_width=narrow,
                    prefer_side=preferred if wall == GATE_WALL else None,
                )
                if cand is None:
                    continue
                # Предпочтённый угол держим до последнего шага: угол мог съесть
                # обход знака камеры (кадр frame12 — дверь убегала в середину
                # стены), а без него — освободиться (снимали от двери, frame13).
                # Если угол занят НАСТОЯЩИМ проёмом и не освободился даже на
                # последнем шаге — берём ранний вариант с нормальным простенком,
                # а не деградировавший.
                if wall == GATE_WALL and cand[2] != preferred:
                    if fallback is None:
                        fallback = (cand, step == len(attempts) - 1)
                    if not last_attempt:
                        continue
                    placement, under_camera = fallback
                    break
                placement, under_camera = cand, step == len(attempts) - 1 and wall == GATE_WALL
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
                "inserted": True,
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
        for wall in _window_walls(plan):
            start, end = wall_span(room, wall)
            spans = usable_spans(start, end, occupied_spans(openings, wall))
            placement = place_opening(
                spans, width, start, end, anchor="center", min_width=MIN_WINDOW_WIDTH_DW
            )
            if placement is None:
                continue
            offset, eff_width, _side = placement
            # Метка inserted: рендер не тянет сектор обзора к выдуманным
            # проёмам — их на фото не видели (schema_lite при пере-парсе
            # план-JSON поле молча отбрасывает, геометрию оно не трогает).
            openings.append({
                "type": "window",
                "wall": wall,
                "offset_dw": round(offset, 3),
                "width_dw": round(eff_width, 3),
                "inserted": True,
            })
            notes.append("window_inserted" if wall == GATE_WALL else "window_inserted:%s" % wall)
            # Вставка могла попасть на стену, которую VLM объявила глухой (для
            # вымышленного окна это не запрет: «глухая» значит «проёма не видно»,
            # а окно и не должно было быть видно). Снимаем метку, чтобы план не
            # противоречил сам себе — schema_lite выбрасывает проёмы на глухих.
            solid = plan.get("solid_walls")
            if solid and wall in solid:
                solid.remove(wall)
                notes.append("solid_wall_opened:%s" % wall)
            break
        else:
            notes.append("window_gate_skipped")

    return notes

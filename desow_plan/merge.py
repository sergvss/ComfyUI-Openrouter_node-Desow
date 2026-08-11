# КАНОНИЧЕСКИЙ КОД. Пары в бэкенде больше нет: `plan2d/merge.py` удалён вместе со
# вторым конвейером построения плана - мерж исполняет только воркфлоу (нода
# DesowPlanRender). Синхронизировать с бэкендом нужно лишь схему (schema_lite.py).
"""Мерж экстракции VLM с проёмами сканера (порт `hybrid-proto/merge.py`).

Правила (утверждены на приёмке Фазы 1):
- состав проёмов по (тип, стена) — приоритет СКАНЕРА; типы нормализуются
  (door|double_door|balcony_door -> door, window|floor_to_ceiling_window -> window);
- позиция / ширина / петля — от VLM (сканер не измеряет геометрию);
- проём VLM, которого нет у сканера, ОСТАЁТСЯ (сканер — пол, не потолок), КРОМЕ
  случая, когда он спаривается с непокрытой записью сканера того же типа на
  другой стене: это один проём, прочитанный дважды, а не два;
- дефолтная позиция считается по свободным интервалам стены (простенок от углов
  и соседей), а не «в центр» — иначе несколько дефолтов садятся друг на друга;
- СТЕНА проёма не обсуждается: не хватило места — сужаем до `min_kept_width_dw`,
  не влез и минимальный — выбрасываем; переезд на другую стену запрещён (он давал
  проёмы-призраки за спиной камеры);
- сканер — «пол, а не потолок», но не приказ: стену, которую VLM ЯВНО назвал
  глухой (`solid_walls`), его запись не восстанавливает — детектор ошибается на
  отражениях зеркального шкафа. Молчание VLM возражением не считается;
- расхождение по составу (у VLM нет проёма сканера) -> вызывающий делает второй
  прогон VLM и передаёт оба; после двух расхождений — состав сканера, позиции от
  ближайшего прогона (или дефолт: центр стены, ширина door=1.0 / window=1.6 dw).

`merge_with_scanner` — чистая функция: повторный вызов экстракции живёт снаружи
(см. `plan2d.service.build_plan`), чтобы мерж оставался тестируемым без сети.
"""
from __future__ import annotations

import copy

from .geometry import occupied_spans, place_opening, usable_spans, wall_span
from .schema_lite import DEFAULT_WIDTH_DW, min_kept_width_dw

# Типы, для которых спор о стене между сканером и VLM означает ОДИН проём,
# прочитанный дважды. `passage` сюда не входит: детектор сканера открытые проходы
# вообще не эмитит, поэтому passage от VLM — всегда законная добавка.
PAIRABLE_TYPES = frozenset({"door", "window"})


def norm_type(t: str) -> str:
    """Тип проёма в терминах сканера: створки, остекление в пол и балкон не различаются.

    Детектор сканера подтипов не знает: любую дверь он репортит как `door`
    (`sliding_door` и прочие compound-типы сворачивает `scanner.normalize_kind`).
    Поэтому балконная дверь VLM обязана покрываться его `door`, иначе состав
    расходится и мерж уходит в деградацию, теряя реальную геометрию проёма.
    """
    if t in ("door", "double_door", "balcony_door"):
        return "door"
    if t in ("window", "floor_to_ceiling_window"):
        return "window"
    return t


def _key(op: dict) -> tuple:
    return (norm_type(op["type"]), op["wall"])


def covers(vlm_ops: list, scanner_ops: list) -> bool:
    """Все проёмы сканера присутствуют у VLM (по мультимножеству (тип, стена))."""
    pool = [_key(o) for o in vlm_ops]
    for so in scanner_ops:
        k = _key(so)
        if k in pool:
            pool.remove(k)
        else:
            return False
    return True


def scanner_openings_from_scan(scan_openings: list[dict]) -> list[dict]:
    """`parse_scan_openings` -> вход мержа.

    Сканерный элемент — `{kind, wall, quantity, display_type}`; мержу нужен
    `{type, wall}`. Берём `kind` (нормализованный effective type), а не
    `display_type`: составные `arched_window` / `sliding_door` уже свёрнуты в
    базовый тип, и сверка состава идёт по нему.
    """
    out: list[dict] = []
    for entry in scan_openings or []:
        kind, wall = entry.get("kind"), entry.get("wall")
        if not kind or not wall:
            continue
        for _ in range(max(1, int(entry.get("quantity") or 1))):
            out.append({"type": kind, "wall": wall})
    return out


def _place_default(room: dict, placed: list[dict], wall: str, width: float, kind: str):
    """Позиция дефолтного проёма с учётом УЖЕ размещённых. `(offset, width, side)`.

    Сканер знает состав проёмов, но не их геометрию; когда VLM не дал позицию,
    её приходится назначать. Раньше это был безусловный центр стены — и на
    деградации мержа два-три дефолта садились в одну точку, а то и поверх проёма
    с реальной геометрией (боевые кадры e4/f1 серии v83).

    **Стена не обсуждается.** Проём ставится только на ту стену, которую назвал
    сканер: не хватило места — сужаем до `min_kept_width_dw`, не помещается и
    минимальный — возвращаем None (проём отбрасывается). Прежний перебор
    «противоположная -> любая свободная» давал стену-призрак: на кадре frame15
    второй сегмент панорамного остекления уезжал на переднюю стену, за спину
    камеры, где ничего подобного нет. Проём за камерой имеет право создавать
    только гейт — он это делает осознанно и помечает.
    """
    # Якорь «по центру свободного места» — прежнее поведение дефолта (центр стены),
    # ограниченное свободными интервалами. Прижимать дверь к углу здесь незачем:
    # это норма ГЕЙТА для двери, которой никто не видел, а тут дверь видел сканер.
    start, end = wall_span(room, wall)
    spans = usable_spans(start, end, occupied_spans(placed, wall))
    return place_opening(
        spans, width, start, end, anchor="center", min_width=min_kept_width_dw(kind)
    )


def merge_with_scanner(scanner: dict, vlm_runs: list[dict]) -> tuple[dict, dict]:
    """scanner: `{"openings":[{type, wall, ...}]}`; vlm_runs: полные JSON экстракций.

    Возврат: `(merged_json, meta)`. meta: `{source_run, consensus_needed,
    fallback_scanner_composition, added_from_vlm, defaults_used, narrowed,
    dropped_no_space, dropped_unconfirmed, paired_wall_dispute}`.
    """
    scanner_ops = scanner.get("openings", [])
    meta = {
        "source_run": None,
        "consensus_needed": False,
        "fallback_scanner_composition": False,
        "added_from_vlm": [],
        "defaults_used": [],
        "narrowed": [],
        "dropped_no_space": [],
        "dropped_unconfirmed": [],
        "paired_wall_dispute": [],
    }

    # 1) Ищем прогон VLM, покрывающий состав сканера.
    chosen = None
    for i, run in enumerate(vlm_runs):
        if covers(run.get("openings", []), scanner_ops):
            chosen, meta["source_run"] = run, i
            if i > 0:
                meta["consensus_needed"] = True   # первый прогон разошёлся, взяли следующий
            break

    if chosen is not None:
        merged = copy.deepcopy(chosen)
        pool = [_key(o) for o in scanner_ops]
        for op in merged.get("openings", []):
            k = _key(op)
            if k in pool:
                pool.remove(k)
            else:
                meta["added_from_vlm"].append(k)
        return merged, meta

    # 2) Все прогоны разошлись: состав сканера, позиции от ближайшего VLM.
    meta["consensus_needed"] = True
    meta["fallback_scanner_composition"] = True
    base = copy.deepcopy(vlm_runs[0])   # комната/пропорции — от первого прогона
    merged_ops: list[dict] = []
    candidates = [op for run in vlm_runs for op in run.get("openings", [])]
    used: list[int] = []

    # Кандидаты VLM разбираются ФАЗАМИ по всему списку сканера, а не «первым
    # подходящим» на каждый проём по очереди. Иначе первый же проём уводит
    # кандидата, который ТОЧНО соответствует другому: scanner [door/left,
    # door/back] + VLM [door/back] → left забирал бы дверь back как «спор о
    # стене», а настоящая back-дверь уходила бы в дефолт с потерей геометрии.
    #   Фаза 1 — точное совпадение (тип + стена): геометрия VLM применима как есть.
    #   Фаза 2 — та же стена, другой тип: геометрия применима, тип берём от сканера.
    #   Фаза 3 — тот же тип, другая стена: это спор о стене (один проём, прочитанный
    #            дважды); геометрия VLM НЕприменима — она про другую стену → дефолт.
    matched: dict[int, dict] = {}
    disputed: dict[int, dict] = {}

    def _assign(predicate, store: dict) -> None:
        for si, so in enumerate(scanner_ops):
            if si in matched or si in disputed:
                continue
            cand = next((c for c in candidates if id(c) not in used and predicate(so, c)), None)
            if cand is not None:
                used.append(id(cand))
                store[si] = cand

    _assign(lambda so, c: _key(c) == _key(so), matched)
    _assign(lambda so, c: c.get("wall") == so["wall"], matched)
    # `passage` не спариваем: сканер открытые проходы вообще не эмитит, поэтому
    # любой passage от VLM — законная добавка, а не спорное чтение.
    _assign(
        lambda so, c: norm_type(so["type"]) in PAIRABLE_TYPES
        and norm_type(c.get("type", "")) == norm_type(so["type"]),
        disputed,
    )

    # Проёмы с реальной геометрией VLM раскладываются ПЕРВЫМИ и целиком: дефолты
    # обязаны обходить их все, а не только те, что оказались раньше по порядку
    # сканера (иначе дефолт садится поверх проёма, который приедет следом).
    resolved: dict[int, dict] = {}
    for si, cand in matched.items():
        so = scanner_ops[si]
        op = copy.deepcopy(cand)
        op["type"] = so["type"] if norm_type(so["type"]) != norm_type(cand.get("type", "")) else cand["type"]
        resolved[si] = op
    merged_ops = list(resolved.values())   # занятость для дефолтов; порядок соберём в конце

    # Стены, которые VLM ЯВНО назвал глухими: его право возразить детектору.
    solid_walls = set(base.get("solid_walls") or [])

    for si, so in enumerate(scanner_ops):
        if si not in matched:
            k = _key(so)
            room = base.get("room", {})
            nt = norm_type(so["type"])
            # Активное противоречие: сканер видит проём там, где VLM написал
            # «стена глухая» (зеркальный шкаф-купе, кадр frame13). Восстанавливать
            # такой проём дефолтом нельзя — это выдумка на ровном месте. Молчание
            # VLM противоречием НЕ считается: там сканер по-прежнему «пол».
            if so["wall"] in solid_walls:
                meta["dropped_unconfirmed"].append(k)
                continue
            # Спор о стене: тот же тип проёма, но VLM и сканер назвали разные стены.
            # Это ОДИН проём, прочитанный дважды, а не два (боевой кадр e5 серии
            # v83: door/left сканера + door/back VLM дали две двери в комнате с
            # одной). Спариваем 1:1, стена — от сканера (состав всегда его),
            # позиция VLM неприменима (она про другую стену) -> дефолт ниже.
            twin = disputed.get(si)
            if twin is not None:
                meta["paired_wall_dispute"].append((nt, twin.get("wall"), so["wall"]))
            wall = so["wall"]
            want = DEFAULT_WIDTH_DW.get(nt, 1.0)
            placement = _place_default(room, merged_ops, wall, want, nt)
            if placement is None:
                meta["dropped_no_space"].append(k)
                continue
            offset, width, side = placement
            if width < want - 1e-6:
                meta["narrowed"].append((k, round(want, 3), round(width, 3)))
            op = {
                "type": so["type"],
                "wall": wall,
                "offset_dw": round(offset, 3),
                "width_dw": round(width, 3),
            }
            if nt == "door":
                # Петля у того угла, к которому проём прижат: на вертикальных
                # стенах ближний к back-углу конец называется "back".
                if wall in ("left", "right"):
                    op["swing"] = {"hinge": "back" if side == "left" else "front", "direction": "in"}
                else:
                    op["swing"] = {"hinge": side, "direction": "in"}
            merged_ops.append(op)
            resolved[si] = op
            meta["defaults_used"].append(k)

    # Итоговый порядок — как у сканера (matched-проёмы выше собирались вперёд ради
    # занятости, а не ради порядка).
    merged_ops = [resolved[si] for si in sorted(resolved)]
    # Добавки VLM сверх сканера (из первого прогона) остаются.
    pool = [_key(o) for o in scanner_ops]
    for op in vlm_runs[0].get("openings", []):
        k = _key(op)
        if k in pool:
            pool.remove(k)
        elif id(op) not in used:
            merged_ops.append(copy.deepcopy(op))
            meta["added_from_vlm"].append(k)
    base["openings"] = merged_ops
    return base, meta


# ── медиана нескольких прогонов экстрактора ──────────────────────────

# Проём другого прогона считается ТЕМ ЖЕ, если тип и стена совпали, а центры
# ближе этого. Толеранс широкий сознательно: прогон-выброс уезжает на ~2 dw
# (lroom: дверь 1.2 против 3.3/3.6), и он обязан попасть в медиану, чтобы быть
# переголосованным — в том числе когда выбросом оказался БАЗОВЫЙ прогон.
# Разные проёмы одного типа на одной стене разводит жадное 1:1-сопоставление
# по ближайшему центру.
SAME_OPENING_TOL_DW = 2.5


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def median_extractions(plans: list[dict]) -> list[str]:
    """Усредняет геометрию базового плана по нескольким прогонам экстрактора.

    Одна экстракция недетерминирована даже при t=0/seed: размеры комнаты и
    offset'ы видимых проёмов дрожат между прогонами на ±0.3-0.5 dw, а изредка
    прогон-выброс уезжает на 1.5+ dw (кадр lroom: дверь 1.2 против 3.3 на одном
    фото). Состав при этом стабилен. Лекарство — медиана: граф зовёт экстрактор
    трижды (разные seed), состав берётся от базового прогона, а размеры комнаты
    и offset/width каждого проёма — медианой по совпавшим прочтениям (тип+стена,
    центры ближе SAME_OPENING_TOL_DW, соответствие 1:1). Медиана из трёх гасит
    одиночный выброс полностью; из двух — хотя бы усредняет.

    Мутирует ПЕРВЫЙ план списка (базовый), возвращает пометки для debug.
    Прогоны с другой формой комнаты в медиану размеров не входят.
    """
    notes: list[str] = []
    if len(plans) < 2:
        return notes
    base = plans[0]
    base_room = base.get("room") or {}
    for key in ("width_dw", "depth_dw"):
        values = []
        for plan in plans:
            room = plan.get("room") or {}
            if room.get("shape") == base_room.get("shape") and room.get(key) is not None:
                values.append(float(room[key]))
        if len(values) >= 2:
            med = round(_median(values), 3)
            if abs(med - float(base_room[key])) > 1e-9:
                notes.append("room.%s %.2f->%.2f (медиана %d прогонов)"
                             % (key, float(base_room[key]), med, len(values)))
                base_room[key] = med

    used: dict[int, set] = {i: set() for i in range(1, len(plans))}
    for op in base.get("openings") or []:
        try:
            offsets = [float(op["offset_dw"])]
            widths = [float(op["width_dw"])]
        except (KeyError, TypeError, ValueError):
            continue
        for i, other in enumerate(plans[1:], start=1):
            best, best_d = None, SAME_OPENING_TOL_DW
            for cand in other.get("openings") or []:
                if (cand.get("type") != op.get("type") or cand.get("wall") != op.get("wall")
                        or id(cand) in used[i]):
                    continue
                try:
                    d = abs(float(cand["offset_dw"]) - offsets[0])
                except (KeyError, TypeError, ValueError):
                    continue
                if d <= best_d:
                    best, best_d = cand, d
            if best is not None:
                used[i].add(id(best))
                offsets.append(float(best["offset_dw"]))
                widths.append(float(best["width_dw"]))
        if len(offsets) >= 2:
            new_offset = round(_median(offsets), 3)
            if abs(new_offset - offsets[0]) > 1e-3:
                notes.append("%s/%s offset %.2f->%.2f (медиана %d)"
                             % (op.get("type"), op.get("wall"), offsets[0], new_offset,
                                len(offsets)))
            op["offset_dw"] = new_offset
            op["width_dw"] = round(_median(widths), 3)
    return notes


# ── коридорная стена — боковая (поворот) ─────────────────────────────

# Насколько далеко дальний косяк пассажа может отстоять от угла глухой стены,
# чтобы всё ещё читаться «коридор начинается прямо в углу».
CORRIDOR_CORNER_TOL_DW = 0.3
# Дверь, чей косяк ближе этого к устью коридора, стоит в самом коридоре — по
# фото она видна СКВОЗЬ проём и комнате не принадлежит (вердикт пользователя по
# fin463: «этой двери нет в комнате, она в коридоре»). Вход в комнату при этом
# остаётся — им становится сам пассаж. Допуск широкий: расстояние дверь-устье
# дрожит между прогонами экстрактора на ±0.5 dw (контрольный прогон бенча 16
# пропустил ту же дверь при 0.5), а настоящая дверь главной зоны стоит от устья
# заметно дальше.
CORRIDOR_DOOR_TOL_DW = 1.5


def reorient_corridor_wall(plan: dict) -> list[str]:
    """Коридор вдоль глухой (зеркальной) стены: стена с проёмами — боковая.

    Паттерн кадра fin463: съёмка в угол. Колонна/пилястра на стыке склеивает
    для VLM две плоскости в одну «back»-стену: туда попадают и проём коридора,
    прижатый к углу глухой (зеркальной) стены, и дверь, которая на самом деле
    стоит в коридоре. Разметка пользователя: большая пустая плоскость — это
    back-стена комнаты, а стена с проёмами — БОКОВАЯ (вдоль неё уходит коридор),
    то есть комната глубже, чем широка. Промптовое исполнение провалилось дважды
    (модель не переносит проём со стены, где видит его плоскость, и путает
    пересчёт камеры), поэтому преобразование делает код:

    - триггер: rectangle + пассаж на back, чей дальний косяк ближе
      CORRIDOR_CORNER_TOL_DW к углу глухой (solid_walls) боковой стены;
    - width_dw и depth_dw меняются местами (стена с проёмами — боковая);
    - пассаж встаёт в СЕРЕДИНУ этой боковой стены: точную глубину устья с фото
      не измерить (стена уходит в перспективу, VLM сжимает её конец), середина —
      наименее ложное утверждение (подтверждена двумя скетчами пользователя);
    - дверь впритык к устью (ближе CORRIDOR_DOOR_TOL_DW) — коридорная,
      выбрасывается; остальные проёмы старой back-стены переезжают на боковую
      с тем же отсчётом от общего угла;
    - глухость: старая противоположная боковая стена (большая пустая плоскость)
      становится back и остаётся глухой; зеркальная метка снимается — её плоскость
      после поворота не образует целой стены;
    - camera.position сохраняется: смещение камеры к коридорной стороне по
      смыслу то же на новой front-стене.

    Сложные случаи поворот пропускает без изменений: проёмы на большой пустой
    плоскости или на front, перегородки — там переотсчёт не однозначен, а плана
    честнее не трогать. Вызывается ПОСЛЕ мержа (пассаж уже подтверждён составом)
    и ДО развода конфликтов: угловой пассаж иначе гибнет там как «не влезающий».
    Возвращает пометки для debug; пустой список = триггер не сработал.
    """
    room = plan.get("room") or {}
    if room.get("shape") != "rectangle":
        return []
    solid = plan.get("solid_walls") or []
    openings = plan.get("openings") or []
    try:
        width = float(room["width_dw"])
        depth = float(room["depth_dw"])
    except (KeyError, TypeError, ValueError):
        return []

    for side in ("right", "left"):
        if side not in solid:
            continue
        far_wall = "left" if side == "right" else "right"   # большая пустая плоскость
        corner = width if side == "right" else 0.0
        for op in openings:
            # Устье коридора модель читает то «passage», то «door» (прогоны
            # бенча 16 дали оба варианта на одном фото) — тип здесь не признак:
            # признак — проём-вход, прижатый к углу глухой (зеркальной) стены.
            if op.get("type") not in ("passage", "door", "double_door") or op.get("wall") != "back":
                continue
            try:
                offset = float(op["offset_dw"])
                op_width = float(op["width_dw"])
            except (KeyError, TypeError, ValueError):
                continue
            if op_width >= width / 2:
                continue   # проём в полстены — не устье коридора
            far_jamb = offset + op_width / 2 if side == "right" else offset - op_width / 2
            if abs(corner - far_jamb) > CORRIDOR_CORNER_TOL_DW:
                continue
            if plan.get("partitions"):
                return []
            if any(o.get("wall") in (far_wall, "front") for o in openings):
                return []
            # Отсчёт вдоль старой back-стены от ОБЩЕГО угла с коридорной стеной:
            # для right-коридора это её левый угол (offset как есть), для left —
            # правый (width - offset). Тот же угол — back-конец новой боковой
            # стены, так что позиции переезжают без пересчёта системы координат.
            def from_corner(x: float) -> float:
                return x if side == "right" else width - x

            notes = ["corridor_wall_to_side:%s, комната %.2fx%.2f -> %.2fx%.2f dw"
                     % (side, width, depth, depth, width)]
            room["width_dw"], room["depth_dw"] = round(depth, 3), round(width, 3)

            kept: list[dict] = []
            passage_jamb = from_corner(offset) - op_width / 2
            for other in openings:
                if other is op:
                    continue
                if other.get("wall") == "back":
                    half = float(other.get("width_dw", 0)) / 2
                    centre = from_corner(float(other.get("offset_dw", 0)))
                    if (other.get("type") in ("door", "double_door")
                            and passage_jamb - (centre + half) <= CORRIDOR_DOOR_TOL_DW):
                        notes.append("corridor_door_dropped:%s/back" % other.get("type"))
                        continue
                    other["wall"] = side
                    other["offset_dw"] = round(centre, 3)
                kept.append(other)
            # Устье — в середину новой боковой стены (её длина = старая width).
            # Полотна у входа в коридор нет — тип нормализуется в passage.
            if op.get("type") != "passage":
                notes.append("corridor_mouth:%s->passage" % op.get("type"))
                op["type"] = "passage"
                op.pop("swing", None)
            op["wall"] = side
            op["offset_dw"] = round(width / 2, 3)
            kept.append(op)
            notes.append("passage_mid_side:%s %.2f" % (side, width / 2))
            plan["openings"] = kept

            new_solid = ["back"] if far_wall in solid else []
            if side in solid:
                notes.append("mirror_wall_dissolved:%s" % side)
            plan["solid_walls"] = new_solid
            return notes
    return []

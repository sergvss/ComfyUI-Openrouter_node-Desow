# ВЕНДОРЕННЫЙ КОД. Источник: desow/plan2d/merge.py - синхронизировать при правках.
# Отличие от источника: только импорт схемы (schema -> schema_lite). Двойное
# ведение осознанное: на машине ComfyUI бэкенда Desow нет (README, «Ноды Desow»).
"""Мерж экстракции VLM с проёмами сканера (порт `hybrid-proto/merge.py`).

Правила (утверждены на приёмке Фазы 1):
- состав проёмов по (тип, стена) — приоритет СКАНЕРА; типы нормализуются
  (door|double_door -> door, window|floor_to_ceiling_window -> window);
- позиция / ширина / петля — от VLM (сканер не измеряет геометрию);
- проём VLM, которого нет у сканера, ОСТАЁТСЯ (сканер — пол, не потолок), КРОМЕ
  случая, когда он спаривается с непокрытой записью сканера того же типа на
  другой стене: это один проём, прочитанный дважды, а не два;
- дефолтная позиция считается по свободным интервалам стены (простенок от углов
  и соседей), а не «в центр» — иначе несколько дефолтов садятся друг на друга;
- расхождение по составу (у VLM нет проёма сканера) -> вызывающий делает второй
  прогон VLM и передаёт оба; после двух расхождений — состав сканера, позиции от
  ближайшего прогона (или дефолт: центр стены, ширина door=1.0 / window=1.6 dw).

`merge_with_scanner` — чистая функция: повторный вызов экстракции живёт снаружи
(см. `plan2d.service.build_plan`), чтобы мерж оставался тестируемым без сети.
"""
from __future__ import annotations

import copy

from .geometry import occupied_spans, place_opening, usable_spans, wall_span
from .schema_lite import DEFAULT_WIDTH_DW

# Стена той же ориентации — первый кандидат при переносе проёма, которому не
# хватило места на назначенной сканером стене.
OPPOSITE_WALL = {"back": "front", "front": "back", "left": "right", "right": "left"}
ALL_WALLS = ("back", "front", "left", "right")

# Типы, для которых спор о стене между сканером и VLM означает ОДИН проём,
# прочитанный дважды. `passage` сюда не входит: детектор сканера открытые проходы
# вообще не эмитит, поэтому passage от VLM — всегда законная добавка.
PAIRABLE_TYPES = frozenset({"door", "window"})


def norm_type(t: str) -> str:
    """Тип проёма в терминах сканера: створки и остекление в пол не различаются."""
    if t in ("door", "double_door"):
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
    """Позиция дефолтного проёма с учётом УЖЕ размещённых. `(wall, offset, width, side)`.

    Сканер знает состав проёмов, но не их геометрию; когда VLM не дал позицию,
    её приходится назначать. Раньше это был безусловный центр стены — и на
    деградации мержа два-три дефолта садились в одну точку, а то и поверх проёма
    с реальной геометрией (боевые кадры e4/f1 серии v83).

    Порядок поиска стены: назначенная сканером -> противоположная (та же
    ориентация, чаще всего это его ошибка привязки) -> любая свободная.
    Нигде не поместилось -> None: проём отбрасывается, потому что план без
    проёма честнее плана с проёмом внахлёст.
    """
    # Якорь «по центру свободного места» — прежнее поведение дефолта (центр стены),
    # ограниченное свободными интервалами. Прижимать дверь к углу здесь незачем:
    # это норма ГЕЙТА для двери, которой никто не видел, а тут дверь видел сканер.
    order = [wall, OPPOSITE_WALL.get(wall)] + list(ALL_WALLS)
    tried: list[str] = []
    for candidate in order:
        if candidate is None or candidate in tried:
            continue
        tried.append(candidate)
        start, end = wall_span(room, candidate)
        spans = usable_spans(start, end, occupied_spans(placed, candidate))
        placement = place_opening(spans, width, start, end, anchor="center")
        if placement is not None:
            offset, eff_width, side = placement
            return candidate, offset, eff_width, side
    return None


def merge_with_scanner(scanner: dict, vlm_runs: list[dict]) -> tuple[dict, dict]:
    """scanner: `{"openings":[{type, wall, ...}]}`; vlm_runs: полные JSON экстракций.

    Возврат: `(merged_json, meta)`. meta: `{source_run, consensus_needed,
    fallback_scanner_composition, added_from_vlm, defaults_used, moved_to_wall,
    dropped_no_space, paired_wall_dispute}`.
    """
    scanner_ops = scanner.get("openings", [])
    meta = {
        "source_run": None,
        "consensus_needed": False,
        "fallback_scanner_composition": False,
        "added_from_vlm": [],
        "defaults_used": [],
        "moved_to_wall": [],
        "dropped_no_space": [],
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
    for so in scanner_ops:
        k = _key(so)
        cand = next((c for c in candidates if _key(c) == k and id(c) not in used), None)
        if cand is None:
            # Та же стена, любой тип — берём геометрию, тип от сканера.
            cand = next((c for c in candidates if c["wall"] == so["wall"] and id(c) not in used), None)
        if cand is not None:
            used.append(id(cand))
            op = copy.deepcopy(cand)
            op["type"] = so["type"] if norm_type(so["type"]) != norm_type(cand.get("type", "")) else cand["type"]
            merged_ops.append(op)
        else:
            room = base.get("room", {})
            nt = norm_type(so["type"])
            # Спор о стене: тот же тип проёма, но VLM и сканер назвали разные стены.
            # Это ОДИН проём, прочитанный дважды, а не два (боевой кадр e5 серии
            # v83: door/left сканера + door/back VLM дали две двери в комнате с
            # одной). Спариваем 1:1, стена — от сканера (состав всегда его),
            # позиция VLM неприменима (она про другую стену) -> дефолт ниже.
            # `passage` не спариваем: сканер их не эмитит вообще, а значит любой
            # passage VLM — законная добавка, а не спорное чтение.
            twin = None
            if nt in PAIRABLE_TYPES:
                twin = next(
                    (c for c in candidates
                     if id(c) not in used and norm_type(c.get("type", "")) == nt),
                    None,
                )
            if twin is not None:
                used.append(id(twin))
                meta["paired_wall_dispute"].append((nt, twin.get("wall"), so["wall"]))
            placement = _place_default(
                room, merged_ops, so["wall"], DEFAULT_WIDTH_DW.get(nt, 1.0), nt
            )
            if placement is None:
                meta["dropped_no_space"].append(k)
                continue
            wall, offset, width, side = placement
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
            meta["defaults_used"].append(k)
            if wall != so["wall"]:
                meta["moved_to_wall"].append((k, wall))
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

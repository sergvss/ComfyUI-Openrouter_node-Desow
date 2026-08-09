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

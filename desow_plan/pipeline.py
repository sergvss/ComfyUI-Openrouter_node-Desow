"""Конвейер пустого 2D-плана внутри воркфлоу: JSON -> PNG + канонический план.

Порядок шагов повторяет `plan2d.service.build_empty_plan` (бэкенд), из которого
выброшено всё сетевое: экстракцией занимается OpenRouter-нода графа, сюда её
ответ приходит уже строкой.

    разбор экстракции -> мерж со сканером -> развод конфликтов -> гейт -> рендер

Инвариант ноды: **ошибка данных не роняет воркфлоу.** Любая фатальная проблема
даёт белый лист, пустой `plan_json` и человекочитаемый `debug` с причиной — скан
обязан завершиться даже при сбое плана (`docs/design/SCAN_PLAN_INTEGRATION.md`).
Fail-loud остаётся на бэкенде, где есть ретрай и отказ до списания кредитов.
"""
from __future__ import annotations

import io
import json

from PIL import Image

from .gate import ensure_door_and_window, resolve_opening_conflicts
from .merge import merge_with_scanner, scanner_openings_from_scan
from .render import CANVAS, PAGE, render_plan
from .scanner import extract_json_object, parse_scanner_openings
from .schema_lite import DW_M, PlanDataError, validate_plan


def blank_png():
    """Белый лист формата графстандарта — заглушка на случай сбоя."""
    buf = io.BytesIO()
    Image.new("L", CANVAS, PAGE).save(buf, format="PNG")
    return buf.getvalue()


def render_camera_png(plan_json):
    """Тот же план, но с маркером камеры: PNG-байты (четвёртый выход ноды).

    Отдельным проходом по уже собранному `plan_json`, а не вторым возвратом
    `build_empty_plan`: основной выход обязан остаться прежним БАЙТ-В-БАЙТ — на
    нём бэкенд расставляет мебель, и цветной маркер там был бы помехой.

    Сбой камеры не имеет права задеть основные выходы, поэтому исключения
    гасятся здесь целиком: пустой план -> белый лист.
    """
    if not plan_json:
        return blank_png()
    try:
        png, _meta = render_plan(json.loads(plan_json), with_furniture=False, draw_camera=True)
    except Exception:
        return blank_png()
    return png


def _fmt_ops(openings):
    """`[door/front, window/back]` — компактный состав проёмов для debug."""
    return "[%s]" % ", ".join("%s/%s" % (o.get("type"), o.get("wall")) for o in openings) if openings else "[]"


def build_empty_plan(extraction_json, scanner_openings_json="", room_type=""):
    """`(png_bytes, plan_json, debug)` по ответу экстрактора и проёмам сканера.

    `plan_json` — компактный канонический план ПОСЛЕ мержа и гейта; при сбое
    пустая строка. `debug` — построчный отчёт: что разобрано, что дал сканер, что
    решил мерж, что вставил гейт, с каким масштабом отрисовано.
    """
    debug = []

    # 1) Экстракция VLM: единственный обязательный вход.
    try:
        raw = extract_json_object(extraction_json)
    except (ValueError, TypeError) as exc:
        debug.append("plan: ОШИБКА extract_invalid_json: %s" % exc)
        debug.append("extract_raw: %r" % ((extraction_json or "")[:200],))
        return blank_png(), "", "\n".join(debug)
    try:
        plan, plan_notes = validate_plan(raw)
    except PlanDataError as exc:
        debug.append("plan: ОШИБКА extract_invalid_schema: %s" % exc)
        return blank_png(), "", "\n".join(debug)

    room = plan["room"]
    debug.append(
        "extract: комната %s %.2fx%.2f dw (%.2fx%.2f м), проёмов %d %s, простенков %d"
        % (room["shape"], room["width_dw"], room["depth_dw"],
           room["width_dw"] * DW_M, room["depth_dw"] * DW_M,
           len(plan["openings"]), _fmt_ops(plan["openings"]), len(plan.get("partitions", [])))
    )
    for note in plan_notes:
        debug.append("extract_fix: %s" % note)

    # 2) Сканер: состав проёмов по (тип, стена) — его приоритет, геометрия — VLM.
    scan_entries, scan_room, scan_notes = parse_scanner_openings(scanner_openings_json)
    scanner_ops = scanner_openings_from_scan(scan_entries)
    debug.append("scanner: проёмов %d %s%s" % (
        len(scanner_ops), _fmt_ops(scanner_ops),
        (" | комната сканера: %s" % json.dumps(scan_room, ensure_ascii=False)) if scan_room else "",
    ))
    for note in scan_notes:
        debug.append("scanner_fix: %s" % note)

    plan, merge_meta = merge_with_scanner({"openings": scanner_ops}, [plan])
    debug.append(
        "merge: состав %s, добавлено от VLM %s, дефолтных позиций %s"
        % ("сканера (расхождение)" if merge_meta["fallback_scanner_composition"] else "VLM покрыл сканер",
           merge_meta["added_from_vlm"] or "[]", merge_meta["defaults_used"] or "[]")
    )
    for kind, vlm_wall, scan_wall in merge_meta["paired_wall_dispute"]:
        # Один проём, прочитанный с разными стенами, а не два разных.
        debug.append("merge_pair: %s %s→%s (спор о стене, взята стена сканера)" % (kind, vlm_wall, scan_wall))
    for key, wall in merge_meta["moved_to_wall"]:
        debug.append("merge_move: %s -> стена %s (на своей нет места)" % (key, wall))
    for key in merge_meta["dropped_no_space"]:
        debug.append("merge_drop: %s выброшен (места нет ни на одной стене)" % (key,))

    # 3) Развод конфликтов: наложения и проёмы впритык к углу приходят и от VLM,
    # и из дефолтных позиций мержа. До гейта, чтобы он считал свободные интервалы
    # стены по уже вычищенной картине.
    conflict_notes = resolve_opening_conflicts(plan)
    debug.append("conflicts: %s" % (", ".join(conflict_notes) if conflict_notes else "нет"))

    # 4) Гейт: комната без двери/окна дальше не идёт (решение принимается здесь,
    # кодом, и въезжает в сохранённый план — инструкции генератору не нужны).
    gate_notes = ensure_door_and_window(plan)
    debug.append("gate: %s" % (", ".join(gate_notes) if gate_notes else "дверь и окно уже есть"))
    debug.append("openings_final: %d %s" % (len(plan["openings"]), _fmt_ops(plan["openings"])))

    # 5) Рендер. Геометрические ошибки тоже не роняют воркфлоу.
    try:
        png, render_meta = render_plan(plan, with_furniture=False)
    except (ValueError, KeyError, ZeroDivisionError, OverflowError) as exc:
        debug.append("plan: ОШИБКА render_failed: %s: %s" % (exc.__class__.__name__, exc))
        return blank_png(), "", "\n".join(debug)
    debug.append("render: %s" % json.dumps(render_meta, ensure_ascii=False))

    debug.append("camera: %s" % json.dumps(plan["camera"], ensure_ascii=False))

    if room_type:
        # Тип комнаты сквозной: геометрию не меняет, но нужен фазе расстановки.
        plan["room_type"] = room_type
        debug.append("room_type: %s" % room_type)

    plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    debug.insert(0, "plan: ok (%d символов JSON)" % len(plan_json))
    return png, plan_json, "\n".join(debug)

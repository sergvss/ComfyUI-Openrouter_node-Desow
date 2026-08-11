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

from .furnish import STYLE_HINT_LIMIT, FurnishError, place_furniture
from .gate import (
    enforce_min_opening_widths,
    ensure_door_and_window,
    keep_side_front_pier,
    resolve_opening_conflicts,
    snap_front_door_to_camera,
)
from .masks import measure_openings_from_masks
from .merge import (
    median_extractions,
    merge_with_scanner,
    reorient_corridor_wall,
    scanner_openings_from_scan,
)
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


class FurnishFailed(RuntimeError):
    """Расстановку построить не удалось (терминально).

    Несёт накопленный до сбоя `debug`: обёртка ноды при `fail_soft=True` отдаёт
    его вместе с белым листом, иначе исключение летит наружу и роняет прогон.
    """

    def __init__(self, reason, debug_lines=()):
        self.reason = reason
        self.debug_lines = list(debug_lines)
        super().__init__(reason)


def _fmt_ops(openings):
    """`[door/front, window/back]` — компактный состав проёмов для debug."""
    return "[%s]" % ", ".join("%s/%s" % (o.get("type"), o.get("wall")) for o in openings) if openings else "[]"


def _probe_position(text) -> float | None:
    """Позиция 0..1 из ответа камера-пробы; None на мусоре/вне диапазона."""
    if not text:
        return None
    try:
        value = float(extract_json_object(text)["position"])
    except (ValueError, TypeError, KeyError):
        return None
    return value if 0.0 <= value <= 1.0 else None


# Пороги консенсус-каскада камера-проб (калибровка: полигон бенча 16).
PROBE_VS_EXTRACTOR_TOL = 0.3   # проба «спорит» с экстрактором -> зовём арбитра
PROBE_AGREEMENT_TOL = 0.25     # пробы согласны -> верим первичной
CAMERA_CENTER_SNAP = 0.10      # почти-центр (0.40..0.60) прижимается к 0.5
# Порог расширен 0.07 -> 0.10 по кадру kitchen (вердикт пользователя «по центру
# и прямо» + гомографический решатель 0.504 против кривого ручного замера 0.754).


def resolve_camera_position(extractor_pos, primary, secondary) -> tuple[float | None, str]:
    """Консенсус позиций камеры: (значение | None, объяснение для debug).

    Первичная проба (1Мп) точнее всех поодиночке (эталон: |Δ| 0.085 против
    0.117 у экстрактора и 0.139 у 2Мп-пробы), но изредка перетягивает в край
    (frame12: 0.75 при истинных ~0.35). Каскад: расходится с экстрактором
    больше PROBE_VS_EXTRACTOR_TOL — спрашиваем арбитра (2Мп-проба); пробы
    согласны между собой — правы пробы (экстрактор тянет к центру, fin463);
    пробы разошлись — берём ту, что ближе к экстрактору. На эталонных кадрах
    каскад совпадает с первичной пробой, менять его пороги — только с новым
    замером по camera_gt.
    """
    if primary is None:
        return (secondary, "первичной пробы нет, арбитр") if secondary is not None \
            else (None, "проб нет, позиция экстрактора")
    if extractor_pos is None or abs(primary - extractor_pos) <= PROBE_VS_EXTRACTOR_TOL:
        return primary, "первичная проба"
    if secondary is None:
        return primary, "спор с экстрактором, арбитра нет — первичная проба"
    if abs(primary - secondary) <= PROBE_AGREEMENT_TOL:
        return primary, "пробы согласны против экстрактора"
    if abs(secondary - extractor_pos) < abs(primary - extractor_pos):
        return secondary, "пробы разошлись, арбитр ближе к экстрактору"
    return primary, "пробы разошлись, первичная ближе к экстрактору"


def build_empty_plan(extraction_json, scanner_openings_json="", room_type="",
                     camera_probe_json="", camera_probe2_json="",
                     extraction2_json="", extraction3_json="",
                     segmentation_json=""):
    """`(png_bytes, plan_json, debug)` по ответу экстрактора и проёмам сканера.

    `plan_json` — компактный канонический план ПОСЛЕ мержа и гейта; при сбое
    пустая строка. `debug` — построчный отчёт: что разобрано, что дал сканер, что
    решил мерж, что вставил гейт, с каким масштабом отрисовано.

    `camera_probe_json` / `camera_probe2_json` — ответы двух камера-проб графа
    (1Мп и 2Мп-арбитр, `{"reason": ..., "position": 0..1}`): консенсус точнее
    числа из большой экстракции и стабилен между прогонами
    (`resolve_camera_position`). Пустые строки / мусор — остаётся экстракторская.

    `extraction2_json` / `extraction3_json` — дублирующие прогоны экстрактора
    (другие seed): состав берётся от первого, а размеры и позиции проёмов —
    медианой совпавших прочтений (`median_extractions`), это гасит прогоны-
    выбросы. Пустые строки / мусор — молча пропускаются.

    `segmentation_json` — ответ сегментации (gemini-2.5-flash, маски пола/стен/
    проёмов): масочная опора ПЕРЕМЕРЯЕТ offset/width проёмов геометрией
    (`measure_openings_from_masks`) — измерение вместо мнения модели. Состав не
    меняется; сбой любого шага откатывает проём на экстракторскую геометрию.
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

    extra_plans = []
    for label, extra_json in (("2", extraction2_json), ("3", extraction3_json)):
        if not extra_json:
            continue
        try:
            extra_plan, _notes = validate_plan(extract_json_object(extra_json))
            extra_plans.append(extra_plan)
        except (ValueError, TypeError, PlanDataError) as exc:
            debug.append("extract%s: нечитаемый дубль, пропущен (%s)" % (label, exc))
    if extra_plans:
        med_notes = median_extractions([plan, *extra_plans])
        debug.append("median: прогонов %d%s" % (
            1 + len(extra_plans),
            (", " + ", ".join(med_notes)) if med_notes else ", геометрия совпала"))

    # 1б) Масочная опора: перемер offset/width проёмов геометрией по маскам
    # сегментации (измерение вместо мнения). До мержа и поворота: работаем в
    # исходной системе фото, состав не трогаем.
    if segmentation_json:
        try:
            for note in measure_openings_from_masks(segmentation_json, plan):
                debug.append("masks: %s" % note)
        except Exception as exc:  # noqa: BLE001 - масочный слой не роняет план
            debug.append("masks: ОШИБКА %s: %s (пропуск)" % (exc.__class__.__name__, exc))

    if camera_probe_json or camera_probe2_json:
        old = plan.get("camera", {}).get("position")
        resolved, why = resolve_camera_position(
            old, _probe_position(camera_probe_json), _probe_position(camera_probe2_json))
        if resolved is not None and abs(resolved - (old or -1)) > 1e-9:
            plan.setdefault("camera", {})["position"] = round(resolved, 3)
            debug.append("camera_probe: позиция %s -> %.2f (%s)" % (old, resolved, why))
        else:
            debug.append("camera_probe: %s (позиция %s)" % (why, old))

    # Почти-центральная камера прижимается к центру: фронтальные кадры снимают
    # из середины комнаты, а дрожание оценок вокруг 0.5 (0.45/0.48/0.53) — шум
    # (вердикт пользователя по bedroom; на эталоне правило УЛУЧШАЕТ среднюю
    # ошибку 0.085 -> 0.074). Порог не трогает уверенно смещённые кадры (0.6+).
    try:
        cam_pos = float(plan.get("camera", {}).get("position", 0.5))
    except (TypeError, ValueError):
        cam_pos = 0.5
    if abs(cam_pos - 0.5) <= CAMERA_CENTER_SNAP and abs(cam_pos - 0.5) > 1e-9:
        plan.setdefault("camera", {})["position"] = 0.5
        debug.append("camera_centered: %.2f -> 0.5 (фронтальный кадр)" % cam_pos)

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
    for key, was, now in merge_meta["narrowed"]:
        debug.append("merge_narrow: %s сужен %.2f->%.2f dw (на своей стене места меньше)" % (key, was, now))
    for key in merge_meta["dropped_no_space"]:
        debug.append("merge_drop: %s выброшен (на своей стене места нет)" % (key,))
    for key in merge_meta["dropped_unconfirmed"]:
        debug.append("merge_dropped_unconfirmed: %s выброшен (VLM назвал стену глухой)" % (key,))

    # 2б) Коридор вдоль глухой (зеркальной) стены: стена с проёмами на самом
    # деле боковая — размеры меняются местами, пассаж встаёт в её середину.
    # Строго ДО развода конфликтов — угловой пассаж иначе гибнет там как
    # «не влезающий».
    corridor_notes = reorient_corridor_wall(plan)
    if corridor_notes:
        debug.append("corridor: %s" % ", ".join(corridor_notes))

    # 2в) Дверь экстрактора на невидимой front-стене — под камеру (offset там
    # всё равно догадка; снимают, как правило, от входа).
    snap_notes = snap_front_door_to_camera(plan)
    if snap_notes:
        debug.append("front_door: %s" % ", ".join(snap_notes))

    # 2г) Боковой проём не впритык к невидимому front-концу стены: простенок со
    # стороны камеры не меньше MIN_FRONT_PIER_DW (гадание модели, кадр lroom).
    pier_notes = keep_side_front_pier(plan)
    if pier_notes:
        debug.append("side_pier: %s" % ", ".join(pier_notes))

    # 2д) Санитария ширин: проём уже физического минимума (дверь 0.7 м, окно
    # 0.3 м) расширяется до него — модель занизила, щели не бывает (fin424).
    width_notes = enforce_min_opening_widths(plan)
    if width_notes:
        debug.append("min_width: %s" % ", ".join(width_notes))

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


# Сколько нарушений одной попытки печатать в debug. Полный список уезжает модели
# ре-промптом; в отчёт нужен характер проблемы, а не простыня на сто строк.
_MAX_VIOLATIONS_IN_DEBUG = 12


def build_furnished_plan(
    plan_json,
    room_type,
    complete,
    *,
    style_hint="",
    max_attempts=3,
    draw_camera=True,
    seed=0,
    model_label="",
):
    """`(png_bytes, furniture_json, debug)` — план с расставленной мебелью.

    Порядок: разбор канонического плана -> расстановка LLM с валидатором
    эргономики и ре-промптами -> рендер плана с мебелью (и с маркером камеры,
    если `draw_camera`). `complete(messages) -> str` — один вызов модели.

    `max_attempts` — это число ВЫЗОВОВ модели (1 = ни одного ре-промпта), тогда
    как у `place_furniture` (порт бэкенда) счёт идёт по ре-промптам. Пересчёт
    здесь, чтобы вендоренный модуль остался построчной копией источника.

    `furniture_json` — массив предметов, а не план целиком: план у потребителя
    уже есть отдельным входом, дублировать его в втором выходе незачем.

    Raises:
        FurnishFailed: терминальный сбой (битый план, модель недоступна, ни одной
        разобранной расстановки, рендер упал). Мягкую отдачу белого листа делает
        обёртка ноды — здесь решение не принимается.
    """
    debug = []

    # 1) Канонический план: вход ноды — выход DesowPlanRender, но приходить он
    # может и из хранилища/руками, поэтому проверяется той же схемой.
    try:
        raw = extract_json_object(plan_json)
    except (ValueError, TypeError) as exc:
        debug.append("plan_raw: %r" % ((plan_json or "")[:200],))
        raise FurnishFailed("plan_invalid_json: %s" % exc, debug) from exc
    try:
        plan, plan_notes = validate_plan(raw)
    except PlanDataError as exc:
        raise FurnishFailed("plan_invalid_schema: %s" % exc, debug) from exc

    room = plan["room"]
    debug.append(
        "plan: комната %s %.2fx%.2f dw (%.2fx%.2f м), проёмов %d %s, простенков %d"
        % (room["shape"], room["width_dw"], room["depth_dw"],
           room["width_dw"] * DW_M, room["depth_dw"] * DW_M,
           len(plan["openings"]), _fmt_ops(plan["openings"]), len(plan.get("partitions", [])))
    )
    for note in plan_notes:
        debug.append("plan_fix: %s" % note)

    hint = (style_hint or "").strip()
    if len(hint) > STYLE_HINT_LIMIT:
        debug.append("style_hint: обрезан %d -> %d символов" % (len(hint), STYLE_HINT_LIMIT))
    attempts = max(1, int(max_attempts))
    debug.append(
        "furnish_in: model=%s, room_type=%r, попыток<=%d, seed=%d, style_hint=%s, draw_camera=%s"
        % (model_label or "?", room_type or "", attempts, int(seed),
           ("%d символов" % len(hint)) if hint else "нет", draw_camera)
    )

    # 2) Расстановка: LLM -> валидатор эргономики -> ре-промпт нарушений.
    try:
        furniture, meta = place_furniture(
            complete, plan, room_type or "",
            max_retries=attempts - 1, style_hint=hint, seed=seed,
        )
    except FurnishError as exc:
        raise FurnishFailed("%s: %s" % (exc.code, exc), debug) from exc
    except Exception as exc:
        # Сюда попадает всё сетевое: `complete` — это уже исчерпавший ретраи
        # вызов OpenRouter. Отдельной ветки на класс ошибки нет намеренно:
        # для графа любая из них означает одно — расстановки не будет.
        raise FurnishFailed("llm_unavailable: %s: %s" % (exc.__class__.__name__, str(exc)[:200]), debug) from exc

    for i, errs in enumerate(meta["violations_by_attempt"], 1):
        debug.append("attempt %d: нарушений %d" % (i, len(errs)))
        for err in errs[:_MAX_VIOLATIONS_IN_DEBUG]:
            debug.append("  violation: %s" % err)
        if len(errs) > _MAX_VIOLATIONS_IN_DEBUG:
            debug.append("  ... ещё %d" % (len(errs) - _MAX_VIOLATIONS_IN_DEBUG))
    debug.append(
        "furnish: предметов %d, вызовов модели %d, ре-промптов %d, нарушений осталось %d"
        % (len(furniture), meta["calls"], meta["retries"], len(meta["violations"]))
    )
    debug.append("furniture: [%s]" % ", ".join(f["kind"] for f in furniture))

    # 3) Рендер плана с мебелью. Геометрия предметов уже проверена схемой, но
    # уронить прогон рендер всё равно не имеет права.
    try:
        png, render_meta = render_plan(
            dict(plan, furniture=furniture), with_furniture=True, draw_camera=draw_camera
        )
    except (ValueError, KeyError, ZeroDivisionError, OverflowError) as exc:
        raise FurnishFailed("render_failed: %s: %s" % (exc.__class__.__name__, exc), debug) from exc
    debug.append("render: %s" % json.dumps(render_meta, ensure_ascii=False))
    debug.append(
        "camera: %s" % (json.dumps(plan["camera"], ensure_ascii=False) if draw_camera else "не рисуется")
    )

    furniture_json = json.dumps(furniture, ensure_ascii=False, separators=(",", ":"))
    debug.insert(0, "furnish: ok (%d предметов, %d символов JSON)" % (len(furniture), len(furniture_json)))
    return png, furniture_json, "\n".join(debug)

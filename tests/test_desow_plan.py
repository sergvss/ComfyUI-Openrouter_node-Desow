"""Тесты пакета `desow_plan` (нода DesowPlanRender).

Запускаются без ComfyUI: логика проверяется на dict и PNG-байтах, конвертация в
тензор — отдельным тестом, который скипается без torch/numpy.
"""
import io
import json

import pytest
from PIL import Image

from desow_plan import blank_png, build_empty_plan
from desow_plan.gate import ensure_door_and_window, resolve_opening_conflicts
from desow_plan.merge import merge_with_scanner, scanner_openings_from_scan
from desow_plan.render import render_plan
from desow_plan.scanner import parse_scanner_openings
from desow_plan.schema_lite import MIN_CORNER_CLEARANCE_DW, PlanDataError, validate_plan


ROOM = {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0}


WALL_LEN_KEY = {"back": "width_dw", "front": "width_dw", "left": "depth_dw", "right": "depth_dw"}


def plan_with(**kwargs):
    """Минимальный сырой план экстрактора с переопределениями."""
    base = {"room": dict(ROOM), "openings": []}
    base.update(kwargs)
    return base


def assert_no_overlaps(plan, *, clearance=0.0):
    """Проёмы внутри своих стен и не налезают друг на друга.

    `clearance` — требуемый простенок от углов и между соседями (0 = проверяем
    только сам факт наложения). Допуск 1.5e-3 dw — план хранится с округлением
    координат до трёх знаков, это чуть больше половины его единицы.
    """
    tol = 1.5e-3
    room = plan["room"]
    by_wall = {}
    for op in plan["openings"]:
        by_wall.setdefault(op["wall"], []).append(
            (op["type"], op["offset_dw"] - op["width_dw"] / 2, op["offset_dw"] + op["width_dw"] / 2)
        )
    for wall, items in by_wall.items():
        length = float(room[WALL_LEN_KEY[wall]])
        items.sort(key=lambda i: i[1])
        for kind, a, b in items:
            assert a >= clearance - tol, "%s/%s начинается в %.3f (стена 0..%.2f)" % (kind, wall, a, length)
            assert b <= length - clearance + tol, "%s/%s кончается в %.3f (стена 0..%.2f)" % (kind, wall, b, length)
        for (k1, _, b1), (k2, a2, _) in zip(items, items[1:]):
            assert a2 >= b1 + clearance - tol, "%s и %s на %s внахлёст" % (k1, k2, wall)


# ── schema_lite: комната ─────────────────────────────────────────────

def test_minimal_plan_is_canonical():
    plan, notes = validate_plan(plan_with())
    assert plan == {"room": ROOM, "openings": []}
    assert notes == []


@pytest.mark.parametrize("room", [
    None,
    "rectangle",
    {},                                              # нет габаритов
    {"width_dw": 4.0},                               # нет глубины
    {"width_dw": 0, "depth_dw": 5.0},                # gt=0 в источнике
    {"width_dw": 4.0, "depth_dw": 31.0},             # > MAX_LENGTH_DW
    {"width_dw": float("nan"), "depth_dw": 5.0},
    {"width_dw": float("inf"), "depth_dw": 5.0},
    {"width_dw": "широкая", "depth_dw": 5.0},
    {"width_dw": True, "depth_dw": 5.0},             # bool числом не считается
])
def test_broken_room_is_fatal(room):
    with pytest.raises(PlanDataError):
        validate_plan({"room": room})


def test_numeric_strings_are_accepted():
    # Паритет с lax-режимом pydantic на бэкенде: VLM иногда шлёт числа строками.
    plan, _ = validate_plan({"room": {"width_dw": "4.0", "depth_dw": " 5 "}})
    assert plan["room"]["width_dw"] == 4.0 and plan["room"]["depth_dw"] == 5.0


def test_unknown_shape_falls_back_to_rectangle():
    plan, notes = validate_plan(plan_with(room={**ROOM, "shape": "square"}))
    assert plan["room"]["shape"] == "rectangle"
    assert any("shape" in n for n in notes)


def test_valid_l_shape_polygon_is_kept():
    poly = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 5], [0, 5]]
    plan, notes = validate_plan(plan_with(
        room={"shape": "l_shape", "width_dw": 4.0, "depth_dw": 5.0, "polygon_dw": poly}
    ))
    assert plan["room"]["shape"] == "l_shape"
    assert plan["room"]["polygon_dw"] == poly
    assert notes == []


@pytest.mark.parametrize("polygon", [
    None,
    [[0, 0], [4, 0], [4, 5], [0, 5]],                             # меньше 6 вершин
    [[0, 0], [4, 0], [4, 2], [2, 2], [2, 5], [0, float("nan")]],  # не конечное
    [[0, 0], [4, 0], [4, 2], [2, 2], [2, 5], [0, 5900]],          # вне диапазона
    [[0, 0], [9, 0], [9, 2], [5, 2], [5, 5], [0, 5]],             # габарит не сходится
])
def test_broken_l_shape_degrades_to_rectangle(polygon):
    # У бэкенда это fail-loud; нода рисует прямоугольник по width/depth и пишет причину.
    plan, notes = validate_plan(plan_with(
        room={"shape": "l_shape", "width_dw": 4.0, "depth_dw": 5.0, "polygon_dw": polygon}
    ))
    assert plan["room"]["shape"] == "rectangle"
    assert "polygon_dw" not in plan["room"]
    assert any("polygon" in n for n in notes)


# ── schema_lite: проёмы и простенки ──────────────────────────────────

def test_good_opening_survives_bad_neighbour():
    plan, notes = validate_plan(plan_with(openings=[
        {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5, "confidence": 0.9},
        {"type": "mirror", "wall": "back", "offset_dw": 1.0, "width_dw": 1.0},
    ]))
    assert len(plan["openings"]) == 1
    assert plan["openings"][0]["type"] == "window"
    assert any("выброшен" in n for n in notes)


@pytest.mark.parametrize("opening", [
    {"type": "door", "wall": "north", "offset_dw": 1.0, "width_dw": 1.0},     # стена не из словаря
    {"type": "door", "wall": "back", "width_dw": 1.0},                        # нет offset
    {"type": "door", "wall": "back", "offset_dw": 1.0},                       # нет ширины
    {"type": "door", "wall": "back", "offset_dw": -1.0, "width_dw": 1.0},     # ge=0
    {"type": "door", "wall": "back", "offset_dw": 1.0, "width_dw": 0},        # gt=0
    {"type": "door", "wall": "back", "offset_dw": 1.0, "width_dw": 99},       # > MAX
    {"type": "door", "wall": "back", "offset_dw": float("nan"), "width_dw": 1.0},
    "не объект",
])
def test_broken_openings_are_dropped(opening):
    plan, notes = validate_plan(plan_with(openings=[opening]))
    assert plan["openings"] == []
    assert notes


def test_wall_index_is_allowed_for_l_shape():
    # WallRef в источнике допускает индекс ребра полигона вместо имени стены.
    plan, _ = validate_plan(plan_with(openings=[
        {"type": "door", "wall": 2, "offset_dw": 1.0, "width_dw": 1.0},
    ]))
    assert plan["openings"][0]["wall"] == 2


def test_broken_swing_is_sanitized_not_dropped():
    plan, notes = validate_plan(plan_with(openings=[
        {"type": "door", "wall": "back", "offset_dw": 1.0, "width_dw": 1.0,
         "swing": {"hinge": "north", "direction": "sideways"}},
    ]))
    assert plan["openings"][0]["swing"] == {"hinge": "left", "direction": "in"}
    assert len(notes) == 2


def test_partitions_valid_and_broken():
    plan, notes = validate_plan(plan_with(partitions=[
        {"attach": "left", "offset_dw": 1.5, "length_dw": 1.0},
        {"attach": "free", "start_dw": [1.0, 2.0], "direction": "right", "length_dw": 2.0},
        {"attach": "free", "start_dw": [1.0, 2.0], "length_dw": 2.0},          # нет direction
        {"attach": "left", "length_dw": 1.0},                                  # нет offset
        {"attach": "ceiling", "offset_dw": 1.0, "length_dw": 1.0},             # attach не из словаря
    ]))
    assert len(plan["partitions"]) == 2
    assert len(notes) == 3


def test_furniture_is_ignored_with_note():
    plan, notes = validate_plan(plan_with(furniture=[{"kind": "bed", "center_dw": [1, 1], "size_m": [2, 1.6]}]))
    assert "furniture" not in plan
    assert any("furniture" in n for n in notes)


# ── scanner ──────────────────────────────────────────────────────────

SCANNER_TEXT = json.dumps({
    "openings": [
        {"type": "window", "wall": "back", "position_on_wall": "center", "confidence": 0.95},
        {"type": "arched_window", "wall": "left_wall", "position_on_wall": "left", "confidence": 0.8},
        {"type": "sliding_door", "wall": "right", "position_on_wall": "single", "confidence": 0.9},
        {"type": "floor_to_ceiling_window", "wall": "back", "position_on_wall": "right", "confidence": 0.7},
        {"type": "window", "wall": "unknown", "position_on_wall": "single", "confidence": 0.9},
        {"type": "mirror", "wall": "back", "position_on_wall": "single", "confidence": 0.9},
    ],
    "room": {"shape": "rectangle", "type": "bedroom"},
})


def test_scanner_normalizes_types_and_walls():
    entries, room, notes = parse_scanner_openings(SCANNER_TEXT)
    assert [(e["kind"], e["wall"]) for e in entries] == [
        ("window", "back"),
        ("window", "left"),          # arched_window -> window, left_wall -> left
        ("door", "right"),           # sliding_door -> door
        ("floor_to_ceiling_window", "back"),
    ]
    assert room["type"] == "bedroom"
    assert len(notes) == 2           # unknown-стена и mirror


@pytest.mark.parametrize("entry, kept", [
    ({"type": "window", "wall": "front", "confidence": 0.35}, False),   # галлюцинация за камерой
    ({"type": "window", "wall": "front", "confidence": 0.95}, True),
    ({"type": "window", "wall": "front"}, True),                        # поля нет -> фильтр не работает
    ({"type": "window", "wall": "front", "confidence": "мусор"}, True),
])
def test_front_wall_confidence_filter(entry, kept):
    entries, _, _ = parse_scanner_openings(json.dumps({"openings": [entry]}))
    assert bool(entries) is kept


def test_scanner_accepts_fence_and_preamble():
    text = 'Here is the JSON:\n```json\n{"openings": [{"type": "door", "wall": "back", "confidence": 0.9}]}\n```'
    entries, _, _ = parse_scanner_openings(text)
    assert [e["kind"] for e in entries] == ["door"]


@pytest.mark.parametrize("payload", [
    {"openings_data": {"openings": [{"type": "door", "wall": "back"}]}},   # запись скана бэкенда
    {"openings_data": [{"type": "door", "wall": "back"}]},                 # legacy-список
])
def test_scanner_accepts_scan_record_wrappers(payload):
    entries, _, _ = parse_scanner_openings(json.dumps(payload))
    assert [e["kind"] for e in entries] == ["door"]


@pytest.mark.parametrize("text", ["", "   ", "не JSON", "{}", '{"openings": "нет"}'])
def test_scanner_failures_are_soft(text):
    entries, room, notes = parse_scanner_openings(text)
    assert entries == [] and room == {} and notes


def test_scanner_entries_expand_by_quantity():
    entries = [{"kind": "window", "wall": "back", "quantity": 2, "display_type": "window"}]
    assert scanner_openings_from_scan(entries) == [
        {"type": "window", "wall": "back"}, {"type": "window", "wall": "back"}
    ]


# ── merge ────────────────────────────────────────────────────────────

def test_merge_keeps_vlm_geometry_when_composition_covers():
    vlm = plan_with(openings=[
        {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5},
        {"type": "door", "wall": "left", "offset_dw": 1.0, "width_dw": 1.0},
    ])
    merged, meta = merge_with_scanner({"openings": [{"type": "window", "wall": "back"}]}, [vlm])
    assert merged["openings"][0]["offset_dw"] == 2.0        # позиция от VLM
    assert meta["fallback_scanner_composition"] is False
    assert meta["added_from_vlm"] == [("door", "left")]     # лишнее у VLM остаётся


def test_merge_falls_back_to_scanner_composition_with_defaults():
    vlm = plan_with(openings=[])
    merged, meta = merge_with_scanner({"openings": [{"type": "door", "wall": "front"}]}, [vlm])
    assert meta["fallback_scanner_composition"] is True
    assert meta["defaults_used"] == [("door", "front")]
    door = merged["openings"][0]
    assert door["offset_dw"] == 2.0 and door["width_dw"] == 1.0   # центр стены 4.0 dw
    assert door["swing"] == {"hinge": "left", "direction": "in"}


def test_merge_defaults_do_not_stack_on_each_other():
    # Боевой кадр e4 серии v83: три проёма сканера на одной стене, у VLM позиции
    # только для одного — два дефолта садились в центр стены поверх всего.
    vlm = plan_with(openings=[{"type": "window", "wall": "right", "offset_dw": 2.8, "width_dw": 1.2}])
    scanner = {"openings": [
        {"type": "door", "wall": "right"}, {"type": "door", "wall": "right"},
        {"type": "window", "wall": "right"},
    ]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    assert meta["fallback_scanner_composition"] is True
    assert_no_overlaps(merged)


def test_merge_moves_default_to_another_wall_when_no_room():
    # Боевой кадр f1: панорама занимает почти всю стену, дефолтной двери места нет.
    vlm = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "left", "offset_dw": 2.5, "width_dw": 4.6},
    ])
    scanner = {"openings": [
        {"type": "floor_to_ceiling_window", "wall": "left"}, {"type": "door", "wall": "left"},
    ]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    door = next(o for o in merged["openings"] if o["type"] == "door")
    assert door["wall"] == "right"                       # та же ориентация
    assert meta["moved_to_wall"] == [(("door", "left"), "right")]
    assert_no_overlaps(merged)


def test_merge_pairs_same_type_on_disputed_wall():
    # Боевой кадр e5: дверь одна, но сканер видит её на left, а VLM на back —
    # раньше «добавка VLM» превращала её во вторую дверь.
    vlm = plan_with(openings=[{"type": "door", "wall": "back", "offset_dw": 1.3, "width_dw": 1.0}])
    merged, meta = merge_with_scanner({"openings": [{"type": "door", "wall": "left"}]}, [vlm])
    assert [o["wall"] for o in merged["openings"]] == ["left"]
    assert meta["added_from_vlm"] == []
    assert meta["paired_wall_dispute"] == [("door", "back", "left")]


def test_merge_does_not_pair_passage_or_across_types():
    # passage сканер не эмитит вообще, а окно и дверь — разные проёмы: обе записи
    # VLM обязаны остаться добавками, иначе спаривание съест реальные проёмы.
    vlm = plan_with(openings=[
        {"type": "passage", "wall": "back", "offset_dw": 2.0, "width_dw": 1.0},
        {"type": "window", "wall": "back", "offset_dw": 3.4, "width_dw": 1.0},
    ])
    merged, meta = merge_with_scanner({"openings": [{"type": "door", "wall": "left"}]}, [vlm])
    assert meta["paired_wall_dispute"] == []
    assert sorted(meta["added_from_vlm"]) == [("passage", "back"), ("window", "back")]
    assert len(merged["openings"]) == 3


def test_merge_keeps_balcony_door_inside_glazing_on_same_wall():
    # Регресс scan_424: остекление в пол и балконная дверь на ОДНОЙ стене —
    # состав покрыт обычным covers, спаривание не должно вмешиваться.
    vlm = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 1.45, "width_dw": 1.3},
        {"type": "door", "wall": "back", "offset_dw": 2.55, "width_dw": 0.9},
    ])
    scanner = {"openings": [
        {"type": "floor_to_ceiling_window", "wall": "back"}, {"type": "door", "wall": "back"},
    ]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    assert meta["fallback_scanner_composition"] is False
    assert meta["paired_wall_dispute"] == []
    assert len(merged["openings"]) == 2


# ── gate ─────────────────────────────────────────────────────────────

def test_gate_inserts_door_at_front_corner():
    plan = plan_with(openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5}])
    notes = ensure_door_and_window(plan)
    assert notes == ["door_inserted"]
    door = plan["openings"][-1]
    assert door["type"] == "door" and door["wall"] == "front"
    assert door["swing"] == {"hinge": "left", "direction": "in"}
    # 0.2 м от угла до косяка при ширине 1 dw: центр = 0.2/0.85 + 0.5
    assert door["offset_dw"] == pytest.approx(0.2 / 0.85 + 0.5, abs=1e-3)


def test_gate_inserts_window_at_front_center():
    plan = plan_with(openings=[{"type": "door", "wall": "left", "offset_dw": 1.0, "width_dw": 1.0}])
    notes = ensure_door_and_window(plan)
    assert notes == ["window_inserted"]
    window = plan["openings"][-1]
    assert window["type"] == "window" and window["wall"] == "front"
    assert window["offset_dw"] == pytest.approx(2.0)     # центр стены 4.0 dw
    assert window["width_dw"] == 1.6


def test_gate_inserts_both_without_overlap():
    plan = plan_with(openings=[])
    assert ensure_door_and_window(plan) == ["door_inserted", "window_inserted"]
    door, window = plan["openings"]
    door_right = door["offset_dw"] + door["width_dw"] / 2
    window_left = window["offset_dw"] - window["width_dw"] / 2
    assert window_left >= door_right - 1e-6


def test_gate_is_noop_when_door_and_window_present():
    plan = plan_with(openings=[
        {"type": "door", "wall": "left", "offset_dw": 1.0, "width_dw": 1.0},
        {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 2.0, "width_dw": 2.0},
    ])
    assert ensure_door_and_window(plan) == []
    assert len(plan["openings"]) == 2


def test_gate_insert_keeps_clearance_from_existing_opening():
    # Окно уже стоит на front-стене; вставленная дверь обязана оставить простенок.
    plan = plan_with(openings=[{"type": "window", "wall": "front", "offset_dw": 1.2, "width_dw": 1.6}])
    assert ensure_door_and_window(plan) == ["door_inserted"]
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


def test_gate_skips_when_front_wall_is_full():
    plan = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "front", "offset_dw": 2.0, "width_dw": 4.0},
    ])
    assert ensure_door_and_window(plan) == ["door_gate_skipped"]


# ── развод конфликтов ────────────────────────────────────────────────

def test_resolve_moves_opening_away_from_corner():
    # Боевой кадр e4: passage вплотную к углу «открывал» его на чертеже.
    plan = plan_with(openings=[{"type": "passage", "wall": "back", "offset_dw": 3.5, "width_dw": 1.0}])
    notes = resolve_opening_conflicts(plan)
    assert notes and notes[0].startswith("moved:passage/back")
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


def test_resolve_separates_overlapping_openings():
    plan = plan_with(openings=[
        {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5},
        {"type": "door", "wall": "back", "offset_dw": 2.2, "width_dw": 1.0},
    ])
    resolve_opening_conflicts(plan)
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


def test_resolve_narrows_opening_wider_than_wall():
    plan = plan_with(openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 5.0}])
    notes = resolve_opening_conflicts(plan)
    assert any(n.startswith("narrowed:window/back") for n in notes)
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


def test_resolve_drops_opening_when_wall_is_full():
    plan = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 2.0, "width_dw": 3.5},
        {"type": "door", "wall": "back", "offset_dw": 3.8, "width_dw": 1.0},
    ])
    notes = resolve_opening_conflicts(plan)
    assert "dropped:door/back" in notes
    assert [o["type"] for o in plan["openings"]] == ["floor_to_ceiling_window"]


def test_resolve_leaves_clean_plan_untouched():
    plan = plan_with(openings=[
        {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5},
        {"type": "door", "wall": "left", "offset_dw": 1.0, "width_dw": 1.0},
    ])
    before = json.dumps(plan, sort_keys=True)
    assert resolve_opening_conflicts(plan) == []
    assert json.dumps(plan, sort_keys=True) == before


def test_resolve_ignores_openings_anchored_to_polygon_edge():
    # wall задан индексом ребра (l_shape): длину такой стены здесь не резолвим,
    # трогать проём вслепую нельзя.
    plan = plan_with(openings=[{"type": "door", "wall": 2, "offset_dw": 99.0, "width_dw": 1.0}])
    assert resolve_opening_conflicts(plan) == []
    assert plan["openings"][0]["offset_dw"] == 99.0


# ── render ───────────────────────────────────────────────────────────

def png_size(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def test_render_produces_png_of_graph_standard_size():
    plan = plan_with(openings=[{"type": "door", "wall": "front", "offset_dw": 1.0, "width_dw": 1.0}])
    png, meta = render_plan(plan, with_furniture=False)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_size(png) == (1152, 928)
    assert meta["px_per_dw"] > 0 and meta["scale_fallback"] is False


def test_render_is_deterministic():
    plan = plan_with(openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5}])
    assert render_plan(plan, with_furniture=False)[0] == render_plan(plan, with_furniture=False)[0]


def test_blank_png_is_white_sheet():
    with Image.open(io.BytesIO(blank_png())) as image:
        assert image.size == (1152, 928)
        assert image.convert("L").getextrema() == (255, 255)


def test_rendered_plan_is_not_blank():
    plan = plan_with(openings=[{"type": "door", "wall": "front", "offset_dw": 1.0, "width_dw": 1.0}])
    png, _ = render_plan(plan, with_furniture=False)
    with Image.open(io.BytesIO(png)) as image:
        assert image.convert("L").getextrema()[0] < 40      # есть чёрные стены


# ── конвейер целиком ─────────────────────────────────────────────────

EXTRACTION = json.dumps({
    "room": {"shape": "rectangle", "width_dw": 3.8, "depth_dw": 6.8},
    "openings": [{"type": "window", "wall": "back", "offset_dw": 1.9, "width_dw": 1.8, "confidence": 0.95}],
})


def test_pipeline_happy_path():
    png, plan_json, debug = build_empty_plan(EXTRACTION, "", "bedroom")
    plan = json.loads(plan_json)
    types = {o["type"] for o in plan["openings"]}
    assert "window" in types and "door" in types          # дверь вставил гейт
    assert plan["room_type"] == "bedroom"
    assert png_size(png) == (1152, 928)
    assert debug.startswith("plan: ok")
    assert "gate: door_inserted" in debug


def test_pipeline_uses_scanner_composition():
    scanner = json.dumps({"openings": [
        {"type": "door", "wall": "left", "confidence": 0.9},
        {"type": "window", "wall": "back", "confidence": 0.9},
    ]})
    _, plan_json, debug = build_empty_plan(EXTRACTION, scanner, "")
    walls = {(o["type"], o["wall"]) for o in json.loads(plan_json)["openings"]}
    assert ("door", "left") in walls                      # дверь пришла от сканера, не от гейта
    assert "gate: дверь и окно уже есть" in debug
    assert "room_type" not in plan_json


@pytest.mark.parametrize("text", ["", "   ", "совсем не JSON", "[1, 2, 3]", '{"room": {"width_dw": "нет"}}'])
def test_pipeline_never_raises_on_bad_extraction(text):
    png, plan_json, debug = build_empty_plan(text, "", "")
    assert plan_json == ""
    assert "ОШИБКА" in debug
    with Image.open(io.BytesIO(png)) as image:
        assert image.convert("L").getextrema() == (255, 255)   # белый лист-заглушка


def test_pipeline_handles_openrouter_soft_failure_string():
    """Вход `OPENROUTER_ERROR: ...` от OpenRouterNode с fail_soft=True.

    Так экстрактор отдаёт свою ошибку, не роняя прогон: нода плана обязана увести
    её в белый лист и debug, а не сделать вид, что план построен.
    """
    soft = "OPENROUTER_ERROR: RuntimeError: OpenRouter API unreachable after 3 retries"
    png, plan_json, debug = build_empty_plan(soft, "", "")
    assert plan_json == ""
    assert "ОШИБКА" in debug and "OPENROUTER_ERROR" in debug   # причина видна в отчёте
    with Image.open(io.BytesIO(png)) as image:
        assert image.convert("L").getextrema() == (255, 255)


def test_pipeline_survives_broken_scanner_json():
    _, plan_json, debug = build_empty_plan(EXTRACTION, "}{ сломано", "")
    assert plan_json and "scanner_fix:" in debug


def test_pipeline_reports_dropped_openings():
    extraction = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0},
        "openings": [{"type": "window", "wall": "back", "offset_dw": 5900, "width_dw": 1.5}],
    })
    _, plan_json, debug = build_empty_plan(extraction, "", "")
    assert plan_json                                       # план построен без битого проёма
    assert "extract_fix:" in debug and "выброшен" in debug


# ── боевые кадры серии v83 (preview, 2026-08-06) ─────────────────────
# Ответы экстрактора в артефактах прогонов не сохранены (наружу выводятся только
# openings, plan_json и debug), поэтому входы восстановлены из plan_json + debug
# каждого прогона. Все четыре кадра до фикса давали дефектный план.

def run_case(extraction, scanner):
    _, plan_json, debug = build_empty_plan(json.dumps(extraction), json.dumps(scanner), "")
    assert plan_json, debug
    return json.loads(plan_json), debug


def test_v83_e4_three_scanner_openings_on_one_wall():
    """e4: две двери и окно на правой стене + passage впритык к углу."""
    plan, debug = run_case(
        {"room": {"shape": "rectangle", "width_dw": 4.8, "depth_dw": 4.5},
         "openings": [
             {"type": "door", "wall": "back", "offset_dw": 2.9, "width_dw": 1.0,
              "swing": {"hinge": "left", "direction": "out"}, "confidence": 0.9},
             {"type": "passage", "wall": "back", "offset_dw": 4.3, "width_dw": 1.0, "confidence": 0.85},
             {"type": "window", "wall": "right", "offset_dw": 2.8, "width_dw": 1.2, "confidence": 0.7},
         ]},
        {"openings": [
            {"type": "door", "wall": "right", "position_on_wall": "left", "confidence": 0.9},
            {"type": "door", "wall": "right", "position_on_wall": "right", "confidence": 0.85},
            {"type": "window", "wall": "right", "position_on_wall": "single", "confidence": 0.55},
        ]},
    )
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)
    assert "merge_move:" in debug           # окну не хватило места на правой стене
    assert "moved:passage/back" in debug    # угол закрыт


def test_v83_e5_single_door_read_on_two_walls():
    """e5: одна дверь, разные стены у сканера и VLM — раньше выходило две."""
    plan, debug = run_case(
        {"room": {"shape": "rectangle", "width_dw": 4.5, "depth_dw": 6.8},
         "openings": [{"type": "door", "wall": "back", "offset_dw": 1.3, "width_dw": 1.0,
                       "confidence": 0.85}]},
        {"openings": [{"type": "door", "wall": "left", "position_on_wall": "center",
                       "confidence": 0.9}]},
    )
    doors = [o for o in plan["openings"] if o["type"] == "door"]
    assert len(doors) == 1 and doors[0]["wall"] == "left"       # стена — от сканера
    assert len(plan["openings"]) == 2                           # + окно от гейта
    assert "merge_pair: door back→left" in debug
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


def test_v83_f1_default_door_inside_panoramic_window():
    """f1: дефолтная дверь садилась внутрь панорамного остекления."""
    plan, _ = run_case(
        {"room": {"shape": "rectangle", "width_dw": 6.8, "depth_dw": 5.0},
         "openings": [
             {"type": "floor_to_ceiling_window", "wall": "left", "offset_dw": 2.5,
              "width_dw": 3.2, "confidence": 0.92},
             {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 2.1,
              "width_dw": 3.0, "confidence": 0.88},
         ]},
        {"openings": [
            {"type": "floor_to_ceiling_window", "wall": "left", "confidence": 0.92},
            {"type": "door", "wall": "left", "confidence": 0.72},
        ]},
    )
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)
    assert len([o for o in plan["openings"] if o["type"] == "door"]) == 1


@pytest.mark.parametrize("tag, extraction, scanner", [
    ("e3_bedroom",
     {"room": {"shape": "rectangle", "width_dw": 4.2, "depth_dw": 4.8},
      "openings": [{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 2.4,
                    "confidence": 0.95}]},
     {"openings": [{"type": "window", "wall": "back", "confidence": 0.97}]}),
    ("f4_living",
     {"room": {"shape": "rectangle", "width_dw": 4.8, "depth_dw": 5.8},
      "openings": [{"type": "window", "wall": "right", "offset_dw": 3.5, "width_dw": 2.4,
                    "confidence": 0.92}]},
     {"openings": [{"type": "window", "wall": "right", "confidence": 0.92}]}),
])
def test_v83_clean_frames_do_not_regress(tag, extraction, scanner):
    """Кадры со штатным мержем: фикс не должен ничего в них двигать."""
    plan, debug = run_case(extraction, scanner)
    assert "conflicts: нет" in debug
    assert len(plan["openings"]) == 2                  # окно VLM + дверь гейта
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


# ── обёртка ComfyUI (нужен torch) ────────────────────────────────────

def test_node_returns_image_tensor():
    torch = pytest.importorskip("torch", reason="torch есть только внутри ComfyUI")
    pytest.importorskip("numpy")
    if not hasattr(torch, "from_numpy"):
        # test_node_prompt_echo подменяет torch заглушкой, когда настоящего нет;
        # заглушка тензоры не умеет — проверять на ней нечего.
        pytest.skip("в sys.modules заглушка torch, а не настоящий пакет")
    from desow_plan_node import DesowPlanRender

    image, plan_json, debug = DesowPlanRender().render(EXTRACTION, "", "bedroom")
    assert image.shape == (1, 928, 1152, 3)                # [batch, H, W, RGB]
    assert str(image.dtype) == "torch.float32"
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert json.loads(plan_json)["room"]["width_dw"] == 3.8
    assert debug.startswith("plan: ok")

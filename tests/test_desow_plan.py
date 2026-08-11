"""Тесты пакета `desow_plan` (нода DesowPlanRender).

Запускаются без ComfyUI: логика проверяется на dict и PNG-байтах, конвертация в
тензор — отдельным тестом, который скипается без torch/numpy.
"""
import io
import json

import pytest
from PIL import Image, ImageChops

from desow_plan import blank_png, build_empty_plan, render_camera_png
from desow_plan.gate import (
    ensure_door_and_window,
    resolve_opening_conflicts,
    snap_front_door_to_camera,
)
from desow_plan.merge import merge_with_scanner, reorient_corridor_wall, scanner_openings_from_scan
from desow_plan.render import CANVAS, render_plan
from desow_plan.scanner import parse_scanner_openings
from desow_plan.validate import validate_structure
from desow_plan.schema_lite import (
    DEFAULT_CAMERA,
    MIN_CORNER_CLEARANCE_DW,
    PlanDataError,
    is_double_leaf,
    validate_plan,
)


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
    assert plan == {"room": ROOM, "openings": [], "camera": DEFAULT_CAMERA}
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


def test_balcony_door_is_a_first_class_opening_type():
    """`balcony_door` разбирается как дверь: тип сохраняется, swing принимается."""
    plan, notes = validate_plan(plan_with(openings=[
        {"type": "balcony_door", "wall": "back", "offset_dw": 2.785, "width_dw": 0.9,
         "swing": {"hinge": "right", "direction": "in"}, "confidence": 0.95},
    ]))
    assert notes == []
    opening = plan["openings"][0]
    assert opening["type"] == "balcony_door"
    assert opening["swing"] == {"hinge": "right", "direction": "in"}


@pytest.mark.parametrize("kind, width_dw, double", [
    ("door", 1.0, False),           # 0.85 м — рядовое межкомнатное полотно
    ("door", 1.29, False),          # 1.10 м — предел однопольного полотна по ГОСТ
    ("door", 1.42, True),           # 1.21 м — за порогом, полотна такой ширины нет
    ("balcony_door", 0.9, False),   # 0.77 м — балконная створка обычного размера
    ("balcony_door", 1.6, True),    # 1.36 м — садовая дверь боевого кадра kitchen
    ("double_door", 0.9, True),     # тип форсирует две створки при любой ширине
])
def test_double_leaf_is_decided_by_width_not_by_type(kind, width_dw, double):
    """Число створок — свойство ширины проёма, а не его подтипа.

    Общий предикат для рендера и валидатора: одностворчатых полотен шире
    1.0-1.1 м не бывает, поэтому широкий проём любого дверного типа собирается
    из двух створок.
    """
    assert is_double_leaf({"type": kind, "width_dw": width_dw}) is double


def test_wall_index_is_allowed_for_l_shape():
    # WallRef в источнике допускает индекс ребра полигона вместо имени стены.
    plan, _ = validate_plan(plan_with(openings=[
        {"type": "door", "wall": 2, "offset_dw": 1.0, "width_dw": 1.0},
    ]))
    assert plan["openings"][0]["wall"] == 2


@pytest.mark.parametrize("wall, offset, hinge", [
    ("back", 1.0, "left"),      # комната 4.0 dw в ширину: ближний угол — левый
    ("back", 3.0, "right"),
    ("front", 1.0, "left"),
    ("left", 1.0, "back"),      # вертикальная стена 5.0 dw: ближний угол — back
    ("left", 4.0, "front"),
    ("right", 4.0, "front"),
])
def test_door_without_swing_gets_the_default_one(wall, offset, hinge):
    """Дверь без `swing` получает норму: внутрь, петли у ближнего угла.

    Раньше поле просто отсутствовало: `validate_structure` ругался «дверь без
    swing», а рендер молча брал свои дефолты — план расходился со своей проверкой.
    """
    plan, notes = validate_plan(plan_with(openings=[
        {"type": "door", "wall": wall, "offset_dw": offset, "width_dw": 1.0}]))
    assert plan["openings"][0]["swing"] == {"hinge": hinge, "direction": "in"}
    assert any("swing_defaulted" in n for n in notes)
    assert validate_structure(plan) == []


def test_balcony_door_without_swing_opens_into_the_room():
    plan, _ = validate_plan(plan_with(openings=[
        {"type": "balcony_door", "wall": "back", "offset_dw": 2.0, "width_dw": 0.9}]))
    assert plan["openings"][0]["swing"]["direction"] == "in"


def test_explicit_swing_is_not_overwritten():
    plan, notes = validate_plan(plan_with(openings=[
        {"type": "door", "wall": "back", "offset_dw": 1.0, "width_dw": 1.0,
         "swing": {"hinge": "right", "direction": "out"}}]))
    assert plan["openings"][0]["swing"] == {"hinge": "right", "direction": "out"}
    assert not any("swing_defaulted" in n for n in notes)


def test_solid_walls_are_parsed_and_garbage_dropped():
    plan, notes = validate_plan(plan_with(solid_walls=["right", "ceiling", "right"]))
    assert plan["solid_walls"] == ["right"]
    assert any("solid_walls" in n for n in notes)


def test_opening_on_a_wall_the_model_called_solid_is_dropped():
    """Самопротиворечие внутри одного ответа решается в пользу «глухой».

    Так отвечают на зеркальной стене: проём вычитан из отражения, а стена при
    этом честно названа глухой (кадры fin463, frame13).
    """
    plan, notes = validate_plan(plan_with(
        openings=[
            {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.2},
            {"type": "door", "wall": "right", "offset_dw": 2.0, "width_dw": 1.6},
        ],
        solid_walls=["right"],
    ))
    assert [(o["type"], o["wall"]) for o in plan["openings"]] == [("window", "back")]
    assert any("объявлена глухой" in n for n in notes)


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


def test_camera_defaults_to_the_front_wall_centre():
    """Блок камеры есть у КАЖДОГО плана: конвенция camera-relative — это данные."""
    plan, notes = validate_plan(plan_with())
    assert plan["camera"] == DEFAULT_CAMERA
    assert plan["camera"] == {"wall": "front", "position": 0.5, "direction": "up",
                              "marker": "orange_sector"}
    assert notes == []


def test_camera_values_from_input_are_kept():
    plan, notes = validate_plan(plan_with(camera={"wall": "left", "position": "0.25", "direction": "right"}))
    assert plan["camera"]["wall"] == "left"
    assert plan["camera"]["position"] == 0.25          # числовая строка принимается
    assert plan["camera"]["direction"] == "right"
    assert notes == []


@pytest.mark.parametrize("camera", [
    "front",                                   # не объект
    {"wall": "ceiling"},                       # стены такой нет
    {"position": 1.4},                         # доля стены вне 0..1
    {"position": "нет"},
    {"direction": "backwards"},
    {"marker": "red_arrow"},                   # символ рисует рендер, не модель
])
def test_broken_camera_degrades_to_default(camera):
    """Камера не имеет права уронить план: любое битое поле -> дефолт + пометка."""
    plan, notes = validate_plan(plan_with(camera=camera))
    assert plan["camera"] == DEFAULT_CAMERA
    assert any("camera" in n for n in notes), notes


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


def test_merge_never_moves_an_opening_to_another_wall():
    """Боевой кадр f1: панорама заняла стену, дефолтной двери места нет.

    Стену называют оба источника, поэтому переезд запрещён: дверь либо сужается
    в пределах своей стены, либо выбрасывается. Прежний перенос на соседнюю
    стену давал проём-призрак (кадр frame15 — остекление за спиной камеры).
    """
    vlm = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "left", "offset_dw": 2.5, "width_dw": 4.6},
    ])
    scanner = {"openings": [
        {"type": "floor_to_ceiling_window", "wall": "left"}, {"type": "door", "wall": "left"},
    ]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    assert [o["wall"] for o in merged["openings"]] == ["left"]   # ни один не уехал
    assert meta["dropped_no_space"] == [("door", "left")]
    assert "moved_to_wall" not in meta
    assert_no_overlaps(merged)


def test_merge_narrows_an_opening_that_almost_fits():
    """Место на своей стене есть, но меньше дефолтной ширины — проём сужается.

    Окно доживает до 0.3 м (0.35 dw), и это правильнее переезда: стена верная,
    ошибка только в обмере ширины.
    """
    vlm = plan_with(openings=[
        {"type": "window", "wall": "back", "offset_dw": 1.4, "width_dw": 1.8},
    ])
    scanner = {"openings": [{"type": "window", "wall": "back"}, {"type": "window", "wall": "back"}]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    assert [o["wall"] for o in merged["openings"]] == ["back", "back"]
    assert meta["narrowed"] and meta["narrowed"][0][0] == ("window", "back")
    narrowed = merged["openings"][1]
    assert 0.35 <= narrowed["width_dw"] < 1.6
    assert_no_overlaps(merged)


def test_merge_keeps_the_scanner_opening_when_vlm_is_silent():
    """Молчание VLM возражением не считается: сканер остаётся «полом».

    Возразить может только явная запись `solid_walls` — см. соседний тест.
    """
    vlm = plan_with(openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5}])
    scanner = {"openings": [{"type": "window", "wall": "back"}, {"type": "door", "wall": "right"}]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    assert ("door", "right") in [(o["type"], o["wall"]) for o in merged["openings"]]
    assert meta["dropped_unconfirmed"] == []


def test_merge_drops_the_scanner_door_on_a_wall_vlm_called_solid():
    """Кадр frame13: сканер принял зеркальный шкаф-купе за дверь на правой стене.

    VLM назвал ту стену глухой (`solid_walls`), и это активное противоречие —
    дефолтная дверь не восстанавливается, стена остаётся глухой.
    """
    vlm = plan_with(
        openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.1}],
        solid_walls=["right"],
    )
    plan, notes = validate_plan(vlm)
    assert plan["solid_walls"] == ["right"] and notes == []
    scanner = {"openings": [{"type": "window", "wall": "back"}, {"type": "door", "wall": "right"}]}
    merged, meta = merge_with_scanner(scanner, [plan])
    assert [o["wall"] for o in merged["openings"]] == ["back"]
    assert meta["dropped_unconfirmed"] == [("door", "right")]


def test_merge_pairs_same_type_on_disputed_wall():
    # Боевой кадр e5: дверь одна, но сканер видит её на left, а VLM на back —
    # раньше «добавка VLM» превращала её во вторую дверь.
    vlm = plan_with(openings=[{"type": "door", "wall": "back", "offset_dw": 1.3, "width_dw": 1.0}])
    merged, meta = merge_with_scanner({"openings": [{"type": "door", "wall": "left"}]}, [vlm])
    assert [o["wall"] for o in merged["openings"]] == ["left"]
    assert meta["added_from_vlm"] == []
    assert meta["paired_wall_dispute"] == [("door", "back", "left")]


def test_merge_exact_match_wins_over_wall_dispute():
    # Точные совпадения резервируются ДО спаривания: сканер [door/left, door/back],
    # VLM [door/back] — геометрию VLM получает back, а left уходит в дефолт.
    # Регресс: одноproходный матчинг отдавал VLM-дверь первому проёму сканера.
    vlm = plan_with(openings=[
        {"type": "door", "wall": "back", "offset_dw": 3.1, "width_dw": 1.0,
         "swing": {"hinge": "left", "direction": "in"}},
    ])
    scanner = {"openings": [{"type": "door", "wall": "left"}, {"type": "door", "wall": "back"}]}
    merged, meta = merge_with_scanner(scanner, [vlm])

    by_wall = {o["wall"]: o for o in merged["openings"]}
    assert set(by_wall) == {"left", "back"}
    assert by_wall["back"]["offset_dw"] == 3.1
    assert meta["paired_wall_dispute"] == []
    assert meta["defaults_used"] == [("door", "left")]
    assert_no_overlaps(merged)


def test_merge_default_avoids_later_exact_match():
    # Дефолт обходит проёмы с реальной геометрией, даже если те идут позже по списку.
    vlm = plan_with(openings=[{"type": "window", "wall": "right", "offset_dw": 2.8, "width_dw": 1.2}])
    scanner = {"openings": [{"type": "door", "wall": "right"}, {"type": "window", "wall": "right"}]}
    merged, _meta = merge_with_scanner(scanner, [vlm])
    window = next(o for o in merged["openings"] if o["type"].endswith("window"))
    assert window["offset_dw"] == 2.8
    assert_no_overlaps(merged)


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


def test_merge_scanner_door_covers_vlm_balcony_door():
    # Сканер подтипов не знает и репортит любую дверь как `door`. Балконная дверь
    # VLM обязана считаться покрытой, иначе состав расходится, мерж уходит в
    # деградацию и настоящая геометрия проёма подменяется дефолтом.
    vlm = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 1.45, "width_dw": 1.3},
        {"type": "balcony_door", "wall": "back", "offset_dw": 2.785, "width_dw": 0.9},
    ])
    scanner = {"openings": [
        {"type": "floor_to_ceiling_window", "wall": "back"}, {"type": "door", "wall": "back"},
    ]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    assert meta["fallback_scanner_composition"] is False
    assert meta["added_from_vlm"] == []
    assert [o["type"] for o in merged["openings"]] == ["floor_to_ceiling_window", "balcony_door"]
    assert merged["openings"][1]["offset_dw"] == 2.785      # геометрия от VLM


def test_merge_fallback_keeps_the_balcony_subtype_on_the_same_wall():
    # Деградация состава (окна сканера у VLM нет): дверь сканера и балконная дверь
    # VLM стоят на одной стене — подтип VLM переживает подстановку состава.
    vlm = plan_with(openings=[
        {"type": "balcony_door", "wall": "back", "offset_dw": 1.4, "width_dw": 0.9},
    ])
    scanner = {"openings": [{"type": "door", "wall": "back"}, {"type": "window", "wall": "left"}]}
    merged, meta = merge_with_scanner(scanner, [vlm])
    assert meta["fallback_scanner_composition"] is True
    assert [o["type"] for o in merged["openings"]] == ["balcony_door", "window"]


# ── медиана прогонов экстрактора ─────────────────────────────────────

def test_median_extractions_kills_the_outlier_run():
    from desow_plan.merge import median_extractions
    # lroom: дверь left в трёх прогонах 3.3 / 1.2 / 3.6 — выброс 1.2 гасится.
    def run(door_offset, width):
        return {"room": {"shape": "rectangle", "width_dw": width, "depth_dw": 5.2},
                "openings": [{"type": "door", "wall": "left", "offset_dw": door_offset,
                              "width_dw": 1.0}]}
    base, r2, r3 = run(3.3, 4.6), run(1.2, 4.4), run(3.6, 4.8)
    median_extractions([base, r2, r3])
    door = base["openings"][0]
    assert door["offset_dw"] == pytest.approx(3.3)       # медиана 1.2/3.3/3.6
    assert base["room"]["width_dw"] == pytest.approx(4.6)  # медиана 4.4/4.6/4.8
    # Выброс в БАЗОВОМ прогоне тоже переголосуется двумя нормальными.
    base2, r22, r32 = run(1.2, 4.6), run(3.3, 4.6), run(3.6, 4.6)
    median_extractions([base2, r22, r32])
    assert base2["openings"][0]["offset_dw"] == pytest.approx(3.3)
    # А совсем чужое прочтение (дальше SAME_OPENING_TOL_DW) в медиану не идёт.
    base3 = run(3.3, 4.6)
    median_extractions([base3, run(0.5, 4.6)])
    assert base3["openings"][0]["offset_dw"] == pytest.approx(3.3)


def test_median_extractions_single_run_is_untouched():
    from desow_plan.merge import median_extractions
    plan = {"room": {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0},
            "openings": [{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.6}]}
    assert median_extractions([plan]) == []
    assert plan["openings"][0]["offset_dw"] == pytest.approx(2.0)


# ── коридорная стена — боковая (поворот) ─────────────────────────────

def fin463_like_plan(**overrides):
    """Кадр fin463: зеркало сделало right глухой, пассаж коридора в углу back."""
    base = {
        "room": {"shape": "rectangle", "width_dw": 6.2, "depth_dw": 4.5},
        "openings": [
            {"type": "door", "wall": "back", "offset_dw": 4.3, "width_dw": 1.0,
             "swing": {"hinge": "left", "direction": "in"}, "confidence": 0.95},
            {"type": "passage", "wall": "back", "offset_dw": 5.6, "width_dw": 0.95,
             "confidence": 0.9},
        ],
        "camera": {**DEFAULT_CAMERA, "position": 0.65},
        "solid_walls": ["left", "right"],
    }
    base.update(overrides)
    return base


def test_corridor_wall_becomes_the_side_wall():
    plan = fin463_like_plan()
    notes = reorient_corridor_wall(plan)
    # Размеры меняются местами: комната уже и вытянутая (вердикт пользователя).
    assert plan["room"]["width_dw"] == pytest.approx(4.5)
    assert plan["room"]["depth_dw"] == pytest.approx(6.2)
    # Пассаж — в середину боковой стены: точную глубину устья с фото не измерить.
    passage = next(o for o in plan["openings"] if o["type"] == "passage")
    assert passage["wall"] == "right"
    assert passage["offset_dw"] == pytest.approx(3.1)
    # Дверь стояла впритык к устью (косяк 4.8 при устье 5.125) — она видна
    # сквозь проём и принадлежит коридору, не комнате.
    assert not any(o["type"] == "door" for o in plan["openings"])
    assert "corridor_door_dropped:door/back" in notes
    # Камера сохраняет смещение к коридорной стороне.
    assert plan["camera"]["position"] == pytest.approx(0.65)
    # Большая пустая плоскость становится back и остаётся глухой; зеркальная
    # метка расформирована — её плоскость больше не образует целой стены.
    assert plan["solid_walls"] == ["back"]
    assert notes[0].startswith("corridor_wall_to_side:right")
    assert "mirror_wall_dissolved:right" in notes


def test_corridor_wall_keeps_a_door_far_from_the_mouth():
    # Дверь в глубине стены (косяк дальше 0.5 dw от устья) — своя, переезжает
    # на боковую стену с тем же отсчётом от общего угла.
    plan = fin463_like_plan()
    plan["openings"][0]["offset_dw"] = 2.0
    reorient_corridor_wall(plan)
    door = next(o for o in plan["openings"] if o["type"] == "door")
    assert door["wall"] == "right"
    assert door["offset_dw"] == pytest.approx(2.0)


def test_corridor_wall_mirrored_on_the_left_side():
    plan = fin463_like_plan(
        openings=[
            {"type": "door", "wall": "back", "offset_dw": 3.6, "width_dw": 1.0,
             "swing": {"hinge": "right", "direction": "in"}, "confidence": 0.95},
            {"type": "passage", "wall": "back", "offset_dw": 0.6, "width_dw": 0.95,
             "confidence": 0.9},
        ],
        camera={**DEFAULT_CAMERA, "position": 0.35},
        solid_walls=["left"],
    )
    notes = reorient_corridor_wall(plan)
    assert plan["room"]["width_dw"] == pytest.approx(4.5)
    assert plan["room"]["depth_dw"] == pytest.approx(6.2)
    passage = next(o for o in plan["openings"] if o["type"] == "passage")
    assert passage["wall"] == "left"
    assert passage["offset_dw"] == pytest.approx(3.1)
    # Отсчёт от общего угла — правого угла старой back-стены: 6.2 - 3.6 = 2.6
    # (дальше CORRIDOR_DOOR_TOL_DW от устья — дверь своя, не коридорная).
    door = next(o for o in plan["openings"] if o["type"] == "door")
    assert door["wall"] == "left"
    assert door["offset_dw"] == pytest.approx(2.6)
    assert plan["solid_walls"] == []      # back глухой не объявляли
    assert "passage_mid_side:left 3.10" in notes


def test_corridor_mouth_read_as_a_door_still_reorients():
    # Прогоны бенча 16 читали устье коридора то passage, то door: тип не признак,
    # признак — вход, прижатый к углу глухой стены. Дверь-устье нормализуется в
    # passage (полотна у входа в коридор нет), коридорная дверь рядом выброшена.
    plan = fin463_like_plan(openings=[
        {"type": "door", "wall": "back", "offset_dw": 4.3, "width_dw": 1.0,
         "swing": {"hinge": "left", "direction": "in"}, "confidence": 0.95},
        {"type": "door", "wall": "back", "offset_dw": 5.6, "width_dw": 0.95,
         "swing": {"hinge": "right", "direction": "in"}, "confidence": 0.9},
    ])
    notes = reorient_corridor_wall(plan)
    assert "corridor_mouth:door->passage" in notes
    passage = next(o for o in plan["openings"] if o["type"] == "passage")
    assert passage["wall"] == "right" and "swing" not in passage
    assert not any(o["type"] == "door" for o in plan["openings"])
    assert "corridor_door_dropped:door/back" in notes


def test_corridor_wall_needs_a_solid_side_wall():
    # Без вердикта «стена глухая» пассаж в углу — обычный проём, не поворачиваем.
    plan = fin463_like_plan(solid_walls=[])
    assert reorient_corridor_wall(plan) == []
    assert plan["room"]["width_dw"] == pytest.approx(6.2)


def test_corridor_wall_needs_the_passage_at_the_corner():
    # Пассаж в середине стены — это проход в соседнее помещение, а не коридор.
    plan = fin463_like_plan()
    plan["openings"][1]["offset_dw"] = 3.0
    assert reorient_corridor_wall(plan) == []


def test_corridor_wall_skips_ambiguous_compositions():
    # Проём на большой пустой плоскости (будущей back) — переотсчёт не однозначен,
    # честнее не трогать. То же для front и перегородок.
    plan = fin463_like_plan()
    plan["openings"].append({"type": "window", "wall": "left", "offset_dw": 2.0, "width_dw": 1.0})
    assert reorient_corridor_wall(plan) == []
    plan = fin463_like_plan(partitions=[{"attach": "back", "offset_dw": 2.0, "length_dw": 1.0}])
    assert reorient_corridor_wall(plan) == []


def test_corridor_wall_ignores_a_wall_wide_passage():
    # Проём в полстены — не устье коридора.
    plan = fin463_like_plan(openings=[
        {"type": "passage", "wall": "back", "offset_dw": 3.6, "width_dw": 5.0, "confidence": 0.9},
    ])
    assert reorient_corridor_wall(plan) == []


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


def test_gate_narrows_the_entrance_instead_of_leaving_the_front_wall():
    """Кадр frame14: коридор 1.15 м шириной — вход остаётся на front, но уже.

    Дверь 1.0 dw с простенками 0.2 м требует 1.4 dw; гейт сужает саму дверь, а
    простенок 0.2 м держит ЖЁСТКО (вердикт пользователя по аудиту архитектора
    2026-08-10: норма важнее, знак камеры отступает первым).
    """
    plan = {"room": {"shape": "rectangle", "width_dw": 1.35, "depth_dw": 4.8},
            "openings": [{"type": "window", "wall": "back", "offset_dw": 0.675, "width_dw": 0.88}]}
    plan["camera"] = dict(DEFAULT_CAMERA)
    notes = ensure_door_and_window(plan)
    door = plan["openings"][-1]
    assert door["wall"] == "front"                       # стена не сменилась
    assert "door_inserted" in notes
    assert door["width_dw"] >= 0.82                      # не уже 0.7 м
    # Норма 0.2 м от обоих углов — без всяких «в обрез».
    assert door["offset_dw"] - door["width_dw"] / 2 >= MIN_CORNER_CLEARANCE_DW - 1e-3
    assert door["offset_dw"] + door["width_dw"] / 2 <= 1.35 - MIN_CORNER_CLEARANCE_DW + 1e-3


def test_gate_leaves_the_front_wall_only_when_even_a_narrow_door_does_not_fit():
    """Стена уже минимальной двери — только тогда вход уходит на соседнюю."""
    plan = {"room": {"shape": "rectangle", "width_dw": 0.9, "depth_dw": 4.8}, "openings": []}
    notes = ensure_door_and_window(plan)
    door = next(o for o in plan["openings"] if o["type"] == "door")
    assert door["wall"] != "front"
    assert any(n.startswith("door_inserted:") for n in notes)


def test_gate_inserts_window_on_front_by_default():
    # Боковых входов нет (вход на back): обе боковые видели в кадре — окно на
    # них противоречит фото (вердикт пользователя по fin380 «откуда появилось
    # окно?»); наименее ложное место — невидимая front-стена.
    plan = plan_with(openings=[{"type": "door", "wall": "back", "offset_dw": 1.0, "width_dw": 1.0}])
    notes = ensure_door_and_window(plan)
    assert notes == ["window_inserted"]
    window = plan["openings"][-1]
    assert window["type"] == "window" and window["wall"] == "front"
    assert window["width_dw"] == 1.6
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


def test_gate_window_goes_opposite_a_side_door():
    # Спальня v85 (вердикт пользователя + floorplan-expert): дверь жилой
    # комнаты ведёт из коридора (внутренняя стена), наружная стена с окном —
    # напротив. Правило распространено с прохода на всё семейство входов.
    plan = plan_with(openings=[
        {"type": "door", "wall": "right", "offset_dw": 1.2, "width_dw": 1.0,
         "swing": {"hinge": "back", "direction": "in"}},
    ])
    notes = ensure_door_and_window(plan)
    assert "window_inserted:left" in notes
    window = next(o for o in plan["openings"] if o["type"] == "window")
    assert window["wall"] == "left"


def test_gate_window_avoids_the_solid_opposite_wall():
    # Противоположная входу боковая объявлена глухой: видимая глухая стена с
    # окном противоречит фото — сначала front, глухая остаётся крайним запасным.
    plan = plan_with(
        openings=[{"type": "door", "wall": "right", "offset_dw": 1.2, "width_dw": 1.0,
                   "swing": {"hinge": "back", "direction": "in"}}],
        solid_walls=["left"],
    )
    notes = ensure_door_and_window(plan)
    assert "window_inserted" in notes
    window = next(o for o in plan["openings"] if o["type"] == "window")
    assert window["wall"] == "front"
    assert plan["solid_walls"] == ["left"]     # глухость не тронута


def test_gate_window_goes_opposite_a_side_passage():
    # Проход на правой стене (коридорный кейс fin463): напротив него — зона,
    # которую кадр толком не видел, окно-вставка честнее всего там.
    plan = plan_with(openings=[
        {"type": "passage", "wall": "right", "offset_dw": 2.5, "width_dw": 1.0},
    ])
    notes = ensure_door_and_window(plan)
    assert "window_inserted:left" in notes
    window = next(o for o in plan["openings"] if o["type"] == "window")
    assert window["wall"] == "left"
    assert window["offset_dw"] == pytest.approx(2.5)     # центр стены 5.0 dw


def test_gate_window_unseals_the_solid_wall():
    # Глухая противоположная стена — крайний запасной вариант: когда на front
    # окну места нет (узкая комната, дверь в центре), окно всё же встаёт на
    # глухую left, и метка глухости снимается — иначе schema_lite при следующем
    # парсе выбросит этот же проём как противоречие.
    plan = plan_with(
        room={"shape": "rectangle", "width_dw": 2.0, "depth_dw": 5.0},
        openings=[
            {"type": "passage", "wall": "right", "offset_dw": 2.5, "width_dw": 1.0},
            {"type": "door", "wall": "front", "offset_dw": 1.0, "width_dw": 1.0,
             "swing": {"hinge": "left", "direction": "in"}},
        ],
        solid_walls=["left"],
    )
    notes = ensure_door_and_window(plan)
    assert "window_inserted:left" in notes and "solid_wall_opened:left" in notes
    assert plan["solid_walls"] == []


def test_narrow_openings_are_widened_to_the_physical_minimum():
    from desow_plan.gate import enforce_min_opening_widths
    # fin424: балконная дверь 0.64 м от экстрактора — дверей уже 0.7 м не бывает.
    plan = plan_with(openings=[
        {"type": "balcony_door", "wall": "back", "offset_dw": 2.0, "width_dw": 0.75,
         "swing": {"hinge": "left", "direction": "in"}},
        {"type": "window", "wall": "left", "offset_dw": 2.0, "width_dw": 1.0},
    ])
    notes = enforce_min_opening_widths(plan)
    assert notes == ["widened:balcony_door/back 0.75->0.82"]
    assert plan["openings"][0]["width_dw"] == pytest.approx(0.7 / 0.85, abs=1e-2)
    assert plan["openings"][1]["width_dw"] == pytest.approx(1.0)   # норма не тронута


def test_camera_icon_dodges_the_door_arc():
    from desow_plan.render import CAM_DOOR_DODGE_DW, _dodge_door_arc
    data = {"openings": [
        {"type": "door", "wall": "front", "offset_dw": 3.0, "width_dw": 1.0},
    ]}
    # Знак в проёме двери — уходит за ближний ДОСТУПНЫЙ косяк: на стене 4.0
    # правый кандидат (4.0) вылезает за край, знак уходит влево от двери.
    assert _dodge_door_arc(data, "front", 0.0, 4.0, 3.1) == pytest.approx(2.5 - CAM_DOOR_DODGE_DW)
    # На длинной стене — к ближнему косяку (справа).
    assert _dodge_door_arc(data, "front", 0.0, 6.0, 3.4) == pytest.approx(3.5 + CAM_DOOR_DODGE_DW)
    # Вне дуги — не трогается; чужая стена — не трогается.
    assert _dodge_door_arc(data, "front", 0.0, 4.0, 1.0) == pytest.approx(1.0)
    assert _dodge_door_arc(data, "back", 0.0, 4.0, 3.1) == pytest.approx(3.1)


def test_side_opening_keeps_a_pier_before_the_unseen_wall_end():
    from desow_plan.gate import MIN_FRONT_PIER_DW, keep_side_front_pier
    # lroom: проход на правой стене нарисован впритык к невидимому концу стены
    # у камеры — сдвигается к back до минимального простенка 0.8 dw.
    plan = plan_with(openings=[
        {"type": "passage", "wall": "right", "offset_dw": 4.4, "width_dw": 1.6},
        {"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 1.0},
    ])
    notes = keep_side_front_pier(plan)
    passage = plan["openings"][0]
    # depth 5.0: предел = 5.0 - 0.8 - 0.8 = 3.4
    assert passage["offset_dw"] == pytest.approx(5.0 - MIN_FRONT_PIER_DW - 0.8)
    assert notes and notes[0].startswith("side_pier:passage/right")
    # Дверь с нормальным простенком не тронута.
    assert plan["openings"][1]["offset_dw"] == pytest.approx(2.0)


def test_front_door_from_the_extractor_snaps_to_the_camera():
    # frame13: дверь на front-стене (не видна в кадре) — offset модели дрожит;
    # снимали из двери, дверь встаёт под камеру. Вставку гейта правило не трогает.
    plan = plan_with(
        openings=[{"type": "door", "wall": "front", "offset_dw": 1.2, "width_dw": 1.0,
                   "swing": {"hinge": "left", "direction": "in"}}],
        camera={**DEFAULT_CAMERA, "position": 0.75},
    )
    notes = snap_front_door_to_camera(plan)
    assert notes and notes[0].startswith("front_door_to_camera:1.20->3.00")
    assert plan["openings"][0]["offset_dw"] == pytest.approx(0.75 * 4.0)
    inserted = plan_with(
        openings=[{"type": "door", "wall": "front", "offset_dw": 0.735, "width_dw": 1.0,
                   "swing": {"hinge": "left", "direction": "in"}, "inserted": True}],
        camera={**DEFAULT_CAMERA, "position": 0.75},
    )
    assert snap_front_door_to_camera(inserted) == []
    assert inserted["openings"][0]["offset_dw"] == pytest.approx(0.735)


def test_gate_window_prefers_a_blank_side_wall_over_a_busy_one():
    # Кейс бенча 16 (fin463): камера дрогнула к 0.35, «дальней» стала правая
    # стена с пассажем коридора — окно всё равно обязано уйти на свободную
    # левую, а не в стену, за которой коридор.
    plan = plan_with(
        openings=[{"type": "passage", "wall": "right", "offset_dw": 2.5, "width_dw": 1.0}],
        camera={**DEFAULT_CAMERA, "position": 0.35},
    )
    notes = ensure_door_and_window(plan)
    assert "window_inserted:left" in notes
    window = next(o for o in plan["openings"] if o["type"] == "window")
    assert window["wall"] == "left"


def test_gate_door_corner_prefers_the_solid_side():
    # frame13: зеркальный шкаф сделал right глухой — реальный вход рядом с ним,
    # дверь гейта прижимается к правому углу front-стены.
    plan = plan_with(
        openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5}],
        solid_walls=["right"],
    )
    ensure_door_and_window(plan)
    door = next(o for o in plan["openings"] if o["type"] == "door")
    assert door["wall"] == "front"
    assert door["offset_dw"] > ROOM["width_dw"] / 2       # правая половина
    assert door["inserted"] is True


def test_gate_door_corner_avoids_the_side_with_glazing():
    # frame12: балконная дверь на right — вход не делают вплотную к остеклению,
    # дверь гейта уходит к левому углу.
    plan = plan_with(openings=[
        {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5},
        {"type": "balcony_door", "wall": "right", "offset_dw": 2.5, "width_dw": 0.9},
    ])
    ensure_door_and_window(plan)
    door = next(o for o in plan["openings"] if o["type"] == "door")
    assert door["wall"] == "front"
    assert door["offset_dw"] < ROOM["width_dw"] / 2       # левая половина


def test_gate_door_corner_tie_breaks_away_from_the_camera():
    # Обе боковые пустые и не глухие: угол — дальний от камеры (его не видно
    # в кадре, дверь там правдоподобнее всего).
    plan = plan_with(
        openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5}],
        camera={**DEFAULT_CAMERA, "position": 0.8},
    )
    ensure_door_and_window(plan)
    door = next(o for o in plan["openings"] if o["type"] == "door")
    assert door["offset_dw"] < ROOM["width_dw"] / 2       # камера справа -> дверь слева


def test_camera_consensus_cascade():
    from desow_plan.pipeline import resolve_camera_position
    # Согласие с экстрактором - первичная проба (обычный случай).
    assert resolve_camera_position(0.5, 0.6, None)[0] == pytest.approx(0.6)
    # frame12: проба спорит с экстрактором, арбитр разошёлся с пробой и ближе
    # к экстрактору - берём арбитра.
    assert resolve_camera_position(0.35, 0.75, 0.35)[0] == pytest.approx(0.35)
    # fin463: проба спорит с экстрактором, но арбитр согласен с пробой -
    # экстрактор неправ (тянет к центру).
    assert resolve_camera_position(0.35, 0.85, 0.75)[0] == pytest.approx(0.85)
    # Проб нет - остаётся экстрактор.
    assert resolve_camera_position(0.4, None, None)[0] is None


def test_camera_probe_overrides_the_extractor_position():
    extraction = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0},
        "openings": [], "camera": {"position": 0.5},
    })
    probe = '```json\n{"reason": "правая стена крупно", "position": 0.82}\n```'
    _png, plan_json, debug = build_empty_plan(extraction, camera_probe_json=probe)
    plan = json.loads(plan_json)
    assert plan["camera"]["position"] == pytest.approx(0.82)
    assert any(line.startswith("camera_probe: позиция") for line in debug.splitlines())


def test_camera_probe_garbage_keeps_the_extractor_position():
    # 0.35 - вне порога центрирования (CAMERA_CENTER_SNAP), позиция не трогается.
    extraction = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0},
        "openings": [], "camera": {"position": 0.35},
    })
    for garbage in ("не json", '{"position": 7.5}', '{"reason": "без числа"}'):
        _png, plan_json, _debug = build_empty_plan(extraction, camera_probe_json=garbage)
        assert json.loads(plan_json)["camera"]["position"] == pytest.approx(0.35)


def test_sector_angles_are_symmetric_and_cover_side_openings():
    from desow_plan.render import CAM_FOV_DEG, _sector_angles
    room = {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0}
    base = {"room": room, "openings": []}
    apex = (2.0, 4.6)
    # Пустая комната: раскрыв симметричен вокруг оси (как настоящий объектив).
    a0, a1 = _sector_angles(base, room, "front", apex, -90.0)
    assert (a0 + a1) / 2 == pytest.approx(-90.0)
    assert a1 - a0 >= CAM_FOV_DEG - 1e-9
    # Реальный проём у камеры на правой стене раскрывает клин шире, но
    # СИММЕТРИЧНО: полууглы влево и вправо равны (вердикт пользователя).
    with_passage = {"room": room, "openings": [
        {"type": "passage", "wall": "right", "offset_dw": 3.5, "width_dw": 1.0}]}
    b0, b1 = _sector_angles(with_passage, room, "front", apex, -90.0)
    assert (b0 + b1) / 2 == pytest.approx(-90.0)
    assert b1 - b0 > a1 - a0 + 10
    # Вставка гейта клин не раскрывает: её на фото не видели.
    with_inserted = {"room": room, "openings": [
        {"type": "window", "wall": "right", "offset_dw": 3.5, "width_dw": 1.0,
         "inserted": True}]}
    c0, c1 = _sector_angles(with_inserted, room, "front", apex, -90.0)
    assert (c0, c1) == pytest.approx((a0, a1))


def test_camera_position_snaps_to_center():
    extraction = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0},
        "openings": [], "camera": {"position": 0.45},
    })
    _png, plan_json, debug = build_empty_plan(extraction)
    assert json.loads(plan_json)["camera"]["position"] == pytest.approx(0.5)
    assert any(line.startswith("camera_centered") for line in debug.splitlines())
    # Уверенно смещённая камера не прижимается.
    extraction2 = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 4.0, "depth_dw": 5.0},
        "openings": [], "camera": {"position": 0.65},
    })
    _png, plan_json2, _d = build_empty_plan(extraction2)
    assert json.loads(plan_json2)["camera"]["position"] == pytest.approx(0.65)


def test_gate_window_falls_back_to_front_when_sides_are_full():
    plan = plan_with(openings=[
        {"type": "door", "wall": "back", "offset_dw": 2.0, "width_dw": 1.0},
        {"type": "passage", "wall": "left", "offset_dw": 2.5, "width_dw": 4.4},
        {"type": "passage", "wall": "right", "offset_dw": 2.5, "width_dw": 4.4},
    ])
    notes = ensure_door_and_window(plan)
    assert notes == ["window_inserted"]
    window = plan["openings"][-1]
    assert window["wall"] == "front"
    assert window["offset_dw"] == pytest.approx(2.0)     # центр стены 4.0 dw


def test_gate_inserts_both_without_overlap():
    plan = plan_with(openings=[])
    assert ensure_door_and_window(plan) == ["door_inserted", "window_inserted"]
    door, window = plan["openings"]
    assert door["wall"] == "front" and window["wall"] == "front"
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


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


def test_gate_moves_the_entrance_to_a_blank_wall_when_front_is_full():
    # Front занят панорамой во всю стену: вход уходит на ближайшую глухую стену,
    # а не пропадает (раньше здесь был door_gate_skipped и план без входа).
    plan = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "front", "offset_dw": 2.0, "width_dw": 4.0},
    ])
    assert ensure_door_and_window(plan) == ["door_inserted:left"]
    door = plan["openings"][-1]
    assert door["type"] == "door" and door["wall"] == "left"
    assert door["swing"] == {"hinge": "back", "direction": "in"}   # петля у угла прижатия
    # Простенок от угла отсчитан по своей стене (комната 5.0 dw в глубину).
    assert door["offset_dw"] == pytest.approx(0.2 / 0.85 + 0.5, abs=1e-3)


def test_gate_prefers_a_blank_wall_over_a_busy_one():
    # Порядок «ближайших» стен — left, right, back; занятая уступает свободной.
    plan = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "front", "offset_dw": 2.0, "width_dw": 4.0},
        {"type": "window", "wall": "left", "offset_dw": 2.5, "width_dw": 1.6},
    ])
    assert ensure_door_and_window(plan) == ["door_inserted:right"]


def test_gate_skips_when_every_wall_is_full():
    plan = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "front", "offset_dw": 2.0, "width_dw": 4.0},
        {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 2.0, "width_dw": 4.0},
        {"type": "floor_to_ceiling_window", "wall": "left", "offset_dw": 2.5, "width_dw": 5.0},
        {"type": "floor_to_ceiling_window", "wall": "right", "offset_dw": 2.5, "width_dw": 5.0},
    ])
    assert ensure_door_and_window(plan) == ["door_gate_skipped"]


def test_gate_does_not_count_the_balcony_door_as_an_entrance():
    # Кадр fin424: дверь на балкон внутри остекления ведёт НАРУЖУ, поэтому вход
    # помещения гейт обязан поставить отдельно.
    plan = plan_with(openings=[
        {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 1.45, "width_dw": 1.3},
        {"type": "balcony_door", "wall": "back", "offset_dw": 2.785, "width_dw": 0.9,
         "swing": {"hinge": "right", "direction": "in"}},
    ])
    assert ensure_door_and_window(plan) == ["door_inserted"]
    door = plan["openings"][-1]
    assert door["type"] == "door" and door["wall"] == "front"
    assert [o["type"] for o in plan["openings"]].count("balcony_door") == 1   # балконная на месте


def test_gate_keeps_the_entrance_away_from_the_camera():
    """Дверь не режется под знаком камеры: её отрезок стены гейт считает занятым.

    Камера сдвинута в тот угол, куда гейт ставит вход по умолчанию, — дверь
    обязана уйти к противоположному.
    """
    plan = plan_with(
        openings=[{"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5}],
        camera={**DEFAULT_CAMERA, "position": 0.1},
    )
    assert ensure_door_and_window(plan) == ["door_inserted"]
    door = plan["openings"][-1]
    assert door["wall"] == "front"
    camera_at = 0.1 * ROOM["width_dw"]
    assert abs(door["offset_dw"] - camera_at) > door["width_dw"] / 2 + 0.35


def test_gate_counts_the_passage_as_an_entrance():
    # Проход в соседнее помещение — законный вход, вторую дверь рисовать незачем.
    plan = plan_with(openings=[
        {"type": "passage", "wall": "left", "offset_dw": 2.0, "width_dw": 1.2},
        {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5},
    ])
    assert ensure_door_and_window(plan) == []
    assert len(plan["openings"]) == 2


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


def test_front_attached_partition_grows_from_the_bottom_wall():
    """Простенок от front-стены (той, что за спиной камеры) — полноправный вход.

    Кадр l_room_synthetic: у правого края кадра стена видна ребром, растёт
    из-за спины камеры и на плане описывается как `attach: front`. Сторону
    роста задаёт геометрия, поэтому проверяем пикселями: блок front-простенка
    целиком ниже блока такого же back-простенка.
    """
    base = plan_with(openings=[{"type": "door", "wall": "left", "offset_dw": 1.0, "width_dw": 1.0,
                                "swing": {"hinge": "back", "direction": "in"}}])
    plain, _ = render_plan(base, with_furniture=False)

    def sheet(attach):
        plan, notes = validate_plan(
            dict(base, partitions=[{"attach": attach, "offset_dw": 3.0, "length_dw": 2.0}]))
        assert notes == [] and plan["partitions"][0]["attach"] == attach
        return render_plan(plan, with_furniture=False)[0]

    front, back = sheet("front"), sheet("back")
    assert front != back
    with Image.open(io.BytesIO(plain)) as p, Image.open(io.BytesIO(front)) as f, \
            Image.open(io.BytesIO(back)) as b:
        box_f = ImageChops.difference(p, f).getbbox()
        box_b = ImageChops.difference(p, b).getbbox()
    assert box_f[1] > box_b[3], (box_f, box_b)      # комната 5.0 dw: блоки не пересекаются


def test_balcony_door_is_drawn_with_the_door_symbol():
    """Символ у `balcony_door` дверной: разрыв + полотно + дуга, лист тот же.

    Отличие типов чисто смысловое (вход/выход), и на чертеже его не должно быть
    видно — иначе картиночная модель ниже по конвейеру прочтёт вторую нотацию.
    """
    swing = {"hinge": "right", "direction": "in"}
    door = plan_with(openings=[
        {"type": "door", "wall": "back", "offset_dw": 2.0, "width_dw": 0.9, "swing": swing}])
    balcony = plan_with(openings=[
        {"type": "balcony_door", "wall": "back", "offset_dw": 2.0, "width_dw": 0.9, "swing": swing}])
    blank = plan_with(openings=[])
    assert render_plan(balcony, with_furniture=False)[0] == render_plan(door, with_furniture=False)[0]
    assert render_plan(balcony, with_furniture=False)[0] != render_plan(blank, with_furniture=False)[0]
    # Балконная дверь — тоже дверное полотно, то есть годная линейка масштаба.
    assert render_plan(balcony, with_furniture=False)[1]["scale_fallback"] is False


def door_sheet(kind, width_dw, *, offset_dw=2.0, hinge="left"):
    """Лист с единственной дверью на back-стене (комната 4.0 dw шириной)."""
    return render_plan(plan_with(openings=[
        {"type": kind, "wall": "back", "offset_dw": offset_dw, "width_dw": width_dw,
         "swing": {"hinge": hinge, "direction": "in"}}]), with_furniture=False)[0]


def mirrored_diff_box(png):
    """bbox расхождения листа со своим зеркалом; None — лист симметричен.

    Комната, стены и пол симметричны относительно вертикальной оси, поэтому
    асимметрию даёт только символ двери: две створки навстречу симметричны,
    одна на всю ширину проёма — нет.
    """
    with Image.open(io.BytesIO(png)) as image:
        grey = image.convert("L")
        return ImageChops.difference(grey, grey.transpose(Image.FLIP_LEFT_RIGHT)).getbbox()


@pytest.mark.parametrize("kind, width_dw", [("balcony_door", 1.8), ("door", 1.6)])
def test_wide_door_is_drawn_with_two_leaves(kind, width_dw):
    """Проём шире порога рисуется двустворчатым независимо от подтипа двери.

    Эталон двустворчатости — `double_door` той же ширины: символ обязан совпасть
    с ним побайтно, иначе на чертеже окажется одна створка в 1.4-1.5 м, каких не
    бывает, и дуга радиусом во всю ширину проёма.
    """
    assert door_sheet(kind, width_dw) == door_sheet("double_door", width_dw)


@pytest.mark.parametrize("kind, width_dw", [("balcony_door", 0.9), ("door", 1.0)])
def test_narrow_door_keeps_one_leaf(kind, width_dw):
    """Дверь обычной ширины остаётся одностворчатой (регресс кадра fin424).

    Тот же лист у `double_door` этой ширины другой: тип форсирует две створки
    там, где ширина их не требует.
    """
    assert door_sheet(kind, width_dw) != door_sheet("double_door", width_dw)


def test_two_leaves_meet_in_the_middle_of_the_opening():
    """Пиксельно: у широкой двери две дуги навстречу, у обычной — одна.

    Дверь стоит по центру стены, так что двустворчатый символ делает лист
    зеркально симметричным, а одностворчатый — нет.
    """
    assert mirrored_diff_box(door_sheet("balcony_door", 1.8)) is None
    assert mirrored_diff_box(door_sheet("balcony_door", 0.9)) is not None


def test_rendered_plan_is_not_blank():
    plan = plan_with(openings=[{"type": "door", "wall": "front", "offset_dw": 1.0, "width_dw": 1.0}])
    png, _ = render_plan(plan, with_furniture=False)
    with Image.open(io.BytesIO(png)) as image:
        assert image.convert("L").getextrema()[0] < 40      # есть чёрные стены


# ── камера на листе ──────────────────────────────────────────────────

CAM_PLAN = plan_with(
    openings=[{"type": "door", "wall": "left", "offset_dw": 1.2, "width_dw": 1.0},
              {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.6}],
    camera=dict(DEFAULT_CAMERA),
)


def chromatic(png, step=3):
    """Цветные пиксели листа. На чистом плане их нет: он строго трёхтоновый."""
    with Image.open(io.BytesIO(png)) as image:
        rgb = image.convert("RGB")
        px = rgb.load()
        w, h = rgb.size
        return [(x, y) for y in range(0, h, step) for x in range(0, w, step)
                if max(px[x, y]) - min(px[x, y]) > 20]


def test_clean_sheet_stays_three_tone_greyscale():
    """Без флага камера не рисуется, даже если блок в плане есть."""
    png, _ = render_plan(CAM_PLAN, with_furniture=False)
    with Image.open(io.BytesIO(png)) as image:
        assert image.mode == "L"
    assert chromatic(png) == []
    without_block = {k: v for k, v in CAM_PLAN.items() if k != "camera"}
    assert render_plan(without_block, with_furniture=False)[0] == png


def test_camera_sector_is_the_only_colour_on_the_sheet():
    png, _ = render_plan(CAM_PLAN, with_furniture=False, draw_camera=True)
    spots = chromatic(png)
    assert spots, "сектор обзора не нарисован"
    with Image.open(io.BytesIO(png)) as image:
        px = image.convert("RGB").load()
    r, g, b = px[spots[len(spots) // 2]]
    assert r > g > b, (r, g, b)          # персиково-оранжевый, а не любой цвет


def marker_mask(plain, cam):
    """Маска маркера: пиксели, где на чистом листе был пол, а стал чёрный.

    Так знак отделяется от сектора без отдельного стиля «маркер без сектора»:
    сектор пол КРАСИТ (персиковый ~180 по яркости), а чернит его только маркер.
    """
    with Image.open(io.BytesIO(plain)) as a, Image.open(io.BytesIO(cam)) as b:
        floor = a.convert("L").point(lambda v: 255 if v > 200 else 0)
        dark = b.convert("L").point(lambda v: 255 if v < 100 else 0)
        return ImageChops.multiply(floor, dark)


def row_span(mask, y):
    """Ширина маркера в строке y (0 — строка пустая)."""
    box = mask.crop((0, y, mask.width, y + 1)).getbbox()
    return 0 if box is None else box[2] - box[0]


def test_camera_icon_is_the_default_marker():
    """Дефолтный знак — иконка камеры; ромб остался legacy-вариантом."""
    plain, _ = render_plan(CAM_PLAN, with_furniture=False)
    icon, _ = render_plan(CAM_PLAN, with_furniture=False, draw_camera=True)
    legacy, _ = render_plan(CAM_PLAN, with_furniture=False, draw_camera=True, camera_style="sector")
    assert icon != legacy
    icon_px = marker_mask(plain, icon).histogram()[255]
    legacy_px = marker_mask(plain, legacy).histogram()[255]
    assert icon_px > legacy_px * 2, (icon_px, legacy_px)     # силуэт крупнее точки
    assert chromatic(icon), "сектор рисуется и с иконкой"


def test_camera_icon_lens_points_along_the_view_direction():
    """Узкий объектив — со стороны взгляда, широкий корпус — со стороны стены."""
    plain, _ = render_plan(CAM_PLAN, with_furniture=False)
    icon, _ = render_plan(CAM_PLAN, with_furniture=False, draw_camera=True)
    mask = marker_mask(plain, icon)
    x0, y0, x1, y1 = mask.getbbox()
    height = y1 - y0
    lens = row_span(mask, y0 + height // 8)          # взгляд «вверх» -> объектив сверху
    body = row_span(mask, y1 - height // 8)
    assert 0 < lens < body * 0.6, (lens, body)
    # Знак стоит у своей стены и внутрь комнаты глубоко не лезет.
    assert y1 > CANVAS[1] * 0.75 and height < CANVAS[1] * 0.12, (y0, y1)


def test_camera_dot_sits_at_the_front_wall_centre():
    """Legacy-ромб: маркер в нижней зоне листа, по центру front-стены."""
    plain, _ = render_plan(CAM_PLAN, with_furniture=False)
    dot, _ = render_plan(CAM_PLAN, with_furniture=False, draw_camera=True, camera_style="dot")
    assert chromatic(dot) == []          # style="dot" — без сектора
    with Image.open(io.BytesIO(plain)) as a, Image.open(io.BytesIO(dot)) as b:
        box = ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox()
    w, h = CANVAS
    x0, y0, x1, y1 = box
    assert y0 > h * 0.6, box                             # нижняя кромка: камера за кадром
    assert abs((x0 + x1) / 2 - w / 2) < w * 0.05, box    # центр стены
    assert (x1 - x0) < w * 0.1 and (y1 - y0) < h * 0.1, box


def test_camera_position_moves_the_marker_along_the_wall():
    left = {**CAM_PLAN, "camera": {**DEFAULT_CAMERA, "position": 0.2}}
    right = {**CAM_PLAN, "camera": {**DEFAULT_CAMERA, "position": 0.8}}
    boxes = []
    for plan in (left, right):
        png, _ = render_plan(plan, with_furniture=False, draw_camera=True, camera_style="dot")
        plain, _ = render_plan(plan, with_furniture=False)
        with Image.open(io.BytesIO(plain)) as a, Image.open(io.BytesIO(png)) as b:
            boxes.append(ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox())
    assert boxes[0][0] < CANVAS[0] / 2 < boxes[1][0], boxes


def test_camera_sector_goes_under_the_furniture():
    """Красится только пол: предмет, стоящий в секторе, остаётся нетронутым."""
    empty, _ = render_plan(CAM_PLAN, with_furniture=False, draw_camera=True)
    furnished, _ = render_plan(
        {**CAM_PLAN, "furniture": [{"kind": "bed_double", "center_dw": [2.0, 2.4],
                                    "size_m": [1.6, 2.05], "rotation": 0}]},
        with_furniture=True, draw_camera=True,
    )
    assert len(chromatic(furnished)) < len(chromatic(empty)) * 0.9


def test_render_camera_png_matches_the_plan_json():
    _, plan_json, _ = build_empty_plan(EXTRACTION, "", "")
    png = render_camera_png(plan_json)
    assert png_size(png) == (1152, 928)
    assert chromatic(png), "камеры на четвёртом выходе нет"


@pytest.mark.parametrize("plan_json", ["", "не json", '{"room": {}}'])
def test_render_camera_png_never_raises(plan_json):
    """Сбой камеры не имеет права задеть основные выходы ноды."""
    with Image.open(io.BytesIO(render_camera_png(plan_json))) as image:
        assert image.convert("L").getextrema() == (255, 255)


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
    assert plan["camera"] == DEFAULT_CAMERA               # камера уезжает в сохранённый план
    assert png_size(png) == (1152, 928)
    assert debug.startswith("plan: ok")
    assert "gate: door_inserted" in debug
    assert '"marker": "orange_sector"' in debug


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
    # Второй двери сканера места на правой стене нет — она выбрасывается, а не
    # переезжает на чужую стену (переезд запрещён).
    assert "merge_drop:" in debug
    assert not any(o["wall"] == "left" for o in plan["openings"])
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


def test_fin424_balcony_door_does_not_replace_the_entrance():
    """fin424: панорама с балконной дверью на back; вход должен появиться отдельно.

    До различения типов гейт видел на плане `door` и считал, что вход есть, —
    комната оставалась без входа, хотя единственная её дверь ведёт на балкон.
    """
    plan, debug = run_case(
        {"room": {"shape": "rectangle", "width_dw": 3.8, "depth_dw": 5.6},
         "openings": [
             {"type": "floor_to_ceiling_window", "wall": "back", "offset_dw": 1.45,
              "width_dw": 1.3, "confidence": 0.95},
             {"type": "balcony_door", "wall": "back", "offset_dw": 2.785, "width_dw": 0.9,
              "swing": {"hinge": "right", "direction": "in"}, "confidence": 0.95},
         ]},
        {"openings": [
            {"type": "floor_to_ceiling_window", "wall": "back", "confidence": 0.95},
            {"type": "door", "wall": "back", "confidence": 0.9},
        ]},
    )
    placed = [(o["type"], o["wall"]) for o in plan["openings"]]
    assert ("balcony_door", "back") in placed        # балконная осталась в остеклении
    assert ("door", "front") in placed               # вход поставил гейт
    assert "merge: состав VLM покрыл сканер" in debug
    assert "gate: door_inserted" in debug
    assert_no_overlaps(plan, clearance=MIN_CORNER_CLEARANCE_DW)


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

    image, plan_json, debug, plan_camera, _camera_json = DesowPlanRender().render(
        EXTRACTION, "", "bedroom")
    assert image.shape == (1, 928, 1152, 3)                # [batch, H, W, RGB]
    assert str(image.dtype) == "torch.float32"
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert json.loads(plan_json)["room"]["width_dw"] == 3.8
    # Четвёртый выход — тот же лист с камерой: та же форма, но не тот же тензор.
    assert plan_camera.shape == image.shape
    assert not bool((plan_camera == image).all())


def test_node_output_contract_is_append_only():
    """Связи в JSON воркфлоу позиционные: порядок выходов менять нельзя."""
    pytest.importorskip("torch", reason="обёртка импортирует torch")
    pytest.importorskip("numpy")
    from desow_plan_node import DesowPlanRender

    assert DesowPlanRender.RETURN_NAMES == (
        "image", "plan_json", "debug", "plan_camera", "camera_json")
    assert DesowPlanRender.RETURN_TYPES == ("IMAGE", "STRING", "STRING", "IMAGE", "STRING")


def test_node_camera_json_output():
    """camera_json - блок camera из плана отдельной строкой; пустой план -> ''."""
    torch = pytest.importorskip("torch", reason="обёртка импортирует torch")
    pytest.importorskip("numpy")
    if not hasattr(torch, "from_numpy"):
        pytest.skip("в sys.modules заглушка torch, а не настоящий пакет")
    from desow_plan_node import DesowPlanRender

    raw = json.dumps(plan_with(openings=[
        {"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 1.0},
        {"type": "window", "wall": "back", "offset_dw": 2.0, "width_dw": 1.5},
    ]))
    out = DesowPlanRender().render(raw)
    plan = json.loads(out[1])
    assert json.loads(out[4]) == plan["camera"]
    broken = DesowPlanRender().render("не json ни разу")
    assert broken[1] == "" and broken[4] == ""

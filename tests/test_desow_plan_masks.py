"""Тесты масочной опоры (`desow_plan/masks.py`).

Синтетическая сцена с аналитически известной геометрией: центральная
перспектива, back-плинтус горизонталь y=300, боковые плинтусы под 45° к нижним
углам растра 1200x900. На такой сцене замер сверяется с точными числами.
"""
import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from desow_plan.masks import (
    GRID_H,
    GRID_W,
    build_floor_h_inv,
    diagnose_diagonal,
    floor_support,
    floor_xy,
    mask_to_grid,
    measure_openings_from_masks,
    parse_segmentation,
)

# Синтетика: back-стена [300..900]x{300}, левый плинтус y=-x+600, правый y=x-600.
XL, XR, Y_BACK = 300, 900, 300


def _png_item(label, mask_arr):
    """Сегмент в формате модели: полнокадровый box_2d + PNG-маска base64."""
    buf = io.BytesIO()
    Image.fromarray((mask_arr * 255).astype(np.uint8), "L").save(buf, "PNG")
    return {"label": label, "box_2d": [0, 0, 1000, 1000],
            "mask": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def synth_floor():
    ys = np.arange(GRID_H)[:, None]
    xs = np.arange(GRID_W)[None, :]
    top = np.where(xs < XL, -xs + 600, np.where(xs > XR, xs - 600, Y_BACK))
    return (ys >= top).astype(np.uint8)


def rect_mask(x0, x1, y0, y1):
    m = np.zeros((GRID_H, GRID_W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def seg_text(items):
    return json.dumps(items)


def base_plan(openings):
    return {"room": {"shape": "rectangle", "width_dw": 3.0, "depth_dw": 4.0},
            "openings": openings}


def test_floor_support_recovers_synthetic_lines():
    sup = floor_support(mask_to_grid(_png_item("the floor", synth_floor())))
    assert sup is not None
    pts, err, lines = sup
    # Углы back-стены восстановлены с точностью до пары пикселей растра.
    assert abs(pts["back_left"][0] * GRID_W - XL) < 15
    assert abs(pts["back_right"][0] * GRID_W - XR) < 15
    assert abs(pts["back_left"][1] * GRID_H - Y_BACK) < 8
    assert err < 3.0


def test_homography_back_wall_is_linear():
    # На самой back-линии X интерполируется линейно между углами: точка на 40%
    # ширины обязана дать floor-X == 0.4.
    pts = {"back_left": [XL / GRID_W, Y_BACK / GRID_H],
           "back_right": [XR / GRID_W, Y_BACK / GRID_H],
           "left_base": [0.0, 600 / GRID_H],
           "right_base": [1.0, 600 / GRID_H]}
    Hinv = build_floor_h_inv(pts)
    assert Hinv is not None
    x40 = (XL + 0.4 * (XR - XL)) / GRID_W
    fx, fy = floor_xy(Hinv, [x40, Y_BACK / GRID_H])
    assert abs(fx - 0.4) < 1e-6
    assert abs(fy) < 1e-6


def test_measure_door_ruler_and_back_passage():
    # Дверь на левой стене - линейка (ровно 1.0 dw); пассаж на back-стене
    # аналитичен: столбцы 540..660 -> центр 1.5 dw, ширина 0.6 dw.
    items = [_png_item("the floor", synth_floor()),
             _png_item("door on the left wall", rect_mask(100, 181, 150, 420)),
             _png_item("passage on the back wall", rect_mask(540, 661, 100, 300))]
    plan = base_plan([
        {"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9},
        {"type": "passage", "wall": "back", "offset_dw": 1.2, "width_dw": 0.8},
    ])
    notes = measure_openings_from_masks(seg_text(items), plan)
    assert any("линейка" in n for n in notes)
    door = plan["openings"][0]
    # Дверь откалибровала масштаб на себе: ширина обязана вернуться ровно 1.0.
    assert abs(door["width_dw"] - 1.0) < 0.02
    passage = plan["openings"][1]
    assert abs(passage["offset_dw"] - 1.5) < 0.06
    assert abs(passage["width_dw"] - 0.6) < 0.06


def test_clipped_mask_not_applied():
    # Дверь, упирающаяся в край кадра, - не линейка и не перемеряется.
    items = [_png_item("the floor", synth_floor()),
             _png_item("door on the left wall", rect_mask(0, 120, 150, 500))]
    plan = base_plan([{"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9}])
    notes = measure_openings_from_masks(seg_text(items), plan)
    assert any("обрезан кадром" in n for n in notes)
    assert plan["openings"][0]["offset_dw"] == 2.0
    assert plan["openings"][0]["width_dw"] == 0.9


def test_side_opening_without_ruler_skipped():
    # На боковой стене только пассаж - линейки нет, глубина неизмерима.
    items = [_png_item("the floor", synth_floor()),
             _png_item("passage on the right wall", rect_mask(1000, 1101, 200, 500))]
    plan = base_plan([{"type": "passage", "wall": "right", "offset_dw": 1.0, "width_dw": 1.1}])
    notes = measure_openings_from_masks(seg_text(items), plan)
    assert any("нет линейки" in n for n in notes)
    assert plan["openings"][0]["offset_dw"] == 1.0


def test_garbage_and_guards():
    plan = base_plan([])
    assert parse_segmentation("ни разу не json") is None
    assert measure_openings_from_masks("мусор", plan) == \
        ["segmentation: нечитаемый ответ, пропуск"]
    # l_shape масочная опора не берёт.
    lplan = {"room": {"shape": "l_shape", "width_dw": 3.0}, "openings": []}
    assert measure_openings_from_masks(seg_text([_png_item("floor", synth_floor())]),
                                       lplan) == ["segmentation: l_shape, пропуск"]
    # Сегментация без пола - честный пропуск.
    only_door = [_png_item("door on the left wall", rect_mask(100, 180, 150, 420))]
    assert measure_openings_from_masks(seg_text(only_door), plan) == \
        ["segmentation: нет маски пола, пропуск"]


def test_noisy_support_rejected():
    # Пол не из трёх прямых (L-форма, зеркало): фит шумит - кадр пропускается
    # целиком, экстракция остаётся как была.
    rng = np.random.default_rng(7)
    ys = np.arange(GRID_H)[:, None]
    xs = np.arange(GRID_W)[None, :]
    wave = 320 + 90 * np.sin(xs / 90.0) + rng.normal(0, 18, (1, GRID_W))
    floor = (ys >= wave).astype(np.uint8)
    items = [_png_item("the floor", floor),
             _png_item("door on the left wall", rect_mask(100, 181, 150, 420))]
    plan = base_plan([{"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9}])
    notes = measure_openings_from_masks(seg_text(items), plan)
    # Достаточно любого из двух отказов опоры: шумный фит либо развал углов.
    assert any("опора шумная" in n or "не построилась" in n for n in notes)
    assert plan["openings"][0]["offset_dw"] == 2.0


def test_occluded_opening_not_applied():
    # Маска видит лишь часть проёма (шторы/мебель): ширина расходится с
    # экстракцией больше 35% - измерению не верим (кейс окна living_room).
    items = [_png_item("the floor", synth_floor()),
             _png_item("door on the left wall", rect_mask(100, 181, 150, 420)),
             _png_item("window on the right wall", rect_mask(1020, 1101, 200, 400))]
    plan = base_plan([
        {"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9},
        {"type": "window", "wall": "right", "offset_dw": 1.5, "width_dw": 2.5},
    ])
    notes = measure_openings_from_masks(seg_text(items), plan)
    assert any("перекрыт" in n for n in notes)
    assert plan["openings"][1]["width_dw"] == 2.5
    assert plan["openings"][1]["offset_dw"] == 1.5


def test_out_of_range_box_is_clamped_not_fatal():
    # Модель порой отдаёт box_2d за пределами 0..1000: маска клампится к
    # растру, а не роняет разбор (репро из ревью: broadcast ValueError).
    bad = {"label": "door on the left wall", "box_2d": [100, 100, 200, 1200],
           "mask": _png_item("x", rect_mask(0, 100, 0, 100))["mask"]}
    assert mask_to_grid(bad) is not None
    # Кривая маска одного проёма не отменяет измерение остальных.
    items = [_png_item("the floor", synth_floor()),
             _png_item("door on the left wall", rect_mask(100, 181, 150, 420)), bad]
    plan = base_plan([{"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9}])
    notes = measure_openings_from_masks(seg_text(items), plan)
    assert any("->" in n for n in notes)


def test_remaining_guards():
    floor = _png_item("the floor", synth_floor())
    door = _png_item("door on the left wall", rect_mask(100, 181, 150, 420))
    # Нет ширины комнаты.
    plan = {"room": {"shape": "rectangle"}, "openings": []}
    assert measure_openings_from_masks(seg_text([floor, door]), plan) == \
        ["segmentation: нет ширины комнаты, пропуск"]
    # Маска пола нечитаема (парсимый сегмент с битым PNG).
    broken_floor = {"label": "the floor", "box_2d": [0, 0, 1000, 1000], "mask": "не base64"}
    assert measure_openings_from_masks(seg_text([broken_floor, door]),
                                       base_plan([])) == \
        ["segmentation: маска пола нечитаема, пропуск"]
    # Слишком мелкая маска пола (мало валидных столбцов) - floor_support None.
    tiny = _png_item("the floor", rect_mask(500, 560, 800, 900))
    assert measure_openings_from_masks(seg_text([tiny, door]), base_plan([])) == \
        ["segmentation: опора из маски пола не построилась, пропуск"]
    # Ширина неправдоподобна (> width_dw * 1.2): боковой пассаж во всю глубину
    # кадра - перспектива раздувает span глубины (кейс lroom: 6.59/421).
    wide = _png_item("passage on the right wall", rect_mask(910, 1191, 200, 500))
    plan = base_plan([
        {"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9},
        {"type": "passage", "wall": "right", "offset_dw": 2.0, "width_dw": 1.0},
    ])
    notes = measure_openings_from_masks(seg_text([floor, door, wide]), plan)
    assert any("неправдоподобна" in n for n in notes)
    assert plan["openings"][1]["width_dw"] == 1.0


def test_match_tolerance_rejects_far_candidate():
    # Измерение далеко от единственного кандидата (> MATCH_TOL_DW) не должно
    # «подхватываться» ближайшим проёмом - это другой объект (двери в зеркале).
    items = [_png_item("the floor", synth_floor()),
             _png_item("door on the left wall", rect_mask(100, 181, 150, 420)),
             _png_item("passage on the back wall", rect_mask(540, 661, 100, 300))]
    plan = base_plan([
        {"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9},
        # Измеренный центр пассажа = 1.5; кандидат в 2.6 dw от него.
        {"type": "passage", "wall": "back", "offset_dw": -1.1, "width_dw": 0.8},
    ])
    notes = measure_openings_from_masks(seg_text(items), plan)
    assert any("не сопоставлен" in n for n in notes)
    assert plan["openings"][1]["offset_dw"] == -1.1


def synth_diag_floor():
    """Пол кадра «в угол»: два ската без горизонтального back-сегмента."""
    ys = np.arange(GRID_H)[:, None]
    xs = np.arange(GRID_W)[None, :]
    # Видимый угол на x=800: слева пологий скат вниз, справа крутой подъём.
    top = np.where(xs < 800, 500 - 0.25 * xs, 300 + 0.55 * (xs - 800))
    return (ys >= top).astype(np.uint8)


def _diag_items():
    # back-стена - два крошечных фрагмента, левой боковой почти нет.
    return [
        _png_item("the floor", synth_diag_floor()),
        _png_item("the back wall", rect_mask(760, 830, 240, 380)),
        _png_item("the back wall", rect_mask(600, 650, 300, 420)),
        _png_item("the left wall", rect_mask(0, 60, 500, 700)),
        _png_item("the right wall", rect_mask(830, 1200, 100, 700)),
        _png_item("door on the right wall", rect_mask(860, 1000, 200, 560)),
    ]


def test_diagnose_diagonal_detects_corner_shot():
    assert diagnose_diagonal(seg_text(_diag_items())) == "left"
    # Фронтальная сцена (горизонтальный back-плинтус) - не диагональ.
    frontal = [_png_item("the floor", synth_floor()),
               _png_item("the back wall", rect_mask(300, 900, 100, 300)),
               _png_item("the left wall", rect_mask(0, 300, 100, 600)),
               _png_item("the right wall", rect_mask(900, 1200, 100, 600))]
    assert diagnose_diagonal(seg_text(frontal)) is None
    # Скат есть, но back-стена видна широко (зеркало/L-пол) - не диагональ.
    trap = list(_diag_items())
    trap[1] = _png_item("the back wall", rect_mask(200, 830, 150, 500))
    assert diagnose_diagonal(seg_text(trap)) is None


def test_diagonal_mode_places_camera_and_door():
    from desow_plan import build_empty_plan
    from desow_plan.schema_lite import MIN_CORNER_CLEARANCE_DW

    extraction = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 4.2, "depth_dw": 4.8},
        "openings": [{"type": "door", "wall": "right", "offset_dw": 2.4,
                      "width_dw": 1.0,
                      "swing": {"hinge": "back", "direction": "in"}}],
        "camera": {"position": 0.55},
    })
    _png, plan_json, debug = build_empty_plan(
        extraction, "", "bedroom",
        # Пробы дают согласованно неверный 0.85 - в диагональном режиме они
        # обязаны игнорироваться.
        json.dumps({"reason": "x", "position": 0.85}),
        json.dumps({"reason": "x", "position": 0.85}),
        "", "", seg_text(_diag_items()))
    plan = json.loads(plan_json)
    assert "camera_diag" in debug and "door_diag" in debug
    assert plan["camera"]["position"] == pytest.approx(0.1)
    door = next(o for o in plan["openings"] if o["type"] == "door")
    assert door["wall"] == "right"
    assert door["offset_dw"] == pytest.approx(MIN_CORNER_CLEARANCE_DW + 0.5, abs=1e-3)


def test_diagonal_mode_unseals_the_unseen_wall():
    # Экстрактор назвал невидимую боковую глухой - на диагональном кадре это
    # гадание (стены нет в кадре): метка снимается, окно встаёт напротив входа
    # (боевой прогон v86: solid_walls=['left'] уводил окно на front).
    from desow_plan import build_empty_plan

    extraction = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 4.2, "depth_dw": 4.8},
        "openings": [{"type": "door", "wall": "right", "offset_dw": 2.4,
                      "width_dw": 1.0,
                      "swing": {"hinge": "back", "direction": "in"}}],
        "camera": {"position": 0.55},
        "solid_walls": ["left"],
    })
    _png, plan_json, debug = build_empty_plan(
        extraction, "", "bedroom", "", "", "", "", seg_text(_diag_items()))
    plan = json.loads(plan_json)
    assert "solid_diag" in debug
    window = next(o for o in plan["openings"] if o["type"] == "window")
    assert window["wall"] == "left"
    assert "left" not in (plan.get("solid_walls") or [])


def test_camera_probe_softened_without_floor_support():
    # A/B кухня 2026-08-11: опора отказала (мебель), проба 0.62 без спора с
    # экстрактором 0.45 тянула камеру вправо при истине ~0.5. Без опоры слабое
    # решение усредняется; среднее 0.535 попадает под снап к центру -> 0.5.
    from desow_plan import build_empty_plan
    from desow_plan.pipeline import resolve_camera_position

    pos, why = resolve_camera_position(0.45, 0.62, None, support_ok=False)
    assert pos == pytest.approx(0.535)
    assert "смягчена" in why
    # С построенной опорой поведение прежнее - эталон бенча не задет.
    assert resolve_camera_position(0.45, 0.62, None, support_ok=True) == (0.62, "первичная проба")
    # Сильная ветка «пробы согласны против экстрактора» не смягчается.
    pos2, why2 = resolve_camera_position(0.2, 0.62, 0.6, support_ok=False)
    assert pos2 == 0.62 and "согласны" in why2

    # Интеграционно: шумный пол (опора не строится) + проба без спора.
    rng = np.random.default_rng(7)
    ys = np.arange(GRID_H)[:, None]
    xs = np.arange(GRID_W)[None, :]
    wave = 320 + 90 * np.sin(xs / 90.0) + rng.normal(0, 18, (1, GRID_W))
    noisy = [_png_item("the floor", (ys >= wave).astype(np.uint8))]
    extraction = json.dumps({
        "room": {"shape": "rectangle", "width_dw": 5.6, "depth_dw": 5.0},
        "openings": [{"type": "window", "wall": "back", "offset_dw": 4.0, "width_dw": 1.5}],
        "camera": {"position": 0.45},
    })
    _png, plan_json, debug = build_empty_plan(
        extraction, "", "kitchen",
        json.dumps({"reason": "x", "position": 0.62}), "", "", "",
        seg_text(noisy))
    plan = json.loads(plan_json)
    assert plan["camera"]["position"] == pytest.approx(0.5)   # среднее + снап
    assert "смягчена" in debug


def test_corner_fit_finds_edge():
    from desow_plan.masks import classify_floor_geometry, floor_support_corner

    # Кадр «в угол»: горизонтальный плинтус слева + скат справа, излом на 0.7
    # (геометрия пустой гостиной declutter-эксперимента, разметка пользователя).
    ys = np.arange(GRID_H)[:, None]
    xs = np.arange(GRID_W)[None, :]
    top = np.where(xs < 840, 500, 500 + 0.65 * (xs - 840))
    corner_floor = (ys >= top).astype(np.uint8)
    r = floor_support_corner(mask_to_grid(_png_item("the floor", corner_floor)))
    assert r is not None
    corner_x, err, _lines = r
    assert corner_x == pytest.approx(0.7, abs=0.02)
    assert err < 3.0
    # Классификатор выбирает угловую модель на этом профиле...
    mode, _err, detail = classify_floor_geometry(
        seg_text([_png_item("the floor", corner_floor)]))
    assert mode == "corner" and detail["corner_x"] == pytest.approx(0.7, abs=0.02)
    # ...и фронтальную - на фронтальном (не уезжает в угол из-за шума).
    mode2, _e2, _d2 = classify_floor_geometry(
        seg_text([_png_item("the floor", synth_floor())]))
    assert mode2 == "frontal"


def test_composition_not_changed_by_masks():
    # Сегмент без пары в экстракции состав НЕ пополняет.
    items = [_png_item("the floor", synth_floor()),
             _png_item("door on the left wall", rect_mask(100, 181, 150, 420)),
             _png_item("window on the right wall", rect_mask(1000, 1101, 200, 400))]
    plan = base_plan([{"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9}])
    measure_openings_from_masks(seg_text(items), plan)
    assert len(plan["openings"]) == 1

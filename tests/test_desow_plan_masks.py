"""Тесты масочной опоры (`desow_plan/masks.py`).

Синтетическая сцена с аналитически известной геометрией: центральная
перспектива, back-плинтус горизонталь y=300, боковые плинтусы под 45° к нижним
углам растра 1200x900. На такой сцене замер сверяется с точными числами.
"""
import base64
import io
import json

import numpy as np
from PIL import Image

from desow_plan.masks import (
    GRID_H,
    GRID_W,
    build_floor_h_inv,
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


def test_composition_not_changed_by_masks():
    # Сегмент без пары в экстракции состав НЕ пополняет.
    items = [_png_item("the floor", synth_floor()),
             _png_item("door on the left wall", rect_mask(100, 181, 150, 420)),
             _png_item("window on the right wall", rect_mask(1000, 1101, 200, 400))]
    plan = base_plan([{"type": "door", "wall": "left", "offset_dw": 2.0, "width_dw": 0.9}])
    measure_openings_from_masks(seg_text(items), plan)
    assert len(plan["openings"]) == 1

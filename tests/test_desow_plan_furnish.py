"""Тесты расстановки мебели (нода DesowPlanFurnish).

LLM всюду замокана: `complete` — обычная функция, отдающая заготовленные ответы
по очереди (тот же приём, что в бэкендовом `tests/test_plan2d.py::_FakeClient`).
Сеть в пакете `desow_plan` не живёт вообще, поэтому мокать нечего.

Тесты обёртки ComfyUI требуют torch и скипаются вне ComfyUI — как и у
DesowPlanRender. Всё, что можно проверить без тензоров (цикл ре-промптов, состав
промпта, белый лист, отчёт), проверяется на уровне пакета.
"""
import io
import json

import pytest
from PIL import Image

from desow_plan import blank_png, build_furnished_plan
from desow_plan.furnish import (
    STYLE_HINT_LIMIT,
    FurnishError,
    build_messages,
    parse_furniture,
    place_furniture,
)
from desow_plan.pipeline import FurnishFailed
from desow_plan.render import CANVAS
from desow_plan.schema_lite import CONTACT_TOL_DW, DW_M, PlanDataError, validate_furniture_item
from desow_plan.validate import validate_furniture

# Боевой кадр living_room (прототип hybrid-proto): 4.8x4.6 dw, окно слева, дверь справа.
PLAN = {
    "room": {"shape": "rectangle", "width_dw": 4.8, "depth_dw": 4.6},
    "openings": [
        {"type": "window", "wall": "left", "offset_dw": 2.1, "width_dw": 1.3},
        {"type": "door", "wall": "right", "offset_dw": 3.4, "width_dw": 1.0,
         "swing": {"hinge": "back", "direction": "in"}},
    ],
    "camera": {"wall": "front", "position": 0.5, "direction": "up", "marker": "orange_sector"},
}
PLAN_JSON = json.dumps(PLAN, ensure_ascii=False)

# Чистая расстановка: комод у back-стены + ковёр по центру (ковёр освобождён от
# проверок пересечений и зон, комод далеко от двери и от полосы перед окном).
GOOD_FURNITURE = [
    {"kind": "dresser", "center_dw": [2.4, 0.4], "size_m": [1.0, 0.5], "rotation": 0},
    {"kind": "rug", "center_dw": [2.4, 2.3], "size_m": [2.0, 2.5], "rotation": 0},
]
# Шкаф вплотную к дверной петле: гарантированные нарушения (дуга + подход + коробка).
BAD_FURNITURE = [
    {"kind": "wardrobe", "center_dw": [4.4, 2.9], "size_m": [0.6, 1.6], "rotation": 0},
]


class FakeLLM:
    """Дублёр текстовой модели: отдаёт заготовленные ответы, помнит диалоги."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def furniture_reply(items):
    return json.dumps(items, ensure_ascii=False)


def run_pipeline(responses, **kwargs):
    """Конвейер ноды на замоканной модели -> (png, furniture_json, debug, llm)."""
    llm = FakeLLM(responses)
    png, furniture_json, debug = build_furnished_plan(
        PLAN_JSON, kwargs.pop("room_type", "living room"), llm, model_label="fake/model", **kwargs
    )
    return png, furniture_json, debug, llm


def chromatic(png, step=3):
    """Цветные пиксели листа. На чистом плане их нет: он строго трёхтоновый."""
    with Image.open(io.BytesIO(png)) as image:
        rgb = image.convert("RGB")
        px = rgb.load()
        w, h = rgb.size
        return [(x, y) for y in range(0, h, step) for x in range(0, w, step)
                if max(px[x, y]) - min(px[x, y]) > 20]


# ── схема предмета (порт FurnitureItem) ──────────────────────────────

@pytest.mark.parametrize("given, snapped", [
    (0, 0), (90, 90), (270, 270), (360, 0),
    (-90, 270),      # отрицательный угол приводится к обороту
    (100, 90),       # ближайшая четверть
    (45, 0),         # ровно посередине: round() в Python банковское (0.5 -> 0)
])
def test_validate_furniture_item_snaps_rotation(given, snapped):
    """Порт FurnitureItem: рендер знает только 0/90/180/270, кривой угол не бракует предмет."""
    item = {"kind": "bed", "center_dw": [2, 2], "size_m": [1.6, 2.0], "rotation": given}
    assert validate_furniture_item(item)["rotation"] == snapped


@pytest.mark.parametrize("bad", [
    {"kind": "bed", "center_dw": [2, 2], "size_m": [500, 2.0]},              # шкаф в полкилометра
    {"kind": "bed", "center_dw": [2, 2], "size_m": [float("nan"), 2.0]},     # NaN в размере
    {"kind": "bed", "center_dw": [2, float("inf")], "size_m": [1.6, 2.0]},   # inf в центре
    {"kind": "", "center_dw": [2, 2], "size_m": [1.6, 2.0]},                 # без вида
    {"kind": "bed", "center_dw": [2], "size_m": [1.6, 2.0]},                 # центр не пара
])
def test_validate_furniture_item_rejects_garbage(bad):
    with pytest.raises(PlanDataError):
        validate_furniture_item(bad)


def test_parse_furniture_skips_invalid_but_keeps_the_rest():
    """Один битый предмет не хоронит расстановку (паритет с бэкендом)."""
    text = furniture_reply([{"kind": "ghost"}, GOOD_FURNITURE[0]])
    assert [i["kind"] for i in parse_furniture(text, PLAN)] == ["dresser"]


def test_parse_furniture_drops_item_far_outside_room():
    text = furniture_reply([
        {"kind": "bed", "center_dw": [900, 900], "size_m": [1.6, 2.0], "rotation": 0},
        GOOD_FURNITURE[1],
    ])
    assert [i["kind"] for i in parse_furniture(text, PLAN)] == ["rug"]


# ── допуск касания (мебель вплотную к стене) ─────────────────────────

# Комната боевого кадра fin379: на нём валидатор браковал флеш-постановку.
TOL_ROOM = {"room": {"shape": "rectangle", "width_dw": 3.8, "depth_dw": 7.2}, "openings": []}
WARDROBE_HALF_DW = 0.6 / DW_M / 2      # полуглубина шкафа 0.6 м


def wardrobe_at(cx, cy, rotation=0):
    return {"kind": "wardrobe", "center_dw": [cx, cy], "size_m": [0.6, 1.6], "rotation": rotation}


@pytest.mark.parametrize("label, center, rotation", [
    ("right", [3.8 - WARDROBE_HALF_DW, 3.0], 0),
    ("right, центр округлён моделью до сотых", [3.45, 3.0], 0),
    ("left", [WARDROBE_HALF_DW, 3.0], 0),
    ("left, центр округлён", [0.35, 3.0], 0),
    ("back", [2.0, WARDROBE_HALF_DW], 90),
    ("back, центр округлён", [2.0, 0.35], 90),
    ("front", [2.0, 7.2 - WARDROBE_HALF_DW], 90),
    ("front, центр округлён", [2.0, 6.85], 90),
])
def test_flush_against_every_wall_is_legal(label, center, rotation):
    """Мебель вплотную к стене — норма жизни, а не выход за пределы комнаты.

    Регресс серии v2: шкаф `[3.10,4.26,3.80,6.14]` при ширине 3.8 браковался как
    вышедший наружу. Точная флеш-постановка модели недоступна — центр она
    округляет до 0.01 dw, а 0.6 м в dw это 0.70588..., поэтому хвост в миллиметры
    неизбежен.
    """
    assert validate_furniture(TOL_ROOM, [wardrobe_at(*center, rotation=rotation)]) == [], label


@pytest.mark.parametrize("over_m", [0.02, 0.05, 0.5])
def test_real_overflow_is_still_caught(over_m):
    """2 см и больше — уже нарушение, и величина выхода видна в тексте."""
    item = wardrobe_at(3.8 - WARDROBE_HALF_DW + over_m / DW_M, 3.0)
    errs = validate_furniture(TOL_ROOM, [item])
    assert len(errs) == 1 and "выходит за пределы комнаты" in errs[0]
    assert "на %.0f мм" % (over_m * 1000) in errs[0], errs


def test_tolerance_is_one_centimetre():
    """Допуск заявлен явно: меньше сантиметра — не нарушение, больше двух — нарушение."""
    assert 0.009 < CONTACT_TOL_DW * DW_M < 0.011


L_ROOM = {
    "room": {"shape": "l_shape", "width_dw": 6.0, "depth_dw": 5.0,
             "polygon_dw": [[0, 0], [6, 0], [6, 3], [4, 3], [4, 5], [0, 5]]},
    "openings": [],
}


def test_flush_against_l_polygon_edge_is_legal():
    """У L-комнаты граница проверяется полигоном — допуск должен работать и там."""
    flush = {"kind": "wardrobe", "center_dw": [6.0 - WARDROBE_HALF_DW, 1.5],
             "size_m": [0.6, 1.6], "rotation": 0}
    assert validate_furniture(L_ROOM, [flush]) == []
    outside = {"kind": "wardrobe", "center_dw": [5.0, 4.0], "size_m": [1.0, 1.0], "rotation": 0}
    assert any("L-полигона" in e for e in validate_furniture(L_ROOM, [outside]))


PART_ROOM = {
    "room": {"shape": "rectangle", "width_dw": 6.8, "depth_dw": 5.2},
    "openings": [],
    "partitions": [{"attach": "back", "offset_dw": 4.1, "length_dw": 1.8}],
}


def test_furniture_touching_partition_is_legal_but_overlap_is_not():
    """Простенок толщиной 0.25 dw: встать вплотную можно, залезть на него нельзя."""
    edge = 4.1 - 0.25 / 2                       # левая грань простенка
    half_w = 0.5 / DW_M / 2                     # полуширина комода вдоль x
    touching = {"kind": "dresser", "center_dw": [edge - half_w, 0.9],
                "size_m": [0.5, 1.0], "rotation": 0}
    assert validate_furniture(PART_ROOM, [touching]) == []
    overlapping = {"kind": "dresser", "center_dw": [edge - half_w + 0.05 / DW_M, 0.9],
                   "size_m": [0.5, 1.0], "rotation": 0}
    assert any("простенок" in e for e in validate_furniture(PART_ROOM, [overlapping]))


def test_furniture_side_by_side_is_legal_but_overlap_is_not():
    """Два предмета борт к борту — норма; допуск не должен прощать реальный нахлёст."""
    left = {"kind": "dresser", "center_dw": [1.0, 1.0], "size_m": [1.0, 0.5], "rotation": 0}
    side = 1.0 / DW_M                            # ширина комода в dw
    right = {"kind": "desk", "center_dw": [1.0 + side, 1.0], "size_m": [1.0, 0.5], "rotation": 0}
    assert validate_furniture(TOL_ROOM, [left, right]) == []
    over = {"kind": "desk", "center_dw": [1.0 + side - 0.1 / DW_M, 1.0],
            "size_m": [1.0, 0.5], "rotation": 0}
    assert any("пересечение" in e for e in validate_furniture(TOL_ROOM, [left, over]))


# ── зоны широкой (двустворчатой) двери ───────────────────────────────

# Боевой кадр kitchen: садовая остеклённая дверь 1.6 dw (1.36 м) на правой стене.
WIDE_DOOR_ROOM = {
    "room": {"shape": "rectangle", "width_dw": 6.0, "depth_dw": 5.0},
    "openings": [
        {"type": "balcony_door", "wall": "right", "offset_dw": 3.8, "width_dw": 1.6,
         "swing": {"hinge": "back", "direction": "in"}},
    ],
}
# Та же дверь на back-стене: проверяем дугу дальней от петель створки.
WIDE_DOOR_BACK = {
    "room": {"shape": "rectangle", "width_dw": 6.0, "depth_dw": 5.0},
    "openings": [
        {"type": "balcony_door", "wall": "back", "offset_dw": 3.0, "width_dw": 1.6,
         "swing": {"hinge": "left", "direction": "in"}},
    ],
}


def test_wide_door_zones_are_measured_by_the_leaf():
    """Зоны широкой двери считаются по створке, как её и рисует рендер.

    Створка вдвое короче проёма, поэтому подход к двери 1.6 dw кончается в
    0.8 + 0.7 м от стены. Комод в 1.65 dw от неё стоит свободно — при расчёте
    по полной ширине проёма он попадал бы в зону и съедал ре-промпт.
    """
    far = {"kind": "dresser", "center_dw": [4.0, 3.8], "size_m": [0.6, 0.6], "rotation": 0}
    assert validate_furniture(WIDE_DOOR_ROOM, [far]) == []
    near = {"kind": "dresser", "center_dw": [5.5, 3.8], "size_m": [0.6, 0.6], "rotation": 0}
    errs = validate_furniture(WIDE_DOOR_ROOM, [near])
    assert any("подход к двери" in e for e in errs), errs
    assert any("дугу двери" in e and "< 0.80 dw" in e for e in errs), errs


def test_both_leaves_of_a_wide_door_sweep_their_arc():
    """У двустворчатой двери две дуги: дальняя от петель тоже держит место.

    С одной дугой на весь проём место у дальнего края не резервировалось вовсе —
    предмет вставал вплотную к створке, которая туда открывается.
    """
    at_far_edge = {"kind": "side_table", "center_dw": [4.3, 0.3], "size_m": [0.3, 0.3],
                   "rotation": 0}
    assert any("дугу двери" in e for e in validate_furniture(WIDE_DOOR_BACK, [at_far_edge]))
    beyond = {"kind": "side_table", "center_dw": [5.0, 0.3], "size_m": [0.3, 0.3], "rotation": 0}
    assert validate_furniture(WIDE_DOOR_BACK, [beyond]) == []


def test_narrow_door_zones_are_unchanged():
    """Обычная дверь 1.0 dw одностворчатая: дуга во всю ширину, подход прежний."""
    errs = validate_furniture(PLAN, BAD_FURNITURE)
    assert any("дугу двери" in e and "< 1.00 dw" in e for e in errs), errs
    assert any("подход к двери" in e for e in errs), errs


# ── цикл расстановки ─────────────────────────────────────────────────

def test_single_call_when_layout_is_clean():
    llm = FakeLLM([furniture_reply(GOOD_FURNITURE)])
    furniture, meta = place_furniture(llm, PLAN, "living room")
    assert len(furniture) == 2
    assert meta["calls"] == 1 and meta["retries"] == 0 and meta["violations"] == []


def test_reprompts_with_violations_listed():
    """Нарушения уезжают модели списком, чистый второй ответ принимается."""
    llm = FakeLLM([furniture_reply(BAD_FURNITURE), furniture_reply(GOOD_FURNITURE)])
    furniture, meta = place_furniture(llm, PLAN, "living room")
    assert meta["calls"] == 2 and meta["retries"] == 1 and meta["violations"] == []
    assert [f["kind"] for f in furniture] == ["dresser", "rug"]
    retry_prompt = llm.calls[1][-1]["content"]
    assert "Validator found violations" in retry_prompt
    assert "дугу двери" in retry_prompt


def test_unparsable_first_answer_is_reprompted():
    llm = FakeLLM(["извини, не могу", furniture_reply(GOOD_FURNITURE)])
    furniture, meta = place_furniture(llm, PLAN, "living room")
    assert meta["calls"] == 2 and len(furniture) == 2
    assert "JSON array" in llm.calls[1][-1]["content"]


def test_attempt_limit_keeps_last_layout_with_violations():
    """После лимита возвращается последняя расстановка КАК ЕСТЬ, с нарушениями."""
    llm = FakeLLM([furniture_reply(BAD_FURNITURE)] * 3)
    furniture, meta = place_furniture(llm, PLAN, "living room", max_retries=2)
    assert meta["calls"] == 3 and meta["retries"] == 2
    assert meta["violations"], "нарушения обязаны остаться в meta"
    assert len(furniture) == 1


def test_never_parsable_answer_raises():
    llm = FakeLLM(["нет"] * 4)
    with pytest.raises(FurnishError) as exc:
        place_furniture(llm, PLAN, "living room", max_retries=3)
    assert exc.value.code == "plan_furnish_invalid_json"


# ── состав промпта ───────────────────────────────────────────────────

def test_camera_is_not_sent_to_the_placer():
    """Паритет с бэкендом: ключ camera вырезан — у размещающей модели нет правил про неё."""
    user = build_messages(PLAN, "living room")[1]["content"]
    assert "camera" not in user and "orange_sector" not in user
    assert "LIVING ROOM" in build_messages(PLAN, "living room")[0]["content"]


def test_seed_reaches_the_prompt():
    """Вариативность: разный seed -> разный промпт (и мимо кеша провайдера)."""
    with_seed = build_messages(PLAN, "living room", "", 4242)[1]["content"]
    assert "LAYOUT VARIANT ID: 4242" in with_seed
    other = build_messages(PLAN, "living room", "", 777)[1]["content"]
    assert other != with_seed
    # Нулевой seed — секции нет вообще: не сбиваем модель служебной строкой.
    assert "LAYOUT VARIANT" not in build_messages(PLAN, "living room", "", 0)[1]["content"]


def test_style_hint_is_a_separate_section_and_is_clamped():
    user = build_messages(PLAN, "living room", "тёплый минимализм, дерево", 0)[1]["content"]
    assert "STYLE PREFERENCES" in user and "тёплый минимализм" in user
    # Пожелания идут ПОСЛЕ плана и не смешиваются с системными правилами.
    assert user.index("Room JSON") < user.index("STYLE PREFERENCES")
    assert "STYLE PREFERENCES" not in build_messages(PLAN, "living room", "   ", 0)[1]["content"]

    long_hint = "ар-деко " * 500
    clamped = build_messages(PLAN, "living room", long_hint, 0)[1]["content"]
    assert len(clamped) < len("Room JSON") + len(long_hint)
    assert clamped.count("ар-деко") <= STYLE_HINT_LIMIT


# ── конвейер ноды ────────────────────────────────────────────────────

def test_pipeline_renders_furnished_plan_and_reports():
    png, furniture_json, debug, llm = run_pipeline([furniture_reply(GOOD_FURNITURE)])
    assert Image.open(io.BytesIO(png)).size == CANVAS
    assert [i["kind"] for i in json.loads(furniture_json)] == ["dresser", "rug"]
    assert debug.startswith("furnish: ok (2 предметов")
    assert "attempt 1: нарушений 0" in debug
    assert "furnish: предметов 2, вызовов модели 1" in debug
    assert "model=fake/model" in debug
    assert llm.calls[0][0]["role"] == "system"


def test_pipeline_debug_lists_violations_per_attempt():
    png, _, debug, _ = run_pipeline(
        [furniture_reply(BAD_FURNITURE), furniture_reply(GOOD_FURNITURE)]
    )
    assert "attempt 1: нарушений" in debug and "attempt 2: нарушений 0" in debug
    assert "  violation: " in debug


def test_pipeline_draws_camera_by_flag():
    """Маркер камеры — единственный цвет на листе, поэтому виден по цветным пикселям."""
    with_cam, _, debug_on, _ = run_pipeline([furniture_reply(GOOD_FURNITURE)], draw_camera=True)
    without, _, debug_off, _ = run_pipeline([furniture_reply(GOOD_FURNITURE)], draw_camera=False)
    assert chromatic(with_cam), "сектор обзора не нарисован"
    assert chromatic(without) == []
    assert "orange_sector" in debug_on and "camera: не рисуется" in debug_off


def test_pipeline_max_attempts_counts_model_calls():
    """max_attempts — это вызовы модели: 1 значит «без ре-промптов»."""
    _, _, debug, llm = run_pipeline([furniture_reply(BAD_FURNITURE)], max_attempts=1)
    assert len(llm.calls) == 1
    assert "вызовов модели 1, ре-промптов 0" in debug


@pytest.mark.parametrize("plan_json, reason", [
    ("не json вовсе", "plan_invalid_json"),
    ('{"room": {"width_dw": "широкая"}}', "plan_invalid_schema"),
])
def test_pipeline_raises_on_broken_plan(plan_json, reason):
    llm = FakeLLM([])
    with pytest.raises(FurnishFailed) as exc:
        build_furnished_plan(plan_json, "living room", llm)
    assert exc.value.reason.startswith(reason)


def test_pipeline_wraps_llm_failure():
    """Сбой сети после ретраев — терминальный: конвейер отдаёт причину наверх."""
    llm = FakeLLM([RuntimeError("OpenRouter API unreachable after 3 retries")])
    with pytest.raises(FurnishFailed) as exc:
        build_furnished_plan(PLAN_JSON, "living room", llm)
    assert exc.value.reason.startswith("llm_unavailable: RuntimeError")
    assert any("plan: комната" in line for line in exc.value.debug_lines)


def test_pipeline_wraps_unparsable_model_answer():
    llm = FakeLLM(["нет"] * 3)
    with pytest.raises(FurnishFailed) as exc:
        build_furnished_plan(PLAN_JSON, "living room", llm, max_attempts=3)
    assert exc.value.reason.startswith("plan_furnish_invalid_json")


# ── обёртка ComfyUI (нужен torch) ────────────────────────────────────

def _node_class():
    torch = pytest.importorskip("torch", reason="torch есть только внутри ComfyUI")
    pytest.importorskip("numpy")
    if not hasattr(torch, "from_numpy"):
        # test_node_prompt_echo подменяет torch заглушкой, когда настоящего нет.
        pytest.skip("в sys.modules заглушка torch, а не настоящий пакет")
    from desow_plan_furnish_node import DesowPlanFurnish

    return DesowPlanFurnish


def _patched_node(monkeypatch, responses):
    """Нода с подменённым сетевым вызовом."""
    cls = _node_class()
    import desow_plan_furnish_node as module

    llm = FakeLLM(responses)
    monkeypatch.setattr(module, "chat_complete", lambda messages, **kw: llm(messages))
    return cls(), llm


def test_node_returns_furnished_tensor(monkeypatch):
    node, _ = _patched_node(monkeypatch, [furniture_reply(GOOD_FURNITURE)])
    image, furniture_json, debug = node.furnish(PLAN_JSON, room_type="living room")
    assert image.shape == (1, CANVAS[1], CANVAS[0], 3)     # [batch, H, W, RGB]
    assert str(image.dtype) == "torch.float32"
    assert len(json.loads(furniture_json)) == 2
    assert debug.startswith("furnish: ok")


def test_node_fail_soft_returns_blank_sheet_and_reason(monkeypatch):
    """Лимит попыток исчерпан -> белый лист + PLACEMENT_FAILED, прогон живой."""
    node, llm = _patched_node(monkeypatch, ["нет"] * 3)
    image, furniture_json, debug = node.furnish(
        PLAN_JSON, room_type="living room", max_attempts=3, fail_soft=True
    )
    assert len(llm.calls) == 3
    assert furniture_json == ""
    assert debug.startswith("PLACEMENT_FAILED: plan_furnish_invalid_json")
    assert "plan: комната" in debug, "ход работы до сбоя обязан остаться в отчёте"
    assert float(image.min()) == 1.0, "белый лист"
    assert image.shape == (1, CANVAS[1], CANVAS[0], 3)


def test_node_fail_soft_off_raises(monkeypatch):
    node, _ = _patched_node(monkeypatch, ["нет"] * 3)
    with pytest.raises(FurnishFailed):
        node.furnish(PLAN_JSON, room_type="living room", max_attempts=3, fail_soft=False)


def test_node_fail_soft_survives_broken_plan(monkeypatch):
    node, _ = _patched_node(monkeypatch, [])
    image, furniture_json, debug = node.furnish("", room_type="living room")
    assert furniture_json == "" and debug.startswith("PLACEMENT_FAILED: plan_invalid_json")
    assert float(image.min()) == 1.0


def test_node_blank_output_matches_render_node_blank():
    """Тот же белый лист, что у DesowPlanRender: гейт ниже по графу один на оба."""
    assert Image.open(io.BytesIO(blank_png())).size == CANVAS


def test_node_output_contract_is_append_only():
    """Связи в JSON воркфлоу позиционные: порядок выходов менять нельзя."""
    cls = _node_class()
    assert cls.RETURN_NAMES == ("furnished_plan", "furniture_json", "debug")
    assert cls.RETURN_TYPES == ("IMAGE", "STRING", "STRING")


def test_node_widget_order_is_fixed():
    """widgets_values мапятся позиционно: порядок виджетов — часть контракта графа."""
    cls = _node_class()
    spec = cls.INPUT_TYPES()
    assert list(spec["required"]) == [
        "plan_json", "room_type", "model", "max_attempts", "draw_camera", "seed", "fail_soft",
    ]
    assert list(spec["optional"]) == ["style_hint"]
    # Сокеты в widgets_values не попадают — их два, значит виджетов ровно шесть.
    assert spec["required"]["plan_json"][1]["forceInput"] is True
    assert spec["optional"]["style_hint"][1]["forceInput"] is True
    assert spec["required"]["fail_soft"][1]["default"] is True

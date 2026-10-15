import pytest
from PIL import Image

from placement_guide import parse_plan, box_px, plan_text, draw_guide, COLORS

PLAN = {"room": "4 x 5 m, ceiling 270 cm", "items": [
    {"n": 1, "name": "Sofa", "colors": "olive velvet", "place": "against the back wall", "facing": "camera",
     "size_cm": [220, 90, 85], "box_2d": [400, 300, 600, 700]},
    {"n": 2, "name": "Lamp", "box_2d": [100, 50, 600, 200]},
]}


# ── разбор ответа планировщика ───────────────────────────────────────
def test_parse_plan_strips_code_fence_and_text():
    raw = "Here is the plan:\n```json\n" + __import__("json").dumps(PLAN) + "\n```\nDone."
    assert parse_plan(raw)["items"][0]["name"] == "Sofa"


@pytest.mark.parametrize("raw", ["", "   ", None, "no json here", "{not json}", '{"room": "x"}', '{"items": []}'])
def test_parse_plan_rejects_bad_input(raw):
    with pytest.raises(ValueError):
        parse_plan(raw)


# ── рамка 0-1000 -> пиксели ──────────────────────────────────────────
def test_box_px_scales_to_image():
    assert box_px([100, 200, 500, 800], 2000, 1000) == (400, 100, 1600, 500)


def test_box_px_swaps_inverted_and_clamps():
    # перевёрнутые края меняются местами, выход за 0-1000 обрезается
    assert box_px([600, 900, -50, 1200], 1000, 1000) == (900, 0, 1000, 600)


# ── план текстом ─────────────────────────────────────────────────────
def test_plan_text_lines():
    t = plan_text(PLAN).split("\n")
    assert t[0] == "Room: 4 x 5 m, ceiling 270 cm"
    assert t[1] == "1. Sofa, olive velvet: against the back wall ; camera ; 220 x 90 x 85 cm"
    assert t[2] == "2. Lamp"


# ── схема ────────────────────────────────────────────────────────────
def test_draw_guide_keeps_size_and_draws_box_edges():
    scene = Image.new("RGB", (1000, 800), (0, 0, 0))
    g = draw_guide(scene, PLAN)
    assert g.size == (1000, 800)
    # левый край рамки предмета 1 (x=300, середина по высоте) - цвет первой рамки
    assert g.getpixel((300, 400)) == COLORS[0]
    # вне рамок сцена осветлена, но не белая
    r, gg, b = g.getpixel((950, 50))
    assert 90 < r < 140 and r == gg == b


def test_draw_guide_skips_item_without_box():
    plan = {"items": [{"n": 1, "name": "Plant"}]}
    g = draw_guide(Image.new("RGB", (200, 100), (0, 0, 0)), plan)
    assert g.size == (200, 100)

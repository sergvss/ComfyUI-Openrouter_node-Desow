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


def test_plan_text_positions_mode():
    # Furnish 2.5: без номеров, место в кадре текстом в процентах
    t = plan_text(PLAN, positions=True).split("\n")
    assert t[1] == ("- Sofa, olive velvet: against the back wall ; camera ; 220 x 90 x 85 cm ; in the photo: "
                    "from 30% to 70% of the width (left to right), from 40% to 60% of the height (top to bottom)")
    assert t[2].startswith("- Lamp: in the photo: from 5% to 20% of the width")
    assert not any(line[:1].isdigit() for line in t)


def test_plan_text_positions_lists_openings():
    plan = dict(PLAN, openings=[{"type": "window", "box_2d": [200, 400, 600, 600]}, {"type": "glass door", "box_2d": [300, 100, 900, 250]}])
    t = plan_text(plan, positions=True).split("\n")
    assert t[1].startswith("FIXED OPENINGS - the room has exactly 1 window(s) and 1 door(s), no others")
    assert t[2] == "- opening 1: window ; in the photo: from 40% to 60% of the width (left to right), from 20% to 60% of the height (top to bottom)"
    assert t[3].startswith("- opening 2: glass door ; in the photo: from 10% to 25%")
    # в режиме Moodboard проёмы не выводятся
    assert "FIXED OPENINGS" not in plan_text(plan)


def test_plan_text_positions_lists_walls_before_openings():
    plan = dict(PLAN, walls=[{"what": "back wall", "box_2d": [100, 330, 700, 650]}, {"what": "bad"}],
                openings=[{"type": "window", "box_2d": [200, 400, 600, 600]}])
    t = plan_text(plan, positions=True).split("\n")
    assert t[1].startswith("FIXED WALLS")
    assert t[2] == "- back wall ; in the photo: from 33% to 65% of the width (left to right), from 10% to 70% of the height (top to bottom)"
    assert t[3].startswith("FIXED OPENINGS")       # стена без рамки пропущена
    assert "FIXED WALLS" not in plan_text(plan)    # Moodboard - без опор


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

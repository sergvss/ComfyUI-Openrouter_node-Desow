"""Схема расстановки для воркфлоу «Moodboard» (NW3): план планировщика -> картинка с рамками + текст плана.

Зачем: генератор (NB2.1) ставит мебель мельче, чем нужно, если размер задан только
сантиметрами в тексте. Когда у каждого предмета есть рамка в кадре, нарисованная на
самой сцене, масштаб заметно реалистичнее (слепой судья: 4 из 6 коллажей лучше,
2026-10-15). Планировщик (gemini-3.8-flash) отдаёт JSON, этот модуль рисует по нему
схему и собирает план текстом для промпта генератора.

Модуль намеренно без ComfyUI и torch (только Pillow) - тестируется вне ComfyUI.

Формат плана (ответ планировщика, допускаются обёртка ```json и текст вокруг):
    {"room": "...", "items": [{"n": 1, "name": "...", "colors": "...", "place": "...",
      "facing": "...", "size_cm": [w, d, h], "box_2d": [ymin, xmin, ymax, xmax]}]}
`box_2d` - рамка в шкале 0-1000 от размеров сцены (формат Gemini).
"""

import json
import re

from PIL import Image, ImageDraw, ImageFont

# Цвета рамок по порядку предметов (RGB); повторяются по кругу.
COLORS = [(230, 0, 0), (0, 160, 0), (0, 0, 230), (230, 140, 0), (180, 0, 180), (0, 170, 170), (120, 70, 0), (60, 60, 60)]
# Насколько осветлить сцену под схемой: рамки читаются, а комната узнаётся.
LIGHTEN = 0.45


def parse_plan(raw):
    """Текст ответа планировщика -> dict плана. Нет JSON или нет предметов - ValueError с причиной."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("placement plan is empty")
    m = re.search(r"\{.*\}", raw, re.S)          # первый «{» .. последний «}»: снимает ```json и текст вокруг
    if not m:
        raise ValueError("placement plan has no JSON object")
    try:
        plan = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError("placement plan is not valid JSON: %s" % e)
    items = plan.get("items") if isinstance(plan, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("placement plan has no items")
    return plan


def box_px(box, width, height):
    """box_2d (0-1000, [ymin, xmin, ymax, xmax]) -> (x0, y0, x1, y1) в пикселях; перевёрнутые края меняются местами, выход за кадр обрезается."""
    y0, x0, y1, x1 = [min(1000.0, max(0.0, float(v))) for v in box[:4]]
    y0, y1 = sorted((y0, y1))
    x0, x1 = sorted((x0, x1))
    return (round(x0 * width / 1000), round(y0 * height / 1000), round(x1 * width / 1000), round(y1 * height / 1000))


def plan_text(plan):
    """План строками для промпта генератора: комната, затем по строке на предмет."""
    lines = []
    if plan.get("room"):
        lines.append("Room: %s" % plan["room"])
    for k, it in enumerate(plan["items"], 1):
        size = it.get("size_cm") or []
        parts = [str(it.get("n", k)) + ". " + str(it.get("name", "item")).strip()]
        if it.get("colors"):
            parts[0] += ", " + str(it["colors"]).strip()
        body = " ; ".join(str(it[f]).strip() for f in ("place", "facing") if it.get(f))
        if size:
            body += (" ; " if body else "") + " x ".join(str(v) for v in size) + " cm"
        lines.append(parts[0] + (": " + body if body else ""))
    return "\n".join(lines)


def _font(size):
    try:
        return ImageFont.load_default(size=size)    # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_guide(scene, plan):
    """Сцена (PIL RGB) + план -> схема: сцена осветлена, у каждого предмета рамка своего цвета и его номер."""
    img = Image.blend(scene.convert("RGB"), Image.new("RGB", scene.size, (255, 255, 255)), LIGHTEN)
    d = ImageDraw.Draw(img)
    w, h = img.size
    line = max(3, w // 400)
    font = _font(max(24, w // 40))
    for k, it in enumerate(plan["items"]):
        box = it.get("box_2d")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue                                  # предмет без рамки - остаётся только в тексте плана
        x0, y0, x1, y1 = box_px(box, w, h)
        c = COLORS[k % len(COLORS)]
        d.rectangle((x0, y0, x1, y1), outline=c, width=line)
        d.text((x0 + line + 4, y0 + line + 2), str(it.get("n", k + 1)), fill=c, font=font)
    return img

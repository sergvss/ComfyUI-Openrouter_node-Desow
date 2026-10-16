"""ComfyUI-обёртка над `placement_guide`: сцена + JSON плана -> схема рамок (IMAGE) и план текстом (STRING)."""

import numpy as np
import torch
from PIL import Image

try:
    from .placement_guide import parse_plan, plan_text, draw_guide
except ImportError:  # импорт модуля вне пакета (тесты, ручная проверка)
    from placement_guide import parse_plan, plan_text, draw_guide


class DesowPlacementGuide:
    """Схема расстановки для «Moodboard»: рамки предметов поверх сцены по плану планировщика.

    Битый план (нет JSON, нет предметов) - ошибка прогона с причиной: генерировать
    без схемы хуже, чем честно упасть до дорогого шага.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("IMAGE",),
                "plan": ("STRING", {"forceInput": True}),
            },
            "optional": {
                # numbered - Moodboard (схема картинкой + план с номерами); positions - Furnish 2.5 (без схемы, место в кадре текстом)
                "text_style": (["numbered", "positions"], {"default": "numbered"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("guide", "plan_text")

    FUNCTION = "build"
    CATEGORY = "Desow/Moodboard"

    def build(self, scene, plan, text_style="numbered"):
        try:
            p = parse_plan(plan)
        except ValueError as e:
            raise RuntimeError("PLACEMENT_PLAN_ERROR: %s" % e)
        # IMAGE в ComfyUI: тензор [B, H, W, C], float 0..1; берём первый кадр
        arr = (scene[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        guide = draw_guide(Image.fromarray(arr[:, :, :3]), p)
        out = torch.from_numpy(np.asarray(guide).astype(np.float32) / 255.0)[None, ...]
        return (out, plan_text(p, positions=(text_style == "positions")))


NODE_CLASS_MAPPINGS = {
    "DesowPlacementGuide": DesowPlacementGuide
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DesowPlacementGuide": "Placement Guide (Desow)"
}

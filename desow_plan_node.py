"""ComfyUI-обёртка над пакетом `desow_plan`. Вся логика — в чистом пакете."""

import io

import numpy as np
import torch
from PIL import Image

try:
    from .desow_plan import blank_png, build_empty_plan, render_camera_png
except ImportError:  # импорт модуля вне пакета (тесты, ручная проверка)
    from desow_plan import blank_png, build_empty_plan, render_camera_png


def _png_to_tensor(png_bytes):
    """PNG-байты -> тензор ComfyUI IMAGE `[1, H, W, 3]`, float32 0..1."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ]


class DesowPlanRender:
    """Рисует пустой 2D-план комнаты кодом: экстракция VLM + проёмы сканера -> чертёж.

    Порядок: разбор JSON экстрактора -> мерж со сканером (состав проёмов — от
    сканера, геометрия — от VLM) -> гейт (нет двери — вставить во front-стену у
    угла, нет окна — по центру front) -> детерминированный рендер.

    Ошибка данных НЕ роняет воркфлоу: на выходе белый лист, пустой `plan_json` и
    причина в `debug`. Скан обязан завершаться даже при сбое плана.

    Выходов четыре. `image` — чистый трёхтоновый план, на нём бэкенд расставляет
    мебель. `plan_camera` — тот же план с маркером точки съёмки (якорь ракурса для
    картиночной модели); отдельным выходом, а не флагом, потому что нужны оба
    одновременно и в одном прогоне.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Ответ VLM-экстрактора структуры (OpenRouter-нода графа).
                "extraction_json": ("STRING", {"forceInput": True}),
            },
            "optional": {
                # Выход детектора проёмов (`output_text_openings` из segments).
                # Пустая строка допустима: план построится по одной экстракции.
                "scanner_openings_json": ("STRING", {"forceInput": True}),
                # Сквозное поле: геометрию не меняет, нужно фазе расстановки.
                "room_type": ("STRING", {"default": "", "multiline": False}),
            },
        }

    # Порядок выходов — часть контракта графов: связи в JSON воркфлоу позиционные,
    # поэтому новый выход дописывается СТРОГО в конец.
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "plan_json", "debug", "plan_camera")

    FUNCTION = "render"
    CATEGORY = "Desow/Plan"

    def render(self, extraction_json, scanner_openings_json="", room_type=""):
        try:
            png, plan_json, debug = build_empty_plan(
                extraction_json, scanner_openings_json or "", (room_type or "").strip()
            )
        except Exception as exc:
            # Страховка на непредвиденное: ожидаемые сбои конвейер разбирает сам,
            # но уронить прогон из-за плана нельзя ни при каких данных.
            png, plan_json = blank_png(), ""
            debug = "plan: ОШИБКА unexpected: %s: %s" % (exc.__class__.__name__, exc)
        return (_png_to_tensor(png), plan_json, debug, _png_to_tensor(render_camera_png(plan_json)))


NODE_CLASS_MAPPINGS = {
    "DesowPlanRender": DesowPlanRender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DesowPlanRender": "Plan Render (Desow)"
}

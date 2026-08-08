"""ComfyUI-обёртка расстановки мебели на 2D-плане. Логика — в `desow_plan`, сеть — в `openrouter_api`."""

import io

import numpy as np
import torch
from PIL import Image

try:
    from .desow_plan import PLAN_FURNISH_MODEL, FurnishFailed, blank_png, build_furnished_plan
    from .openrouter_api import chat_complete
except ImportError:  # импорт модуля вне пакета (тесты, ручная проверка)
    from desow_plan import PLAN_FURNISH_MODEL, FurnishFailed, blank_png, build_furnished_plan
    from openrouter_api import chat_complete

# Расстановка — короткий текстовый вызов: низкая температура, детерминизм важнее
# фантазии (вариативность даёт seed).
FURNISH_TEMPERATURE = 0.4
# Потолок токенов — как в прототипе (hybrid-proto/furnish.py), а НЕ 4000 как у
# бэкенда: `google/gemini-3.6-flash` reasoning-модель, и размышление тратит тот
# же бюджет. Замер 2026-08-08: 3.8-7.1k токенов на reasoning при ~400 символах
# ответа. С лимитом 4000 ответ обрывается на середине массива, парсер честно
# сообщает «не JSON», и нода жжёт ре-промпты на ровном месте.
FURNISH_MAX_TOKENS = 16000
# Сетевые ретраи одного вызова — это НЕ попытки расстановки: первые чинят сбой
# транспорта (429/5xx), вторые чинят содержание ответа (нарушения эргономики).
FURNISH_HTTP_RETRIES = 3
FURNISH_TIMEOUT_S = 120

FAIL_PREFIX = "PLACEMENT_FAILED"


def _png_to_tensor(png_bytes):
    """PNG-байты -> тензор ComfyUI IMAGE `[1, H, W, 3]`, float32 0..1."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ]


class DesowPlanFurnish:
    """Расставляет мебель на готовом 2D-плане: LLM предлагает — код проверяет.

    Порядок: разбор канонического плана -> текстовая модель отдаёт координаты
    предметов -> детерминированный валидатор эргономики (containment, дуга и
    подход двери, полоса перед окном, проходы, простенки) -> список нарушений
    уезжает модели ре-промптом (всего не более `max_attempts` вызовов) ->
    рендер плана с мебелью, при `draw_camera` — с маркером точки съёмки.

    Судья расстановки — код, а не вторая модель: нарушения объяснимы и
    воспроизводимы. Планка качества «правдоподобно, не идеально»: оставшиеся
    после лимита попыток нарушения не отменяют результат, они видны в `debug`.

    `fail_soft=True` (по умолчанию): терминальный сбой не роняет прогон — на
    выходе белый лист, пустой `furniture_json` и причина `PLACEMENT_FAILED: ...`
    в `debug`. Гейт `DesowIsBlank` дальше по графу уведёт генерацию в ветку
    «без плана». `fail_soft=False` — исключение, для отладки.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Канонический план из скана (выход `plan_json` DesowPlanRender):
                # полигон, проёмы, простенки, камера.
                "plan_json": ("STRING", {"forceInput": True}),
                "room_type": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": PLAN_FURNISH_MODEL, "multiline": False}),
                # Число ВЫЗОВОВ модели: 1 — без ре-промптов, 3 — как на бэкенде.
                "max_attempts": ("INT", {"default": 3, "min": 1, "max": 5}),
                "draw_camera": ("BOOLEAN", {"default": True}),
                # Вариативность расстановки между прогонами и обход кеша ComfyUI
                # (тот же граф с тем же seed отдаёт результат из кеша, не платя).
                # control_after_generate выключен: лишнее значение в
                # widgets_values сдвигает маппинг виджетов, как у OpenRouterNode.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": False}),
                "fail_soft": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Краткие пожелания стиля: влияют на СОСТАВ мебели, не на
                # геометрию. Не обязателен — без него расставляется типовой набор.
                "style_hint": ("STRING", {"forceInput": True}),
            },
        }

    # Порядок выходов — часть контракта графов: связи в JSON воркфлоу позиционные,
    # поэтому новый выход дописывается СТРОГО в конец.
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("furnished_plan", "furniture_json", "debug")

    FUNCTION = "furnish"
    CATEGORY = "Desow/Plan"

    def furnish(self, plan_json, room_type="", model=PLAN_FURNISH_MODEL, max_attempts=3,
                draw_camera=True, seed=0, fail_soft=True, style_hint=""):
        model_name = (model or "").strip() or PLAN_FURNISH_MODEL

        def complete(messages):
            """Один вызов текстовой модели. Сеть, ключ и ретраи — общий слой."""
            return chat_complete(
                messages,
                model=model_name,
                temperature=FURNISH_TEMPERATURE,
                max_tokens=FURNISH_MAX_TOKENS,
                seed=seed,
                max_retries=FURNISH_HTTP_RETRIES,
                timeout=FURNISH_TIMEOUT_S,
                log_prefix="DesowPlanFurnish",
            )

        try:
            png, furniture_json, debug = build_furnished_plan(
                plan_json,
                (room_type or "").strip(),
                complete,
                style_hint=style_hint or "",
                max_attempts=max_attempts,
                draw_camera=draw_camera,
                seed=seed,
                model_label=model_name,
            )
        except FurnishFailed as exc:
            if not fail_soft:
                raise
            png, furniture_json = blank_png(), ""
            debug = self._fail_debug(exc.debug_lines, exc.reason)
        except Exception as exc:
            # Страховка на непредвиденное: ожидаемые сбои конвейер разбирает сам,
            # но при fail_soft уронить прогон нельзя ни при каких данных.
            if not fail_soft:
                raise
            png, furniture_json = blank_png(), ""
            debug = self._fail_debug([], "unexpected: %s: %s" % (exc.__class__.__name__, exc))
        return (_png_to_tensor(png), furniture_json, debug)

    @classmethod
    def _fail_debug(cls, lines, reason):
        """Отчёт мягкого отказа: причина первой строкой, накопленный ход — под ней."""
        message = "%s: %s" % (FAIL_PREFIX, reason)
        print("[DesowPlanFurnish] fail_soft=True — прогон не роняем: %s" % message)
        return "\n".join([message] + list(lines))


NODE_CLASS_MAPPINGS = {
    "DesowPlanFurnish": DesowPlanFurnish
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DesowPlanFurnish": "Plan Furnish (Desow)"
}

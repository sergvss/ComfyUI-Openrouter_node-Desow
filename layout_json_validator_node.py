"""ComfyUI-обёртка над `layout_json_validator`. Вся логика — в чистом модуле."""

try:
    from .layout_json_validator import repair_and_validate
except ImportError:  # импорт модуля вне пакета (тесты, ручная проверка)
    from layout_json_validator import repair_and_validate


class LayoutJsonValidator:
    """Чинит и валидирует layout-JSON от VLM; невалидный layout роняет prompt.

    mode:
        strict — количество объектов обязано быть в [10, 22] (основной проход);
        soft   — достаточно одного объекта (уточняющие/частичные ответы).
    label — метка узла в тексте ошибки, чтобы в логе ComfyDeploy было видно,
    какой именно валидатор графа сработал.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout_json": ("STRING", {"forceInput": True}),
                "mode": (["strict", "soft"], {"default": "soft"}),
                "label": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layout_json",)

    FUNCTION = "validate"
    CATEGORY = "Desow/Layout"

    def validate(self, layout_json, mode, label=""):
        # ComfyUI валидирует COMBO до исполнения, так что через граф сюда чужое
        # значение не придёт. Но молчаливый fallback в soft ослабил бы проверку
        # незаметно — а это прямой путь мусорного layout в билдер промпта.
        if mode not in ("strict", "soft"):
            raise ValueError(
                "LayoutJsonValidator: unknown mode {!r} (expected 'strict' or 'soft')".format(mode)
            )
        result = repair_and_validate(
            layout_json, strict_count=(mode == "strict"), label=label
        )
        return (result,)


NODE_CLASS_MAPPINGS = {
    "LayoutJsonValidator": LayoutJsonValidator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayoutJsonValidator": "Layout JSON Validator (Desow)"
}

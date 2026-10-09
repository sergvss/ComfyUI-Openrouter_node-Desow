"""Сторож ошибки OpenRouter: роняет прогон с текстом ошибки, который нода
OpenRouter отдала в мягком режиме (`fail_soft=True`, префикс `OPENROUTER_ERROR`).

Зачем: ComfyDeploy не передаёт текст исключения ни в API прогона, ни в webhook
`failed` - там только статус и выходы нод. Поэтому ошибку сначала выводим
(OpenRouterNode с fail_soft -> showAnything: текст попадает в выходы прогона),
а уже потом этот сторож роняет прогон - статус остаётся `failed`, кредиты
возвращаются, а бэк берёт текст из выходов. Без сторожа fail_soft пустил бы
дальше чёрный кадр, и следующие проходы платили бы за генерацию по нему.
"""

FAIL_PREFIX = "OPENROUTER_ERROR"


# Wildcard-тип: ComfyUI проверяет совместимость через `!=`, подкласс str
# отвечает «совпадает» на любой тип. Тот же приём, что в node.py / is_blank_node.py.
class AnyType(str):
    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


class OpenRouterErrorGuardError(RuntimeError):
    """Прогон уронен сторожем: в тексте - исходная ошибка OpenRouter."""


def check_text(text):
    # Ошибка = строка, начинающаяся с префикса мягкого отказа (после trim)
    if isinstance(text, (list, tuple)):
        text = "".join(str(x) for x in text)
    s = "" if text is None else str(text).strip()
    return s if s.startswith(FAIL_PREFIX) else None


class OpenRouterErrorGuard:
    """Пропускает `value` насквозь; если `text` - мягкая ошибка OpenRouter, роняет прогон."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Текст-выход OpenRouterNode (лучше через showAnything, чтобы ошибка
                # гарантированно попала в выходы прогона ДО падения)
                "text": ("STRING", {"forceInput": True}),
                # Любое значение, которое идёт дальше по графу (картинка или текст)
                "value": (ANY_TYPE,),
            }
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "guard"
    CATEGORY = "OpenRouter"

    def guard(self, text, value):
        err = check_text(text)
        if err:
            raise OpenRouterErrorGuardError(err)
        return (value,)


NODE_CLASS_MAPPINGS = {"OpenRouterErrorGuard": OpenRouterErrorGuard}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenRouterErrorGuard": "OpenRouter error guard (Desow)"}

"""Сторож ошибки OpenRouter: роняет прогон с текстом ошибки, который нода
OpenRouter отдала в мягком режиме (`fail_soft=True`, префикс `OPENROUTER_ERROR`).

Зачем: ComfyDeploy не передаёт текст исключения ни в API прогона, ни в webhook
`failed` - там только статус и выходы нод. Поэтому сторож сначала публикует
ошибку событием `executed` (текст попадает в выходы прогона), а потом роняет
прогон - статус остаётся `failed`, кредиты возвращаются, а бэк берёт текст из
выходов. Без сторожа fail_soft пустил бы дальше чёрный кадр, и следующие
проходы платили бы за генерацию по нему.
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


def publish_error(node_id, err):
    """Отправить текст ошибки событием `executed` от этой ноды, как будто она вывела текст.

    Плагин ComfyDeploy записывает в выходы прогона каждое `executed` (кроме PreviewImage),
    а обычные выходные ноды для этого не годятся: ComfyUI исполняет их всегда, и в
    ленивых ветках они тянули бы платный вызов модели. Вне ComfyUI (тесты) - тихо пропускаем.
    """
    try:
        from server import PromptServer
        ps = PromptServer.instance
        ps.send_sync("executed", {"node": str(node_id), "display_node": str(node_id), "output": {"text": [err]},
                                  "prompt_id": getattr(ps, "last_prompt_id", None)}, getattr(ps, "client_id", None))
        return True
    except Exception as exc:  # отсутствие сервера не должно подменять исходную ошибку
        print("[OpenRouterErrorGuard] не удалось опубликовать ошибку: %s" % exc)
        return False


class OpenRouterErrorGuard:
    """Пропускает `value` насквозь; если `text` - мягкая ошибка OpenRouter, публикует её
    в выходы прогона и роняет прогон с тем же текстом."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Текст-выход Output ноды OpenRouterNode с fail_soft=True
                "text": ("STRING", {"forceInput": True}),
                # Любое значение, которое идёт дальше по графу (картинка или текст)
                "value": (ANY_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "guard"
    CATEGORY = "OpenRouter"

    def guard(self, text, value, unique_id=None):
        err = check_text(text)
        if err:
            publish_error(unique_id, err)
            raise OpenRouterErrorGuardError(err)
        return (value,)


NODE_CLASS_MAPPINGS = {"OpenRouterErrorGuard": OpenRouterErrorGuard}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenRouterErrorGuard": "OpenRouter error guard (Desow)"}

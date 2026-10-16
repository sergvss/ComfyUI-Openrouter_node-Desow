"""ComfyUI-обёртка над `judge_verdict`: ответ судьи -> ok (BOOLEAN) и problems (STRING)."""

try:
    from .judge_verdict import parse_verdict
except ImportError:  # импорт модуля вне пакета (тесты, ручная проверка)
    from judge_verdict import parse_verdict


class DesowJudgeVerdict:
    """Вердикт судьи геометрии: `ok` для ленивого ветвления (повтор только при нарушении), `problems` - текст для повтора.

    Неразборчивый ответ - ok=True (судья не роняет прогон и не запускает лишний платный повтор).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "verdict": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("ok", "problems")

    FUNCTION = "parse"
    CATEGORY = "Desow/Furnish"

    def parse(self, verdict):
        return parse_verdict(verdict)


NODE_CLASS_MAPPINGS = {
    "DesowJudgeVerdict": DesowJudgeVerdict
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DesowJudgeVerdict": "Judge Verdict (Desow)"
}

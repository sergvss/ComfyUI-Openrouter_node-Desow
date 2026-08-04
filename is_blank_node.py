"""ComfyUI-обёртка над `is_blank`. Вся логика — в чистом модуле."""

try:
    from .is_blank import is_blank
except ImportError:  # импорт модуля вне пакета (тесты, ручная проверка)
    from is_blank import is_blank


# Wildcard-тип входа: ComfyUI проверяет совместимость через `!=`, а Python отдаёт
# приоритет методу подкласса str, поэтому любое соединение проходит проверку.
# Тот же приём, что `AnyType` в node.py; дублируется намеренно — node.py тянет
# torch/requests/tiktoken, а эта обёртка обязана импортироваться без них.
class AnyType(str):
    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


class DesowIsBlank:
    """`True`, если на входе «ничего нет»: None, пробельная строка, `null` /
    `none` / `undefined` / `nan`, пустой `{}` / `[]` (текстом или объектом).

    Ноль (`0`, `0.0`) и `False` считаются НЕ пустыми — осознанное отличие от
    `easy isNone`. Полный список правил и мотивация — в докстринге `is_blank`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY_TYPE,),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_blank",)

    FUNCTION = "check"
    CATEGORY = "Desow/Logic"

    def check(self, value):
        return (is_blank(value),)


NODE_CLASS_MAPPINGS = {
    "DesowIsBlank": DesowIsBlank
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DesowIsBlank": "Is Blank (Desow)"
}

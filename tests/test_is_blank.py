import json

import pytest

from is_blank import is_blank


# ── пусто: None и пробельные строки ──────────────────────────────────
@pytest.mark.parametrize("value", [None, "", " ", "   ", "\n", "\t", " \r\n\t "])
def test_none_and_whitespace_are_blank(value):
    assert is_blank(value) is True


# ── пусто: строки-заглушки ───────────────────────────────────────────
@pytest.mark.parametrize(
    "value",
    ["null", "NULL", "Null", "none", "None", " None ", "undefined", "UNDEFINED", "nan", "NaN"],
)
def test_placeholder_strings_are_blank(value):
    assert is_blank(value) is True


# ── пусто: пустые JSON-контейнеры текстом ────────────────────────────
@pytest.mark.parametrize("value", ["{}", "[]", "{ }", "[ ]", " {} ", "[\n]", "{\t}", "\n[]\n"])
def test_empty_json_containers_as_text_are_blank(value):
    assert is_blank(value) is True


# ── пусто: пустые контейнеры объектом ────────────────────────────────
@pytest.mark.parametrize("value", [[], {}, (), set(), frozenset()])
def test_empty_containers_are_blank(value):
    assert is_blank(value) is True


# ── НЕ пусто: ноль и булевы ──────────────────────────────────────────
@pytest.mark.parametrize("value", [0, 0.0, -0.0, 1, -1, 0.5])
def test_numbers_are_not_blank(value):
    # Осознанное отличие от `easy isNone` (там `any == 0` ⇒ пусто): ноль сплошь
    # и рядом валидное значение (upscale=0, creativity=0), а не «значения нет».
    assert is_blank(value) is False


@pytest.mark.parametrize("value", [False, True])
def test_booleans_are_not_blank(value):
    # `False == 0`, поэтому в `easy isNone` он пустой. У нас это ответ «нет».
    assert is_blank(value) is False


def test_float_nan_is_not_blank():
    # Пуста только СТРОКА "nan"; настоящий float('nan') — число.
    assert is_blank(float("nan")) is False


# ── НЕ пусто: строки ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value",
    ["0", "false", "no", "-", "n/a", "nullable", "none of them", "{", "[", "{}{}", "[] []"],
)
def test_non_empty_strings_are_not_blank(value):
    assert is_blank(value) is False


def test_json_with_empty_objects_array_is_not_blank():
    # Контейнер не пуст: пустой внутри — уже забота валидатора layout, не ветвления.
    assert is_blank('{"objects": []}') is False


def test_real_layout_json_is_not_blank():
    payload = {"objects": [{"name": "sofa", "position": "center of the room", "wall": "back"}]}
    assert is_blank(json.dumps(payload, ensure_ascii=False)) is False


def test_arbitrary_text_is_not_blank():
    assert is_blank("Here is the layout: nothing found") is False


# ── НЕ пусто: непустые контейнеры и прочие объекты ───────────────────
@pytest.mark.parametrize("value", [[""], {"a": 1}, (0,), {0}])
def test_non_empty_containers_are_not_blank(value):
    assert is_blank(value) is False


def test_unknown_object_is_not_blank():
    # Про чужой объект (тензор, модель) ничего не известно — «не пусто»
    # безопаснее, чем молча увести граф в ветку «значения нет».
    assert is_blank(object()) is False

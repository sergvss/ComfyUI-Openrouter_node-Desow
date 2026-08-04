"""Починка + валидация канонического layout-JSON внутри графа ComfyUI.

Порт правил `layout_generation._parse_vlm_json` из бэкенда Desow: два валидатора
обязаны браковать одно и то же, иначе layout, принятый воркфлоу, развалится на
бэкенде (и наоборот). Модуль намеренно чистый stdlib — без ComfyUI, torch,
folder_paths, чтобы его можно было импортировать и тестировать вне ComfyUI.

Отличие от бэкенда только в tolerance-слое (см. `_repair`) и в том, что нижняя
граница количества объектов включается флагом `strict_count`.
"""

import json
import re

# Паритет с `layout_generation._ALLOWED_WALLS` / `routers/layouts.ALLOWED_FURNITURE_WALLS`:
# `wall` — поверхность размещения, а не только вертикальная стена.
_ALLOWED_WALLS = {"back", "front", "left", "right", "center", "ceiling", "floor"}

# Приёмочный диапазон для strict-режима (паритет с бэкендом).
_MIN_OBJECTS = 10
_MAX_OBJECTS = 22

_PREVIEW_LIMIT = 200

# Регексы портированы из `layout_generation` без изменений.
_MARKDOWN_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)
# Известное ограничение (разделяется с бэкендом): регекс не различает контекст,
# поэтому строковый литерал вида "a, }" будет испорчен. Сужать нельзя —
# расхождение с `layout_generation._TRAILING_COMMA_RE` опаснее этого кейса.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_markdown_fence(content):
    """Снимает ```json ... ``` обёртку если есть, иначе возвращает как есть."""
    stripped = content.strip()
    m = _MARKDOWN_FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def _extract_json_object(text):
    """Срезает preamble/постскриптум модели: от первой `{` до последней `}`.

    Срез безусловный: для чистого `{...}` это no-op, а для «{...} Note: ...»
    (постскриптум без преамбулы) — единственный способ его отрезать.
    Шире бэкенда: в графе нет retry-слоя, и «Here is the layout: {...} Note: ...»
    не повод ронять прогон.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in content")
    return text[start:end + 1]


def _repair(raw):
    """Tolerance-слой: фенсы → срез до JSON-объекта → висячие запятые."""
    text = _strip_markdown_fence(raw)
    text = _extract_json_object(text)
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _validate(cleaned, strict_count):
    """Порт `_parse_vlm_json`: парсит и проверяет канон, нормализует на месте."""
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON: {}".format(exc)) from exc

    if not isinstance(data, dict):
        raise ValueError("Expected dict, got {}".format(type(data).__name__))
    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Missing or invalid 'objects' array")
    if strict_count:
        if not (_MIN_OBJECTS <= len(objects) <= _MAX_OBJECTS):
            raise ValueError(
                "objects count {} out of range [{}, {}]".format(
                    len(objects), _MIN_OBJECTS, _MAX_OBJECTS
                )
            )
    elif not objects:
        raise ValueError("objects count 0 out of range [1, inf)")

    # Ссылки `on` резолвим вторым проходом: вперёд-ссылка легальна (аксессуар
    # может стоять в списке раньше своей опоры), поэтому копим их по пути.
    names = set()
    on_refs = []  # (index, name, on)

    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError("objects[{}] is not an object".format(i))

        name = obj.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("objects[{}] missing 'name'".format(i))
        name = name.strip()
        obj["name"] = name
        if name in names:
            raise ValueError(
                "objects[{}] duplicate 'name': {!r} "
                "(names must be unique, 'on' resolves objects by name)".format(i, name)
            )
        names.add(name)

        position = obj.get("position")
        if not isinstance(position, str) or not position.strip():
            raise ValueError("objects[{}] missing 'position'".format(i))

        # Не-строка или пустая строка = «поле не задано»: модели любят слать null
        # для неиспользуемых опциональных полей.
        raw_wall = obj.get("wall")
        wall = raw_wall.strip().lower() if isinstance(raw_wall, str) else ""
        raw_on = obj.get("on")
        on = raw_on.strip() if isinstance(raw_on, str) else ""
        if not wall and not on:
            raise ValueError("objects[{}] requires 'wall' or 'on'".format(i))

        if wall:
            if wall not in _ALLOWED_WALLS:
                raise ValueError("objects[{}] invalid 'wall': {!r}".format(i, raw_wall))
            obj["wall"] = wall
        elif "wall" in obj:
            # Пустой/null `wall` рядом с валидным `on` — выкидываем ключ, чтобы
            # наружу уходил чистый канон без `"wall": null`.
            del obj["wall"]

        if on:
            obj["on"] = on
            on_refs.append((i, name, on))
        elif "on" in obj:
            del obj["on"]

        # `style` и `group` опциональны: проходят как есть.

    for i, name, on in on_refs:
        if on == name:
            raise ValueError("objects[{}] {!r}: 'on' references itself".format(i, name))
        if on not in names:
            raise ValueError(
                "objects[{}] {!r}: 'on' references unknown object {!r} "
                "(must be the exact 'name' of another object in the list)".format(i, name, on)
            )

    return data


def repair_and_validate(raw, *, strict_count, label=""):
    """Чинит, валидирует и канонизирует layout-JSON.

    Args:
        raw: сырой ответ VLM (возможно в markdown-фенсе, с preamble, с висячей
            запятой).
        strict_count: True → количество объектов обязано быть в
            [`_MIN_OBJECTS`, `_MAX_OBJECTS`]; False → достаточно одного объекта.
        label: метка узла графа, попадает в начало текста ошибки.

    Returns:
        Канонический JSON-текст (нормализованные `wall`/`on`/`name`).

    Raises:
        ValueError: любая ошибка починки или валидации, включая не-строку на
            входе. Fail-loud: воркфлоу должен падать, а не тащить битый layout
            дальше по графу.
    """
    # Тип проверяем до `_repair`: иначе не-строка даёт AttributeError мимо
    # `except ValueError` ниже, и в логе теряются label и raw_preview.
    if not isinstance(raw, str):
        raise ValueError(
            _format_error(
                "Expected string input, got {}".format(type(raw).__name__), label, raw
            )
        )
    try:
        data = _validate(_repair(raw), strict_count)
    except ValueError as exc:
        raise ValueError(_format_error(str(exc), label, raw)) from exc
    return json.dumps(data, ensure_ascii=False)


def _format_error(message, label, raw):
    """`[label] message | raw_preview='...'` — админ читает это в логе ComfyDeploy."""
    prefix = "[{}] ".format(label.strip()) if label and label.strip() else ""
    preview = raw[:_PREVIEW_LIMIT] if isinstance(raw, str) else str(raw)[:_PREVIEW_LIMIT]
    return "{}{} | raw_preview={!r}".format(prefix, message, preview)

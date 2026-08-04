import json

import pytest

from layout_json_validator import repair_and_validate


def _obj(name, **kw):
    obj = {"name": name, "position": "center of the room"}
    obj.update(kw)
    return obj


def _layout(count, wall="back"):
    """Layout из `count` однотипных объектов с уникальными именами."""
    return {"objects": [_obj("item {}".format(i), wall=wall) for i in range(count)]}


def _dumps(payload):
    return json.dumps(payload, ensure_ascii=False)


# ── happy path ───────────────────────────────────────────────────────
def test_strict_happy_path_14_objects():
    objects = [_obj("item {}".format(i), wall="left") for i in range(11)]
    objects.append(_obj("chandelier", wall="ceiling"))
    objects.append(_obj("area rug", wall="floor"))
    objects.append(_obj("table lamp", on="item 0", style="brass"))

    result = json.loads(repair_and_validate(_dumps({"objects": objects}), strict_count=True))

    assert len(result["objects"]) == 14
    assert result["objects"][-1]["on"] == "item 0"
    assert result["objects"][-1]["style"] == "brass"
    assert {o.get("wall") for o in result["objects"]} == {"left", "ceiling", "floor", None}


def test_markdown_fence_is_stripped():
    raw = "```json\n" + _dumps(_layout(12)) + "\n```"
    assert len(json.loads(repair_and_validate(raw, strict_count=True))["objects"]) == 12


def test_preamble_and_postscript_are_cut():
    raw = "Here is the layout: " + _dumps(_layout(12)) + " Note: adjust as needed."
    assert len(json.loads(repair_and_validate(raw, strict_count=True))["objects"]) == 12


def test_postscript_without_preamble_is_cut():
    # Срез до первой `{` / последней `}` безусловный: раньше fast-path на
    # `startswith("{")` пропускал такой хвост в json.loads и валил прогон.
    raw = _dumps(_layout(12)) + " Note: adjust as needed."
    assert len(json.loads(repair_and_validate(raw, strict_count=True))["objects"]) == 12


def test_clean_object_passes_through_unchanged():
    # Регресс на безусловный срез: чистый `{...}` без мусора должен давать
    # ровно тот же канон, что и до правки.
    payload = _layout(12)
    assert repair_and_validate(_dumps(payload), strict_count=True) == _dumps(payload)


def test_trailing_comma_is_repaired():
    raw = _dumps(_layout(12)).replace("}]}", "},]}")
    assert "},]}" in raw
    assert len(json.loads(repair_and_validate(raw, strict_count=True))["objects"]) == 12


# ── нормализация ─────────────────────────────────────────────────────
def test_wall_is_lowercased():
    payload = {"objects": [_obj("sofa", wall="LEFT"), _obj("lamp", wall=" Back ")]}
    result = json.loads(repair_and_validate(_dumps(payload), strict_count=False))
    assert [o["wall"] for o in result["objects"]] == ["left", "back"]


def test_null_wall_with_valid_on_drops_the_key():
    payload = {"objects": [_obj("sofa", wall="back"), _obj("lamp", wall=None, on="sofa")]}
    result = json.loads(repair_and_validate(_dumps(payload), strict_count=False))
    assert "wall" not in result["objects"][1]
    assert result["objects"][1]["on"] == "sofa"


def test_forward_reference_in_on_is_legal():
    payload = {"objects": [_obj("lamp", on="table"), _obj("table", wall="center")]}
    result = json.loads(repair_and_validate(_dumps(payload), strict_count=False))
    assert result["objects"][0]["on"] == "table"


# ── ошибки схемы ─────────────────────────────────────────────────────
def test_unknown_wall_value_fails():
    payload = {"objects": [_obj("sofa", wall="north")]}
    with pytest.raises(ValueError, match="invalid 'wall'"):
        repair_and_validate(_dumps(payload), strict_count=False)


def test_missing_both_wall_and_on_fails():
    payload = {"objects": [_obj("sofa")]}
    with pytest.raises(ValueError, match="requires 'wall' or 'on'"):
        repair_and_validate(_dumps(payload), strict_count=False)


def test_missing_name_fails():
    payload = {"objects": [{"position": "by the window", "wall": "back"}]}
    with pytest.raises(ValueError, match="missing 'name'"):
        repair_and_validate(_dumps(payload), strict_count=False)


def test_missing_position_fails():
    payload = {"objects": [{"name": "sofa", "wall": "back"}]}
    with pytest.raises(ValueError, match="missing 'position'"):
        repair_and_validate(_dumps(payload), strict_count=False)


def test_duplicate_name_fails():
    payload = {"objects": [_obj("sofa", wall="back"), _obj("sofa", wall="left")]}
    with pytest.raises(ValueError, match="duplicate 'name'"):
        repair_and_validate(_dumps(payload), strict_count=False)


def test_on_referencing_unknown_object_fails():
    payload = {"objects": [_obj("lamp", on="ghost table")]}
    with pytest.raises(ValueError, match="references unknown object"):
        repair_and_validate(_dumps(payload), strict_count=False)


def test_on_self_reference_fails():
    payload = {"objects": [_obj("lamp", on="lamp")]}
    with pytest.raises(ValueError, match="references itself"):
        repair_and_validate(_dumps(payload), strict_count=False)


def test_garbage_input_fails():
    with pytest.raises(ValueError, match="No JSON object found"):
        repair_and_validate("I cannot help with that request.", strict_count=False)


def test_broken_json_fails():
    with pytest.raises(ValueError, match="Invalid JSON"):
        repair_and_validate('{"objects": [{"name": }]}', strict_count=False)


def test_truncated_output_without_closing_brace_fails():
    # Оборванная генерация: закрывающей `}` нет, срез не находит объект.
    # Бэкенд на том же входе даёт "Invalid JSON" — вердикт тот же, текст другой.
    with pytest.raises(ValueError, match="No JSON object found"):
        repair_and_validate('{"objects": [{"name": ', strict_count=False)


def test_top_level_list_fails():
    # Массив не начинается с `{` → срез до первого объекта, дальше нет 'objects'.
    with pytest.raises(ValueError, match="Missing or invalid 'objects'"):
        repair_and_validate('[{"name": "sofa"}]', strict_count=False)


def test_missing_objects_key_fails():
    with pytest.raises(ValueError, match="Missing or invalid 'objects'"):
        repair_and_validate('{"items": []}', strict_count=False)


# ── счётчик объектов ─────────────────────────────────────────────────
@pytest.mark.parametrize("strict", [True, False])
def test_empty_objects_fails_in_both_modes(strict):
    with pytest.raises(ValueError, match="out of range"):
        repair_and_validate('{"objects": []}', strict_count=strict)


def test_nine_objects_strict_fails_soft_passes():
    raw = _dumps(_layout(9))
    with pytest.raises(ValueError, match=r"count 9 out of range \[10, 22\]"):
        repair_and_validate(raw, strict_count=True)
    assert len(json.loads(repair_and_validate(raw, strict_count=False))["objects"]) == 9


def test_twentythree_objects_strict_fails_soft_passes():
    raw = _dumps(_layout(23))
    with pytest.raises(ValueError, match=r"count 23 out of range \[10, 22\]"):
        repair_and_validate(raw, strict_count=True)
    assert len(json.loads(repair_and_validate(raw, strict_count=False))["objects"]) == 23


# ── диагностика ошибки ───────────────────────────────────────────────
def test_label_and_raw_preview_in_error_message():
    raw = "totally not json, sorry"
    with pytest.raises(ValueError) as exc:
        repair_and_validate(raw, strict_count=False, label="furnish/pass1")
    message = str(exc.value)
    assert message.startswith("[furnish/pass1] ")
    assert "raw_preview=" in message
    assert raw in message


def test_raw_preview_is_truncated_to_200_chars():
    raw = "x" * 500
    with pytest.raises(ValueError) as exc:
        repair_and_validate(raw, strict_count=False)
    preview = str(exc.value).split("raw_preview=", 1)[1]
    assert len(preview.strip("'\"")) == 200


def test_error_without_label_has_no_prefix():
    with pytest.raises(ValueError) as exc:
        repair_and_validate("nope", strict_count=False)
    assert not str(exc.value).startswith("[")


# ── не-строка на входе ───────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, 42, 3.5, ["{}"], {"objects": []}, b"{}"])
def test_non_string_input_raises_value_error(bad):
    # Не AttributeError из `.strip()`: иначе ошибка минует `_format_error`.
    with pytest.raises(ValueError, match="Expected string input"):
        repair_and_validate(bad, strict_count=False)


def test_non_string_input_keeps_label_and_preview():
    with pytest.raises(ValueError) as exc:
        repair_and_validate(None, strict_count=False, label="furnish/pass1")
    message = str(exc.value)
    assert message.startswith("[furnish/pass1] ")
    assert "got NoneType" in message
    assert "raw_preview=" in message


# ── контракт ноды ────────────────────────────────────────────────────
def test_node_contract():
    from layout_json_validator_node import LayoutJsonValidator, NODE_CLASS_MAPPINGS

    assert NODE_CLASS_MAPPINGS["LayoutJsonValidator"] is LayoutJsonValidator
    required = LayoutJsonValidator.INPUT_TYPES()["required"]
    assert list(required.keys()) == ["layout_json", "mode", "label"]
    assert required["mode"][0] == ["strict", "soft"]
    assert LayoutJsonValidator.RETURN_TYPES == ("STRING",)
    assert LayoutJsonValidator.FUNCTION == "validate"

    payload = _dumps({"objects": [_obj("sofa", wall="BACK")]})
    (out,) = LayoutJsonValidator().validate(payload, "soft", "")
    assert json.loads(out)["objects"][0]["wall"] == "back"


@pytest.mark.parametrize("bad_mode", ["Strict", "STRICT", "whatever", "", None])
def test_node_rejects_unknown_mode(bad_mode):
    # Тихий fallback в soft недопустим: ослабление проверки должно быть громким.
    from layout_json_validator_node import LayoutJsonValidator

    payload = _dumps(_layout(12))
    with pytest.raises(ValueError, match="unknown mode"):
        LayoutJsonValidator().validate(payload, bad_mode, "")


def test_node_strict_mode_enforces_count():
    from layout_json_validator_node import LayoutJsonValidator

    node = LayoutJsonValidator()
    raw = _dumps(_layout(9))
    with pytest.raises(ValueError, match=r"count 9 out of range"):
        node.validate(raw, "strict", "")
    (out,) = node.validate(raw, "soft", "")
    assert len(json.loads(out)["objects"]) == 9

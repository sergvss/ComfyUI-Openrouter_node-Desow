import pytest

from judge_verdict import parse_verdict


def test_ok_true():
    assert parse_verdict('{"ok": true, "problems": ""}') == (True, "")


def test_ok_false_with_problems_in_code_fence():
    raw = 'Result:\n```json\n{"ok": false, "problems": "a new window on the right wall"}\n```'
    assert parse_verdict(raw) == (False, "a new window on the right wall")


def test_ok_true_drops_problems_text():
    # при ok=true текст проблем не нужен повтору и не возвращается
    assert parse_verdict('{"ok": true, "problems": "minor shadow"}') == (True, "")


@pytest.mark.parametrize("raw", [None, "", "OPENROUTER_ERROR: 429", "no json", "{bad json}", '{"problems": "x"}', '{"ok": "no"}', "[]"])
def test_unreadable_counts_as_ok(raw):
    # судья - страховка: неразборчивый ответ не роняет прогон и не запускает повтор
    assert parse_verdict(raw) == (True, "")

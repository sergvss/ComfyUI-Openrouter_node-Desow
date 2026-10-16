"""json_mode: текстовая модель получает response_format json_object (планировщик Furnish 2.5); image-моделям не уходит."""
import pytest

from test_node_fail_soft import _call
from test_node_prompt_echo import _FakeResponse, node_mod

TEXT_MODEL = "google/gemini-3.8-flash"
IMAGE_MODEL = "google/gemini-nano-banana-2.1"


@pytest.fixture
def node():
    return node_mod.OpenRouterNode.__new__(node_mod.OpenRouterNode)


@pytest.fixture
def sent(monkeypatch):
    """Перехватывает тело запроса; отвечает текстом JSON."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    box = {}

    def post(url, json=None, **kw):
        box["body"] = json
        return _FakeResponse({"choices": [{"message": {"role": "assistant", "content": '{"items": []}'}}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(node_mod.requests, "post", post)
    monkeypatch.setattr(node_mod.requests, "get", lambda *a, **kw: _FakeResponse({"data": {"total_credits": 5.0, "total_usage": 1.0}}))
    return box


def test_widget_is_last_and_off_by_default():
    spec = node_mod.OpenRouterNode.INPUT_TYPES()["optional"]
    assert list(spec)[-1] == "json_mode"
    assert spec["json_mode"][1]["default"] is False


def test_text_model_gets_response_format(node, sent):
    _call(node, model=TEXT_MODEL, json_mode=True)
    assert sent["body"]["response_format"] == {"type": "json_object"}


def test_off_by_default_no_response_format(node, sent):
    _call(node, model=TEXT_MODEL)
    assert "response_format" not in sent["body"]


def test_image_model_never_gets_response_format(node, sent):
    # заглушка отвечает текстом без картинки - image-ветка ноды падает уже ПОСЛЕ отправки; проверяем тело запроса
    try:
        _call(node, model=IMAGE_MODEL, json_mode=True)
    except RuntimeError:
        pass
    assert sent["body"]["model"] == IMAGE_MODEL
    assert "response_format" not in sent["body"]

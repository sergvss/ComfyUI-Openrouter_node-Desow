"""W6: отказ Gemini по содержанию (IMAGE_RECITATION и т.п.) не повторяется и падает с понятной причиной.

Отказ стабилен: тот же вход даёт тот же отказ при любом seed и температуре, поэтому
повторы только тратят ~1 мин. Пустой ответ без причины по-прежнему повторяется.
"""
import pytest

from test_node_fail_soft import _call
from test_node_prompt_echo import _FakeResponse, node_mod

IMAGE_MODEL = "google/gemini-nano-banana-2.1"


def _empty_image_body(native=None, finish="stop"):
    choice = {"message": {"role": "assistant", "content": "", "images": []}, "finish_reason": finish}
    if native:
        choice["native_finish_reason"] = native
    return {"choices": [choice], "usage": {"prompt_tokens": 10, "completion_tokens": 0}}


@pytest.fixture
def node():
    return node_mod.OpenRouterNode.__new__(node_mod.OpenRouterNode)


@pytest.fixture
def api(monkeypatch):
    """Подменяет сеть: каждый POST отдаёт `api.body`, число вызовов - в `api.calls`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setattr(node_mod, "_retry_sleep", lambda *a, **kw: None)

    class Api:
        body = None
        calls = 0

    def post(*a, **kw):
        Api.calls += 1
        return _FakeResponse(Api.body)

    monkeypatch.setattr(node_mod.requests, "post", post)
    monkeypatch.setattr(node_mod.requests, "get",
                        lambda *a, **kw: _FakeResponse({"data": {"total_credits": 5.0, "total_usage": 1.0}}))
    return Api


@pytest.mark.parametrize("native", ["IMAGE_RECITATION", "IMAGE_SAFETY", "PROHIBITED_CONTENT"])
def test_content_refusal_is_not_retried(node, api, native):
    api.body = _empty_image_body(native)
    with pytest.raises(RuntimeError) as excinfo:
        _call(node, model=IMAGE_MODEL, max_retries=3)
    assert api.calls == 1
    assert native in str(excinfo.value)


def test_content_filter_finish_reason_is_not_retried(node, api):
    api.body = _empty_image_body(finish="content_filter")
    with pytest.raises(RuntimeError) as excinfo:
        _call(node, model=IMAGE_MODEL, max_retries=3)
    assert api.calls == 1
    assert "content_filter" in str(excinfo.value)


def test_refusal_reaches_fail_soft_text(node, api):
    api.body = _empty_image_body("IMAGE_RECITATION")
    result = _call(node, model=IMAGE_MODEL, max_retries=3, fail_soft=True)
    assert result[0].startswith("OPENROUTER_ERROR:")
    assert "IMAGE_RECITATION" in result[0]


def test_empty_answer_without_reason_is_still_retried(node, api):
    api.body = _empty_image_body()
    with pytest.raises(RuntimeError):
        _call(node, model=IMAGE_MODEL, max_retries=3)
    assert api.calls == 4

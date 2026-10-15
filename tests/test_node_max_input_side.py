"""max_input_side: картинки уменьшаются перед отправкой в API (быстрее ответ), а размер ответа «auto»
по-прежнему считается по исходному image_1 (большое фото -> 4K). 0 = выключено (прежнее поведение)."""
import base64
import io
import json

import pytest

from test_node_fail_soft import _call
from test_node_prompt_echo import _FakeResponse, node_mod

torch = pytest.importorskip("torch")
if not hasattr(torch, "zeros") or isinstance(getattr(torch, "float32", None), str):
    pytest.skip("в sys.modules заглушка torch, а не настоящий пакет", allow_module_level=True)
from PIL import Image  # noqa: E402

IMAGE_MODEL = "google/gemini-nano-banana-2.1"


@pytest.fixture
def node():
    return node_mod.OpenRouterNode.__new__(node_mod.OpenRouterNode)


@pytest.fixture
def sent(monkeypatch):
    """Перехватывает тело запроса; отвечает картинкой, чтобы прогон дошёл до конца."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    box = {}

    def post(url, json=None, **kw):
        box["body"] = json
        return _FakeResponse({"choices": [{"message": {"role": "assistant", "content": "", "images": [{"image_url": {"url": data_url}}]}}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(node_mod.requests, "post", post)
    monkeypatch.setattr(node_mod.requests, "get", lambda *a, **kw: _FakeResponse({"data": {"total_credits": 5.0, "total_usage": 1.0}}))
    return box


def sent_image_size(body):
    for block in body["messages"][-1]["content"]:
        if block.get("type") == "image_url":
            raw = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
            return Image.open(io.BytesIO(raw)).size
    return None


def test_widget_is_appended_last_and_off_by_default():
    spec = node_mod.OpenRouterNode.INPUT_TYPES()["optional"]
    assert list(spec)[-1] == "max_input_side"
    assert spec["max_input_side"][1]["default"] == 0


def test_big_image_is_downscaled_but_size_detected_from_original(node, sent):
    big = torch.rand(1, 2448, 3264, 3)
    _call(node, model=IMAGE_MODEL, image_1=big, max_input_side=2048, aspect_ratio="auto", image_resolution="auto")
    assert sent_image_size(sent["body"]) == (2048, 1536)
    assert sent["body"]["image_config"]["image_size"] == "4K"        # как у исходного 3264 px


def test_off_keeps_original_size(node, sent):
    img = torch.rand(1, 600, 800, 3)
    _call(node, model=IMAGE_MODEL, image_1=img, max_input_side=0)
    assert sent_image_size(sent["body"]) == (800, 600)


def test_small_image_not_upscaled(node, sent):
    img = torch.rand(1, 600, 800, 3)
    _call(node, model=IMAGE_MODEL, image_1=img, max_input_side=2048)
    assert sent_image_size(sent["body"]) == (800, 600)

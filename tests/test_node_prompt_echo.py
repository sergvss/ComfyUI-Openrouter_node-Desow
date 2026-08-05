"""Тесты контракта выходов OpenRouterNode и диагностических эхо-выходов.

Проверяем два инварианта:
  1) слоты 0-3 (Output/image/Stats/Credits) не сдвинулись — на них позиционно
     ссылаются связи во всех сохранённых воркфлоу;
  2) слоты 4-5 отдают ФАКТИЧЕСКИ отправленные system/user промпты — то есть уже
     после compat-шима сдвига виджетов и после парсера маркеров [[IMGn]].

node.py тянет torch/tiktoken, которых нет вне ComfyUI, поэтому недостающие модули
подменяются лёгкими стабами, а сам модуль грузится под синтетическим пакетом
(папка ноды содержит дефис -> обычным импортом относительный `.chat_manager`
не резолвится).
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_NODE_DIR = Path(__file__).resolve().parents[1]


# --- Стабы тяжёлых зависимостей (только если реальных нет) ---
if importlib.util.find_spec("torch") is None:
    _torch = types.ModuleType("torch")
    _torch.float32 = "float32"

    class _FakeTensor:  # для isinstance-проверок в node.py
        pass

    _torch.Tensor = _FakeTensor
    _torch.zeros = lambda *args, **kwargs: _FakeTensor()
    sys.modules["torch"] = _torch

if importlib.util.find_spec("tiktoken") is None:
    # Пустой модуль: count_tokens поймает AttributeError и уйдёт в оценку по символам
    sys.modules["tiktoken"] = types.ModuleType("tiktoken")


# --- Загрузка node.py как части синтетического пакета ---
_PKG = "_openrouter_node_under_test"
if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [str(_NODE_DIR)]
    sys.modules[_PKG] = _pkg

_spec = importlib.util.spec_from_file_location(f"{_PKG}.node", _NODE_DIR / "node.py")
node_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = node_mod
_spec.loader.exec_module(node_mod)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, (dict, list)):
            return self._payload
        raise json.JSONDecodeError("not json", self.text, 0)

    def raise_for_status(self):
        return None


def _ok_body(text="hi"):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


@pytest.fixture
def node():
    # __new__ вместо конструктора: ChatSessionManager в __init__ создаёт папку chats/
    return node_mod.OpenRouterNode.__new__(node_mod.OpenRouterNode)


@pytest.fixture
def api(monkeypatch):
    """Мокает сеть и отдаёт словарь с перехваченным payload запроса."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    box = {"payload": None, "response": _FakeResponse(_ok_body())}

    def fake_post(url, headers=None, json=None, timeout=None):
        box["payload"] = json
        return box["response"]

    monkeypatch.setattr(node_mod.requests, "post", fake_post)
    monkeypatch.setattr(
        node_mod.requests, "get",
        lambda *a, **kw: _FakeResponse({"data": {"total_credits": 5.0, "total_usage": 1.0}}),
    )
    return box


def _call(node, **overrides):
    kwargs = dict(
        system_prompt="SYS PROMPT",
        user_message_box="USER BOX",
        model="anthropic/claude-3-haiku",
        web_search=False,
        cheapest=False,
        fastest=False,
        seed=1,
        temperature=1.0,
        pdf_engine="auto",
        chat_mode=False,
        max_retries=0,
    )
    kwargs.update(overrides)
    return node.generate_response(**kwargs)


# --- Контракт выходов ---

def test_slots_0_3_unchanged_and_new_outputs_appended():
    cls = node_mod.OpenRouterNode
    assert cls.RETURN_TYPES[:4] == ("STRING", "IMAGE", "STRING", "STRING")
    assert cls.RETURN_NAMES[:4] == ("Output", "image", "Stats", "Credits")
    assert cls.RETURN_TYPES[4:] == ("STRING", "STRING")
    assert cls.RETURN_NAMES[4:] == ("SystemPromptUsed", "UserPromptUsed")
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES) == 6


def test_result_arity_matches_return_types(node, api):
    result = _call(node)
    assert len(result) == len(node_mod.OpenRouterNode.RETURN_TYPES)


# --- Эхо на успешном пути ---

def test_text_only_echoes_sent_prompts(node, api):
    result = _call(node)
    system_used, user_used = result[4], result[5]
    assert isinstance(system_used, str) and isinstance(user_used, str)
    assert system_used == "SYS PROMPT"
    assert user_used == "USER BOX"
    # эхо совпадает с тем, что реально ушло в payload
    messages = api["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "SYS PROMPT"}
    assert messages[1]["content"] == "USER BOX"


def test_linked_user_message_input_wins_over_widget(node, api):
    result = _call(node, user_message_input="FROM LINK")
    assert result[5] == "FROM LINK"
    assert api["payload"]["messages"][1]["content"] == "FROM LINK"


def test_blank_user_message_input_falls_back_to_widget(node, api):
    result = _call(node, user_message_input="   ")
    assert result[5] == "USER BOX"


def test_echo_is_post_compat_shim(node, api):
    """Легаси-воркфлоу с api_key в widgets_values[0] сдвигает все виджеты на 1.
    Эхо обязано отдать значения ПОСЛЕ восстановления, а не сырые аргументы."""
    result = _call(
        node,
        system_prompt="sk-or-v1-legacy-key",   # мусор от старого виджета
        user_message_box="REAL SYS",           # сюда сдвинулся system_prompt
        model="REAL USER",                     # сюда сдвинулся user_message_box
        web_search="anthropic/claude-3-haiku",  # сюда сдвинулась модель
        cheapest=False,
        fastest=False,
        seed=False,
        temperature=7,
        pdf_engine=1.0,
        chat_mode="auto",
        max_retries=False,
    )
    assert result[4] == "REAL SYS"
    assert result[5] == "REAL USER"
    assert api["payload"]["messages"][0]["content"] == "REAL SYS"
    assert api["payload"]["model"] == "anthropic/claude-3-haiku"


def test_multimodal_echo_keeps_order_and_labels_without_base64(node, api, monkeypatch):
    monkeypatch.setattr(
        node_mod.OpenRouterNode, "image_to_base64", staticmethod(lambda img: "FAKEBASE64"),
    )
    result = _call(
        node,
        user_message_input="MAIN TEXT\n[[IMG1]] source room\n[[IMG2]] style reference",
        image_1=object(),
        image_2=object(),
    )
    assert result[5] == (
        "MAIN TEXT\n"
        "source room\n"
        "[IMAGE image_1]\n"
        "style reference\n"
        "[IMAGE image_2]"
    )
    assert "FAKEBASE64" not in result[5]
    # порядок в эхо совпадает с порядком блоков в реальном запросе
    blocks = api["payload"]["messages"][1]["content"]
    assert [b["type"] for b in blocks] == ["text", "text", "image_url", "text", "image_url"]
    assert blocks[0]["text"] == "MAIN TEXT"
    assert blocks[1]["text"] == "source room"
    assert blocks[3]["text"] == "style reference"


def test_markers_without_images_are_echoed_raw(node, api):
    """Без подключённых картинок запрос уходит простой строкой, и маркеры [[IMGn]]
    остаются в промпте как есть — эхо это честно показывает (полезно для отладки)."""
    text = "MAIN TEXT\n[[IMG1]] source room"
    result = _call(node, user_message_input=text)
    assert result[5] == text
    assert api["payload"]["messages"][1]["content"] == text


def test_pdf_echo_uses_placeholder(node, api):
    result = _call(node, pdf_data={"filename": "plan.pdf", "bytes": b"%PDF-1.4 fake"})
    assert result[5] == "USER BOX\n[PDF plan.pdf]"


def test_chat_mode_without_system_message_echoes_empty_string(node, api):
    """В chat_mode системного сообщения в сессии может не быть — тогда в API оно
    не уходит и эхо честно пустое (а не копия виджета)."""
    class _FakeChatManager:
        def get_or_create_session(self, user_text, system_prompt):
            return "/tmp/session", [{"role": "user", "content": "previous turn"}]

        def save_conversation(self, path, messages):
            self.saved = messages

    node.chat_manager = _FakeChatManager()
    result = _call(node, chat_mode=True)
    assert result[4] == ""
    assert all(m["role"] != "system" for m in api["payload"]["messages"])


# --- Ошибочные пути ---

def test_non_retryable_http_error_still_raises_runtime_error(node, api):
    api["response"] = _FakeResponse({"error": "bad request"}, status_code=400)
    with pytest.raises(RuntimeError) as exc:
        _call(node)
    assert "rejected request" in str(exc.value)


def test_retryable_http_error_exhausted_raises_runtime_error(node, api):
    api["response"] = _FakeResponse({"error": "boom"}, status_code=503)
    with pytest.raises(RuntimeError):
        _call(node, max_retries=0)


def test_missing_api_key_raises_before_request(node, api, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        _call(node)
    assert "OPENROUTER_API_KEY" in str(exc.value)
    assert api["payload"] is None

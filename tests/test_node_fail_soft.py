"""Тесты виджета `fail_soft` OpenRouterNode.

Инварианты:
  1) виджет добавлен СТРОГО в конец — widgets_values сохранённых воркфлоу мапится
     позиционно, вставка в середину сдвинула бы значения во всех боевых графах;
  2) fail_soft=False — прежнее поведение (исключение), байт-в-байт;
  3) fail_soft=True — терминальная ошибка вызова API отдаётся выходами, прогон
     не падает: ветка плана в segments не имеет права уносить с собой скан.

Загрузчик node.py со стабами torch/tiktoken переиспользуется из соседнего модуля,
чтобы не держать две копии одной и той же обвязки.
"""
import pytest

from test_node_prompt_echo import _FakeResponse, _ok_body, node_mod


# Порядок виджетов = порядок widgets_values в сохранённых воркфлоу. Первые 11
# зафиксированы боевыми графами (segments, furnish-2, redesign-2 и т.д.);
# fail_soft — 12-й, добавленный. Ломать этот список нельзя: любая перестановка
# переклеит значения виджетов во всех графах разом.
EXPECTED_WIDGETS = [
    "system_prompt", "user_message_box", "model", "web_search", "cheapest",
    "fastest", "seed", "temperature", "pdf_engine", "chat_mode", "max_retries",
    "fail_soft",
    # append-only: гейт пропуска (скан, «комната уже пустая») - виджет строго
    # ПОСЛЕ fail_soft; gate_text - forceInput-сокет, в виджеты не входит.
    "gate_skip_value",
]
WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}


def widget_names():
    """Имена виджетов ноды в порядке ComfyUI: required, затем optional, без сокетов."""
    spec = node_mod.OpenRouterNode.INPUT_TYPES()
    names = []
    for section in ("required", "optional"):
        for name, decl in (spec.get(section) or {}).items():
            kind = decl[0]
            options = decl[1] if len(decl) > 1 else {}
            if isinstance(kind, list):                       # COMBO (model, pdf_engine)
                names.append(name)
            elif isinstance(kind, str) and kind in WIDGET_TYPES and not options.get("forceInput"):
                names.append(name)
    return names


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


@pytest.fixture
def node():
    return node_mod.OpenRouterNode.__new__(node_mod.OpenRouterNode)


@pytest.fixture
def dead_api(monkeypatch):
    """Сеть, которая всегда падает: терминальная ошибка после исчерпания ретраев."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    def boom(*args, **kwargs):
        raise node_mod.requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(node_mod.requests, "post", boom)
    monkeypatch.setattr(node_mod.requests, "get", boom)


# --- Позиция виджета ---

def test_fail_soft_is_the_last_widget():
    assert widget_names() == EXPECTED_WIDGETS


def test_fail_soft_defaults_to_false():
    # Дефолт False = все существующие воркфлоу сохраняют fail-loud поведение,
    # даже если их пересохранят с новым виджетом.
    decl = node_mod.OpenRouterNode.INPUT_TYPES()["optional"]["fail_soft"]
    assert decl[0] == "BOOLEAN" and decl[1]["default"] is False


def test_old_workflow_widget_values_map_to_same_widgets():
    """11 значений старого графа ложатся на первые 11 виджетов без сдвига."""
    saved = ["SYSTEM", "USER", "google/gemini-3.6-flash", False, False, False, 0, 0, "auto", False, 3]
    names = widget_names()
    assert len(saved) < len(names)                      # новый виджет только дописан
    assert names[:len(saved)] == EXPECTED_WIDGETS[:len(saved)]
    assert names[len(saved)] == "fail_soft"             # ему достанется дефолт


# --- Поведение ---

def test_fail_soft_false_still_raises(node, dead_api):
    with pytest.raises(RuntimeError):
        _call(node)


def test_fail_soft_true_returns_error_tuple(node, dead_api):
    result = _call(node, fail_soft=True)
    assert len(result) == len(node_mod.OpenRouterNode.RETURN_TYPES)
    assert result[0].startswith("OPENROUTER_ERROR:")
    assert "connection refused" in result[0] or "ConnectionError" in result[0]
    assert result[2] == result[3] == result[4] == result[5] == ""   # без silent-fallback


def test_fail_soft_true_covers_errors_outside_inner_try(node, monkeypatch):
    """Отсутствующий ключ падает ДО внутреннего try/except — мягкий отказ обязан ловить и это."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    result = _call(node, fail_soft=True)
    assert result[0].startswith("OPENROUTER_ERROR:")
    with pytest.raises(RuntimeError):
        _call(node, fail_soft=False)


def test_fail_soft_ignored_on_legacy_shifted_workflow(node, dead_api):
    """Легаси-сдвиг кладёт в fail_soft значение max_retries старого графа (int 3).

    Само по себе это включило бы мягкий отказ в воркфлоу, который его не просил,
    поэтому на признаке сдвига fail_soft гасится.

    Класс исключения тут не фиксируем: сдвинутый набор виджетов разваливается
    раньше сетевого вызова. Значимо ровно одно — прогон падает, а не возвращает
    мягкий кортеж.
    """
    with pytest.raises(Exception) as excinfo:
        _call(node, system_prompt="sk-or-v1-legacy-key", fail_soft=3)
    assert "OPENROUTER_ERROR" not in str(excinfo.value)


def test_success_path_untouched_with_fail_soft_true(node, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setattr(node_mod.requests, "post",
                        lambda *a, **kw: _FakeResponse(_ok_body("hello")))
    monkeypatch.setattr(node_mod.requests, "get",
                        lambda *a, **kw: _FakeResponse({"data": {"total_credits": 5.0, "total_usage": 1.0}}))
    result = _call(node, fail_soft=True)
    assert result[0] == "hello"
    assert not result[0].startswith("OPENROUTER_ERROR")
    assert result[4] == "SYS PROMPT"        # диагностические эхо-выходы на месте


def test_fail_soft_image_is_valid_black_tensor():
    torch = pytest.importorskip("torch", reason="torch есть только внутри ComfyUI")
    if not hasattr(torch, "zeros") or not hasattr(torch, "float32") or isinstance(torch.float32, str):
        pytest.skip("в sys.modules заглушка torch, а не настоящий пакет")
    _, image, *_ = node_mod.OpenRouterNode._soft_failure(RuntimeError("boom"))
    assert tuple(image.shape) == (1, 64, 64, 3)          # валидный IMAGE, не вырожденный 1x1
    assert str(image.dtype) == "torch.float32"
    assert float(image.abs().max()) == 0.0               # чёрный кадр

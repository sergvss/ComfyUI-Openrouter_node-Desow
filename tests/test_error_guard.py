"""Сторож ошибки OpenRouter: пропускает значение, роняет прогон на тексте с префиксом OPENROUTER_ERROR."""
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "error_guard_node", pathlib.Path(__file__).resolve().parents[1] / "error_guard_node.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


@pytest.mark.parametrize("text", ["Room type: Bedroom", "", None, "1", "GATE_SKIPPED: yes", "error: OPENROUTER_ERROR not at start"])
def test_passes_value_through(text):
    value = object()
    assert G.OpenRouterErrorGuard().guard(text, value) == (value,)


@pytest.mark.parametrize("text", [
    "OPENROUTER_ERROR: RuntimeError: OpenRouter API failed after 3 retries: HTTP 429: {}",
    "  OPENROUTER_ERROR: OpenRouterCallError: Provider error 429 (rate_limit): quota",
    ["OPENROUTER_ERROR: RuntimeError: ", "image-model returned no images"],  # showAnything отдаёт список
])
def test_raises_with_original_text(text):
    with pytest.raises(G.OpenRouterErrorGuardError) as e:
        G.OpenRouterErrorGuard().guard(text, "x")
    assert str(e.value).startswith("OPENROUTER_ERROR")


def test_publishes_error_as_executed_event(monkeypatch):
    # Ошибка уходит событием executed от самой ноды-сторожа - так её записывает ComfyDeploy
    import sys
    import types
    sent = []

    class PS:
        last_prompt_id = "p1"
        client_id = "c1"

        def send_sync(self, event, data, sid=None):
            sent.append((event, data, sid))
    PS.instance = PS()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=PS))
    with pytest.raises(G.OpenRouterErrorGuardError):
        G.OpenRouterErrorGuard().guard("OPENROUTER_ERROR: RuntimeError: HTTP 429", "x", unique_id="42")
    assert sent == [("executed", {"node": "42", "display_node": "42", "output": {"text": ["OPENROUTER_ERROR: RuntimeError: HTTP 429"]}, "prompt_id": "p1"}, "c1")]


def test_no_event_without_error(monkeypatch):
    import sys
    import types
    sent = []

    class PS:
        def send_sync(self, *a, **k):
            sent.append(a)
    PS.instance = PS()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=PS))
    assert G.OpenRouterErrorGuard().guard("ok", "x", unique_id="42") == ("x",)
    assert sent == []


def test_registered_under_stable_name():
    # Имя класса ноды = class_type в графах: менять нельзя
    assert "OpenRouterErrorGuard" in G.NODE_CLASS_MAPPINGS

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


def test_registered_under_stable_name():
    # Имя класса ноды = class_type в графах: менять нельзя
    assert "OpenRouterErrorGuard" in G.NODE_CLASS_MAPPINGS

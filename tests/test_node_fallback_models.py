"""Запасные модели: NB2.1 -> NB2 списком `models` (OpenRouter сам переключает при ошибке/лимите основной)."""
import importlib
import pathlib
import sys
import types

import pytest


def _mod():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "from_numpy"):
        pytest.skip("в sys.modules заглушка torch, а не настоящий пакет")
    if "ornode_pkg" not in sys.modules:
        pkg = types.ModuleType("ornode_pkg")
        pkg.__path__ = [str(pathlib.Path(__file__).resolve().parents[1])]
        sys.modules["ornode_pkg"] = pkg
    return importlib.import_module("ornode_pkg.node")


@pytest.mark.parametrize("model,expected", [
    ("google/gemini-nano-banana-2.1", ["google/gemini-nano-banana-2.1", "google/gemini-3.1-flash-image-preview"]),
    ("google/gemini-nano-banana-2.1:nitro", ["google/gemini-nano-banana-2.1:nitro", "google/gemini-3.1-flash-image-preview:nitro"]),
    ("google/gemini-3.6-flash", None),
    ("anthropic/claude-haiku-5.5", None),
    ("", None),
    (None, None),
])
def test_fallback_models(model, expected):
    assert _mod()._fallback_models(model) == expected

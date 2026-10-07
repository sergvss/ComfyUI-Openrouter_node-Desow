"""Признак image-модели и расширенные пропорции: Nano Banana 2.1 (google/gemini-nano-banana-2.1)
не содержит "image" в id - без этого нода не шлёт modalities и не ждёт картинку."""
import importlib
import pathlib
import sys
import types

import pytest


def _cls():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "from_numpy"):
        pytest.skip("в sys.modules заглушка torch, а не настоящий пакет")
    # node.py использует относительные импорты (.chat_manager, .openrouter_api) - грузим его как
    # модуль пакета-псевдонима, не выполняя тяжёлый __init__.py ноды
    if "ornode_pkg" not in sys.modules:
        pkg = types.ModuleType("ornode_pkg")
        pkg.__path__ = [str(pathlib.Path(__file__).resolve().parents[1])]
        sys.modules["ornode_pkg"] = pkg
    return importlib.import_module("ornode_pkg.node").OpenRouterNode


@pytest.mark.parametrize("model,expected", [
    ("google/gemini-nano-banana-2.1", True),
    ("google/gemini-3.1-flash-image-preview", True),
    ("google/gemini-3-pro-image-preview", True),
    ("google/gemini-3.1-flash-image-preview:nitro", True),
    ("google/gemini-3.6-flash", False),
    ("anthropic/claude-haiku-4.5", False),
    ("", False),
    (None, False),
])
def test_is_image_model(model, expected):
    assert _cls()._is_image_model(model) is expected


def test_nano_banana_21_gets_extended_aspect_ratios():
    cls = _cls()
    # Панорама 4:1 - есть только в расширенном наборе
    assert cls._detect_aspect_and_size(4000, 1000, "google/gemini-nano-banana-2.1")[0] == "4:1"
    assert cls._detect_aspect_and_size(4000, 1000, "google/gemini-3.1-flash-image-preview")[0] == "4:1"
    # У Pro расширенного набора нет - ближайшее из базового
    assert cls._detect_aspect_and_size(4000, 1000, "google/gemini-3-pro-image-preview")[0] == "21:9"


def test_size_thresholds_unchanged():
    cls = _cls()
    assert cls._detect_aspect_and_size(1600, 1200, "google/gemini-nano-banana-2.1")[1] == "1K"
    assert cls._detect_aspect_and_size(2800, 2100, "google/gemini-nano-banana-2.1")[1] == "2K"
    assert cls._detect_aspect_and_size(2900, 2175, "google/gemini-nano-banana-2.1")[1] == "4K"

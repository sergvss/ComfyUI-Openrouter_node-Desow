"""Тесты гейта пропуска OpenRouterNode (скан: «комната уже пустая - declutter
не нужен»). Скип не ходит в сеть - тестируется офлайн."""
import sys

import pytest


def _node():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "from_numpy"):
        pytest.skip("в sys.modules заглушка torch, а не настоящий пакет")
    from node import OpenRouterNode
    return OpenRouterNode(), torch


def test_gate_skip_passes_image_through(monkeypatch):
    node, torch = _node()
    called = {"n": 0}
    monkeypatch.setattr(node, "_generate_response",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    img = torch.rand((1, 32, 48, 3))
    out = node.generate_response(
        "sys", "user", "google/gemini-3.1-flash-lite-image",
        False, False, False, 0, 1.0, "auto", False, 3,
        image_1=img, fail_soft=True,
        gate_text="  Empty \n", gate_skip_value="empty")
    assert called["n"] == 0                      # API не вызывался
    assert out[0].startswith("GATE_SKIPPED")
    assert out[1] is img                         # оригинал насквозь
    assert out[2] == out[3] == ""


def test_gate_mismatch_calls_api(monkeypatch):
    node, torch = _node()
    sentinel = ("ok", torch.zeros((1, 8, 8, 3)), "", "", "", "")
    monkeypatch.setattr(node, "_generate_response", lambda *a, **k: sentinel)
    out = node.generate_response(
        "sys", "user", "m", False, False, False, 0, 1.0, "auto", False, 3,
        gate_text="furnished", gate_skip_value="empty")
    assert out == sentinel                       # гейт не совпал - обычный путь


def test_gate_disabled_when_widget_empty(monkeypatch):
    node, torch = _node()
    sentinel = ("ok", torch.zeros((1, 8, 8, 3)), "", "", "", "")
    monkeypatch.setattr(node, "_generate_response", lambda *a, **k: sentinel)
    # Пустой gate_skip_value = гейт выключен, даже если gate_text подключен.
    out = node.generate_response(
        "sys", "user", "m", False, False, False, 0, 1.0, "auto", False, 3,
        gate_text="", gate_skip_value="")
    assert out == sentinel

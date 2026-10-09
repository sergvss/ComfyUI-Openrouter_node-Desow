"""FalUpscale: загрузка -> очередь -> скачивание кусками; повторы на 429; ошибка публикуется с префиксом OPENROUTER_ERROR."""
import importlib.util
import io
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "from_numpy"):
    pytest.skip("в sys.modules заглушка torch, а не настоящий пакет", allow_module_level=True)
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("fal_upscale_node", ROOT / "fal_upscale_node.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def _png(w, h):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(b, "PNG")
    return b.getvalue()


class Resp:
    def __init__(self, status=200, js=None, content=b"", headers=None, text=""):
        self.status_code, self._js, self.content, self.headers, self.text = status, js, content, headers or {}, text

    def json(self):
        return self._js


def fake_fal(monkeypatch, result_png, fail_first_submit=0):
    calls = []
    state = {"submit_fail": fail_first_submit}

    def request(method, url, headers=None, timeout=None, **kw):
        calls.append((method, url, dict(headers or {})))
        if url.startswith(M.UPLOAD_INIT):
            return Resp(js={"upload_url": "https://up/put", "file_url": "https://cdn/in.png"})
        if url == "https://up/put":
            return Resp()
        if url.startswith(M.QUEUE) and method == "POST":
            if state["submit_fail"]:
                state["submit_fail"] -= 1
                return Resp(429, text="rate limit")
            assert kw["json"]["image_url"] == "https://cdn/in.png" and kw["json"]["upscale_factor"] == 2.0
            return Resp(js={"status_url": "https://q/status", "response_url": "https://q/resp"})
        if url == "https://q/status":
            return Resp(js={"status": "COMPLETED"})
        if url == "https://q/resp":
            return Resp(js={"image": {"url": "https://cdn/out.png"}})
        if url == "https://cdn/out.png":
            rng = (headers or {}).get("Range")
            a, b = [int(x) for x in rng.split("=")[1].split("-")]
            return Resp(206, content=result_png[a:b + 1], headers={"Content-Range": "bytes %d-%d/%d" % (a, b, len(result_png))})
        raise AssertionError(url)

    monkeypatch.setattr(M.requests, "request", request)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    return calls


def test_upscale_roundtrip(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    monkeypatch.setenv("FAL_DL_CHUNK", "100")  # много кусков - проверяем склейку
    calls = fake_fal(monkeypatch, _png(8, 6))
    out, = M.FalUpscale().upscale(torch.zeros((1, 3, 4, 3)), 2.0, unique_id="7")
    assert tuple(out.shape) == (1, 6, 8, 3)
    assert abs(float(out[0, 0, 0, 0]) - 10 / 255) < 1e-6
    # ключ уходит только в API fal, не в хранилище и не в CDN
    assert all(("Authorization" in h) == (u.startswith(M.UPLOAD_INIT) or u.startswith(M.QUEUE) or u.startswith("https://q/")) for _, u, h in calls)


def test_retries_on_429(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    fake_fal(monkeypatch, _png(4, 4), fail_first_submit=2)
    out, = M.FalUpscale().upscale(torch.zeros((1, 2, 2, 3)), 2.0)
    assert tuple(out.shape) == (1, 4, 4, 3)


def test_error_published_and_raised(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_API_KEY", raising=False)
    sent = []
    monkeypatch.setattr(M, "publish_error", lambda nid, err: sent.append((nid, err)))
    with pytest.raises(M.FalUpscaleError) as e:
        M.FalUpscale().upscale(torch.zeros((1, 2, 2, 3)), 2.0, unique_id="9")
    assert str(e.value).startswith("OPENROUTER_ERROR: FalUpscaleError: FAL_KEY")
    assert sent and sent[0][0] == "9" and sent[0][1].startswith("OPENROUTER_ERROR")


def test_registered_name():
    assert "FalUpscale" in M.NODE_CLASS_MAPPINGS

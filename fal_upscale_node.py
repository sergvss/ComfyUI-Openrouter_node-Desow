"""Апскейл через fal.ai (по умолчанию SeedVR2) - честный 8K: Gemini отдаёт максимум 4K, нода увеличивает x2.

Ключ - только из переменной окружения `FAL_KEY` (на ComfyDeploy - секрет машины). Картинка уходит в хранилище
fal, задача ставится в очередь, результат скачивается кусками Range (CDN fal из части сетей отдаёт мало за
одно соединение). При ошибке текст публикуется так же, как у сторожа OpenRouter (префикс OPENROUTER_ERROR -
бэк кладёт его в generation.message для админов), и прогон падает.
"""
import io
import os
import time

import numpy as np
import requests
import torch
from PIL import Image

try:
    from .error_guard_node import FAIL_PREFIX, publish_error
except ImportError:  # загрузка вне пакета (тесты)
    from error_guard_node import FAIL_PREFIX, publish_error

QUEUE = "https://queue.fal.run/"
UPLOAD_INIT = "https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class FalUpscaleError(RuntimeError):
    pass


def _key():
    k = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not k:
        raise FalUpscaleError("FAL_KEY environment variable is not set")
    return k


def _request(method, url, key=None, retries=3, **kw):
    # Повторы на 429/5xx и сетевых ошибках с нарастающей паузой (5, 15, 30 с)
    headers = kw.pop("headers", {})
    if key:
        headers["Authorization"] = "Key " + key
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.request(method, url, headers=headers, timeout=kw.pop("timeout", 120), **kw)
            if r.status_code not in RETRY_STATUSES:
                return r
            last = "HTTP %s: %s" % (r.status_code, r.text[:300])
        except requests.RequestException as e:
            last = "network: %s" % e
        if attempt < retries:
            time.sleep((5, 15, 30)[min(attempt, 2)])
    raise FalUpscaleError("fal request failed after %d retries: %s" % (retries, last))


def upload(data, content_type, key):
    # Загрузка в хранилище fal: initiate -> PUT по выданной ссылке -> публичный file_url
    r = _request("POST", UPLOAD_INIT, key, json={"content_type": content_type, "file_name": "input.png"})
    if r.status_code >= 400:
        raise FalUpscaleError("fal upload init failed: HTTP %s: %s" % (r.status_code, r.text[:300]))
    j = r.json()
    p = _request("PUT", j["upload_url"], None, data=data, headers={"Content-Type": content_type}, timeout=300)
    if p.status_code >= 400:
        raise FalUpscaleError("fal upload failed: HTTP %s: %s" % (p.status_code, p.text[:300]))
    return j["file_url"]


def run_queue(endpoint, args, key, timeout=900, poll=3):
    # Постановка в очередь и ожидание результата
    r = _request("POST", QUEUE + endpoint, key, json=args)
    if r.status_code >= 400:
        raise FalUpscaleError("fal %s rejected request: HTTP %s: %s" % (endpoint, r.status_code, r.text[:300]))
    q = r.json()
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(poll)
        s = _request("GET", q["status_url"], key)
        if s.status_code >= 400:
            raise FalUpscaleError("fal status failed: HTTP %s: %s" % (s.status_code, s.text[:300]))
        st = s.json().get("status")
        if st == "COMPLETED":
            res = _request("GET", q["response_url"], key)
            if res.status_code >= 400:
                raise FalUpscaleError("fal %s failed: HTTP %s: %s" % (endpoint, res.status_code, res.text[:300]))
            return res.json()
    raise FalUpscaleError("fal %s timeout after %d s" % (endpoint, timeout))


def result_url(resp):
    img = resp.get("image") or (resp.get("images") or [None])[0]
    if not isinstance(img, dict) or not img.get("url"):
        raise FalUpscaleError("fal response without image: %s" % str(resp)[:300])
    return img["url"]


def download(url, chunk=None, workers=16):
    # Скачивание кусками Range параллельно; размер куска - FAL_DL_CHUNK (байт), по умолчанию 1 МБ
    from concurrent.futures import ThreadPoolExecutor
    chunk = int(chunk or os.getenv("FAL_DL_CHUNK") or 1_048_576)
    head = _request("GET", url, None, headers={"Range": "bytes=0-0"}, timeout=60)
    cr = head.headers.get("Content-Range")
    if head.status_code != 206 or not cr:  # сервер без Range - качаем целиком
        full = _request("GET", url, None, timeout=600)
        return full.content
    total = int(cr.split("/")[1])

    def part(start):
        r = _request("GET", url, None, headers={"Range": "bytes=%d-%d" % (start, min(start + chunk, total) - 1)}, timeout=120)
        return r.content

    with ThreadPoolExecutor(workers) as ex:
        data = b"".join(ex.map(part, range(0, total, chunk)))
    if len(data) != total:
        raise FalUpscaleError("fal download incomplete: %d of %d bytes" % (len(data), total))
    return data


def tensor_to_png(t):
    a = (t.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def png_to_tensor(data):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)


class FalUpscale:
    """Увеличить картинку через fal (SeedVR2 по умолчанию). Пакет кадров обрабатывается по одному."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "endpoint": ("STRING", {"default": "fal-ai/seedvr/upscale/image"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "OpenRouter"

    def upscale(self, image, upscale_factor=2.0, endpoint="fal-ai/seedvr/upscale/image", unique_id=None):
        try:
            key = _key()
            out = []
            for i in range(image.shape[0]):
                url = upload(tensor_to_png(image[i]), "image/png", key)
                resp = run_queue(endpoint, {"image_url": url, "upscale_factor": float(upscale_factor), "output_format": "png"}, key)
                out.append(png_to_tensor(download(result_url(resp))))
            return (torch.stack(out, dim=0),)
        except Exception as exc:
            err = "%s: %s: %s" % (FAIL_PREFIX, exc.__class__.__name__, str(exc)[:300])
            publish_error(unique_id, err)
            raise FalUpscaleError(err) from exc


NODE_CLASS_MAPPINGS = {"FalUpscale": FalUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"FalUpscale": "fal.ai Upscale (Desow)"}

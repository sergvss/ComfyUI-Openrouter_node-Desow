"""ComfyUI-обёртка описи объектов (`desow_objects`). Логика - в чистом модуле."""

import hashlib
import json

try:
    from .desow_objects import encode_objects
except ImportError:  # импорт вне пакета (тесты, ручная проверка)
    from desow_objects import encode_objects


class DesowObjectsEncode:
    """Опись объектов скана из ансамбля Gemini-детекций - замена SAM3-ветки.

    Три текстовых входа - ответы трёх прогонов детекции (OpenRouter-ноды,
    gemini-3.5-flash, seed 0/1/2). score = ансамблевое согласие votes/3.
    Настройки классов (thresholds/excludes/sizes) - те же external-входы
    графа, что получала SAM3-нода: админка продолжает управлять сканом.

    Выход `objects_json` - формат sam3EncodeResultsToText без mask_b64
    (маски строятся on-demand при модификации, эпик SCAN_NO_SAM3 Ф3).
    Ошибка данных не роняет скан: пустая опись + причина в debug.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_json": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "detection2_json": ("STRING", {"forceInput": True}),
                "detection3_json": ("STRING", {"forceInput": True}),
                "thresholds_json": ("STRING", {"forceInput": True}),
                "excludes_json": ("STRING", {"forceInput": True}),
                "sizes_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("objects_json", "debug")

    FUNCTION = "encode"
    CATEGORY = "Desow/Scan"

    def encode(self, image, detection_json, detection2_json="", detection3_json="",
               thresholds_json="", excludes_json="", sizes_json=""):
        try:
            # image: тензор ComfyUI [B, H, W, C].
            h, w = int(image.shape[1]), int(image.shape[2])
            try:
                image_hash = hashlib.sha256(
                    image.detach().cpu().numpy().tobytes()).hexdigest()
            except Exception:
                image_hash = ""
            texts = [t for t in (detection_json, detection2_json, detection3_json) if t]
            payload, notes = encode_objects(
                texts, w, h, image_hash,
                thresholds_json or "", excludes_json or "", sizes_json or "")
            return (json.dumps(payload, ensure_ascii=False), "\n".join(notes))
        except Exception as exc:
            # Опись не имеет права уронить скан: пустой результат + причина.
            empty = {"version": 1, "image_w": 0, "image_h": 0, "image_hash": "",
                     "id_grid": 8, "objects": []}
            return (json.dumps(empty),
                    "objects: ОШИБКА %s: %s" % (exc.__class__.__name__, exc))


NODE_CLASS_MAPPINGS = {
    "DesowObjectsEncode": DesowObjectsEncode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DesowObjectsEncode": "Objects Encode (Desow)"
}

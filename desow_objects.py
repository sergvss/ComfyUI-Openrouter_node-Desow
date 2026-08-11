# КАНОНИЧЕСКИЙ КОД. Пары в бэкенде нет: опись объектов скана собирает только
# воркфлоу. Замена SAM3-ветки segments (эпик docs/design/SCAN_NO_SAM3.md, Ф2).
"""Опись объектов скана из ансамбля Gemini-детекций - без GPU.

Схема (канон SCAN_GEMINI_PIPELINE_RESEARCH.md + тест эмуляции score 2026-08-11):
три прогона детекции (gemini-3.5-flash, exhaustive-промпт) -> кластеризация
одинаковых находок (класс + IoU>0.5) -> score = votes/3 (ансамблевое согласие;
самооценка модели не дискриминирует - все 0.8-0.95, проверено) -> та же
CPU-обвязка, что жила в SAM3-ноде (порты из ComfyUI-Easy-Sam3-Desow/nodes.py):
смарт-дедуп родственных лейблов и синонимов, иерархические исключения
(Wardrobe глотает Shelving внутри себя), минимальные размеры. Настройки
классов (thresholds/excludes/sizes) продолжают приходить из админки теми же
входами графа - управление сканом у админки не отбирается.

Формат выхода 1:1 с sam3EncodeResultsToText, МИНУС mask_b64 (решение
пользователя: маски в скане не хранятся, строятся on-demand в момент
модификации - Ф3 эпика). id - тот же контент-квантованный формат.
"""
from __future__ import annotations

import json
import re

# Глобальные умолчания - как у боевой SAM3-ноды (виджеты #230), кроме порога:
# ансамблевый score дискретен {0.33, 0.67, 1.0}, порог 0.5 = «нашли минимум
# два прогона из трёх» (тест: голос 1/3 - шум, 2/3+ - реальные объекты).
GLOBAL_THRESHOLD = 0.5
DEDUP_IOU = 0.35
SYNONYM_IOU = 0.7
EXCLUSION_IOS = 0.7
GLOBAL_MIN_BOX_PCT = 1.0
ID_GRID = 8
# Кластеризация ансамбля: та же граница совпадения, что в тесте эмуляции.
ENSEMBLE_MATCH_IOU = 0.5


def parse_detections(text: str):
    """Список {label, box_2d} из ответа модели; None на мусоре (фенсы терпим)."""
    if not text:
        return None
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e <= s:
        return None
    try:
        items = json.loads(re.sub(r",\s*([}\]])", r"\1", text[s:e + 1]), strict=False)
    except (ValueError, TypeError):
        return None
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            y0, x0, y1, x1 = [float(v) for v in it["box_2d"]]
            label = str(it["label"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if label and y1 > y0 and x1 > x0:
            out.append({"label": label, "box_2d": [y0, x0, y1, x1]})
    return out


def norm_box_to_pixels(box_2d, image_w: int, image_h: int):
    """box_2d [ymin,xmin,ymax,xmax] 0-1000 -> пиксельный [x0,y0,x1,y1] (SAM3-вид)."""
    y0, x0, y1, x1 = box_2d
    return [x0 / 1000.0 * image_w, y0 / 1000.0 * image_h,
            x1 / 1000.0 * image_w, y1 / 1000.0 * image_h]


def box_overlap(box_a, box_b):
    """IoU и IoS (по меньшему боксу); порт _box_overlap SAM3-ноды."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0,
            inter / smaller if smaller > 0 else 0.0)


def labels_are_related(label_a: str, label_b: str) -> bool:
    """Один лейбл входит в другой целыми словами; порт из SAM3-ноды.

    'Table' и 'Coffee table' - родня; 'Bed' и 'Bedside table' - нет.
    """
    a, b = label_a.strip().lower(), label_b.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    words_a, words_b = set(a.split()), set(b.split())
    return words_a <= words_b or words_b <= words_a


def cluster_ensemble(runs: list, match_iou: float = ENSEMBLE_MATCH_IOU):
    """Кластеры одинаковых находок между прогонами -> votes и медианный бокс.

    Жадно: каждая детекция прилипает к первому кластеру с тем же классом и
    IoU>порога; один прогон голосует за кластер не больше одного раза.
    """
    clusters: list[dict] = []
    for run_idx, run in enumerate(runs):
        for det in run or []:
            for cl in clusters:
                if (cl["label"].lower() == det["label"].lower()
                        and run_idx not in cl["runs"]
                        and box_overlap(cl["box"], det["box"])[0] > match_iou):
                    cl["runs"].add(run_idx)
                    cl["boxes"].append(det["box"])
                    # Центр кластера - среднее боксов: устойчивее одного сэмпла.
                    cl["box"] = [sum(b[i] for b in cl["boxes"]) / len(cl["boxes"])
                                 for i in range(4)]
                    break
            else:
                clusters.append({"label": det["label"], "box": list(det["box"]),
                                 "boxes": [det["box"]], "runs": {run_idx}})
    # Слияние синонимных кластеров: один объект, названный прогонами по-разному
    # (Nightstand vs Side table), расщепляет голоса и оба осколка рискуют не
    # пройти порог. Кластеры с сильным пересечением (IoU>SYNONYM_IOU) сливаются,
    # лейбл - от кластера с большинством голосов (при равенстве - длиннее).
    merged: list[dict] = []
    for cl in sorted(clusters, key=lambda c: (-len(c["runs"]), -len(c["label"]))):
        for m in merged:
            if box_overlap(m["box"], cl["box"])[0] > SYNONYM_IOU:
                m["runs"] |= cl["runs"]
                m["boxes"].extend(cl["boxes"])
                m["box"] = [sum(b[i] for b in m["boxes"]) / len(m["boxes"])
                            for i in range(4)]
                break
        else:
            merged.append(cl)
    total = max(1, len(runs))
    for cl in merged:
        cl["votes"] = len(cl["runs"])
        cl["score"] = round(cl["votes"] / total, 3)
    return merged


def deduplicate_boxes(objects: list, iou_threshold: float = DEDUP_IOU):
    """Смарт-дедуп; порт _deduplicate_boxes SAM3-ноды.

    Родственные лейблы + IoS>порога -> остаётся более специфичный (длинный);
    неродственные + IoU>SYNONYM_IOU -> синонимы одного объекта, тот же выбор.
    """
    n = len(objects)
    removed: set[int] = set()
    for i in range(n):
        if i in removed:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue
            iou, ios = box_overlap(objects[i]["box"], objects[j]["box"])
            related = labels_are_related(objects[i]["label"], objects[j]["label"])
            if (related and ios > iou_threshold) or (not related and iou > SYNONYM_IOU):
                if len(objects[i]["label"]) >= len(objects[j]["label"]):
                    removed.add(j)
                else:
                    removed.add(i)
                    break
    return [objects[i] for i in range(n) if i not in removed]


def apply_exclusion_pairs(objects: list, exclusion_json: str,
                          ios_threshold: float = EXCLUSION_IOS):
    """Иерархические исключения; порт _apply_exclusion_pairs SAM3-ноды.

    {'Wardrobe': ['Shelving', ...]}: ребёнок, лежащий внутри родителя
    (IoS>порога), удаляется - это деталь родителя, не отдельный объект.
    """
    try:
        pairs = json.loads(exclusion_json) if exclusion_json and exclusion_json.strip() else {}
    except (ValueError, TypeError):
        return objects, ["excludes: нечитаемый JSON, пропущен"]
    if not isinstance(pairs, dict) or not pairs:
        return objects, []
    parent_children = {str(p).strip().lower(): {str(c).strip().lower() for c in ch}
                       for p, ch in pairs.items() if isinstance(ch, list)}
    notes = []
    removed: set[int] = set()
    for i, parent in enumerate(objects):
        if i in removed:
            continue
        children = parent_children.get(parent["label"].strip().lower())
        if not children:
            continue
        for j, child in enumerate(objects):
            if j == i or j in removed:
                continue
            if child["label"].strip().lower() in children:
                _, ios = box_overlap(parent["box"], child["box"])
                if ios > ios_threshold:
                    removed.add(j)
                    notes.append("excludes: %s внутри %s (IoS %.2f)"
                                 % (child["label"], parent["label"], ios))
    return [objects[i] for i in range(len(objects)) if i not in removed], notes


def make_obj_id(box, grid: int = ID_GRID) -> str:
    """Контент-стабильный id из бокса; порт make_obj_id SAM3-ноды.

    '{cx}_{cy}_{w}_{h}' на решётке grid px: дрожание бокса на пару пикселей
    не меняет id, разные объекты получают разные.
    """
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    cx = round((x0 + x1) / 2 / grid) * grid
    cy = round((y0 + y1) / 2 / grid) * grid
    w = round((x1 - x0) / grid) * grid
    h = round((y1 - y0) / grid) * grid
    return "%d_%d_%d_%d" % (cx, cy, w, h)


def _json_map(text: str):
    try:
        raw = json.loads(text) if text and text.strip() else {}
    except (ValueError, TypeError):
        return {}, False
    if not isinstance(raw, dict):
        return {}, False
    out = {}
    for k, v in raw.items():
        try:
            out[str(k).strip().lower()] = float(v)
        except (TypeError, ValueError):
            continue
    return out, True


def encode_objects(detection_texts: list, image_w: int, image_h: int,
                   image_hash: str = "", thresholds_json: str = "",
                   excludes_json: str = "", sizes_json: str = "") -> tuple[dict, list]:
    """Опись объектов из ансамбля детекций: (payload, notes для debug).

    payload - формат sam3EncodeResultsToText без mask_b64; настройки классов -
    те же JSON-входы графа, что получала SAM3-нода (админка продолжает рулить).
    """
    notes = []
    runs = []
    for idx, text in enumerate(detection_texts):
        dets = parse_detections(text)
        if dets is None:
            notes.append("детекция %d: нечитаемый ответ, пропущена" % (idx + 1))
            continue
        for d in dets:
            d["box"] = norm_box_to_pixels(d["box_2d"], image_w, image_h)
        runs.append(dets)
    if not runs:
        return ({"version": 1, "image_w": int(image_w), "image_h": int(image_h),
                 "image_hash": image_hash, "id_grid": ID_GRID, "objects": []},
                notes + ["объектов нет: все прогоны детекции нечитаемы"])
    notes.append("ансамбль: прогонов %d, находок %s"
                 % (len(runs), "+".join(str(len(r)) for r in runs)))

    clusters = cluster_ensemble(runs)

    thresholds, thr_ok = _json_map(thresholds_json)
    if thresholds_json and thresholds_json.strip() and not thr_ok:
        notes.append("thresholds: нечитаемый JSON, глобальный порог")
    sizes, sz_ok = _json_map(sizes_json)
    if sizes_json and sizes_json.strip() and not sz_ok:
        notes.append("sizes: нечитаемый JSON, глобальный минимум")

    # Пороги уверенности: per-class из админки поверх глобального. Порог >= 1 =
    # класс выключен (точная совместимость с прод-поведением SAM3, где 1.0
    # недостижим; переключение на семантику «нужно единогласие» - решение
    # пользователя, тогда сравнение станет score >= thr).
    kept = []
    for cl in clusters:
        thr = thresholds.get(cl["label"].strip().lower(), GLOBAL_THRESHOLD)
        if thr >= 1.0:
            notes.append("threshold: %s выключен (порог %.2f)" % (cl["label"], thr))
            continue
        if cl["score"] < thr:
            notes.append("threshold: %s score %.2f < %.2f" % (cl["label"], cl["score"], thr))
            continue
        kept.append(cl)

    # Минимальные размеры: per-class из админки поверх глобального 1% кадра.
    image_area = float(image_w) * float(image_h)
    sized = []
    for cl in kept:
        b = cl["box"]
        area_pct = 100.0 * (b[2] - b[0]) * (b[3] - b[1]) / image_area if image_area else 0.0
        min_pct = sizes.get(cl["label"].strip().lower(), GLOBAL_MIN_BOX_PCT)
        if area_pct < min_pct:
            notes.append("min_size: %s %.2f%% < %.1f%%" % (cl["label"], area_pct, min_pct))
            continue
        sized.append(cl)

    before = len(sized)
    sized = deduplicate_boxes(sized)
    if len(sized) != before:
        notes.append("dedup: убрано %d" % (before - len(sized)))
    sized, excl_notes = apply_exclusion_pairs(sized, excludes_json)
    notes.extend(excl_notes)

    objects = [{"id": make_obj_id(cl["box"]),
                "label": cl["label"],
                "box": [round(float(v), 2) for v in cl["box"]],
                "score": cl["score"]} for cl in sized]
    payload = {"version": 1, "image_w": int(image_w), "image_h": int(image_h),
               "image_hash": image_hash, "id_grid": ID_GRID, "objects": objects}
    notes.append("итог: %d объектов" % len(objects))
    return payload, notes

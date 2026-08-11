"""Тесты описи объектов (`desow_objects`) - замены SAM3-ветки скана.

Порты обвязки проверяются на кейсах из докстринов исходной SAM3-ноды
(Table vs Coffee table, Sofa vs Couch, Shelving внутри Wardrobe).
"""
import json

from desow_objects import (
    GLOBAL_THRESHOLD,
    apply_exclusion_pairs,
    box_overlap,
    cluster_ensemble,
    deduplicate_boxes,
    encode_objects,
    labels_are_related,
    make_obj_id,
    norm_box_to_pixels,
    parse_detections,
)


def det(label, y0, x0, y1, x1):
    return {"label": label, "box_2d": [y0, x0, y1, x1]}


def txt(items):
    return json.dumps(items)


def test_labels_related_whole_words():
    assert labels_are_related("Table", "Coffee table")
    assert labels_are_related("Vase", "vase")
    assert not labels_are_related("Bed", "Bedside table")      # Bed - часть слова
    assert not labels_are_related("Chair", "Armchair")


def test_dedup_keeps_more_specific_and_synonyms():
    objs = [
        {"label": "Table", "box": [100, 100, 300, 300]},
        {"label": "Coffee table", "box": [105, 105, 305, 305]},   # родня, IoS высок
        {"label": "Sofa", "box": [500, 500, 900, 900]},
        {"label": "Couch", "box": [505, 505, 905, 905]},          # синоним, IoU > 0.7
    ]
    left = deduplicate_boxes(objs)
    labels = sorted(o["label"] for o in left)
    assert labels == ["Coffee table", "Couch"]    # более специфичный/длинный


def test_exclusion_child_inside_parent():
    objs = [
        {"label": "Wardrobe", "box": [0, 0, 400, 800]},
        {"label": "Shelving", "box": [50, 100, 350, 700]},    # внутри шкафа
        {"label": "Shelving", "box": [900, 100, 1100, 700]},  # отдельно стоящий
    ]
    left, notes = apply_exclusion_pairs(objs, json.dumps({"Wardrobe": ["Shelving"]}))
    assert len(left) == 2
    assert notes and "внутри" in notes[0]
    # Мусорный JSON исключений не роняет и не режет.
    same, notes2 = apply_exclusion_pairs(objs, "не json")
    assert len(same) == 3 and "нечитаемый" in notes2[0]


def test_cluster_votes_and_score():
    runs = [
        [det("Bed", 500, 0, 900, 990), det("Vase", 600, 80, 700, 150)],
        [det("Bed", 505, 5, 905, 995)],
        [det("Bed", 498, 0, 897, 988), det("Vase", 602, 82, 702, 152)],
    ]
    for r in runs:
        for d in r:
            d["box"] = norm_box_to_pixels(d["box_2d"], 1000, 1000)
    clusters = {c["label"]: c for c in cluster_ensemble(runs)}
    assert clusters["Bed"]["votes"] == 3 and clusters["Bed"]["score"] == 1.0
    assert clusters["Vase"]["votes"] == 2


def test_cluster_merges_synonym_labels():
    # Один объект, названный прогонами по-разному: голоса не расщепляются.
    runs = [
        [det("Nightstand", 700, 0, 890, 160)],
        [det("Side table", 702, 2, 888, 158)],
        [det("Side table", 698, 0, 892, 162)],
    ]
    for r in runs:
        for d in r:
            d["box"] = norm_box_to_pixels(d["box_2d"], 1000, 1000)
    clusters = cluster_ensemble(runs)
    assert len(clusters) == 1
    assert clusters[0]["votes"] == 3
    assert clusters[0]["label"] == "Side table"     # большинство голосов


def test_encode_full_pipeline_with_admin_settings():
    runs_texts = [
        txt([det("Bed", 500, 0, 900, 990), det("Rack", 100, 100, 300, 300),
             det("Clock", 10, 10, 60, 60), det("Wardrobe", 0, 0, 400, 800),
             det("Shelving", 50, 100, 350, 700)]),
        txt([det("Bed", 505, 5, 905, 995), det("Wardrobe", 2, 0, 402, 798),
             det("Shelving", 52, 100, 348, 702)]),
        txt([det("Bed", 498, 0, 897, 988), det("Wardrobe", 0, 2, 398, 800),
             det("Shelving", 50, 98, 350, 698)]),
    ]
    payload, notes = encode_objects(
        runs_texts, 1200, 900, "hash123",
        thresholds_json=json.dumps({"Rack": 1}),          # выключен админкой
        excludes_json=json.dumps({"Wardrobe": ["Shelving"]}),
        sizes_json=json.dumps({"Clock": 5}),              # мелкие часы - режутся
    )
    labels = sorted(o["label"] for o in payload["objects"])
    assert labels == ["Bed", "Wardrobe"]
    assert any("Rack выключен" in n for n in notes)
    # Clock нашёл один прогон из трёх - режется ГЛОБАЛЬНЫМ порогом 0.5 раньше,
    # чем дошло бы до min_size (оба гарда стоят на его пути).
    assert any("Clock" in n for n in notes)
    bed = next(o for o in payload["objects"] if o["label"] == "Bed")
    assert bed["score"] == 1.0
    assert payload["image_w"] == 1200 and payload["image_hash"] == "hash123"
    assert "mask_b64" not in bed
    # id - формат SAM3: cx_cy_w_h на решётке 8px.
    assert len(bed["id"].split("_")) == 4


def test_encode_garbage_inputs_fail_soft():
    payload, notes = encode_objects(["мусор", "", None], 1000, 800)
    assert payload["objects"] == []
    assert any("нечитаемы" in n for n in notes)
    assert parse_detections("ни разу не json") is None


def test_make_obj_id_stable_under_jitter():
    a = make_obj_id([100, 200, 300, 400])
    b = make_obj_id([102, 199, 301, 402])   # дрожание пару пикселей
    c = make_obj_id([500, 200, 700, 400])   # другой объект
    assert a == b and a != c


def test_global_threshold_cuts_single_vote():
    # Голос 1/3 - шум (тест эмуляции score): глобальный порог 0.5 его режет.
    assert GLOBAL_THRESHOLD == 0.5
    runs_texts = [txt([det("Vase", 600, 80, 700, 150)]), txt([]), txt([])]
    payload, _notes = encode_objects(runs_texts, 1000, 1000)
    assert payload["objects"] == []


def test_gate_skipped_detections_give_clean_empty_result():
    # Гейт пустой комнаты: все детекции пропущены нодой - опись пустая с
    # честной пометкой, а не «нечитаемый ответ».
    payload, notes = encode_objects(
        ["GATE_SKIPPED: empty", "GATE_SKIPPED: empty", "GATE_SKIPPED: empty"],
        1000, 800)
    assert payload["objects"] == []
    assert notes == ["гейт: комната пустая - детекция объектов пропущена"]

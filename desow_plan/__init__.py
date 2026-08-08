"""Построение пустого 2D-плана комнаты кодом (эпик furnish, фаза A).

Пакет вендорен из бэкенда Desow (`desow/plan2d/` и `layout_openings_match.py`):
на машине ComfyUI бэкенда нет, а логика должна быть одна и та же. Двойное ведение
осознанное — в шапке каждого модуля указан источник, при правках менять парой.
См. README репозитория, раздел «Ноды Desow».

Зависимости: только Pillow (есть в любой сборке ComfyUI). Сети здесь нет:
экстракцию структуры делает OpenRouter-нода графа, а расстановка мебели
(`build_furnished_plan`) получает вызов модели callable'ом снаружи — http-стек
живёт в `openrouter_api.py`.
"""
from .furnish import PLAN_FURNISH_MODEL, FurnishError
from .pipeline import FurnishFailed, blank_png, build_empty_plan, build_furnished_plan, render_camera_png
from .schema_lite import PlanDataError

__all__ = [
    "build_empty_plan",
    "build_furnished_plan",
    "blank_png",
    "render_camera_png",
    "FurnishError",
    "FurnishFailed",
    "PLAN_FURNISH_MODEL",
    "PlanDataError",
]

"""Построение пустого 2D-плана комнаты кодом (эпик furnish, фаза A).

Пакет вендорен из бэкенда Desow (`desow/plan2d/` и `layout_openings_match.py`):
на машине ComfyUI бэкенда нет, а логика должна быть одна и та же. Двойное ведение
осознанное — в шапке каждого модуля указан источник, при правках менять парой.
См. README репозитория, раздел «Ноды Desow».

Зависимости: только Pillow (есть в любой сборке ComfyUI). Сети здесь нет —
экстракцию делает OpenRouter-нода графа.
"""
from .pipeline import blank_png, build_empty_plan
from .schema_lite import PlanDataError

__all__ = ["build_empty_plan", "blank_png", "PlanDataError"]

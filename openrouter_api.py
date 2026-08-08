"""Общий сетевой слой OpenRouter для нод этого репозитория.

Здесь живёт ЕДИНСТВЕННАЯ политика похода в API: откуда берётся ключ, что
считается временной ошибкой, сколько ждать между попытками и какой таймаут.
`OpenRouterNode` импортирует отсюда примитивы (`_get_openrouter_api_key`,
`_is_retryable_status`, `_retry_sleep`, `RETRYABLE_PROVIDER_ERROR_TYPES`), а
ноды, которым нужен обычный текстовый запрос, вызывают `chat_complete`.

**Своего http-стека у нод быть не должно.** Политика ретраев уже стоила
инцидентов (silent fallback на ошибке API утекал reference-картинкой в
результат), и чинить её надо в одном месте, а не в трёх копиях.

Почему `OpenRouterNode._generate_response` не переписан на `chat_complete`:
его цикл в той же итерации разбирает мультимодальный ответ (images, PDF,
chat-сессии, статистику токенов) — перенос требует прогона всех боевых графов
и делается отдельной задачей. Дублирования при этом нет: и он, и
`chat_complete` решают «ретраить или падать» одними и теми же предикатами.
"""
from __future__ import annotations

import json
import os
import random
import time

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 3

# Заголовки апстримной ноды — оставлены как есть, по ним OpenRouter считает
# статистику интеграции (менять смысла нет, а расхождение между нодами вредно).
_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/yourusername/comfyui-openrouter",
    "X-Title": "ComfyUI OpenRouter LLM Node",
}

# Ошибки провайдера внутри choices[0] (OpenRouter отдаёт их с HTTP 200), которые
# имеет смысл повторить. Safety reject и content policy сюда не входят: модель
# не передумает на тот же промпт.
RETRYABLE_PROVIDER_ERROR_TYPES = frozenset(
    {"rate_limit_exceeded", "timeout", "server_error", "internal_server_error"}
)


# API key больше не хранится в widgets_values JSON — это вызывало утечки секрета
# при экспорте воркфлоу (инцидент 2026-05-10). Теперь читается из переменной
# окружения OPENROUTER_API_KEY. На ComfyDeploy задаётся через их Secrets-панель,
# локально — через .env / launcher-скрипт.
def _get_openrouter_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


# Retry с экспоненциальным backoff для retry-able ошибок API.
# 429 (rate limit), 5xx (server errors), timeout, connection errors —
# временные сбои, шансы успеха при повторе высокие. Safety reject и
# другие 4xx — не retry, Gemini не передумает на тот же промпт.
def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _retry_sleep(attempt: int, base_delay: float = 1.0) -> None:
    """Exponential backoff с jitter: 1s, 2s, 4s + random 0-1s.
    Jitter нужен чтобы N параллельных клиентов не retry'ли одновременно после
    общего 429 (это снова даст 429)."""
    wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
    time.sleep(wait)


class OpenRouterCallError(RuntimeError):
    """Терминальная ошибка вызова API: ретраи исчерпаны или ошибка не временная."""


def chat_complete(
    messages,
    *,
    model,
    temperature=0.4,
    max_tokens=4000,
    seed=0,
    max_retries=DEFAULT_MAX_RETRIES,
    timeout=DEFAULT_TIMEOUT_S,
    log_prefix="OpenRouter",
):
    """Текстовый chat-completion -> content ответа (строка).

    Ровно та же политика, что и в `OpenRouterNode`: 429/5xx/сеть/не-JSON при
    HTTP 200/пустые `choices`/временная ошибка провайдера -> повтор с backoff;
    4xx и невременные ошибки -> сразу `OpenRouterCallError`.

    Мультимодальность, chat-сессии и статистика сюда не входят намеренно: это
    вызов «текст на входе, текст на выходе» для служебных нод.

    Raises:
        OpenRouterCallError: ключа нет, либо API не отдал ответ.
    """
    api_key = _get_openrouter_api_key()
    if not api_key:
        raise OpenRouterCallError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it via ComfyDeploy Secrets panel (production) or .env / "
            "launcher script (local ComfyUI)."
        )

    headers = dict(_HEADERS, Authorization="Bearer %s" % api_key)
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # Клэмп seed в диапазон INT32 — Google AI Studio (Gemini) отклоняет
        # значения больше 2^31-1 (та же арифметика, что в OpenRouterNode).
        "seed": int(seed) % 0x80000000,
    }
    if max_tokens:
        data["max_tokens"] = max_tokens

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = "Network error: %s" % exc
            if attempt < max_retries:
                print("[%s] %s — retry %d/%d after backoff" % (log_prefix, last_error, attempt + 1, max_retries))
                _retry_sleep(attempt)
                continue
            raise OpenRouterCallError(
                "OpenRouter API unreachable after %d retries: %s" % (max_retries, last_error)
            )

        if _is_retryable_status(response.status_code):
            last_error = "HTTP %s: %s" % (response.status_code, response.text[:300])
            if attempt < max_retries:
                print("[%s] %s — retry %d/%d after backoff" % (log_prefix, last_error, attempt + 1, max_retries))
                _retry_sleep(attempt)
                continue
            raise OpenRouterCallError(
                "OpenRouter API failed after %d retries: %s" % (max_retries, last_error)
            )

        if response.status_code >= 400:
            raise OpenRouterCallError(
                "OpenRouter API rejected request: HTTP %s: %s" % (response.status_code, response.text[:300])
            )

        # HTTP 200 + НЕ-JSON body — это HTML-страница ошибки шлюза, transient.
        try:
            result = response.json()
        except json.JSONDecodeError as exc:
            last_error = "Non-JSON 200 response (likely upstream gateway error page): %s" % exc
            if attempt < max_retries:
                print("[%s] %s — retry %d/%d after backoff" % (log_prefix, last_error, attempt + 1, max_retries))
                _retry_sleep(attempt)
                continue
            raise OpenRouterCallError(
                "OpenRouter returned non-JSON after %d retries: %s" % (max_retries, last_error)
            )

        if not result.get("choices") or not result["choices"][0].get("message"):
            last_error = "Invalid response format: 'choices' or 'message' missing"
            if attempt < max_retries:
                print("[%s] %s — retry %d/%d after backoff" % (log_prefix, last_error, attempt + 1, max_retries))
                _retry_sleep(attempt)
                continue
            raise OpenRouterCallError(
                "Invalid response format after %d retries: 'choices' or 'message' missing." % max_retries
            )

        choice = result["choices"][0]
        if choice.get("error"):
            err = choice["error"]
            full_error = "Provider error %s (%s): %s" % (
                err.get("code", "?"),
                (err.get("metadata") or {}).get("error_type", ""),
                err.get("message", "Unknown provider error"),
            )
            err_type = str((err.get("metadata") or {}).get("error_type", "")).lower()
            if err_type in RETRYABLE_PROVIDER_ERROR_TYPES and attempt < max_retries:
                last_error = full_error
                print("[%s] %s — retry %d/%d after backoff" % (log_prefix, full_error, attempt + 1, max_retries))
                _retry_sleep(attempt)
                continue
            raise OpenRouterCallError(full_error)

        # Обрыв по лимиту токенов не ошибка транспорта: HTTP 200, тело валидное,
        # оборван ответ МОДЕЛИ. Ретраить бессмысленно (повтор упрётся в тот же
        # потолок), но и молчать нельзя — у reasoning-моделей размышление съедает
        # тот же бюджет, и снаружи это выглядит как «модель прислала не JSON».
        if choice.get("finish_reason") == "length":
            usage = result.get("usage") or {}
            print(
                "[%s] ВНИМАНИЕ: ответ оборван по max_tokens=%s (completion_tokens=%s, из них reasoning=%s). "
                "Поднимите потолок токенов." % (
                    log_prefix, data.get("max_tokens"),
                    usage.get("completion_tokens"),
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                )
            )

        return choice["message"].get("content") or ""

    raise OpenRouterCallError("OpenRouter retry-loop exhausted: %s" % last_error)

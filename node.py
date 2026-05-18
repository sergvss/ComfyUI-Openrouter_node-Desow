import os
import random
import requests
import json
import time
import base64
import io
import numpy as np
import torch
import tiktoken
from PIL import Image
import hashlib # Added for hashing PDF bytes in IS_CHANGED
from .chat_manager import ChatSessionManager


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

# Define a placeholder type name for PDF data.
# The actual input connection will accept '*' but we check the structure.
# Expecting a dictionary: {"filename": str, "bytes": bytes}
PDF_DATA_TYPE = "*" # Use '*' to accept any type, check structure later

# Класс-wildcard для входов, принимающих любой тип. ComfyUI сравнивает типы через __ne__,
# поэтому если всегда возвращать False, любое соединение будет разрешено
class AnyType(str):
    def __ne__(self, other):
        return False

ANY_TYPE = AnyType("*")

class OpenRouterNode:
    """
    A node for interacting with OpenRouter's chat/completion API.
    Supports text, images, and PDFs as input.
    Returns three outputs:
      1) "Output": the text response from the LLM
      2) "Stats": a string detailing tokens per second, input tokens, and output tokens
      3) "Credits": a string showing your remaining OpenRouter account balance
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600  # Cache duration in seconds (1 hour)

    def __init__(self):
        self.chat_manager = ChatSessionManager()

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input specification for this node.
        Includes optional inputs for image and PDF data.
        """
        return {
            "required": {
                # api_key УБРАН из widgets — теперь читается из переменной окружения
                # OPENROUTER_API_KEY (см. _get_openrouter_api_key выше). Это убирает
                # утечку секрета при экспорте воркфлоу в JSON.
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "user_message_box": ("STRING", {
                    "multiline": True,
                    "default": "Hello, how are you?"
                }),
                "model": (cls.fetch_openrouter_models(),),
                "web_search": ("BOOLEAN", {"default": False}),
                "cheapest": ("BOOLEAN", {"default": True}),
                "fastest": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "round": 0.01,
                }),
                 "pdf_engine": (["auto", "mistral-ocr", "pdf-text"], {"default": "auto"}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                # Количество дополнительных попыток при retry-able ошибках API
                # (429 rate-limit, 5xx server, timeout, connection error, empty
                # image response для image-моделей). 0 = без retry (для дебага).
                # 3 — баланс между UX-устойчивостью и временем ожидания юзера.
                "max_retries": ("INT", {"default": 3, "min": 0, "max": 5}),
            },
            "optional": {
                # ANY_TYPE - принимает входы любого типа (STRING, text, enum, External Enum, Power Primitive). При отсутствии подключения используется "auto". Валидация в generate_response
                "aspect_ratio": (ANY_TYPE, {"default": "auto"}),
                "image_resolution": (ANY_TYPE, {"default": "auto"}),
                "pdf_data": (PDF_DATA_TYPE,), # Use '*' and check structure in generate_response
                "user_message_input": ("STRING", {"forceInput": True}),
                # 5 статических input-слотов для изображений. Подключать можно любые, неиспользуемые остаются None и игнорируются в payload
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("Output", "image", "Stats", "Credits")

    FUNCTION = "generate_response"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """
        Fetches a list of model IDs from the OpenRouter API, caching them.
        """
        current_time = time.time()
        if (cls.models_cache is None) or (current_time - cls.last_fetch_time > cls.cache_duration):
            url = "https://openrouter.ai/api/v1/models"
            try:
                response = requests.get(url)
                response.raise_for_status()
                models = response.json()["data"]
                # Filter for models that support chat completions if needed, but API handles this
                model_list = sorted([model['id'] for model in models])
                cls.models_cache = model_list
                cls.last_fetch_time = current_time
            except requests.exceptions.RequestException as e:
                print(f"Error fetching models: {e}")
                # Provide a default list or indicate error if cache is empty
                if cls.models_cache is None:
                    cls.models_cache = ["error_fetching_models", "google/gemma-3-27b-it", "openai/gpt-4o"] # Example fallbacks
        return cls.models_cache if cls.models_cache else ["error_fetching_models"] # Ensure it's never empty

    def validate_temperature(self, temperature):
        """
        Validates and converts temperature value to float within acceptable range.
        """
        try:
            temp = float(temperature)
            return max(0.0, min(2.0, temp))  # Clamp between 0.0 and 2.0
        except (ValueError, TypeError):
            return 1.0  # Return default if conversion fails

    @staticmethod
    def _detect_aspect_and_size(src_w, src_h, model_name):
        """
        Подбирает ближайшие поддерживаемые aspect_ratio и image_size по размерам исходного изображения.
        Для Nano Banana 2 (gemini-3.1-flash-image) включает расширенные соотношения 1:4, 4:1, 1:8, 8:1.
        Возвращает (aspect_ratio_str, image_size_str).
        """
        if src_w <= 0 or src_h <= 0:
            return None, None

        # Базовый набор соотношений, поддерживаемых всеми image-моделями
        candidates = [
            ("1:1", 1, 1),
            ("2:3", 2, 3),
            ("3:2", 3, 2),
            ("3:4", 3, 4),
            ("4:3", 4, 3),
            ("4:5", 4, 5),
            ("5:4", 5, 4),
            ("9:16", 9, 16),
            ("16:9", 16, 9),
            ("21:9", 21, 9),
        ]
        # Nano Banana 2 (gemini-3.1-flash-image*) поддерживает экстремальные соотношения
        if "gemini-3.1-flash-image" in model_name.lower():
            candidates += [
                ("1:4", 1, 4),
                ("4:1", 4, 1),
                ("1:8", 1, 8),
                ("8:1", 8, 1),
            ]

        # Находим ближайшее соотношение через минимум модуля разности logarithm (устойчиво к направлению)
        import math
        src_ratio = src_w / src_h
        best_ratio = min(candidates, key=lambda c: abs(math.log(src_ratio) - math.log(c[1] / c[2])))
        aspect_str = best_ratio[0]

        # image_size по длинной стороне: 1K~1024-1344, 2K~2048, 4K~4096. Границы - геометрические средние
        longest = max(src_w, src_h)
        if longest < 1700:          # до sqrt(1024*2048) ~ 1448, округляем вверх
            size_str = "1K"
        elif longest < 2900:        # до sqrt(2048*4096) ~ 2896
            size_str = "2K"
        else:
            size_str = "4K"

        return aspect_str, size_str

    def fetch_credits(self, api_key):
        """
        Fetches the user's credits information from the OpenRouter API.
        Returns a formatted string with remaining credits.
        """
        if not api_key:
             return "API Key not provided."

        url = "https://openrouter.ai/api/v1/credits"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/comfyui-openrouter",
            "X-Title": "ComfyUI OpenRouter LLM Node",
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            result = response.json()
            # Check if 'data' and expected keys exist
            if "data" in result and "total_credits" in result["data"] and "total_usage" in result["data"]:
                total_credits = result["data"]["total_credits"]
                total_usage = result["data"]["total_usage"]
                remaining = total_credits - total_usage
                credits_text = f"Remaining: ${remaining:.3f}"
            else:
                credits_text = "Could not parse credit data from response."

            return credits_text

        except requests.exceptions.RequestException as e:
            # Provide more context about the error
            error_message = f"Error fetching credits: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                 error_message += f" | Status Code: {e.response.status_code} | Response: {e.response.text[:200]}" # Log part of response
            return error_message
        except json.JSONDecodeError:
             return "Error fetching credits: Could not decode JSON response."

    def generate_response(self, system_prompt, user_message_box, model,
                         web_search, cheapest, fastest, seed, temperature, pdf_engine, chat_mode,
                         max_retries=3,
                         aspect_ratio="auto", image_resolution="auto",
                         pdf_data=None, user_message_input=None,
                         image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                         **kwargs):
        # Сливаем 5 статических image-слотов в kwargs, чтобы существующий цикл по image_* работал без изменений
        for _name, _img in (("image_1", image_1), ("image_2", image_2), ("image_3", image_3), ("image_4", image_4), ("image_5", image_5)):
            if _img is not None:
                kwargs[_name] = _img
        """
        Sends a completion request to the OpenRouter chat completion endpoint.
        Handles text, optional image, and optional PDF inputs.

        Returns four outputs:
          (1) Output: the LLM's text response
          (2) image: an image tensor if the response contains an image, else empty tensor
          (3) Stats: a string with tokens per second, prompt tokens, completion tokens
          (4) Credits: a string with the user's credit information
        """
        # Create empty placeholder image (используется только для text-only моделей
        # или для credit/stats полей при ошибках — НЕ как silent-fallback вместо
        # настоящего сгенерированного изображения).
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        # API key из переменной окружения — больше не из widget'а, чтобы избежать
        # утечки секрета при экспорте воркфлоу. На ComfyDeploy задаётся через
        # Secrets-панель; локально — через .env / launcher-скрипт.
        api_key = _get_openrouter_api_key()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Set it via ComfyDeploy Secrets panel (production) or .env / "
                "launcher script (local ComfyUI)."
            )

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/comfyui-openrouter",
            "X-Title": "ComfyUI OpenRouter LLM Node",
        }

        # Validate and convert temperature
        validated_temp = self.validate_temperature(temperature)

        # Decide whether to use user_message_input or user_message_box
        user_text = user_message_input if user_message_input is not None and user_message_input.strip() else user_message_box

        # Initialize session_path
        session_path = None
        
        # Handle chat mode
        if chat_mode:
            # Get or create a chat session
            session_path, messages = self.chat_manager.get_or_create_session(user_text, system_prompt)
            
            # Check if we need to update the system prompt (for existing sessions)
            if messages and messages[0]["role"] == "system" and messages[0]["content"] != system_prompt:
                # Update system prompt if it has changed
                messages[0]["content"] = system_prompt
        else:
            # Non-chat mode: Build the messages array, starting with a system prompt.
            messages = [
                {"role": "system", "content": system_prompt},
            ]

        # --- Build the user message content ---
        user_content_blocks = []

        # 1. Add Text part (always present)
        user_content_blocks.append({
            "type": "text",
            "text": user_text
        })

        # 2. Add Image parts (optional) - support multiple images from kwargs
        # Process all image_N inputs from kwargs
        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')], 
                           key=lambda x: int(x.split('_')[1]))
        
        for image_key in image_keys:
            if kwargs[image_key] is not None:
                try:
                    img_str = self.image_to_base64(kwargs[image_key])
                    user_content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    })
                except Exception as e:
                    # Fail loud: лучше явная ошибка чем silent placeholder.
                    raise RuntimeError(f"Error processing input {image_key}: {e}") from e

        # 3. Add PDF part (optional)
        pdf_filename = "document.pdf" # Default filename if not provided
        if pdf_data is not None:
            # Validate pdf_data structure (expecting dict with 'filename' and 'bytes')
            if isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
                pdf_bytes = pdf_data["bytes"]
                # Use provided filename if available and valid, otherwise use default
                if "filename" in pdf_data and isinstance(pdf_data["filename"], str) and pdf_data["filename"].strip():
                     pdf_filename = pdf_data["filename"]

                try:
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    data_url = f"data:application/pdf;base64,{base64_pdf}"
                    user_content_blocks.append({
                        "type": "file",
                        "file": {
                            "filename": pdf_filename,
                            "file_data": data_url
                        }
                    })
                except Exception as e:
                    raise RuntimeError(f"Error encoding PDF: {e}") from e
            else:
                # Handle case where pdf_data is not in the expected format
                print(f"Warning: pdf_data input is not in the expected format (dict with 'filename' and 'bytes'). PDF not included.")
                # Optionally return an error or just proceed without the PDF
                # return ("Error: Invalid PDF data format.", "Stats N/A", "Credits N/A")


        # Determine message format based on content type
        # Use simple string format for text-only requests to ensure compatibility
        # Use structured format only when we have multimodal content
        has_multimodal_content = len(user_content_blocks) > 1 or any(block.get("type") != "text" for block in user_content_blocks)
        
        if has_multimodal_content:
            # Use structured format for multimodal content
            new_user_message = {
                "role": "user",
                "content": user_content_blocks
            }
        else:
            # Use simple string format for text-only requests
            new_user_message = {
                "role": "user",
                "content": user_text
            }
        
        if chat_mode:
            # In chat mode, append to existing conversation (but don't save yet - wait for response)
            messages.append(new_user_message)
        else:
            # In non-chat mode, messages array already has system prompt, just append user message
            messages.append(new_user_message)

        # --- Apply model modifiers ---
        modified_model = model
        # Check if model already has modifiers to avoid duplication
        if web_search and ":online" not in modified_model:
            modified_model = f"{modified_model}:online"
        if ":online" not in modified_model:
             if cheapest and ":floor" not in modified_model:
                 modified_model = f"{modified_model}:floor"
             elif fastest and not cheapest and ":nitro" not in modified_model:
                 modified_model = f"{modified_model}:nitro"


        # Клэмп seed в диапазон INT32 - Google AI Studio (Gemini) отклоняет значения больше 2^31-1
        raw_seed = int(seed)
        clamped_seed = raw_seed % 0x80000000
        print(f"[OpenRouter] Seed received: {raw_seed} -> sent: {clamped_seed}")

        # --- Construct the final payload ---
        data = {
            "model": modified_model,
            "messages": messages,
            "temperature": validated_temp,
            "seed": clamped_seed
        }

        # Определяем, является ли модель image-генерирующей по наличию "image" в имени
        is_image_model = "image" in modified_model.lower()

        if is_image_model:
            # Для image-моделей OpenRouter требует явно указать modalities, иначе картинка не возвращается
            data["modalities"] = ["image", "text"]

            # Если оба параметра auto и есть входное изображение - автоматически определяем ratio и size по image_1
            auto_ratio = None
            auto_size = None
            if aspect_ratio == "auto" or image_resolution == "auto":
                first_image = kwargs.get("image_1")
                if first_image is not None and isinstance(first_image, torch.Tensor) and first_image.ndim == 4:
                    # ComfyUI формат [B, H, W, C]
                    src_h = int(first_image.shape[1])
                    src_w = int(first_image.shape[2])
                    auto_ratio, auto_size = self._detect_aspect_and_size(src_w, src_h, modified_model)
                    print(f"[OpenRouter] Auto-detected from image_1 {src_w}x{src_h}: aspect_ratio={auto_ratio}, image_size={auto_size}")

            # Нормализация и валидация aspect_ratio (STRING - может прийти что угодно, включая старый формат "1:1 (1024x1024)")
            valid_ratios = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1"}
            normalized_ratio = (aspect_ratio or "auto").strip()
            if normalized_ratio.lower() == "auto":
                normalized_ratio = "auto"
            else:
                # Обрезаем описание в скобках если есть: "1:1 (1024x1024)" -> "1:1"
                normalized_ratio = normalized_ratio.split(" ", 1)[0]
                if normalized_ratio not in valid_ratios:
                    print(f"[OpenRouter] Warning: invalid aspect_ratio '{aspect_ratio}', falling back to auto")
                    normalized_ratio = "auto"

            # image_config собирается динамически: поля добавляются только если значение не "auto" или есть авто-детект
            image_config = {}
            if normalized_ratio != "auto":
                image_config["aspect_ratio"] = normalized_ratio
            elif auto_ratio:
                image_config["aspect_ratio"] = auto_ratio

            # Нормализация и валидация image_resolution (теперь STRING - может прийти что угодно)
            normalized_resolution = (image_resolution or "auto").strip().upper()
            if normalized_resolution == "AUTO":
                normalized_resolution = "auto"
            if normalized_resolution not in ("auto", "1K", "2K", "4K"):
                print(f"[OpenRouter] Warning: invalid image_resolution '{image_resolution}', falling back to auto")
                normalized_resolution = "auto"

            if normalized_resolution != "auto":
                image_config["image_size"] = normalized_resolution
            elif auto_size:
                image_config["image_size"] = auto_size

            # Отправляем image_config только если там есть хотя бы одно поле - иначе провайдер использует свои дефолты
            if image_config:
                data["image_config"] = image_config

        print(f"Payload: model={modified_model}, modalities={data.get('modalities')}, image_config={data.get('image_config')}")

        # Add plugins if a specific PDF engine is selected
        if pdf_engine != "auto":
             data["plugins"] = [
                 {
                     "id": "file-parser",
                     "pdf": {
                         "engine": pdf_engine
                     }
                 }
             ]

        # --- Pre-calculate text input tokens (rough estimate) ---
        # Note: Actual token count depends on the model and includes parsed PDF/image data.
        # Rely on the API response for accurate usage stats.
        text_token_estimate = 0
        try:
            text_token_estimate = self.count_tokens(system_prompt, model) + self.count_tokens(user_text, model)
        except Exception as e:
            print(f"Warning: Token counting failed - {e}")


        # --- Make API Call and Process Response (с retry-loop на retryable errors) ---
        # Retry стратегия:
        #   * 429/5xx HTTP, timeout, connection error → retry с exponential backoff
        #   * Provider error внутри choices[0] (rate_limit, timeout, server_error) → retry
        #   * Image-модель вернула пустой response без images (soft safety reject /
        #     временный сбой Gemini) → retry. Раньше тут был silent fallback на
        #     placeholder, который через ConditionalBranch в графе мог подставлять
        #     reference-картинку — юзер платил за мусор.
        #   * Safety reject (явный), 4xx (auth, invalid) → НЕ retry, raise сразу.
        # После исчерпания попыток любая retry-able ошибка → raise RuntimeError,
        # ComfyUI/ComfyDeploy корректно помечают run как failed.
        result = None
        response = None
        start_time = None
        end_time = None
        last_error = None
        retryable_provider_error_types = {"rate_limit_exceeded", "timeout", "server_error", "internal_server_error"}

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(url, headers=headers, json=data, timeout=120)
                end_time = time.time()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = f"Network error: {exc}"
                if attempt < max_retries:
                    print(f"[OpenRouter] {last_error} — retry {attempt + 1}/{max_retries} after backoff")
                    _retry_sleep(attempt)
                    continue
                raise RuntimeError(f"OpenRouter API unreachable after {max_retries} retries: {last_error}")

            # Retryable HTTP statuses (429 rate-limit, 5xx server errors)
            if _is_retryable_status(response.status_code):
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if attempt < max_retries:
                    print(f"[OpenRouter] {last_error} — retry {attempt + 1}/{max_retries} after backoff")
                    _retry_sleep(attempt)
                    continue
                raise RuntimeError(f"OpenRouter API failed after {max_retries} retries: {last_error}")

            # Non-retryable 4xx (401 auth, 400 invalid request, 403 forbidden) — fail fast
            if response.status_code >= 400:
                raise RuntimeError(f"OpenRouter API rejected request: HTTP {response.status_code}: {response.text[:300]}")

            # 200 OK — парсим ответ.
            # Случай: HTTP 200 + НЕ-JSON body (HTML 502 error page от CF/upstream
            # routing OpenRouter). Это transient — ретраим, не fail-fast.
            try:
                result = response.json()
            except json.JSONDecodeError as exc:
                last_error = (
                    f"Non-JSON 200 response (likely upstream gateway error page): "
                    f"{exc}; body[:300]={response.text[:300]!r}"
                )
                if attempt < max_retries:
                    print(f"[OpenRouter] {last_error} — retry {attempt + 1}/{max_retries} after backoff")
                    _retry_sleep(attempt)
                    continue
                raise RuntimeError(
                    f"OpenRouter returned non-JSON after {max_retries} retries: {last_error}"
                )

            debug_str = json.dumps(result, default=str)
            print(f"API response ({len(debug_str)} chars): {debug_str[:500]}")

            # Missing choices/message — тоже transient (OpenRouter возвращает HTTP 200
            # с пустым/невалидным телом при некоторых провайдер-сбоях). Ретраим.
            if not result.get("choices") or not result["choices"][0].get("message"):
                last_error = (
                    f"Invalid response format: 'choices' or 'message' missing. "
                    f"Body[:300]={json.dumps(result, default=str)[:300]!r}"
                )
                if attempt < max_retries:
                    print(f"[OpenRouter] {last_error} — retry {attempt + 1}/{max_retries} after backoff")
                    _retry_sleep(attempt)
                    continue
                raise RuntimeError(
                    f"Invalid response format after {max_retries} retries: 'choices' or 'message' missing."
                )

            choice = result["choices"][0]

            # Provider error внутри choices[0] (OpenRouter возвращает HTTP 200 с
            # ошибкой в теле — raise_for_status её не ловит).
            if choice.get("error"):
                err = choice["error"]
                err_code = err.get("code", "?")
                err_msg = err.get("message", "Unknown provider error")
                err_meta = err.get("metadata", {})
                err_type = err_meta.get("error_type", "")
                full_error = f"Provider error {err_code} ({err_type}): {err_msg}"

                if err_type.lower() in retryable_provider_error_types and attempt < max_retries:
                    last_error = full_error
                    print(f"[OpenRouter] {full_error} — retry {attempt + 1}/{max_retries} after backoff")
                    _retry_sleep(attempt)
                    continue
                # Safety reject, content policy, прочие non-retryable — fail
                raise RuntimeError(full_error)

            # Для image-моделей пустой response без images = soft fail (safety reject
            # или временный сбой Gemini). Retry даёт шанс получить картинку при
            # повторной попытке. Если все попытки пусты — raise (НЕ silent placeholder).
            message = choice["message"]
            if is_image_model and not message.get("images"):
                last_error = "image-model returned no images (possibly safety reject or temporary Gemini issue)"
                if attempt < max_retries:
                    print(f"[OpenRouter] {last_error} — retry {attempt + 1}/{max_retries} after backoff")
                    _retry_sleep(attempt)
                    continue
                raise RuntimeError(
                    f"OpenRouter image generation failed after {max_retries} retries: no images in response. "
                    f"Last response content: {(message.get('content') or '')[:200]}"
                )

            # Успех — выходим из retry-loop
            break
        else:
            # for-else: достигли max_retries без break — defensive, теоретически
            # сюда не попадаем (raise срабатывает раньше), но если попали — fail.
            raise RuntimeError(f"OpenRouter retry-loop exhausted: {last_error}")

        # --- Парсинг успешного ответа ---
        # На этой точке result, response, start_time, end_time, choice, message гарантированно валидны.
        try:
            # Parse response for text and image content
            # Gemini при генерации изображения возвращает content: null - приводим к пустой строке, чтобы downstream-ноды не падали
            text_output = message.get("content") or ""
            image_tensor = placeholder_image

            # Check for images in the separate images field (OpenRouter format)
            if message.get("images"):
                print(f"Found {len(message['images'])} image(s) in API response")
                try:
                    # Get the first image from the images array
                    first_image = message["images"][0]
                    image_url = first_image["image_url"]["url"]

                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",", 1)[1]
                        try:
                            # Convert base64 to image tensor
                            image_tensor = self.base64_to_image(base64_str)
                            print(f"Successfully decoded image from API response")
                        except Exception as e:
                            print(f"Error decoding image: {e}")
                    else:
                        print(f"Image URL format not supported: {image_url[:50]}...")
                except Exception as e:
                    print(f"Error processing images from response: {e}")
            else:
                # Text-only модель — placeholder OK. Для image-моделей мы сюда не
                # попадём — выше retry-loop поднял бы raise.
                print("No images found in API response - text-only model, this is normal")
            
            # Also handle legacy multimodal content format as fallback
            if isinstance(text_output, list):
                text_parts = []
                for content in text_output:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            text_parts.append(content.get("text", ""))
                        elif content.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                base64_str = image_url.split(",", 1)[1]
                                try:
                                    # Convert base64 to image tensor
                                    image_tensor = self.base64_to_image(base64_str)
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                text_output = "\n".join(text_parts)

            response_ms = result.get("response_ms", None)
            api_usage = result.get("usage", {})
            prompt_tokens = api_usage.get("prompt_tokens", text_token_estimate) # Use API value if available
            completion_tokens = api_usage.get("completion_tokens", 0)
            if completion_tokens == 0 and text_output: # Estimate completion tokens if API doesn't provide them
                 try:
                     completion_tokens = self.count_tokens(text_output, model)
                 except Exception as e:
                     print(f"Warning: Completion token counting failed - {e}")


            # Calculate tokens per second (TPS)
            tps = 0
            elapsed_time = end_time - start_time
            if response_ms is not None:
                server_elapsed_time = response_ms / 1000.0
                if server_elapsed_time > 0:
                    tps = completion_tokens / server_elapsed_time
            elif elapsed_time > 0:
                # Use client-side timing as fallback, less accurate due to network latency
                 tps = completion_tokens / elapsed_time
                 # Optional: apply a heuristic correction factor if needed, but server time is better
                 # correction_factor = 1.28 # Example factor, might need tuning
                 # tps *= correction_factor

            stats_text = (
                f"TPS: {tps:.2f}, "
                f"Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, "
                f"Temp: {validated_temp:.1f}, "
                f"Model: {modified_model}" # Display the actual model used
            )
            if pdf_engine != "auto":
                 stats_text += f", PDF Engine: {pdf_engine}"


            # Fetch credits information AFTER the main request
            credits_text = self.fetch_credits(api_key)

            # Save conversation in chat mode
            if chat_mode and session_path:
                # Append assistant's response to the conversation
                assistant_message = {
                    "role": "assistant",
                    "content": text_output
                }
                messages.append(assistant_message)
                
                # Save the updated conversation
                self.chat_manager.save_conversation(session_path, messages)

            return (text_output, image_tensor, stats_text, credits_text)

        except RuntimeError:
            # RuntimeError из retry-loop пробрасываем как есть — это финальное
            # состояние после исчерпания попыток или non-retryable ошибки.
            # ComfyUI/ComfyDeploy увидят exception и корректно пометят run failed.
            raise
        except Exception as e:
            # Не-сетевые ошибки парсинга/обработки ответа — тоже raise, НЕ silent
            # fallback на placeholder. Лучше честный fail чем мусор на выходе.
            print(f"ERROR: Node Error: {str(e)}")
            raise RuntimeError(f"OpenRouter node failed: {str(e)}") from e

    @staticmethod
    def image_to_base64(image):
        """
        Converts a ComfyUI IMAGE (torch.Tensor, BHWC, float 0-1)
        into a base64-encoded PNG string.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input 'image' is not a torch.Tensor")

        # Remove batch dimension if present
        if image.ndim == 4:
            if image.shape[0] != 1:
                 print(f"Warning: Image batch size is {image.shape[0]}, using only the first image.")
            image = image.squeeze(0) # Shape HWC

        if image.ndim != 3:
             raise ValueError(f"Unexpected image dimensions: {image.shape}. Expected HWC.")

        # Convert float tensor (0-1) to numpy array (0-255, uint8)
        image_np = image.cpu().numpy()
        if image_np.dtype != np.uint8:
             if image_np.min() < 0 or image_np.max() > 1:
                  print("Warning: Image tensor values outside [0, 1] range. Clamping.")
                  image_np = np.clip(image_np, 0, 1)
             image_np = (image_np * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_np, 'RGB') # Assuming RGB, adjust if needed

        # Save PIL Image to a bytes buffer as PNG
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")

        # Encode the bytes buffer to base64 string
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def base64_to_image(base64_str: str) -> torch.Tensor:
        """
        Converts a base64 image string to a ComfyUI image tensor
        Returns tensor in [1, H, W, 3] format with values in [0, 1]
        """
        try:
            # Decode base64 string to image
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            img = img.convert("RGB")

            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Add batch dimension: [1, H, W, 3]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            print(f"Successfully converted base64 to image tensor: {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            print(f"Error in base64_to_image: {e}")
            # Return a small placeholder image instead of failing
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    @staticmethod
    def count_tokens(text, model):
        """
        Count tokens for a given text using tiktoken.
        Uses model-specific encodings where possible, falls back to cl100k_base.
        Handles potential errors during encoding.
        """
        if not text or not isinstance(text, str):
            return 0

        # Strip any model modifiers like :floor, :nitro, :online
        base_model = model.split(':')[0] if ':' in model else model

        # Simplified mapping, cl100k_base is common for many recent models
        encoding_name = "cl100k_base"
        try:
            # List known models/prefixes that definitely use cl100k_base
            # Add others if known, but cl100k_base is a safe default for many
            cl100k_models = [
                "openai/gpt-4", "openai/gpt-3.5", "openai/gpt-4o",
                "anthropic/claude",
                "google/gemini",
                "meta-llama/llama-2", "meta-llama/llama-3",
                "mistralai/mistral", "mistralai/mixtral",
            ]
            # Check if the base_model or its prefix matches known cl100k models
            is_cl100k = any(base_model.startswith(prefix) for prefix in cl100k_models)

            if is_cl100k:
                 encoding_name = "cl100k_base"
            # else: # Add logic for other encodings if needed, e.g., p50k_base for older models
            #    pass # Stick with cl100k_base as default for now

            encoding = tiktoken.get_encoding(encoding_name)
            token_count = len(encoding.encode(text, disallowed_special=())) # Allow special tokens
            return token_count

        except Exception as e:
            print(f"Warning: Tiktoken error for model '{model}' (base: '{base_model}', encoding: '{encoding_name}'): {e}. Falling back to estimation.")
            # Fallback: Estimate tokens based on characters (rough approximation)
            # Average ~4 chars per token is a common heuristic
            return max(1, round(len(text) / 4))


    @classmethod
    def IS_CHANGED(cls, api_key, system_prompt, user_message_box, model,
                   web_search, cheapest, fastest, temperature, pdf_engine, chat_mode,
                   aspect_ratio="auto", image_resolution="auto", seed=0,
                   pdf_data=None, user_message_input=None,
                   image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                   **kwargs):
        # Сливаем 5 статических image-слотов в kwargs для единого хеширования цикла image_*
        for _name, _img in (("image_1", image_1), ("image_2", image_2), ("image_3", image_3), ("image_4", image_4), ("image_5", image_5)):
            if _img is not None:
                kwargs[_name] = _img
        """
        Check if any input that affects the output has changed.
        Includes hashing image and PDF data.
        """
        # Hash image data if present - handle multiple images from kwargs
        image_hashes = []
        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')], 
                           key=lambda x: int(x.split('_')[1]))
        
        for image_key in image_keys:
            if kwargs[image_key] is not None:
                image = kwargs[image_key]
                if isinstance(image, torch.Tensor):
                    try:
                        hasher = hashlib.sha256()
                        hasher.update(image.cpu().numpy().tobytes())
                        image_hashes.append(hasher.hexdigest())
                    except Exception as e:
                        print(f"Warning: Could not hash {image_key} data for IS_CHANGED: {e}")
                        image_hashes.append(f"{image_key}_hashing_error")
                else:
                    image_hashes.append(None)


        # Hash PDF data if present and valid
        pdf_hash = None
        if pdf_data is not None and isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
             try:
                 hasher = hashlib.sha256()
                 hasher.update(pdf_data["bytes"])
                 pdf_hash = hasher.hexdigest()
                 # Optionally include filename in hash if it affects processing?
                 # if "filename" in pdf_data: hasher.update(pdf_data["filename"].encode())
             except Exception as e:
                 print(f"Warning: Could not hash pdf data for IS_CHANGED: {e}")
                 pdf_hash = "pdf_hashing_error" # Use a placeholder on error
        elif pdf_data is not None:
             # Handle cases where pdf_data is present but not in the expected format
             pdf_hash = "invalid_pdf_data_format"


        # Ensure temperature is consistently represented (e.g., as float)
        try:
            temp_float = float(temperature) if isinstance(temperature, (str, int, float)) else 1.0
            temp_float = max(0.0, min(2.0, temp_float))
        except (ValueError, TypeError):
            temp_float = 1.0


        # Combine all relevant inputs into a tuple for comparison
        # Use primitive types where possible for reliable hashing/comparison
        return (api_key, system_prompt, user_message_box, model,
                web_search, cheapest, fastest, temp_float, pdf_engine, chat_mode,
                aspect_ratio, image_resolution, seed, tuple(image_hashes), pdf_hash, user_message_input)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "OpenRouterNode": OpenRouterNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterNode": "OpenRouter LLM Node (Text/Multi-Image/PDF/Chat)" # Updated name
}

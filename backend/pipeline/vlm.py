"""Описание техники через VLM (Claude): тип, марка, модель, страна, цвет, загрузка, прицеп.

Номер VLM НЕ доверяем — его читает отдельный распознаватель (plates.py).
VLM даёт только описание внешности и оценку марки/модели.
Модуль необязательный: без доступа к API пайплайн работает, поля остаются пустыми.

Экономия: один запрос на кадр (только машина на весах), ответы кэшируются на диске
по хэшу изображения — повторный прогон тех же кадров не стоит денег.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
from pathlib import Path

import cv2
import numpy as np

from backend import config

PROMPT_VERSION = "v2"
CACHE_PATH = Path(config.RESULTS_DIR) / "vlm_cache.json"

_client = None
_available: bool | None = None
# Последняя ошибка обращения к VLM (для панели статуса); сбрасывается при успешном ответе.
last_error: str | None = None
_cache: dict | None = None
_cache_lock = threading.Lock()

SYSTEM = (
    "Ты эксперт по грузовой и сельскохозяйственной технике стран СНГ. "
    "Тебе показывают вырезку с камеры весовой площадки. Отвечай только JSON без пояснений. "
    "Если признак не виден или ты не уверен — ставь null или «низкая», не выдумывай."
)

PROMPT = """Опиши технику на изображении. Верни JSON с полями:
{
  "equipment_type": "самосвал | бортовой грузовик | седельный тягач | трактор | комбайн | автобус | легковой | погрузчик | прицеп | другое",
  "manufacturer": "производитель или null",
  "model": "модель/семейство или null",
  "country": "страна производителя или null",
  "manufacturer_confidence": "высокая | средняя | низкая",
  "model_confidence": "высокая | средняя | низкая",
  "color": "основной цвет кабины и кузова",
  "has_trailer": true/false,
  "load_state": "гружёный | пустой | не видно",
  "cargo": "что в кузове, если видно, иначе null",
  "view": "спереди | сзади | сбоку",
  "appearance": "2-3 предложения: отличительные признаки, по которым машину можно узнать без номера (цвет, кузов, повреждения, надписи, надстройки)"
}
Если подсказки по тексту с кадра есть, учитывай их: {hints}"""


def _anthropic_credentials() -> bool:
    import os
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # Профиль `ant auth login` лежит в ~/.config/anthropic
    return (Path.home() / ".config" / "anthropic").exists()


def provider() -> str | None:
    """Какой провайдер использовать: 'openrouter', 'anthropic' или None."""
    want = config.VLM_PROVIDER.lower()
    if want == "openrouter":
        return "openrouter" if config.OPENROUTER_API_KEY else None
    if want == "anthropic":
        return "anthropic" if _anthropic_credentials() else None
    if config.OPENROUTER_API_KEY:
        return "openrouter"
    if _anthropic_credentials():
        return "anthropic"
    return None


def model_name() -> str:
    return config.OPENROUTER_MODEL if provider() == "openrouter" else config.VLM_MODEL


def enabled() -> bool:
    global _available
    if config.VLM_ENABLED.lower() in ("0", "false", "off", "no"):
        return False
    if _available is False:      # ключ отвергнут — больше не пробуем до перезапуска
        return False
    if config.VLM_ENABLED.lower() in ("1", "true", "on", "yes"):
        return True
    if _available is None:
        _available = provider() is not None
    return _available


def status() -> dict:
    """Состояние VLM для интерфейса: включена ли, провайдер, модели и последняя ошибка (без ключей)."""
    prov = provider()
    return {
        "enabled": enabled(),
        "provider": prov,
        "model": model_name() if prov else None,
        "fallback": (config.OPENROUTER_FALLBACK or None) if prov == "openrouter" else None,
        "last_error": last_error,
    }


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _encode_jpeg(img_bgr: np.ndarray, max_side: int = 1568) -> bytes:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _extract_json(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
        except (OSError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _call_openrouter(jpeg: bytes, prompt: str) -> tuple[str, str] | None:
    """OpenRouter (OpenAI-совместимый API). Возвращает (текст ответа, модель) или None."""
    import urllib.error
    import urllib.request
    global _available, last_error

    body = {
        "model": config.OPENROUTER_MODEL,
        "models": [m for m in (config.OPENROUTER_MODEL, config.OPENROUTER_FALLBACK) if m],
        "max_tokens": 1500,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("utf-8")}},
                {"type": "text", "text": prompt},
            ]},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json",
                 "HTTP-Referer": "http://localhost:8000", "X-Title": "Weighbridge AI (hackathon)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")
        except Exception:
            detail = ""
        if e.code == 401:
            _available = False   # ключ не принят
            last_error = "OpenRouter 401: ключ не принят"
        elif e.code == 403:
            # 403 бывает и от модерации конкретного кадра — тогда ключ рабочий, не отключаем.
            if "moderation" in detail.lower() or "flagged" in detail.lower():
                last_error = "OpenRouter 403: кадр отклонён модерацией"
            else:
                _available = False
                last_error = "OpenRouter 403: доступ запрещён"
        elif e.code == 402:
            # Баланс могут пополнить без перезапуска — не отключаем.
            last_error = "OpenRouter 402: недостаточно средств на балансе"
        elif e.code == 429:
            last_error = "OpenRouter 429: лимит запросов"
        elif e.code >= 500:
            last_error = f"OpenRouter {e.code}: ошибка сервера"
        else:
            last_error = f"OpenRouter {e.code}"
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        last_error = "нет сети или таймаут OpenRouter"
        return None
    except ValueError:
        last_error = "OpenRouter: некорректный ответ"
        return None
    try:
        content = data["choices"][0]["message"].get("content") or ""
        if isinstance(content, list):   # некоторые модели отдают список блоков
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content, data.get("model") or config.OPENROUTER_MODEL
    except (KeyError, IndexError, TypeError, AttributeError):
        last_error = "OpenRouter: неожиданный формат ответа"
        return None


def describe_vehicle(crop_bgr: np.ndarray, hints: str = "") -> dict | None:
    """Возвращает dict с описанием или None, если VLM недоступна/ошиблась."""
    if not enabled():
        return None

    jpeg = _encode_jpeg(crop_bgr)
    prompt = PROMPT.replace("{hints}", hints or "нет")
    key = hashlib.sha1(jpeg + f"|{PROMPT_VERSION}|{model_name()}|{config.VLM_EFFORT}|{hints}".encode()).hexdigest()
    with _cache_lock:
        cached = _load_cache().get(key)
    if cached:
        return {**cached, "_cached": True}

    global _available, last_error
    if provider() == "openrouter":
        # VLM необязательна: любая ошибка здесь не должна ронять обработку кадра.
        try:
            result = _call_openrouter(jpeg, prompt)
            if not result:
                return None
            text, used_model = result
            data = _extract_json(text)
            if not isinstance(data, dict):
                last_error = "OpenRouter: в ответе нет JSON"
                return None
            last_error = None
            data["_model"] = used_model
            with _cache_lock:
                _load_cache()[key] = data
                _save_cache()
            return data
        except Exception as e:
            last_error = f"OpenRouter: {type(e).__name__}"
            return None

    try:
        client = _get_client()
    except Exception as e:
        # Ключа нет или он негоден: пайплайн работает и без описания внешности,
        # ронять из-за этого обработку 22 тысяч кадров нельзя.
        global _available
        _available = False
        last_error = f"Anthropic: {type(e).__name__}"
        return None
    try:
        response = client.beta.messages.create(
            model=config.VLM_MODEL,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": config.VLM_EFFORT},
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(jpeg).decode("utf-8")}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except anthropic.AuthenticationError:
        _available = False
        last_error = "Anthropic 401: ключ не принят"
        return None
    except anthropic.APIStatusError as e:
        last_error = f"Anthropic {e.status_code}"
        return None
    except anthropic.APIConnectionError:
        last_error = "нет сети"
        return None

    if response.stop_reason in ("refusal", "max_tokens"):
        last_error = f"Anthropic: ответ прерван ({response.stop_reason})"
        return None
    text = "".join(b.text for b in response.content if b.type == "text")
    data = _extract_json(text)
    if data:
        last_error = None
        data["_model"] = response.model
        with _cache_lock:
            _load_cache()[key] = data
            _save_cache()
    return data

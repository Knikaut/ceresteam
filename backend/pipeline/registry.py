"""Поиск марки/модели/года по госномеру в открытых данных РК (data.egov.kz).

Набор «Зарегистрированные транспортные средства»: поля reg_number, marka, model,
year_issue, type, number_doc. Два способа доступа:
1. Официальный API v4 — нужен бесплатный ключ (переменная EGOV_API_KEY, выдаётся
   после регистрации на data.egov.kz).
2. Без ключа — тот же поисковый запрос, которым пользуется страница набора.

Набор неполный (снимок регистраций 2022 г.), поэтому «не найдено» — обычный ответ.
Ошибки сети глушатся: без интернета пайплайн работает как раньше.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import os
import urllib.request
from pathlib import Path

from backend import config

# Статусы ответа lookup_status(): различаем «нет в наборе» и «не смогли проверить».
FOUND = "found"
NOT_FOUND = "not_found"
UNAVAILABLE = "unavailable"   # нет сети / таймаут / ошибка сервера
DISABLED = "disabled"

# После сбоя сети не ходим в интернет столько секунд: иначе каждый кадр ждал бы таймауты.
OFFLINE_BACKOFF = 60
_offline_until = 0.0

DATASET = "registered_vehicles"
VERSION = "v2"
API_URL = f"https://data.egov.kz/api/v4/{DATASET}/{VERSION}"
PAGE_URL = f"https://data.egov.kz/datasets/getdata?index={DATASET}&version={VERSION}"
TIMEOUT = 8
CACHE_PATH = Path(config.RESULTS_DIR) / "registry_cache.json"
SOURCE_NAME = "data.egov.kz (открытые данные, «Зарегистрированные транспортные средства»)"

# Коды типов ТС в наборе (по наблюдаемым данным: 2 — легковые, 3 — грузовые).
TYPE_CODES = {"1": "мототранспорт", "2": "легковой автомобиль", "3": "грузовой автомобиль",
              "4": "автобус", "5": "прицеп", "6": "спецтехника"}

_cache: dict[str, dict | None] | None = None

# Снимок набора, скачанный tools/fetch_registry.py: поиск идёт по нему, без интернета.
SNAPSHOT_NAME = "registry/registered_vehicles.jsonl"
# «Не найден», полученный из сети, живёт сутки: набор обновляется, а вечный отказ
# запоминал бы и случайную ошибку поиска.
NEGATIVE_TTL = 24 * 3600
_snapshot: dict[str, dict] | None = None


def snapshot_path() -> Path | None:
    """Путь считаем каждый раз: RESULTS_DIR подменяется в тестах и задаётся окружением."""
    env = os.environ.get("REGISTRY_SNAPSHOT")
    candidates = [Path(env)] if env else []
    candidates += [Path(config.ROOT) / "data" / SNAPSHOT_NAME, Path(config.RESULTS_DIR) / SNAPSHOT_NAME]
    return next((p for p in candidates if p.exists()), None)


def _load_snapshot() -> dict[str, dict]:
    """Номер -> строка набора. Читается один раз, это около 25 тысяч записей и 5 МБ."""
    global _snapshot
    if _snapshot is None:
        _snapshot = {}
        path = snapshot_path()
        if path:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = _normalize_plate(str(row.get("reg_number", "")))
                    if key:
                        _snapshot.setdefault(key, row)
    return _snapshot


def enabled() -> bool:
    return config.REGISTRY_ENABLED.lower() not in ("0", "false", "off", "no")


def _cached(value) -> tuple[dict | None, float | None]:
    """Разбирает запись кэша: (запись или None, когда записан отрицательный ответ)."""
    if isinstance(value, dict) and value.get("__missing__"):
        return None, value.get("at")
    return (value or None), None


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


def _get_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={
        "User-Agent": "weighbridge-hackathon/0.1",
        "Accept": "application/json, text/javascript, */*",
        "X-Requested-With": "XMLHttpRequest",   # без этого страница отдаёт HTML вместо JSON
        "Referer": f"https://data.egov.kz/datasets/view?index={DATASET}",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
        return None


def _normalize_plate(plate: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _split_make_model(marka: str | None, model: str | None) -> tuple[str | None, str | None]:
    """В наборе часто marka == model == «HYUNDAI SANTA FE». Отделяем производителя."""
    marka, model = _clean(marka), _clean(model)
    if not marka and not model:
        return None, None
    full = model or marka
    words = full.split()
    if marka and model and marka.upper() != model.upper() and len(marka.split()) <= 2:
        return marka, model
    manufacturer = words[0] if words else marka
    rest = " ".join(words[1:]) if len(words) > 1 else None
    return manufacturer, rest or full


def _to_record(row: dict, plate: str) -> dict:
    manufacturer, model = _split_make_model(row.get("marka"), row.get("model"))
    return {
        "plate": plate,
        "manufacturer": manufacturer,
        "model": model,
        "marka_raw": _clean(row.get("marka")),
        "model_raw": _clean(row.get("model")),
        "year": _clean(str(row.get("year_issue") or "")),
        "type_code": _clean(str(row.get("type") or "")),
        "type": TYPE_CODES.get(str(row.get("type") or ""), None),
        "source": SOURCE_NAME,
    }


def _query_api(plate: str) -> list[dict] | None:
    """Официальный API v4 (нужен ключ). Фильтруем точным совпадением на своей стороне."""
    source = json.dumps({"size": 20, "query": {"match": {"reg_number": plate}}})
    url = f"{API_URL}?{urllib.parse.urlencode({'apiKey': config.EGOV_API_KEY, 'source': source})}"
    data = _get_json(url)
    if data is None:
        return None
    if isinstance(data, dict):
        rows = data.get("elements") or data.get("data") or data.get("hits") or []
    else:
        rows = data
    return [r for r in rows if isinstance(r, dict)]


def _query_page(plate: str) -> list[dict] | None:
    """Поиск без ключа — тем же запросом, что делает страница набора данных."""
    url = f"{PAGE_URL}&{urllib.parse.urlencode({'page': 1, 'count': 20, 'text': plate, 'column': 'id', 'order': 'ascending'})}"
    data = _get_json(url)
    if not isinstance(data, dict):
        return None
    return [r for r in data.get("elements", []) if isinstance(r, dict)]


def lookup_status(plate: str | None, use_cache: bool = True) -> tuple[str, dict | None]:
    """Возвращает (статус, запись). Статус: found / not_found / unavailable / disabled."""
    global _offline_until
    if not enabled():
        return DISABLED, None
    key = _normalize_plate(plate or "")
    if not key:
        return NOT_FOUND, None
    snapshot = _load_snapshot()
    if key in snapshot:
        return FOUND, _to_record(snapshot[key], key)

    cache = _load_cache()
    if use_cache and key in cache:
        record, stamped = _cached(cache[key])
        # Отрицательный ответ из сети держим сутки, найденную запись — бессрочно.
        if record or stamped is None or time.time() - stamped < NEGATIVE_TTL:
            return (FOUND if record else NOT_FOUND), record
    if snapshot:
        # Снимок набора есть и номера в нём нет — это и есть ответ, в сеть не идём.
        return NOT_FOUND, None
    if time.monotonic() < _offline_until:
        return UNAVAILABLE, None

    rows = _query_api(key) if config.EGOV_API_KEY else None
    if rows is None:
        rows = _query_page(key)
    if rows is None:
        # Сеть недоступна: не кэшируем результат, но какое-то время не пробуем снова.
        _offline_until = time.monotonic() + OFFLINE_BACKOFF
        return UNAVAILABLE, None
    _offline_until = 0.0

    match = next((r for r in rows if _normalize_plate(str(r.get("reg_number", ""))) == key), None)
    record = _to_record(match, key) if match else None
    cache[key] = record if record else {"__missing__": True, "at": time.time()}
    _save_cache()
    return (FOUND if record else NOT_FOUND), record

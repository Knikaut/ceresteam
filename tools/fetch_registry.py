#!/usr/bin/env python3
"""Выкачивает набор data.egov.kz «Зарегистрированные транспортные средства» целиком.

Сайт отдаёт максимум 20 записей за запрос, поэтому набор (~24 759 записей) берётся
постранично с сортировкой по id: так нет пропусков и дублей, в отличие от выгрузки
с сайта по 100 штук вперемешку.

Запуск:
    python tools/fetch_registry.py --out results/registry/registered_vehicles.jsonl

Повторный запуск продолжает с того места, где остановился (файл дописывается).
Только стандартная библиотека: работает и в контейнере организаторов без pip.

Источник данных: интернет-портал «Открытые данные» Республики Казахстан, data.egov.kz,
набор «Зарегистрированные транспортные средства». Данные сохраняются без изменений.
Пользовательское соглашение портала ограничивает частоту обращений: не более 40 запросов
в минуту, поэтому пауза между запросами по умолчанию 1.6 с (--delay менять только в
большую сторону). Ссылка на источник обязательна в проектах, использующих набор.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "registered_vehicles"
VERSION = "v2"
PAGE_URL = f"https://data.egov.kz/datasets/getdata?index={DATASET}&version={VERSION}"
PAGE_SIZE = 20          # больше сервер всё равно не отдаёт
HEADERS = {
    "User-Agent": "weighbridge-hackathon/0.1",
    "Accept": "application/json, text/javascript, */*",
    "X-Requested-With": "XMLHttpRequest",   # без него страница отдаёт HTML вместо JSON
    "Referer": f"https://data.egov.kz/datasets/view?index={DATASET}",
}


def ssl_context() -> ssl.SSLContext:
    """На macOS Python нередко ставится без корневых сертификатов — берём их из certifi."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_CTX = None


def fetch_page(page: int, timeout: float, retries: int = 4) -> dict | None:
    """Одна страница выдачи. None — если сервер так и не ответил."""
    url = f"{PAGE_URL}&" + urllib.parse.urlencode(
        {"page": page, "count": PAGE_SIZE, "column": "id", "order": "ascending"})
    global _CTX
    if _CTX is None:
        _CTX = ssl_context()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as e:
            if attempt == retries - 1:
                print(f"  страница {page}: не удалось ({e})", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/registry/registered_vehicles.jsonl", help="файл JSONL")
    ap.add_argument("--delay", type=float, default=1.6,
                    help="пауза между запросами, с (лимит портала — 40 запросов в минуту)")
    ap.add_argument("--timeout", type=float, default=20, help="таймаут запроса, с")
    ap.add_argument("--limit-pages", type=int, default=0, help="взять только N страниц (для проверки)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Сколько записей уже лежит — столько страниц пропускаем.
    # Если файл оборван на середине страницы, она будет перечитана целиком — дубли уберём в конце.
    have = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    start_page = have // PAGE_SIZE + 1

    first = fetch_page(1, args.timeout)
    if not first:
        print("Сервер не отвечает, попробуйте позже", file=sys.stderr)
        return 1
    total = int(first.get("totalCount") or 0)
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    if args.limit_pages:
        pages = min(pages, args.limit_pages)
    print(f"В наборе {total} записей, это {pages} страниц по {PAGE_SIZE}. Уже есть {have}, начинаю со страницы {start_page}.")

    t0 = time.time()
    written = 0
    with out.open("a", encoding="utf-8") as f:
        for page in range(start_page, pages + 1):
            data = fetch_page(page, args.timeout)
            rows = (data or {}).get("elements") or []
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            if page % 50 == 0 or page == pages:
                done = have + written
                speed = written / max(time.time() - t0, 1)
                left = (total - done) / speed if speed else 0
                print(f"  страница {page}/{pages}: всего {done} записей, осталось ~{left / 60:.1f} мин")
            time.sleep(args.delay)

    # Дубли (после обрыва) и порядок приводим в чувство.
    seen, uniq = set(), []
    for line in out.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("id")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    uniq.sort(key=lambda r: int(r.get("id") or 0))
    with out.open("w", encoding="utf-8") as f:
        for row in uniq:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Готово: {len(uniq)} записей за {(time.time() - t0) / 60:.1f} мин -> {out}")
    if len(uniq) < total:
        print(f"Внимание: получено меньше, чем заявлено ({len(uniq)} из {total}). "
              f"Запустите скрипт ещё раз — он продолжит с места обрыва.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

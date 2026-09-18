"""Демо-история для показа: рабочий день на одном складе (по умолчанию — Склад №1 · Элеватор).

Каждое событие проходит настоящую обработку приложения (детектор, номер, марка/модель, VLM),
вес даёт имитация весов COM3, водителя и груз — имитация путевого листа. Поэтому все цифры
в сводке склада складываются из рейсов, а рейсы — из событий в чате.

Что подстраивается под показ:
* время событий — по ходу дня из плана (часы камеры на части кадров сбиты: показывают 2019 год);
* выезд снимается тем же кадром, что и заезд (пар «перед/зад» одной машины в наборе мало); ответ VLM
  по тому же снимку берётся из кэша, но его оценка груза и описание на выезде убираются — на снимке
  гружёная машина, и без этого VLM поднимал бы тревогу «на выезде кузов гружёный».

Запускать при ОСТАНОВЛЕННОМ приложении (пишет в ту же базу); база стирается, копию делайте заранее.

    python tools/seed_demo.py tools/demo_day.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config, db, service, warehouses  # noqa: E402
from backend.pipeline import analyze  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="JSON: {date, warehouse_id, events: [{frame, time, kind, note?}]}")
    ap.add_argument("--no-reset", action="store_true", help="не стирать базу перед заполнением")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    day = plan.get("date") or datetime.now().strftime("%Y-%m-%d")
    wh = plan.get("warehouse_id", "W1")
    events = sorted(plan["events"], key=lambda e: e["time"])

    db.init()
    if not args.no_reset:
        db.reset()

    real_analyze = analyze.analyze_image
    real_now = service.now_iso
    clock = {"t": None, "exit": False}

    def analyze_at(image_path, use_vlm=True, use_registry=True):
        report = real_analyze(image_path, use_vlm=use_vlm, use_registry=use_registry)
        report["timestamp_on_frame"] = clock["t"].strftime("%Y-%m-%d %H:%M:%S")   # время события — из плана
        if clock["exit"]:
            for d in report["detections"]:
                if d.get("category") == "vehicle" and d.get("appearance"):
                    d["appearance"]["load_state"] = None     # снимок заезда: груз на нём ещё виден
                    d["appearance"]["description"] = None
        return report

    def now_at():
        return (clock["t"] + timedelta(seconds=clock.get("extra", 0))).strftime("%Y-%m-%d %H:%M:%S")

    analyze.analyze_image = analyze_at
    service.now_iso = now_at
    try:
        for i, ev in enumerate(events, 1):
            clock["t"] = datetime.strptime(f"{day} {ev['time']}", "%Y-%m-%d %H:%M" if len(ev["time"]) == 5 else "%Y-%m-%d %H:%M:%S")
            clock["exit"] = ev.get("kind") == "exit"
            clock["extra"] = 0
            t0 = time.time()
            cap = service.save_capture(config.DATA_DIR / ev["frame"], ev["frame"])
            clock["extra"] = 12   # ответ ИИ приходит через несколько секунд после снимка
            res = service.process_capture(cap["capture_id"], {"warehouse_id": wh, "note": ev.get("note") or "",
                                                              "frame": ev["frame"]})
            p = res["ai_message"].get("payload") or {}
            print(f"{i:2d}. {ev['time']} {ev['frame']:>7} {p.get('event_kind') or '?':5s} "
                  f"{(p.get('vehicle') or {}).get('label', '—')} · {(p.get('plate') or {}).get('formatted') or 'номер не прочитан'} · "
                  f"вес {p.get('weight')} · тревоги {p.get('alerts') or '—'} ({time.time() - t0:.0f} с)", flush=True)
    finally:
        analyze.analyze_image = real_analyze
        service.now_iso = real_now

    s = service.warehouse_summary(warehouses.get(wh))
    print(f"\nИтог по складу {s['name']}: выгружено {s['received_t']} т, загружено {s['shipped_t']} т, "
          f"на складе {s['count_now']} машин, рейсов {s['trips_total']} (закрыто {s['trips_closed']}), "
          f"тревоги {s['alert_kinds']}, по культурам {s['by_crop']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
